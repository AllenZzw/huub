//! Engine-resident state and bounds propagation for the difference logic
//! propagator.
//!
//! The engine owns a single difference logic graph (each edge encodes
//! `x − y ≤ d`). [`crate::solver::Solver::add_diff_logic_edge`]
//! registers a constraint into this graph at lowering time; the resulting
//! state is consumed by [`DifferenceLogicBoundsShell`] during search.
//!
//! ## PR scope
//!
//! This file currently supports **globally-active** edges only (i.e.
//! constraints without a boolean gate). Boolean-gated implications,
//! lazy-literal `tighten_difference`, the dedicated brancher, and most
//! configuration knobs land in subsequent PRs and are not represented in
//! the state struct yet. [`DiffLogicState::register_edge`] panics if it
//! is asked to register a gated edge.
//!
//! ## Architecture
//!
//! - [`DiffLogicState`] lives on [`State`]. It owns the master edge list, the
//!   per-node active-edge adjacency lists, the Johnson potential `pi`, and the
//!   per-node bound shadow used for incremental Dijkstra.
//! - [`DifferenceLogicBoundsShell`] is a trampoline propagator with no
//!   per-instance state; its `propagate` body does a [`mem::take`] swap on
//!   `state.diff_logic`, hands off to [`DiffLogicState::propagate_bounds`], and
//!   writes the state back. This avoids holding two simultaneous mutable
//!   borrows into [`State`].
//!
//! ## Known limitations (fixed alongside `tighten_difference`)
//!
//! - **Late-endpoint advisor-subscription gap.** The bounds shell's
//!   `initialize` fires once, at the moment the first edge auto-posts the
//!   shell. Only endpoints already in `state.diff_logic.int_vars` at that
//!   instant get a bounds advisor. Any endpoint introduced by a *later*
//!   [`crate::solver::Solver::add_diff_logic_edge`] call is interned in the
//!   graph but is never wired to wake the shell on bound changes. The
//!   propagator stays correct (it cannot prove false), but it becomes
//!   incomplete for workloads where mid-search SAT decisions tighten the
//!   late-interned endpoint's bounds. The fix is a `subscribe_int_bounds_advisor`
//!   helper on [`crate::solver::Solver`] (~15 lines) plus a lowering-time
//!   pre-pass that interns every endpoint and posts the shell before any
//!   edge registration. Both pieces are also prerequisites for mid-search
//!   `tighten_difference`, so they land together.
//! - **No mid-search endpoint introduction.** Same root cause as above: the
//!   shell carries no protocol to subscribe new advisors after `initialize`.
//!   Once `tighten_difference` exists, the same `subscribe_int_bounds_advisor`
//!   helper handles the case where the asserted difference's endpoints were
//!   not previously in the graph.

use std::{cmp::Reverse, mem};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
	IntVal,
	actions::{
		BoolInspectionActions, InitActions, IntDecisionActions, IntEvent, IntInitActions,
		IntInspectionActions, IntPropCond, IntPropagationActions, PropagationActions,
		ReasoningContext,
	},
	constraints::{Conflict, Propagator},
	helpers::{priority_queue::LazyPriorityQueue, trailed_list::TrailedList},
	solver::{
		Decision, IntLitMeaning, View,
		engine::{Engine, PropRef, State},
		queue::PriorityLevel,
		solving_context::SolvingContext,
	},
};

/// An edge in the engine-resident difference logic graph.
///
/// Represents the constraint `int_vars[from] − int_vars[to] ≤ val`. Edges
/// are globally active in this PR; the `bool_var` slot is retained for
/// compatibility with the planned gated-edge support but is always
/// `None` here.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DiffEdge {
	pub(crate) from: usize,
	pub(crate) to: usize,
	pub(crate) val: IntVal,
	pub(crate) bool_var: Option<usize>,
}

/// Difference logic state owned by the engine.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DiffLogicState {
	// ---- Graph identity & lookup ----
	pub(crate) int_var_to_node: FxHashMap<View<IntVal>, usize>,
	pub(crate) int_vars: Vec<View<IntVal>>,
	pub(crate) edges: Vec<DiffEdge>,

	// ---- Per-node active edge adjacency ----
	/// Active outgoing edges. The trailed length lets a future PR shrink
	/// the active set on backtrack when gated edges become dormant;
	/// global edges in this PR never go inactive.
	pub(crate) active_out: Vec<TrailedList<usize>>,
	pub(crate) active_in: Vec<TrailedList<usize>>,

	// ---- Per-node algorithm working buffers (not trailed) ----
	/// Johnson's potential per node. Computed once on the first call to
	/// [`Self::propagate_bounds`] via one Bellman-Ford pass and not
	/// refreshed afterwards.
	pub(crate) pi: Vec<IntVal>,
	pub(crate) lower_bound: Vec<Option<IntVal>>,
	pub(crate) upper_bound: Vec<Option<IntVal>>,
	/// Predecessor node on the current Dijkstra pass, used to attribute
	/// reasons to a source node.
	pub(crate) backtrace: Vec<Option<usize>>,
	pub(crate) visited: Vec<bool>,

	// ---- Propagation transient scratch (not trailed; cleared each call) ----
	pub(crate) visited_updates: Vec<usize>,
	pub(crate) lower_bound_changes: FxHashSet<usize>,
	pub(crate) upper_bound_changes: FxHashSet<usize>,
	pub(crate) lb_updates: Vec<usize>,
	pub(crate) ub_updates: Vec<usize>,

	// ---- Engine integration ----
	pub(crate) pi_initialized: bool,
	/// Priority level at which to schedule the bounds shell. `None` falls
	/// back to [`PriorityLevel::Medium`]. Threaded in from
	/// `Lowerer::diff_logic_prio_bounds`.
	pub(crate) priority_bounds: Option<PriorityLevel>,
	/// Whether the bounds shell has been registered with the engine
	/// already. Set on the first `Solver::add_diff_logic_edge`.
	pub(crate) propagators_registered: bool,
	/// `PropRef` of the bounds propagator shell, after auto-registration.
	pub(crate) bounds_ref: Option<PropRef>,
}

impl DiffLogicState {
	/// Find or create the node index for an integer decision.
	pub(crate) fn intern_int(
		&mut self,
		trail: &mut crate::solver::trail::Trail,
		x: View<IntVal>,
	) -> usize {
		if let Some(&n) = self.int_var_to_node.get(&x) {
			return n;
		}
		let n = self.int_vars.len();
		self.int_vars.push(x);
		let _ = self.int_var_to_node.insert(x, n);
		self.active_out.push(TrailedList::new(trail, false));
		self.active_in.push(TrailedList::new(trail, false));
		self.pi.push(0);
		self.lower_bound.push(None);
		self.upper_bound.push(None);
		self.backtrace.push(None);
		self.visited.push(false);
		n
	}

	/// Register a difference constraint `x − y ≤ d` in the engine graph.
	pub(crate) fn register_edge(
		&mut self,
		trail: &mut crate::solver::trail::Trail,
		x: View<IntVal>,
		y: View<IntVal>,
		d: IntVal,
		gate: Option<View<bool>>,
	) -> usize {
		assert!(
			gate.is_none(),
			"diff-logic: boolean-gated edges are not supported in this build"
		);
		let from = self.intern_int(trail, x);
		let to = self.intern_int(trail, y);
		let edge = DiffEdge {
			from,
			to,
			val: d,
			bool_var: None,
		};
		let idx = self.edges.len();
		self.active_out[from].push(trail, idx);
		self.active_in[to].push(trail, idx);
		self.edges.push(edge);
		idx
	}

	// ---- Visit bookkeeping ----

	fn visit(&mut self, n: usize) {
		if !self.visited[n] {
			self.visited_updates.push(n);
		}
		self.visited[n] = true;
	}

	fn reset_visit(&mut self) {
		for &n in self.visited_updates.iter() {
			self.visited[n] = false;
		}
		self.visited_updates.clear();
	}

	// ---- Bound shadow helpers ----

	fn get_cur_lower_bound<Ctx>(&self, ctx: &Ctx, n: usize) -> IntVal
	where
		Ctx: ReasoningContext + ?Sized,
		View<IntVal>: IntInspectionActions<Ctx>,
	{
		match self.lower_bound[n] {
			Some(lb) => lb,
			None => self.int_vars[n].min(ctx),
		}
	}

	fn update_lb(&mut self, n: usize, val: IntVal) {
		if self.lower_bound[n].is_none() {
			self.lb_updates.push(n);
		}
		self.lower_bound[n] = Some(val);
	}

	fn get_cur_upper_bound<Ctx>(&self, ctx: &Ctx, n: usize) -> IntVal
	where
		Ctx: ReasoningContext + ?Sized,
		View<IntVal>: IntInspectionActions<Ctx>,
	{
		match self.upper_bound[n] {
			Some(ub) => ub,
			None => self.int_vars[n].max(ctx),
		}
	}

	fn update_ub(&mut self, n: usize, val: IntVal) {
		if self.upper_bound[n].is_none() {
			self.ub_updates.push(n);
		}
		self.upper_bound[n] = Some(val);
	}

	pub(crate) fn notify_lb_change<Ctx>(&mut self, ctx: &Ctx, n: usize) -> bool
	where
		Ctx: ReasoningContext + ?Sized,
		View<IntVal>: IntInspectionActions<Ctx>,
	{
		if self.lower_bound[n].is_none_or(|v| v < self.int_vars[n].min(ctx)) {
			return self.lower_bound_changes.insert(n);
		}
		false
	}

	pub(crate) fn notify_ub_change<Ctx>(&mut self, ctx: &Ctx, n: usize) -> bool
	where
		Ctx: ReasoningContext + ?Sized,
		View<IntVal>: IntInspectionActions<Ctx>,
	{
		if self.upper_bound[n].is_none_or(|v| v > self.int_vars[n].max(ctx)) {
			return self.upper_bound_changes.insert(n);
		}
		false
	}

	pub(crate) fn reset_bounds(&mut self) {
		self.lower_bound_changes.clear();
		self.upper_bound_changes.clear();
		for &n in self.lb_updates.iter() {
			self.lower_bound[n] = None;
		}
		for &n in self.ub_updates.iter() {
			self.upper_bound[n] = None;
		}
		self.lb_updates.clear();
		self.ub_updates.clear();
	}

	// ---- Reason builders (with SAT-vs-huub-lag walk) ----

	/// Walk from the propagation-time bound `lb_val` toward the *currently*
	/// SAT-true lower bound, returning the strongest literal `view >= w`
	/// (with `lb_val <= w <= cur_min`) that is presently assigned true.
	///
	/// SAT-side defining-clause propagation lags huub-level propagation.
	/// On a trail rewound by conflict analysis, the literal at the exact
	/// reason level may still be unassigned even though a stronger literal
	/// is true; the reason atom therefore needs to be the strongest
	/// currently-true relaxation, not the exact level captured at
	/// propagation time.
	fn relaxed_ge_lit(
		ctx: &mut SolvingContext<'_>,
		view: View<IntVal>,
		lb_val: IntVal,
	) -> View<bool> {
		let cur_min = view.min(ctx);
		let mut level = lb_val;
		while level <= cur_min {
			let lit = view.lit(ctx, IntLitMeaning::GreaterEq(level));
			if lit.val(ctx) == Some(true) {
				return lit;
			}
			level += 1;
		}
		view.lit(ctx, IntLitMeaning::GreaterEq(lb_val))
	}

	/// Mirror of [`Self::relaxed_ge_lit`] for `view < strict_ub`.
	fn relaxed_lt_lit(
		ctx: &mut SolvingContext<'_>,
		view: View<IntVal>,
		strict_ub: IntVal,
	) -> View<bool> {
		let cur_max = view.max(ctx);
		let mut level = strict_ub;
		while level > cur_max {
			let lit = view.lit(ctx, IntLitMeaning::Less(level));
			if lit.val(ctx) == Some(true) {
				return lit;
			}
			level -= 1;
		}
		view.lit(ctx, IntLitMeaning::Less(strict_ub))
	}

	fn set_int_lower_bound(
		&mut self,
		ctx: &mut SolvingContext<'_>,
		n: usize,
		value: IntVal,
		lb_var: usize,
		lb_val: IntVal,
	) -> Result<(), Conflict<Decision<bool>>> {
		let from_view = self.int_vars[lb_var];
		let target_view = self.int_vars[n];
		target_view.tighten_min(ctx, value, |ctx: &mut SolvingContext<'_>| {
			vec![Self::relaxed_ge_lit(ctx, from_view, lb_val)]
		})?;
		Ok(())
	}

	fn set_int_upper_bound(
		&mut self,
		ctx: &mut SolvingContext<'_>,
		n: usize,
		value: IntVal,
		ub_var: usize,
		ub_val: IntVal,
	) -> Result<(), Conflict<Decision<bool>>> {
		let from_view = self.int_vars[ub_var];
		let target_view = self.int_vars[n];
		target_view.tighten_max(ctx, value, |ctx: &mut SolvingContext<'_>| {
			vec![Self::relaxed_lt_lit(ctx, from_view, ub_val + 1)]
		})?;
		Ok(())
	}

	// ---- Incremental lower/upper bound propagation ----

	fn inc_lb(&mut self, ctx: &mut SolvingContext<'_>) -> Result<(), Conflict<Decision<bool>>> {
		self.reset_visit();
		let pi0 = self
			.lower_bound_changes
			.iter()
			.map(|&n| self.int_vars[n].min(ctx) + self.pi[n])
			.max()
			.unwrap();
		let mut queue = LazyPriorityQueue::new();
		for &n in self.lower_bound_changes.iter() {
			let _ = queue.push(n, Reverse(pi0 - self.int_vars[n].min(ctx) - self.pi[n]));
		}
		while !queue.is_empty() {
			let (s, Reverse(gamma_s)) = queue.pop().unwrap();
			self.visit(s);
			let bound = pi0 - gamma_s - self.pi[s];
			if bound > self.get_cur_lower_bound(ctx, s) || self.lower_bound_changes.contains(&s) {
				self.update_lb(s, bound);
				if bound > self.int_vars[s].min(ctx) {
					let prev = self.backtrace[s].unwrap();
					let lb = self.get_cur_lower_bound(ctx, prev);
					self.set_int_lower_bound(ctx, s, bound, prev, lb)?;
					let _ = self.lower_bound_changes.insert(s);
				}
				for &e in self.active_out[s].iter(ctx) {
					let edge = &self.edges[e];
					if !self.visited[edge.to] {
						let path = gamma_s + self.pi[s] + edge.val - self.pi[edge.to];
						let old = queue.push_increase(edge.to, Reverse(path));
						if old.is_none_or(|Reverse(old_path)| path < old_path) {
							self.backtrace[edge.to] = Some(s);
						}
					}
				}
			}
		}
		Ok(())
	}

	fn inc_ub(&mut self, ctx: &mut SolvingContext<'_>) -> Result<(), Conflict<Decision<bool>>> {
		self.reset_visit();
		let pi0 = self
			.upper_bound_changes
			.iter()
			.map(|&n| self.int_vars[n].max(ctx) + self.pi[n])
			.min()
			.unwrap();
		let mut queue = LazyPriorityQueue::new();
		for &n in self.upper_bound_changes.iter() {
			let _ = queue.push(n, Reverse(self.pi[n] + self.int_vars[n].max(ctx) - pi0));
		}
		while !queue.is_empty() {
			let (s, Reverse(gamma_s)) = queue.pop().unwrap();
			self.visit(s);
			let bound = pi0 + gamma_s - self.pi[s];
			if bound < self.get_cur_upper_bound(ctx, s) || self.upper_bound_changes.contains(&s) {
				self.update_ub(s, bound);
				if bound < self.int_vars[s].max(ctx) {
					let prev = self.backtrace[s].unwrap();
					let ub = self.get_cur_upper_bound(ctx, prev);
					self.set_int_upper_bound(ctx, s, bound, prev, ub)?;
					let _ = self.upper_bound_changes.insert(s);
				}
				for &e in self.active_in[s].iter(ctx) {
					let edge = &self.edges[e];
					if !self.visited[edge.from] {
						let path = gamma_s + self.pi[edge.from] + edge.val - self.pi[s];
						let old = queue.push_increase(edge.from, Reverse(path));
						if old.is_none_or(|Reverse(old_path)| path < old_path) {
							self.backtrace[edge.from] = Some(s);
						}
					}
				}
			}
		}
		Ok(())
	}

	/// One Bellman-Ford pass to populate the initial Johnson potentials.
	/// Returns a conflict if a negative cycle is reachable.
	fn bellman_ford_init_pi(
		&mut self,
		ctx: &mut SolvingContext<'_>,
	) -> Result<(), Conflict<Decision<bool>>> {
		let num_nodes = self.int_vars.len();
		if self.pi.len() < num_nodes {
			self.pi.resize(num_nodes, 0);
		}
		if self.lower_bound.len() < num_nodes {
			self.lower_bound.resize(num_nodes, None);
			self.upper_bound.resize(num_nodes, None);
			self.backtrace.resize(num_nodes, None);
			self.visited.resize(num_nodes, false);
		}
		let mut changed = true;
		for _ in 0..num_nodes {
			changed = false;
			for n in 0..num_nodes {
				for &e in self.active_out[n].iter(ctx) {
					let edge = &self.edges[e];
					if self.pi[edge.from] + edge.val < self.pi[edge.to] {
						self.pi[edge.to] = self.pi[edge.from] + edge.val;
						changed = true;
					}
				}
			}
			if !changed {
				break;
			}
		}
		if changed {
			for n in 0..num_nodes {
				for &e in self.active_out[n].iter(ctx) {
					let edge = &self.edges[e];
					if self.pi[edge.from] + edge.val < self.pi[edge.to] {
						return Err(ctx.declare_conflict(Vec::<View<bool>>::new()));
					}
				}
			}
		}
		Ok(())
	}

	/// Bounds propagation entry point invoked by the bounds shell.
	pub(crate) fn propagate_bounds(
		&mut self,
		ctx: &mut SolvingContext<'_>,
	) -> Result<(), Conflict<Decision<bool>>> {
		if !self.pi_initialized {
			self.bellman_ford_init_pi(ctx)?;
			self.pi_initialized = true;
			// On first run, seed every node so the initial incremental
			// pass propagates the graph-implied bounds.
			for n in 0..self.int_vars.len() {
				let _ = self.lower_bound_changes.insert(n);
				let _ = self.upper_bound_changes.insert(n);
			}
		}

		if !self.lower_bound_changes.is_empty() {
			self.inc_lb(ctx)?;
		}
		if !self.upper_bound_changes.is_empty() {
			self.inc_ub(ctx)?;
		}

		// Gated-edge closure passes that read these change sets arrive in
		// a later PR; clear them so the next pass starts fresh.
		self.lower_bound_changes.clear();
		self.upper_bound_changes.clear();
		Ok(())
	}

	pub(crate) fn advise_int_change(&mut self, ctx: &State, data: usize, event: IntEvent) -> bool {
		let mut enqueue = false;
		if event == IntEvent::LowerBound || event == IntEvent::Fixed {
			enqueue = self.notify_lb_change(ctx, data);
		}
		if event == IntEvent::UpperBound || event == IntEvent::Fixed {
			enqueue |= self.notify_ub_change(ctx, data);
		}
		enqueue
	}
}

/// Trampoline propagator that drives [`DiffLogicState::propagate_bounds`].
///
/// The shell holds no per-instance state; the graph lives on
/// [`State::diff_logic`]. Auto-registered on the first
/// `Solver::add_diff_logic_edge` call.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct DifferenceLogicBoundsShell;

impl Propagator<Engine> for DifferenceLogicBoundsShell {
	fn initialize(
		&mut self,
		ctx: &mut <Engine as crate::actions::ReasoningEngine>::InitializationContext<'_>,
	) {
		let prio = ctx
			.state
			.diff_logic
			.priority_bounds
			.unwrap_or(PriorityLevel::Medium);
		ctx.set_priority(prio);
		// Subscribe a bounds advisor on every integer endpoint already in
		// the graph. Endpoints introduced by mid-search edge registration
		// would not be picked up; that capability arrives with
		// `tighten_difference` in a later PR.
		let int_vars: Vec<View<IntVal>> = ctx.state.diff_logic.int_vars.clone();
		for (i, n) in int_vars.iter().enumerate() {
			n.advise_when(ctx, IntPropCond::Bounds, i as u64);
		}
		ctx.advise_on_backtrack();
		// Force a propagate at decision level 0 so the graph folds its
		// implied bounds into the variable domains before SAT branching.
		ctx.enqueue_now(true);
	}

	fn advise_of_backtrack(
		&mut self,
		ctx: &mut <Engine as crate::actions::ReasoningEngine>::NotificationContext<'_>,
	) {
		ctx.diff_logic.reset_bounds();
	}

	fn advise_of_int_change(
		&mut self,
		ctx: &mut <Engine as crate::actions::ReasoningEngine>::NotificationContext<'_>,
		data: u64,
		event: IntEvent,
	) -> bool {
		let mut diff = mem::take(&mut ctx.diff_logic);
		let enqueue = diff.advise_int_change(ctx, data as usize, event);
		ctx.diff_logic = diff;
		enqueue
	}

	fn propagate(
		&mut self,
		ctx: &mut <Engine as crate::actions::ReasoningEngine>::PropagationContext<'_>,
	) -> Result<(), <Engine as crate::actions::ReasoningEngine>::Conflict> {
		let mut diff = mem::take(&mut ctx.state.diff_logic);
		let result = diff.propagate_bounds(ctx);
		ctx.state.diff_logic = diff;
		result
	}
}
