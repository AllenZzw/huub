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
//! Supports globally-active edges and Boolean-gated implication edges
//! (`b → x − y ≤ d`). Lazy-literal `tighten_difference` and the
//! diff-logic-aware brancher land in subsequent PRs.
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
//! - **No mid-search endpoint introduction.** The bounds shell's `initialize`
//!   fires once, when the first [`crate::solver::Solver::add_diff_logic_edge`]
//!   auto-posts the shell. Lowering pre-interns every endpoint via
//!   [`crate::solver::Solver::intern_diff_logic_int`] /
//!   [`crate::solver::Solver::intern_diff_logic_bool`] before any edge is
//!   registered, so every lowering-time endpoint gets an advisor. But
//!   `tighten_difference` (future work) would assert a brand-new `x − y ≤ d`
//!   mid-search — possibly with endpoints not previously in the graph. Those
//!   late endpoints would be interned by `register_edge` but would have no
//!   bounds advisor, so subsequent bound changes on them would not wake the
//!   shell. The fix is a `subscribe_int_bounds_advisor` helper on
//!   [`crate::solver::Solver`] (~15 lines) that synthesizes the advisor entry
//!   without needing an `InitializationContext`; it lands together with
//!   `tighten_difference`.

use std::{cmp::Reverse, mem};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
	IntVal,
	actions::{
		BoolInitActions, BoolInspectionActions, BoolPropagationActions, InitActions,
		IntDecisionActions, IntEvent, IntInitActions, IntInspectionActions, IntPropCond,
		IntPropagationActions, PropagationActions, ReasoningContext, Trailed, TrailingActions,
	},
	constraints::{Conflict, Propagator},
	helpers::{
		priority_queue::LazyPriorityQueue, trailed_list::TrailedList,
		trailed_open_list::TrailedOpenList,
	},
	solver::{
		Decision, IntLitMeaning, View,
		engine::{Engine, PropRef, State},
		queue::PriorityLevel,
		solving_context::SolvingContext,
	},
};

/// An edge in the engine-resident difference logic graph.
///
/// Represents the constraint `int_vars[from] − int_vars[to] ≤ val`,
/// either globally (`bool_var = None`) or as an implication gated by the
/// Boolean stored at `bool_vars[bool_var.unwrap()]`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DiffEdge {
	pub(crate) from: usize,
	pub(crate) to: usize,
	pub(crate) val: IntVal,
	pub(crate) bool_var: Option<usize>,
	/// Position of this edge in `bool_implications[bool_var]`. Used by
	/// [`DiffLogicState::close_imp_edge`] for O(1) lookup. Meaningful
	/// only when `bool_var.is_some()`.
	pub(crate) bool_index: usize,
	/// Position of this edge in `open_out[from]`. Meaningful only when
	/// this edge is dormant (gated and not yet activated).
	pub(crate) out_index: usize,
	/// Position of this edge in `open_in[to]`. Mirror of `out_index`.
	pub(crate) in_index: usize,
}

/// Difference logic state owned by the engine.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DiffLogicState {
	// ---- Graph identity & lookup ----
	pub(crate) int_var_to_node: FxHashMap<View<IntVal>, usize>,
	pub(crate) int_vars: Vec<View<IntVal>>,
	pub(crate) bool_var_to_node: FxHashMap<View<bool>, usize>,
	pub(crate) bool_vars: Vec<View<bool>>,
	pub(crate) edges: Vec<DiffEdge>,

	// ---- Per-node active / dormant edge adjacency ----
	pub(crate) active_out: Vec<TrailedList<usize>>,
	pub(crate) active_in: Vec<TrailedList<usize>>,
	/// Per-node dormant outgoing implication edges (gate not yet fixed).
	pub(crate) open_out: Vec<TrailedOpenList<usize>>,
	/// Per-node dormant incoming implication edges.
	pub(crate) open_in: Vec<TrailedOpenList<usize>>,
	/// Per gating-Boolean list of implication edges. Indexed by
	/// `bool_vars` index.
	pub(crate) bool_implications: Vec<TrailedOpenList<usize>>,
	/// Trailed counter for the number of currently open implication
	/// edges. `None` until the first gated edge is registered.
	pub(crate) num_open_edges: Option<Trailed<usize>>,

	// ---- Per-node algorithm working buffers (not trailed) ----
	/// Johnson's potential per node. Computed once on the first call to
	/// [`Self::propagate_bounds`] via one Bellman-Ford pass and not
	/// refreshed afterwards.
	pub(crate) pi: Vec<IntVal>,
	pub(crate) lower_bound: Vec<Option<IntVal>>,
	pub(crate) upper_bound: Vec<Option<IntVal>>,
	/// Predecessor on the current Dijkstra pass: `(prev_node,
	/// optional_bool_idx)`. The optional Boolean index identifies the
	/// gate of the in-edge if it was a gated implication that had been
	/// activated; used by reason construction to attribute the
	/// propagation back through the gate.
	pub(crate) backtrace: Vec<Option<(usize, Option<usize>)>>,
	pub(crate) visited: Vec<bool>,

	// ---- Propagation transient scratch (not trailed; cleared each call) ----
	pub(crate) visited_updates: Vec<usize>,
	pub(crate) lower_bound_changes: FxHashSet<usize>,
	pub(crate) upper_bound_changes: FxHashSet<usize>,
	pub(crate) lb_updates: Vec<usize>,
	pub(crate) ub_updates: Vec<usize>,
	/// Set of gating Boolean indices reported by `advise_of_bool_change`
	/// as fixed since the last `propagate_booleans`.
	pub(crate) fixed_bools: FxHashSet<usize>,

	// ---- Engine integration ----
	pub(crate) pi_initialized: bool,
	/// Priority level at which to schedule the bounds shell. `None`
	/// falls back to [`PriorityLevel::Medium`].
	pub(crate) priority_bounds: Option<PriorityLevel>,
	/// Priority level for the booleans shell. `None` falls back to
	/// [`PriorityLevel::Low`].
	pub(crate) priority_booleans: Option<PriorityLevel>,
	/// Mode for reasons attached to Booleans set false by the propagator:
	/// `0` deferred via
	/// [`crate::actions::PropagationActions::deferred_reason`], `1` lifted lit
	/// reasons, `2` eager full reasons.
	pub(crate) bool_reasons: u8,
	/// Whether the booleans shell should run `inc_imp` (proactive check
	/// of open implication edges after each edge activation).
	pub(crate) use_inc_imp: bool,
	/// Whether the diff-logic propagators have been pushed to the engine
	/// already. Set on the first `Solver::add_diff_logic_edge`.
	pub(crate) propagators_registered: bool,
	pub(crate) bounds_ref: Option<PropRef>,
	pub(crate) booleans_ref: Option<PropRef>,
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
		self.open_out.push(TrailedOpenList::new(trail));
		self.open_in.push(TrailedOpenList::new(trail));
		self.pi.push(0);
		self.lower_bound.push(None);
		self.upper_bound.push(None);
		self.backtrace.push(None);
		self.visited.push(false);
		n
	}

	/// Find or create the node index for a gating Boolean decision.
	pub(crate) fn intern_bool(
		&mut self,
		trail: &mut crate::solver::trail::Trail,
		b: View<bool>,
	) -> usize {
		if let Some(&n) = self.bool_var_to_node.get(&b) {
			return n;
		}
		let n = self.bool_vars.len();
		self.bool_vars.push(b);
		let _ = self.bool_var_to_node.insert(b, n);
		self.bool_implications.push(TrailedOpenList::new(trail));
		n
	}

	/// Register a difference constraint `x − y ≤ d` in the engine graph.
	///
	/// `gate` is `None` for a globally-active edge; the edge enters
	/// `active_out` / `active_in` immediately. `gate = Some(b)` makes the
	/// edge a dormant implication; it lives in `open_out` / `open_in` /
	/// `bool_implications[gate]` until the booleans shell activates it
	/// (gate fixed true) or closes it (gate fixed false).
	pub(crate) fn register_edge(
		&mut self,
		trail: &mut crate::solver::trail::Trail,
		x: View<IntVal>,
		y: View<IntVal>,
		d: IntVal,
		gate: Option<View<bool>>,
	) -> usize {
		let from = self.intern_int(trail, x);
		let to = self.intern_int(trail, y);
		let bool_var = gate.map(|b| self.intern_bool(trail, b));
		// Lazily allocate the trailed `num_open_edges` counter on the very
		// first gated edge — it can't be allocated in `Default::default()`.
		if bool_var.is_some() && self.num_open_edges.is_none() {
			self.num_open_edges = Some(trail.track(0_usize));
		}
		let mut edge = DiffEdge {
			from,
			to,
			val: d,
			bool_var,
			bool_index: 0,
			out_index: 0,
			in_index: 0,
		};
		let idx = self.edges.len();
		if let Some(b_idx) = bool_var {
			edge.bool_index = self.bool_implications[b_idx].len();
			self.bool_implications[b_idx].push(idx);
			edge.out_index = self.open_out[from].len();
			self.open_out[from].push(idx);
			edge.in_index = self.open_in[to].len();
			self.open_in[to].push(idx);
			let cnt = self.num_open_edges.unwrap();
			let cur = trail.trailed(cnt);
			let _ = trail.set_trailed(cnt, cur + 1);
		} else {
			self.active_out[from].push(trail, idx);
			self.active_in[to].push(trail, idx);
		}
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

	// ---- Gated edge activation / closure ----

	/// Move a gated edge from the dormant lists into the active adjacency.
	fn activate_imp_edge(&mut self, ctx: &mut SolvingContext<'_>, index: usize) {
		let edge = &self.edges[index];
		let from = edge.from;
		let to = edge.to;
		self.active_out[from].push(ctx, index);
		self.active_in[to].push(ctx, index);
	}

	/// Close a gated edge: remove it from the dormant lists and decrement
	/// the open-edge counter. Called when the gate is fixed (either true
	/// after activation, or false).
	fn close_imp_edge(&mut self, ctx: &mut SolvingContext<'_>, e: usize) {
		let edge = &self.edges[e];
		let b = edge.bool_var.unwrap();
		let to = edge.to;
		let from = edge.from;
		let bool_index = edge.bool_index;
		let out_index = edge.out_index;
		let in_index = edge.in_index;
		let edges = &mut self.edges;
		let was_open = self.bool_implications[b]
			.close(ctx, bool_index, |&e, i| edges[e].bool_index = i)
			& self.open_out[from].close(ctx, out_index, |&e, i| edges[e].out_index = i)
			& self.open_in[to].close(ctx, in_index, |&e, i| edges[e].in_index = i);
		debug_assert!(was_open);
		let cnt = self.num_open_edges.unwrap();
		let cur = ctx.trailed(cnt);
		let _ = ctx.set_trailed(cnt, cur - 1);
	}

	/// Build the explanation for a negative cycle reaching `node` during
	/// `inc_sat`. Walks the current Dijkstra backtrace and collects the
	/// gating Booleans along the path.
	fn get_cycle_reason(&self, node: usize) -> Vec<View<bool>> {
		let mut reason = Vec::new();
		let mut var = node;
		while let Some((cur, b)) = self.backtrace[var] {
			if let Some(b) = b {
				reason.push(self.bool_vars[b]);
			}
			var = cur;
		}
		reason
	}

	// ---- Bounds-shell deferred-reason encoding ----
	//
	// Bounds-shell propagations register a *lazy* reason. The reason is
	// rebuilt at explain-time (after `goto_assign_lit`) when the SAT trail
	// has caught up to the moment of propagation — at which point the
	// strongest currently-true antecedent literal on the source variable
	// is available via `lit_relaxed`.
	//
	/// Used by the booleans shell to discriminate between bounds-shell
	/// and booleans-shell encodings in the shared lazy-reason path.
	/// Bounds-shell reasons are eager (built at propagation time below)
	/// and never carry this tag.
	pub(crate) const BOUNDS_TAG: u64 = 1 << 63;

	fn set_int_lower_bound(
		&mut self,
		ctx: &mut SolvingContext<'_>,
		n: usize,
		value: IntVal,
		bool_var: Option<usize>,
		lb_var: usize,
		lb_val: IntVal,
	) -> Result<(), Conflict<Decision<bool>>> {
		let target_view = self.int_vars[n];
		let source_view = self.int_vars[lb_var];
		let gate = bool_var.map(|b| self.bool_vars[b]);
		let reason = move |rctx: &mut SolvingContext<'_>| {
			let mut atoms = vec![source_view.lit(rctx, IntLitMeaning::GreaterEq(lb_val))];
			if let Some(g) = gate {
				atoms.push(g);
			}
			atoms
		};
		target_view.tighten_min(ctx, value, reason)?;
		Ok(())
	}

	fn set_int_upper_bound(
		&mut self,
		ctx: &mut SolvingContext<'_>,
		n: usize,
		value: IntVal,
		bool_var: Option<usize>,
		ub_var: usize,
		ub_val: IntVal,
	) -> Result<(), Conflict<Decision<bool>>> {
		let target_view = self.int_vars[n];
		let source_view = self.int_vars[ub_var];
		let gate = bool_var.map(|b| self.bool_vars[b]);
		let reason = move |rctx: &mut SolvingContext<'_>| {
			let mut atoms = vec![source_view.lit(rctx, IntLitMeaning::Less(ub_val + 1))];
			if let Some(g) = gate {
				atoms.push(g);
			}
			atoms
		};
		target_view.tighten_max(ctx, value, reason)?;
		Ok(())
	}

	/// Eager bool-set-false reason, used by [`Self::set_bool_false`] when
	/// `bool_reasons` is 1 (lifted) or 2 (eager). Returns a closure so
	/// callers can defer construction past the `BoolPropagationActions::fix`
	/// borrow boundary.
	fn get_bool_reason<'a, 'b>(
		&'a self,
		edge: usize,
		lb_fixed: bool,
	) -> impl crate::constraints::ReasonBuilder<SolvingContext<'b>> + 'a {
		let bool_reasons = self.bool_reasons;
		move |ctx: &mut SolvingContext<'_>| {
			let e = &self.edges[edge];
			let mut lb = self.get_cur_lower_bound(ctx, e.from);
			let mut ub = self.get_cur_upper_bound(ctx, e.to);
			if bool_reasons == 1 {
				if lb_fixed {
					ub = lb - e.val - 1;
				} else {
					lb = ub + e.val + 1;
				}
			}
			vec![
				self.int_vars[e.from].lit(ctx, IntLitMeaning::GreaterEq(lb)),
				self.int_vars[e.to].lit(ctx, IntLitMeaning::Less(ub + 1)),
			]
		}
	}

	/// Fix a gating Boolean to false (the gated edge can never fire). Used
	/// from the bounds shell when a domain extreme rules out the edge,
	/// and from `inc_sat` / `inc_imp` when a cycle would form. The reason
	/// mode (`bool_reasons`) selects between deferred and eager
	/// explanations.
	fn set_bool_false(
		&mut self,
		ctx: &mut SolvingContext<'_>,
		bool_var: Option<usize>,
		edge: usize,
		lb_fixed: bool,
	) -> Result<(), Conflict<Decision<bool>>> {
		if self.bool_reasons == 0 {
			// Pack `edge` index into the high bits and `lb_fixed` into the
			// low bit so the booleans shell's `explain` handler can
			// recover both.
			let data = ((edge as u64) << 1) | u64::from(lb_fixed);
			if let Some(b) = bool_var {
				let bv = self.bool_vars[b];
				bv.fix(ctx, false, ctx.deferred_reason(data))?;
			} else {
				return Err(ctx.declare_conflict(ctx.deferred_reason(data)));
			}
		} else if let Some(b) = bool_var {
			let bv = self.bool_vars[b];
			bv.fix(ctx, false, self.get_bool_reason(edge, lb_fixed))?;
		} else {
			return Err(ctx.declare_conflict(self.get_bool_reason(edge, lb_fixed)));
		}
		Ok(())
	}

	// ---- Incremental SAT (new-edge consistency check) ----

	/// Check whether the newly-activated edge introduces a negative
	/// cycle, and either set the gate to false (when gated) or declare a
	/// conflict (when global). Returns `Ok(true)` when the edge was
	/// installed cleanly (and `pi` updates were merged), `Ok(false)` when
	/// the cycle was detected and the gate was falsified.
	fn inc_sat(
		&mut self,
		ctx: &mut SolvingContext<'_>,
		new_index: usize,
	) -> Result<bool, Conflict<Decision<bool>>> {
		let new_edge = self.edges[new_index];
		let mut queue = LazyPriorityQueue::new();
		let mut pi_new: FxHashMap<usize, IntVal> = FxHashMap::default();
		self.backtrace[new_edge.to] = None;
		let gamma_v = self.pi[new_edge.from] + new_edge.val - self.pi[new_edge.to];
		if gamma_v < 0 {
			let _ = queue.push(new_edge.to, Reverse(gamma_v));
		}
		while !queue.is_empty() && queue.get_priority(&new_edge.from).is_none() {
			let (s, Reverse(gamma_s)) = queue.pop().unwrap();
			let _ = pi_new.insert(s, self.pi[s] + gamma_s);
			for &e in self.active_out[s].iter(ctx) {
				let edge = &self.edges[e];
				if !pi_new.contains_key(&edge.to) {
					let gamma_t = pi_new[&s] + edge.val - self.pi[edge.to];
					if gamma_t < 0 {
						let old = queue.push_increase(edge.to, Reverse(gamma_t));
						if old.is_none_or(|Reverse(old_gamma)| gamma_t < old_gamma) {
							self.backtrace[edge.to] = Some((s, edge.bool_var));
						}
					}
				}
			}
		}
		if queue.get_priority(&new_edge.from).is_some() {
			let reason = self.get_cycle_reason(new_edge.from);
			if let Some(b) = new_edge.bool_var {
				let bv = self.bool_vars[b];
				bv.fix(ctx, false, reason)?;
			} else {
				return Err(ctx.declare_conflict(reason));
			}
			return Ok(false);
		}
		for (var, val) in pi_new {
			self.pi[var] = val;
		}
		Ok(true)
	}

	// ---- Dijkstra over relevant nodes for `inc_imp` ----

	fn dijkstra_relevant(
		&mut self,
		ctx: &SolvingContext<'_>,
		new_edge: usize,
		reverse: bool,
	) -> FxHashMap<usize, IntVal> {
		self.reset_visit();
		let new_edge = self.edges[new_edge];
		let origin = if reverse { new_edge.to } else { new_edge.from };
		let relevant_target = if reverse { new_edge.from } else { new_edge.to };
		let mut distances: FxHashMap<usize, IntVal> = FxHashMap::default();
		let _ = distances.insert(relevant_target, new_edge.val);
		let mut queue = LazyPriorityQueue::new();
		let _ = queue.push(origin, Reverse((0, false)));
		let _ = queue.push(
			relevant_target,
			Reverse((
				new_edge.val
					+ if reverse {
						self.pi[relevant_target] - self.pi[origin]
					} else {
						self.pi[origin] - self.pi[relevant_target]
					},
				true,
			)),
		);
		let mut relevant_count = 1;
		while !queue.is_empty() && relevant_count > 0 {
			let (s, Reverse((dist, relevant))) = queue.pop().unwrap();
			self.visit(s);
			let it = if reverse {
				self.active_in[s].iter(ctx)
			} else {
				self.active_out[s].iter(ctx)
			};
			for &e in it {
				let edge = &self.edges[e];
				let target = if reverse { edge.from } else { edge.to };
				let new_dist = dist
					+ edge.val + if reverse {
					self.pi[target] - self.pi[s]
				} else {
					self.pi[s] - self.pi[target]
				};
				if !self.visited[target] {
					let new_relevant = relevant || (s == origin && target == relevant_target);
					let new_prio = Reverse((new_dist, new_relevant));
					let prev = queue.push_increase(target, new_prio);
					if prev != Some(new_prio) {
						if new_relevant {
							if distances
								.insert(
									target,
									new_dist
										+ if reverse {
											self.pi[origin] - self.pi[target]
										} else {
											self.pi[target] - self.pi[origin]
										},
								)
								.is_none()
							{
								relevant_count += 1;
							}
						} else if distances.remove(&target).is_some() {
							relevant_count -= 1;
						}
					}
				}
			}
			if relevant {
				relevant_count -= 1;
			}
		}
		distances
	}

	// ---- Implication propagation for a newly-added edge ----

	fn inc_imp(
		&mut self,
		ctx: &mut SolvingContext<'_>,
		new_index: usize,
	) -> Result<(), Conflict<Decision<bool>>> {
		if ctx.trailed(self.num_open_edges.unwrap()) == 0 {
			return Ok(());
		}

		let incoming_u = self.dijkstra_relevant(ctx, new_index, false);
		let outgoing_v = self.dijkstra_relevant(ctx, new_index, true);
		let indegree_u: usize = incoming_u
			.iter()
			.map(|(&n, _)| self.open_in[n].num_open(ctx))
			.sum();
		let outdegree_v: usize = outgoing_v
			.iter()
			.map(|(&n, _)| self.open_out[n].num_open(ctx))
			.sum();

		let new_edge_val = self.edges[new_index].val;

		if indegree_u < outdegree_v {
			let keys: Vec<usize> = incoming_u.keys().copied().collect();
			for n in keys {
				let in_pairs: Vec<usize> = self.open_in[n].open_iter(ctx).collect();
				for i in in_pairs {
					let &e = self.open_in[n].index(ctx, i);
					let edge = self.edges[e];
					if outgoing_v.contains_key(&edge.from)
						&& outgoing_v[&edge.from] + incoming_u[&edge.to] - new_edge_val <= edge.val
					{
						self.close_imp_edge(ctx, e);
					}
				}
				let out_pairs: Vec<usize> = self.open_out[n].open_iter(ctx).collect();
				for i in out_pairs {
					let &e = self.open_out[n].index(ctx, i);
					let edge = self.edges[e];
					if outgoing_v.contains_key(&edge.to)
						&& outgoing_v[&edge.to] + incoming_u[&edge.from] - new_edge_val < -edge.val
					{
						self.close_imp_edge(ctx, e);
						let result = self.inc_sat(ctx, e)?;
						debug_assert!(!result, "Adding {e} should not be possible");
					}
				}
			}
		} else {
			let keys: Vec<usize> = outgoing_v.keys().copied().collect();
			for n in keys {
				let out_pairs: Vec<usize> = self.open_out[n].open_iter(ctx).collect();
				for i in out_pairs {
					let &e = self.open_out[n].index(ctx, i);
					let edge = self.edges[e];
					if incoming_u.contains_key(&edge.to)
						&& outgoing_v[&edge.from] + incoming_u[&edge.to] - new_edge_val <= edge.val
					{
						self.close_imp_edge(ctx, e);
					}
				}
				let in_pairs: Vec<usize> = self.open_in[n].open_iter(ctx).collect();
				for i in in_pairs {
					let &e = self.open_in[n].index(ctx, i);
					let edge = self.edges[e];
					if incoming_u.contains_key(&edge.from)
						&& outgoing_v[&edge.to] + incoming_u[&edge.from] - new_edge_val < -edge.val
					{
						self.close_imp_edge(ctx, e);
						let result = self.inc_sat(ctx, e)?;
						debug_assert!(!result, "Adding {e} should not be possible");
					}
				}
			}
		}

		Ok(())
	}

	/// Drive `inc_sat` + (optional) `inc_imp` + immediate bound propagation
	/// for a freshly-activated edge.
	fn propagate_edge_addition(
		&mut self,
		ctx: &mut SolvingContext<'_>,
		e: usize,
		check_implied: bool,
	) -> Result<(), Conflict<Decision<bool>>> {
		let result = self.inc_sat(ctx, e)?;
		debug_assert!(result, "Adding {e} should be possible or cause a conflict!");
		if check_implied {
			self.inc_imp(ctx, e)?;
		}
		let edge = self.edges[e];
		let source_lb = self.get_cur_lower_bound(ctx, edge.from);
		let lb_y = source_lb - edge.val;
		if lb_y > self.get_cur_lower_bound(ctx, edge.to) {
			self.set_int_lower_bound(ctx, edge.to, lb_y, edge.bool_var, edge.from, source_lb)?;
		}
		let target_ub = self.get_cur_upper_bound(ctx, edge.to);
		let ub_x = target_ub + edge.val;
		if ub_x < self.get_cur_upper_bound(ctx, edge.from) {
			self.set_int_upper_bound(ctx, edge.from, ub_x, edge.bool_var, edge.to, target_ub)?;
		}
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
					let (prev, b) = self.backtrace[s].unwrap();
					let lb = self.get_cur_lower_bound(ctx, prev);
					self.set_int_lower_bound(ctx, s, bound, b, prev, lb)?;
					let _ = self.lower_bound_changes.insert(s);
				}
				for &e in self.active_out[s].iter(ctx) {
					let edge = &self.edges[e];
					if !self.visited[edge.to] {
						let path = gamma_s + self.pi[s] + edge.val - self.pi[edge.to];
						let old = queue.push_increase(edge.to, Reverse(path));
						if old.is_none_or(|Reverse(old_path)| path < old_path) {
							self.backtrace[edge.to] = Some((s, edge.bool_var));
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
					let (prev, b) = self.backtrace[s].unwrap();
					let ub = self.get_cur_upper_bound(ctx, prev);
					self.set_int_upper_bound(ctx, s, bound, b, prev, ub)?;
					let _ = self.upper_bound_changes.insert(s);
				}
				for &e in self.active_in[s].iter(ctx) {
					let edge = &self.edges[e];
					if !self.visited[edge.from] {
						let path = gamma_s + self.pi[edge.from] + edge.val - self.pi[s];
						let old = queue.push_increase(edge.from, Reverse(path));
						if old.is_none_or(|Reverse(old_path)| path < old_path) {
							self.backtrace[edge.from] = Some((s, edge.bool_var));
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

		// Post-pass: for every node whose lower bound was tightened, walk
		// its dormant outgoing / incoming implication edges and either
		// falsify the gate (when the edge can no longer fire) or close
		// the edge silently (when its constraint is already entailed).
		let lb_changes = mem::take(&mut self.lower_bound_changes);
		for n in lb_changes {
			let Some(lb) = self.lower_bound[n] else {
				continue;
			};

			let out_pairs: Vec<usize> = self.open_out[n].open_iter(ctx).collect();
			for i in out_pairs {
				let &e = self.open_out[n].index(ctx, i);
				let edge = self.edges[e];
				let target_ub = self.get_cur_upper_bound(ctx, edge.to);
				if lb - target_ub > edge.val {
					self.set_bool_false(ctx, edge.bool_var, e, false)?;
					self.close_imp_edge(ctx, e);
				}
			}

			let in_pairs: Vec<usize> = self.open_in[n].open_iter(ctx).collect();
			for i in in_pairs {
				let &e = self.open_in[n].index(ctx, i);
				let edge = self.edges[e];
				if self.get_cur_upper_bound(ctx, edge.from) - lb <= edge.val {
					self.close_imp_edge(ctx, e);
				}
			}
		}

		let ub_changes = mem::take(&mut self.upper_bound_changes);
		for n in ub_changes {
			let Some(ub) = self.upper_bound[n] else {
				continue;
			};

			let out_pairs: Vec<usize> = self.open_out[n].open_iter(ctx).collect();
			for j in out_pairs {
				let &e = self.open_out[n].index(ctx, j);
				let edge = self.edges[e];
				if ub - self.get_cur_lower_bound(ctx, edge.to) <= edge.val {
					self.close_imp_edge(ctx, e);
				}
			}

			let in_pairs: Vec<usize> = self.open_in[n].open_iter(ctx).collect();
			for j in in_pairs {
				let &e = self.open_in[n].index(ctx, j);
				let edge = self.edges[e];
				let source_lb = self.get_cur_lower_bound(ctx, edge.from);
				if source_lb - ub > edge.val {
					self.set_bool_false(ctx, edge.bool_var, e, true)?;
					self.close_imp_edge(ctx, e);
				}
			}
		}

		Ok(())
	}

	/// Boolean propagation entry point invoked by the booleans shell.
	///
	/// For each gating Boolean newly fixed since the last call, either
	/// activate every edge it gates (gate fixed true) and propagate its
	/// consequences via `propagate_edge_addition`, or close every edge it
	/// gates (gate fixed false) so the rest of search ignores them.
	pub(crate) fn propagate_booleans(
		&mut self,
		ctx: &mut SolvingContext<'_>,
	) -> Result<(), Conflict<Decision<bool>>> {
		let fixed_bools = mem::take(&mut self.fixed_bools);
		let check_implied = self.use_inc_imp;
		for b in fixed_bools {
			let bv = self.bool_vars[b];
			let val = bv.val(ctx).unwrap();
			if val {
				let edges: Vec<usize> = {
					let mut out = Vec::new();
					for i in self.bool_implications[b].open_iter(ctx) {
						if let Some(&e) = self.bool_implications[b].index_opt(ctx, i) {
							out.push(e);
						}
					}
					out
				};
				for e in edges {
					self.close_imp_edge(ctx, e);
					self.activate_imp_edge(ctx, e);
					self.propagate_edge_addition(ctx, e, check_implied)?;
				}
			} else {
				let edges: Vec<usize> = {
					let mut out = Vec::new();
					for i in self.bool_implications[b].open_iter(ctx) {
						out.push(*self.bool_implications[b].index(ctx, i));
					}
					out
				};
				for e in edges {
					self.close_imp_edge(ctx, e);
				}
			}
		}
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

	/// Mark a gating Boolean as fixed since the last `propagate_booleans`.
	pub(crate) fn advise_bool_fixed(&mut self, data: usize) -> bool {
		self.fixed_bools.insert(data)
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
		// Subscribe a bounds advisor on every integer endpoint in the
		// graph. Lowering pre-interns all endpoints via
		// `Solver::intern_diff_logic_int` before the first edge auto-posts
		// the shell, so every lowering-time endpoint is reachable here.
		// Mid-search endpoints (introduced later by `tighten_difference`)
		// would require a separate subscription path.
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

	/// `propagate_bounds` may call `set_bool_false` in its post-pass when
	/// a tightened bound falsifies a dormant implication edge; that path
	/// registers a deferred reason whose owner is *this* bounds shell.
	/// We dispatch through the shared booleans-shell encoder.
	fn explain(
		&mut self,
		ctx: &mut <Engine as crate::actions::ReasoningEngine>::ExplanationContext<'_>,
		_lit: <Engine as crate::actions::ReasoningEngine>::Atom,
		data: u64,
	) -> crate::Conjunction<<Engine as crate::actions::ReasoningEngine>::Atom> {
		let diff = mem::take(&mut ctx.diff_logic);
		let reason = explain_diff_logic_lazy(&diff, ctx, data);
		ctx.diff_logic = diff;
		reason
	}
}

/// Shared dispatcher for the booleans shell's lazy reasons. The
/// bounds shell builds its reasons eagerly (see
/// [`DiffLogicState::set_int_lower_bound`] /
/// [`DiffLogicState::set_int_upper_bound`]), so this dispatcher only
/// handles the booleans-shell `set_bool_false` encoding.
fn explain_diff_logic_lazy(
	diff: &DiffLogicState,
	ctx: &mut <Engine as crate::actions::ReasoningEngine>::ExplanationContext<'_>,
	data: u64,
) -> crate::Conjunction<<Engine as crate::actions::ReasoningEngine>::Atom> {
	use crate::actions::IntExplanationActions;

	debug_assert_eq!(
		data & DiffLogicState::BOUNDS_TAG,
		0,
		"bounds-shell reasons are eager; lazy dispatcher should only see booleans-shell encodings"
	);

	// Booleans-shell encoding (lazy reason for `set_bool_false`).
	//
	// The falsification `bv = false` is justified by the inconsistency
	// `source.lb - target.ub > edge.val`. The tightest reason captures
	// the exact boundary values that triggered the inconsistency, but
	// under Eager order encoding the literal at the exact boundary may
	// have never been directly assigned by SAT — only a strictly
	// stronger literal at the source's current lb (or weaker than the
	// target's current ub) is on the trail. To stay sound under both
	// encoding strategies, we report the source's *current* lb and the
	// target's *current* ub at explain-time as the antecedents.
	// `goto_assign_lit` positions the trail at the moment of
	// propagation, so these bounds are exactly the historical ones, and
	// the literals at those bounds are the ones that were assigned via
	// `tighten_min` / `tighten_max` and therefore on the SAT trail.
	let _lb_fixed = (data & 1) != 0;
	let edge_idx = (data >> 1) as usize;
	let edge = &diff.edges[edge_idx];
	let source_lb = diff.int_vars[edge.from].min(ctx);
	let target_ub = diff.int_vars[edge.to].max(ctx);
	let (lit_from, _) =
		diff.int_vars[edge.from].lit_relaxed(ctx, IntLitMeaning::GreaterEq(source_lb));
	let (lit_to, _) = diff.int_vars[edge.to].lit_relaxed(ctx, IntLitMeaning::Less(target_ub + 1));
	vec![lit_from, lit_to]
}

/// Trampoline propagator that drives [`DiffLogicState::propagate_booleans`].
///
/// Mirror of [`DifferenceLogicBoundsShell`] for gating Boolean changes.
/// Auto-registered alongside the bounds shell on the first
/// `Solver::add_diff_logic_edge` call.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct DifferenceLogicBooleansShell;

impl Propagator<Engine> for DifferenceLogicBooleansShell {
	fn initialize(
		&mut self,
		ctx: &mut <Engine as crate::actions::ReasoningEngine>::InitializationContext<'_>,
	) {
		let prio = ctx
			.state
			.diff_logic
			.priority_booleans
			.unwrap_or(PriorityLevel::Low);
		ctx.set_priority(prio);
		let bool_vars: Vec<View<bool>> = ctx.state.diff_logic.bool_vars.clone();
		for (i, b) in bool_vars.iter().enumerate() {
			b.advise_when_fixed(ctx, i as u64);
		}
		ctx.advise_on_backtrack();
		// Force a propagate at decision level 0 so gating Booleans
		// already fixed by upstream simplification flow through into
		// edge activation before the SAT solver makes its first decision.
		ctx.enqueue_now(true);
	}

	fn advise_of_bool_change(
		&mut self,
		ctx: &mut <Engine as crate::actions::ReasoningEngine>::NotificationContext<'_>,
		data: u64,
	) -> bool {
		ctx.diff_logic.advise_bool_fixed(data as usize)
	}

	fn advise_of_backtrack(
		&mut self,
		ctx: &mut <Engine as crate::actions::ReasoningEngine>::NotificationContext<'_>,
	) {
		ctx.diff_logic.fixed_bools.clear();
	}

	fn propagate(
		&mut self,
		ctx: &mut <Engine as crate::actions::ReasoningEngine>::PropagationContext<'_>,
	) -> Result<(), <Engine as crate::actions::ReasoningEngine>::Conflict> {
		let mut diff = mem::take(&mut ctx.state.diff_logic);
		let result = diff.propagate_booleans(ctx);
		ctx.state.diff_logic = diff;
		result
	}

	fn explain(
		&mut self,
		ctx: &mut <Engine as crate::actions::ReasoningEngine>::ExplanationContext<'_>,
		_lit: <Engine as crate::actions::ReasoningEngine>::Atom,
		data: u64,
	) -> crate::Conjunction<<Engine as crate::actions::ReasoningEngine>::Atom> {
		// Both `set_bool_false` (this shell) and `set_int_lower_bound` /
		// `set_int_upper_bound` (invoked indirectly via
		// `propagate_edge_addition` inside `propagate_booleans`) can
		// register lazy reasons attributed to this propagator. The shared
		// dispatcher tells the two encodings apart by the top bit of
		// `data`.
		let diff = mem::take(&mut ctx.diff_logic);
		let reason = explain_diff_logic_lazy(&diff, ctx, data);
		ctx.diff_logic = diff;
		reason
	}
}
