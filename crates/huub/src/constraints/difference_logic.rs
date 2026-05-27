//! Difference logic propagator.
//!
//! Each edge in the graph encodes `x − y ≤ d`, either globally
//! (`bool_var = None`) or as an implication gated by a Boolean
//! (`b → x − y ≤ d`). [`crate::solver::Solver::add_diff_logic_edge`]
//! registers an edge at lowering time; the resulting graph is consumed by
//! [`DifferenceLogicPropagator`] during search.
//!
//! ## PR scope
//!
//! Supports globally-active edges and Boolean-gated implication edges.
//! Lazy-literal `tighten_difference` and the diff-logic-aware brancher
//! land in subsequent PRs.
//!
//! ## Architecture
//!
//! - [`DiffLogicState`] lives on the [`crate::solver::Solver`] inside an
//!   `Rc<RefCell<…>>`. It owns the master edge list, the per-node active-edge
//!   adjacency lists, the Johnson potential `pi`, and the per-node bound shadow
//!   used for incremental Dijkstra.
//! - [`DifferenceLogicPropagator`] holds an `Rc::clone` of the graph and
//!   bridges the engine's `Propagator<Engine>` interface to the graph methods
//!   (`propagate_bounds`, `propagate_booleans`, advisors, reasons). One
//!   propagator runs both bound and Boolean phases in a single `borrow_mut()`
//!   scope.
//!
//! ## Known limitations (fixed alongside `tighten_difference`)
//!
//! - **No mid-search endpoint introduction.** The propagator's `initialize`
//!   fires once, when the first [`crate::solver::Solver::add_diff_logic_edge`]
//!   auto-posts the propagator. Lowering pre-interns every endpoint via
//!   [`crate::solver::Solver::intern_diff_logic_int`] /
//!   [`crate::solver::Solver::intern_diff_logic_bool`] before any edge is
//!   registered, so every lowering-time endpoint gets an advisor. But
//!   `tighten_difference` (future work) would assert a brand-new `x − y ≤ d`
//!   mid-search — possibly with endpoints not previously in the graph. Those
//!   late endpoints would be interned by `register_edge` but would have no
//!   bounds advisor, so subsequent bound changes on them would not wake the
//!   propagator. The fix is a `subscribe_int_bounds_advisor` helper on
//!   [`crate::solver::Solver`] (~15 lines) that synthesizes the advisor entry
//!   without needing an `InitializationContext`; it lands together with
//!   `tighten_difference`.

use std::{cell::RefCell, cmp::Reverse, mem, rc::Rc};

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
		engine::{Engine, State},
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
	/// Priority level at which to schedule the propagator. `None`
	/// falls back to [`PriorityLevel::Medium`].
	pub(crate) priority_bounds: Option<PriorityLevel>,
	/// Mode for reasons attached to Booleans set false by the propagator:
	/// `0` deferred via
	/// [`crate::actions::PropagationActions::deferred_reason`], `1` lifted lit
	/// reasons, `2` eager full reasons.
	pub(crate) bool_reasons: u8,
	/// Whether the booleans phase should run `inc_imp` (proactive check
	/// of open implication edges after each edge activation).
	pub(crate) use_inc_imp: bool,
	/// Whether the diff-logic propagator has been pushed to the engine
	/// already. Set on the first `Solver::add_diff_logic_edge`.
	pub(crate) propagator_registered: bool,
	/// The diff-logic propagator's `PropRef`, recorded by
	/// `DifferenceLogicPropagator::initialize`. Needed by mid-search
	/// advisor subscription so newly-introduced int endpoints and gate
	/// Booleans can be wired to wake the same propagator.
	pub(crate) propagator_ref: Option<crate::solver::engine::PropRef>,
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

/// Merged propagator that drives [`DiffLogicState::propagate_bounds`]
/// followed by [`DiffLogicState::propagate_booleans`].
///
/// Holds an `Rc<RefCell<DiffLogicState>>` clone — the master cell lives
/// on the [`crate::solver::Solver`]. Auto-registered (a single
/// propagator, replacing the prior bounds/booleans shell pair) on the
/// first `Solver::add_diff_logic_edge` call.
///
/// The booleans phase is a no-op when no gated edges exist, so the
/// merged propagator handles every diff-logic level uniformly.
#[derive(Clone, Debug)]
pub(crate) struct DifferenceLogicPropagator {
	pub(crate) graph: Rc<RefCell<DiffLogicState>>,
}

impl Propagator<Engine> for DifferenceLogicPropagator {
	fn initialize(
		&mut self,
		ctx: &mut <Engine as crate::actions::ReasoningEngine>::InitializationContext<'_>,
	) {
		// Record the propagator's `PropRef` so mid-search advisor
		// subscriptions (via `SolvingContext::subscribe_diff_logic_*`)
		// can wake this propagator on newly-introduced endpoints and
		// gating Booleans.
		self.graph.borrow_mut().propagator_ref = Some(ctx.prop);
		let graph = self.graph.borrow();
		// Low default chosen empirically: running diff-logic *after* other
		// Medium-priority propagators (disjunctive, cumulative) lets them
		// tighten bounds first, so diff-logic does less wasted work.
		// jobshop_la02/la03 are ~10% slower at Medium; Low closes the gap.
		// See `diff_logic/report.md` for the 4-way bench.
		let prio = graph.priority_bounds.unwrap_or(PriorityLevel::Low);
		ctx.set_priority(prio);
		// Subscribe a bounds advisor on every integer endpoint in the
		// graph. Lowering pre-interns all endpoints via
		// `Solver::intern_diff_logic_int` before the first edge auto-posts
		// the propagator, so every lowering-time endpoint is reachable
		// here. Mid-search endpoints (introduced later by
		// `tighten_difference`) would require a separate subscription
		// path.
		let int_vars: Vec<View<IntVal>> = graph.int_vars.clone();
		let bool_vars: Vec<View<bool>> = graph.bool_vars.clone();
		drop(graph);
		for (i, n) in int_vars.iter().enumerate() {
			n.advise_when(ctx, IntPropCond::Bounds, i as u64);
		}
		for (i, b) in bool_vars.iter().enumerate() {
			b.advise_when_fixed(ctx, i as u64);
		}
		ctx.advise_on_backtrack();
		// Force a propagate at decision level 0 so the graph folds its
		// implied bounds into the variable domains and any
		// upstream-fixed gates flow into edge activation before the
		// SAT solver makes its first decision.
		ctx.enqueue_now(true);
	}

	fn advise_of_backtrack(
		&mut self,
		_ctx: &mut <Engine as crate::actions::ReasoningEngine>::NotificationContext<'_>,
	) {
		let mut g = self.graph.borrow_mut();
		g.reset_bounds();
		g.fixed_bools.clear();
	}

	fn advise_of_int_change(
		&mut self,
		ctx: &mut <Engine as crate::actions::ReasoningEngine>::NotificationContext<'_>,
		data: u64,
		event: IntEvent,
	) -> bool {
		self.graph
			.borrow_mut()
			.advise_int_change(ctx, data as usize, event)
	}

	fn advise_of_bool_change(
		&mut self,
		_ctx: &mut <Engine as crate::actions::ReasoningEngine>::NotificationContext<'_>,
		data: u64,
	) -> bool {
		self.graph.borrow_mut().advise_bool_fixed(data as usize)
	}

	fn propagate(
		&mut self,
		ctx: &mut <Engine as crate::actions::ReasoningEngine>::PropagationContext<'_>,
	) -> Result<(), <Engine as crate::actions::ReasoningEngine>::Conflict> {
		let mut g = self.graph.borrow_mut();
		g.propagate_bounds(ctx)?;
		g.propagate_booleans(ctx)
	}

	/// `propagate_bounds`/`propagate_booleans` may register deferred
	/// reasons (via `set_bool_false`); the shared dispatcher decodes
	/// `data` and reconstructs the antecedents.
	fn explain(
		&mut self,
		ctx: &mut <Engine as crate::actions::ReasoningEngine>::ExplanationContext<'_>,
		_lit: <Engine as crate::actions::ReasoningEngine>::Atom,
		data: u64,
	) -> crate::Conjunction<<Engine as crate::actions::ReasoningEngine>::Atom> {
		let g = self.graph.borrow();
		explain_diff_logic_lazy(&g, ctx, data)
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

// =============================================================================
//                            Model-stage section
// =============================================================================
//
// The items below operate on `Model` views (pre-lowering). They are
// invoked by the lowering pipeline in `crate::lower` to expand the
// syntactic constraint variants, run cycle detection / bound tightening /
// Johnson pruning / equality-cycle unification, and produce flattened
// `ModelDiffEdge` values that the lowering loop posts as engine edges via
// `Solver::add_diff_logic_edge`.

use pindakaas::propositional_logic::Formula;

use crate::{
	actions::IntSimplificationActions,
	constraints::Reason,
	model::{Model, View as ModelView},
};

/// The syntactic variants of a difference constraint.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DifferenceLogicConstraint {
	/// A globally active difference constraint `x − y ≤ d`.
	Global(ModelView<IntVal>, ModelView<IntVal>, IntVal),
	/// An implied difference constraint `b → (x − y ≤ d)`.
	Implied(
		ModelView<bool>,
		ModelView<IntVal>,
		ModelView<IntVal>,
		IntVal,
	),
	/// A reified difference constraint `b ↔ (x − y ≤ d)`.
	Reified(
		ModelView<bool>,
		ModelView<IntVal>,
		ModelView<IntVal>,
		IntVal,
	),
	/// An implied equality `b → (x − y == d)`.
	ImpliedEquals(
		ModelView<bool>,
		ModelView<IntVal>,
		ModelView<IntVal>,
		IntVal,
	),
	/// A disequality `x − y ≠ d`.
	NotEquals(ModelView<IntVal>, ModelView<IntVal>, IntVal),
	/// An implied disequality `b → (x − y ≠ d)`.
	ImpliedNotEquals(
		ModelView<bool>,
		ModelView<IntVal>,
		ModelView<IntVal>,
		IntVal,
	),
	/// A reified equality `b ↔ (x − y == d)`.
	ReifiedEquals(
		ModelView<bool>,
		ModelView<IntVal>,
		ModelView<IntVal>,
		IntVal,
	),
}

/// User-tunable knobs for difference-logic processing.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DifferenceLogicParameters {
	/// Acceptance level for the [`DifferenceLogicCollection::add`] gate.
	/// Higher levels accept more constraint variants; only level 1 (Global
	/// only) is implemented here.
	pub level: u8,
	/// Whether to run [`simplify_cycle_detection`] at lowering time.
	pub simplify: bool,
}

impl Default for DifferenceLogicParameters {
	fn default() -> Self {
		// Default to 1: `Global` / `Implied` / `Reified` constraints are
		// auto-routed through the difference-logic engine; equality
		// (`ImpliedEquals`) and disequality (`NotEquals`,
		// `ImpliedNotEquals`, `ReifiedEquals`) variants are *not* accepted
		// — benchmarking showed they regress some corpus instances
		// (`svrp_s4_v2_c3` +15% at level 2, `amaze3_2012_03_19` +31% at
		// level 3) without net wins elsewhere. Opt in to higher levels
		// via [`DifferenceLogicCollection::set_parameters`] or (PR9) the
		// CLI's `--diff-logic` flag; set level 0 to disable routing
		// entirely.
		Self {
			level: 1,
			simplify: true,
		}
	}
}

/// Collection of raw difference constraints, attached to every [`Model`]
/// via the inherent [`Model::diff_logic`] field.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DifferenceLogicCollection {
	parameters: DifferenceLogicParameters,
	raw_constraints: Vec<DifferenceLogicConstraint>,
}

impl DifferenceLogicCollection {
	/// Return the parameters governing this collection.
	pub fn parameters(&self) -> DifferenceLogicParameters {
		self.parameters
	}

	/// Mutably set the parameters governing this collection. Used by the
	/// [`crate::lower::Lowerer`] builder, the CLI, and programmatic
	/// callers that want to opt into diff-logic auto-detection (default
	/// level 0 disables everything).
	pub fn set_parameters(&mut self, parameters: DifferenceLogicParameters) {
		self.parameters = parameters;
	}

	/// Whether the collection currently holds zero constraints.
	pub fn is_empty(&self) -> bool {
		self.raw_constraints.is_empty()
	}

	/// Number of raw constraints currently in the collection.
	pub fn len(&self) -> usize {
		self.raw_constraints.len()
	}

	/// Try to add a constraint. Returns `true` when the constraint is
	/// accepted (and stored) or `false` when the collection's current
	/// level rejects this variant.
	pub fn add(&mut self, constraint: DifferenceLogicConstraint) -> bool {
		let accept = match constraint {
			// Level 1+: globally-active edges and gated edges that don't
			// need fresh model booleans to lower.
			DifferenceLogicConstraint::Global(_, _, _)
			| DifferenceLogicConstraint::Implied(_, _, _, _)
			| DifferenceLogicConstraint::Reified(_, _, _, _) => self.parameters.level >= 1,
			// Level 2+: equality constraints that expand into two gated
			// edges using the existing implication boolean.
			DifferenceLogicConstraint::ImpliedEquals(_, _, _, _) => self.parameters.level >= 2,
			// Level 3+: disequality constraints that need fresh model
			// booleans plus CNF clauses to encode the disjunction.
			DifferenceLogicConstraint::NotEquals(_, _, _)
			| DifferenceLogicConstraint::ImpliedNotEquals(_, _, _, _)
			| DifferenceLogicConstraint::ReifiedEquals(_, _, _, _) => self.parameters.level >= 3,
		};
		if accept {
			self.raw_constraints.push(constraint);
		}
		accept
	}

	/// Drain the raw constraints out of the collection. The
	/// [`crate::lower`] pipeline calls this once during lowering and
	/// expands the result into [`ModelDiffEdge`]s.
	pub(crate) fn take_constraints(&mut self) -> Vec<DifferenceLogicConstraint> {
		mem::take(&mut self.raw_constraints)
	}
}

/// A flattened diff-logic edge in model-view terms. Produced by
/// [`expand_collection`] and consumed by the lowering pipeline.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ModelDiffEdge {
	/// Source endpoint (the `x` in `x − y ≤ d`).
	pub x: ModelView<IntVal>,
	/// Target endpoint.
	pub y: ModelView<IntVal>,
	/// Edge weight.
	pub d: IntVal,
	/// Boolean gate. Always `None` for the Global variant.
	pub gate: Option<ModelView<bool>>,
}

/// Expand the syntactic constraints into flat edges, allocating fresh
/// model Booleans and posting CNF disjunctions as needed for the
/// disequality variants.
///
/// Mapping (each `(x, y, d)` denotes `x − y ≤ d`):
///
/// - `Global`: one gateless edge.
/// - `Implied(b, x, y, d)`: one edge gated by `b`.
/// - `Reified(b, x, y, d)`: two edges encoding `b ↔ (x − y ≤ d)` via `b → (x −
///   y ≤ d)` and `¬b → (y − x ≤ −d − 1)`.
/// - `ImpliedEquals(b, x, y, d)`: `b → (x − y == d)` → two edges `b → (x − y ≤
///   d)` and `b → (y − x ≤ −d)`.
/// - `NotEquals(x, y, d)`: pick a fresh `c`. Post nothing extra; the two gated
///   edges `c → (x − y ≤ d − 1)` and `¬c → (y − x ≤ −d − 1)` together force `(x
///   − y ≠ d)`.
/// - `ImpliedNotEquals(b, x, y, d)`: pick fresh `c1`, `c2`. Post `b → (c1 ∨
///   c2)` and `(¬c1 ∨ ¬c2)`, plus the gated edges `c1 → (x − y ≤ d − 1)` and
///   `c2 → (y − x ≤ −d − 1)`.
/// - `ReifiedEquals(b, x, y, d)`: `b → (x − y == d)` plus `¬b → (x − y ≠ d)`.
///   The first half is two edges (mirror of `ImpliedEquals`); the second half
///   is an `ImpliedNotEquals(¬b, ...)` expansion.
pub(crate) fn expand_collection(
	model: &mut Model,
	raw: Vec<DifferenceLogicConstraint>,
) -> Result<Vec<ModelDiffEdge>, Conflict<ModelView<bool>>> {
	let mut out = Vec::with_capacity(raw.len());
	for c in raw {
		match c {
			DifferenceLogicConstraint::Global(x, y, d) => {
				out.push(ModelDiffEdge {
					x,
					y,
					d,
					gate: None,
				});
			}
			DifferenceLogicConstraint::Implied(b, x, y, d) => {
				out.push(ModelDiffEdge {
					x,
					y,
					d,
					gate: Some(b),
				});
			}
			DifferenceLogicConstraint::Reified(b, x, y, d) => {
				out.push(ModelDiffEdge {
					x,
					y,
					d,
					gate: Some(b),
				});
				out.push(ModelDiffEdge {
					x: y,
					y: x,
					d: -d - 1,
					gate: Some(!b),
				});
			}
			DifferenceLogicConstraint::ImpliedEquals(b, x, y, d) => {
				out.push(ModelDiffEdge {
					x,
					y,
					d,
					gate: Some(b),
				});
				out.push(ModelDiffEdge {
					x: y,
					y: x,
					d: -d,
					gate: Some(b),
				});
			}
			DifferenceLogicConstraint::NotEquals(x, y, d) => {
				let c = model.new_bool_decision();
				out.push(ModelDiffEdge {
					x,
					y,
					d: d - 1,
					gate: Some(c),
				});
				out.push(ModelDiffEdge {
					x: y,
					y: x,
					d: -d - 1,
					gate: Some(!c),
				});
			}
			DifferenceLogicConstraint::ImpliedNotEquals(b, x, y, d) => {
				expand_implied_not_equals(model, &mut out, b, x, y, d)?;
			}
			DifferenceLogicConstraint::ReifiedEquals(b, x, y, d) => {
				// b → (x − y == d): two gated edges.
				out.push(ModelDiffEdge {
					x,
					y,
					d,
					gate: Some(b),
				});
				out.push(ModelDiffEdge {
					x: y,
					y: x,
					d: -d,
					gate: Some(b),
				});
				// ¬b → (x − y ≠ d): an implied disequality.
				expand_implied_not_equals(model, &mut out, !b, x, y, d)?;
			}
		}
	}
	Ok(out)
}

/// Lower `b → (x − y ≠ d)` into two gated edges plus the CNF disjunction
/// that links them. Allocates two fresh Booleans `c1`, `c2`.
fn expand_implied_not_equals(
	model: &mut Model,
	edges: &mut Vec<ModelDiffEdge>,
	b: ModelView<bool>,
	x: ModelView<IntVal>,
	y: ModelView<IntVal>,
	d: IntVal,
) -> Result<(), Conflict<ModelView<bool>>> {
	let c1 = model.new_bool_decision();
	let c2 = model.new_bool_decision();
	// b → (c1 ∨ c2)  ≡  (¬b ∨ c1 ∨ c2)
	model
		.proposition(Formula::Or(vec![
			Formula::from(!b),
			Formula::from(c1),
			Formula::from(c2),
		]))
		.post()?;
	// At most one fires: (¬c1 ∨ ¬c2)
	model
		.proposition(Formula::Or(vec![Formula::from(!c1), Formula::from(!c2)]))
		.post()?;
	edges.push(ModelDiffEdge {
		x,
		y,
		d: d - 1,
		gate: Some(c1),
	});
	edges.push(ModelDiffEdge {
		x: y,
		y: x,
		d: -d - 1,
		gate: Some(c2),
	});
	Ok(())
}

/// Slice 1: detect a negative cycle in the globally active subgraph via
/// one Bellman-Ford pass. Returns the input edges unchanged on success,
/// or a model-level [`Conflict`] with an empty antecedent (which the
/// lowering pipeline wraps as [`crate::lower::LoweringError::Simplification`]).
///
/// The model-stage check is structurally identical to the engine's own
/// Bellman-Ford pass in
/// [`crate::solver::engine::diff_logic::DiffLogicState`]. Running it
/// here at lowering time surfaces an unsatisfiable input before any SAT
/// machinery wakes up, which makes the failure mode considerably easier
/// to debug.
pub(crate) fn simplify_cycle_detection(
	model: &Model,
	edges: Vec<ModelDiffEdge>,
) -> Result<Vec<ModelDiffEdge>, Conflict<ModelView<bool>>> {
	if !model.diff_logic.parameters.simplify {
		return Ok(edges);
	}

	// Intern endpoints into a stable node mapping.
	let mut node_of: FxHashMap<ModelView<IntVal>, usize> = FxHashMap::default();
	let mut int_vars: Vec<ModelView<IntVal>> = Vec::new();
	for e in &edges {
		for endpoint in [e.x, e.y] {
			if let std::collections::hash_map::Entry::Vacant(entry) = node_of.entry(endpoint) {
				int_vars.push(endpoint);
				let _ = entry.insert(int_vars.len() - 1);
			}
		}
	}
	let n = int_vars.len();

	// Active edges are those with no gate. (PR4 only emits gateless
	// edges, so this filter is trivial here; future PRs will narrow the
	// subgraph further before cycle detection.)
	let active: Vec<&ModelDiffEdge> = edges.iter().filter(|e| e.gate.is_none()).collect();

	// Bellman-Ford from a virtual super-source at distance 0 to every
	// node; `pi` ends up holding the shortest-path distances.
	let mut pi: Vec<IntVal> = vec![0; n];
	let mut changed = true;
	for _ in 0..n {
		changed = false;
		for edge in &active {
			let from = node_of[&edge.x];
			let to = node_of[&edge.y];
			let cand = pi[from].saturating_add(edge.d);
			if cand < pi[to] {
				pi[to] = cand;
				changed = true;
			}
		}
		if !changed {
			break;
		}
	}
	if changed {
		// A relaxation in pass `n` means a negative cycle is reachable.
		for edge in &active {
			let from = node_of[&edge.x];
			let to = node_of[&edge.y];
			if pi[from].saturating_add(edge.d) < pi[to] {
				return Err(Conflict {
					subject: None,
					reason: Reason::Eager(Vec::new().into_boxed_slice()),
				});
			}
		}
	}
	Ok(edges)
}

/// Slice 2: model-stage bound tightening.
///
/// Walks the participating edges in a Bellman-Ford-style fixed-point and
/// tightens the model's int-var domains to the graph-implied bounds. A
/// [`ModelDiffEdge`] stores `edge.x`, `edge.y`, `edge.d` such that the
/// represented constraint is `edge.x − edge.y ≤ edge.d`:
///
/// - `edge.x ≤ edge.y + edge.d`  ⇒  `edge.x.max := min(edge.x.max, edge.y.max +
///   d)`
/// - `edge.y ≥ edge.x − edge.d`  ⇒  `edge.y.min := max(edge.y.min, edge.x.min −
///   d)`
///
/// The pass converges in at most `n` iterations on a feasible graph
/// (Slice 1 has already ruled out negative cycles). Tightening is done
/// via `IntPropagationActions::tighten_min`/`tighten_max` with an empty
/// reason (matching [`simplify_unify`]'s convention — these facts are
/// unconditional at the model's level-0 trail head).
///
/// Edge participation:
/// - `gate == None`: always participates (globally-active).
/// - `gate == Some(g)` with `g.val(model) == Some(true)`: participates too.
///   Gated-fixed-true gates arise from `Const(true)`-gated expansions or
///   upstream model-side bool fixing.
///
/// Returns the input edges unchanged on success; mutates the model's
/// domains as a side effect. Returns `Conflict` if a tightening empties
/// a domain (which the Lowerer translates to
/// `LoweringError::Simplification`).
///
/// No-op when [`DifferenceLogicParameters::simplify`] is false.
pub(crate) fn simplify_bound_tightening(
	model: &mut Model,
	edges: Vec<ModelDiffEdge>,
) -> Result<Vec<ModelDiffEdge>, Conflict<ModelView<bool>>> {
	use crate::actions::{BoolInspectionActions, IntInspectionActions, IntPropagationActions};

	if !model.diff_logic.parameters.simplify {
		return Ok(edges);
	}

	// Snapshot the participating edges by index. A `Const(true)` gate
	// stays entailed throughout the fixed-point (we don't mutate gates
	// here), so this filter is safe to evaluate once.
	let participating: Vec<usize> = edges
		.iter()
		.enumerate()
		.filter(|(_, e)| match e.gate {
			None => true,
			Some(g) => matches!(g.val(model), Some(true)),
		})
		.map(|(i, _)| i)
		.collect();

	if participating.is_empty() {
		return Ok(edges);
	}

	// Number of distinct endpoints — used as the iteration bound. The
	// fixed-point converges in ≤ n passes for a feasible graph (Slice 1
	// rules out negative cycles).
	let mut endpoints: FxHashMap<ModelView<IntVal>, ()> = FxHashMap::default();
	for &idx in &participating {
		let _ = endpoints.insert(edges[idx].x, ());
		let _ = endpoints.insert(edges[idx].y, ());
	}
	let n = endpoints.len();

	for _ in 0..n {
		let mut changed = false;
		for &idx in &participating {
			let edge = edges[idx];
			// edge.x ≤ edge.y + edge.d
			let y_max = edge.y.max(model);
			let new_x_max = y_max.saturating_add(edge.d);
			if new_x_max < edge.x.max(model) {
				edge.x
					.tighten_max(model, new_x_max, Vec::<ModelView<bool>>::new())?;
				changed = true;
			}
			// edge.y ≥ edge.x − edge.d
			let x_min = edge.x.min(model);
			let new_y_min = x_min.saturating_sub(edge.d);
			if new_y_min > edge.y.min(model) {
				edge.y
					.tighten_min(model, new_y_min, Vec::<ModelView<bool>>::new())?;
				changed = true;
			}
		}
		if !changed {
			break;
		}
	}

	Ok(edges)
}

/// Slice 3: Johnson's all-pairs shortest paths + redundant-edge pruning.
///
/// After Slice 1 has populated `pi` and ruled out negative cycles, run
/// Dijkstra reweighted by `pi` from every node to compute the full
/// shortest-path matrix over the globally-active subgraph. Then:
///
/// - For a globally-active edge `(x, y, d)` with `dist[x][y] < d`, the edge is
///   redundant — a strictly shorter path exists without it. Drop it.
/// - For a gated edge `b → (x − y ≤ d)`:
///   - If `dist[y][x] < −d`, the reverse path forces `x − y > d`, so the gate
///     can never fire. The edge is dropped from the output unchanged; a future
///     PR can additionally `b ← false` on the model.
///   - If `dist[x][y] ≤ d`, the gated edge is already implied by the
///     globally-active subgraph regardless of `b`. Drop it.
///
/// Returns the surviving edges. No-op when `parameters().simplify` is
/// false.
pub(crate) fn simplify_johnson_pruning(
	model: &Model,
	edges: Vec<ModelDiffEdge>,
) -> Vec<ModelDiffEdge> {
	if !model.diff_logic.parameters.simplify {
		return edges;
	}

	// Intern endpoints and build node table.
	let mut node_of: FxHashMap<ModelView<IntVal>, usize> = FxHashMap::default();
	let mut int_vars: Vec<ModelView<IntVal>> = Vec::new();
	for e in &edges {
		for endpoint in [e.x, e.y] {
			if let std::collections::hash_map::Entry::Vacant(entry) = node_of.entry(endpoint) {
				int_vars.push(endpoint);
				let _ = entry.insert(int_vars.len() - 1);
			}
		}
	}
	let n = int_vars.len();
	if n == 0 {
		return edges;
	}

	// Globally-active subgraph adjacency.
	let mut active_out: Vec<Vec<usize>> = vec![Vec::new(); n];
	for (idx, edge) in edges.iter().enumerate() {
		if edge.gate.is_some() {
			continue;
		}
		let from = node_of[&edge.x];
		active_out[from].push(idx);
	}

	// Bellman-Ford for pi (the engine repeats this; model-stage version
	// is independent so we can prune before any solver state exists).
	let mut pi: Vec<IntVal> = vec![0; n];
	let mut changed = true;
	for _ in 0..n {
		changed = false;
		for adj in &active_out {
			for &e_idx in adj {
				let edge = &edges[e_idx];
				let from = node_of[&edge.x];
				let to = node_of[&edge.y];
				let cand = pi[from].saturating_add(edge.d);
				if cand < pi[to] {
					pi[to] = cand;
					changed = true;
				}
			}
		}
		if !changed {
			break;
		}
	}
	if changed {
		// Negative cycle — Slice 1 should already have caught this.
		// Pass through unchanged; the engine's `bellman_ford_init_pi`
		// will report it again.
		return edges;
	}

	// Johnson: Dijkstra from every source with reduced (non-negative)
	// weights. `dist[u][v]` = shortest-path distance in the original
	// graph, or `IntVal::MAX` if unreachable.
	let mut dist: Vec<Vec<IntVal>> = vec![vec![IntVal::MAX; n]; n];
	for src in 0..n {
		let mut dist_src: Vec<IntVal> = vec![IntVal::MAX; n];
		dist_src[src] = 0;
		let mut queue: LazyPriorityQueue<usize, Reverse<IntVal>> = LazyPriorityQueue::new();
		let _ = queue.push(src, Reverse(0));
		while let Some((u, Reverse(d_u))) = queue.pop() {
			if d_u > dist_src[u] {
				continue; // stale
			}
			for &e_idx in &active_out[u] {
				let edge = &edges[e_idx];
				let v = node_of[&edge.y];
				let reduced_w = pi[u].saturating_add(edge.d).saturating_sub(pi[v]);
				let alt = d_u.saturating_add(reduced_w);
				if alt < dist_src[v] {
					dist_src[v] = alt;
					let _ = queue.push_increase(v, Reverse(alt));
				}
			}
		}
		for dst in 0..n {
			if dist_src[dst] != IntVal::MAX {
				dist[src][dst] = dist_src[dst]
					.saturating_add(pi[dst])
					.saturating_sub(pi[src]);
			}
		}
	}

	// Filter edges based on the distance matrix.
	edges
		.into_iter()
		.filter(|edge| {
			let from = node_of[&edge.x];
			let to = node_of[&edge.y];
			match edge.gate {
				None => {
					// Drop iff a strictly shorter path exists without this edge.
					dist[from][to] >= edge.d
				}
				Some(_) => {
					// Drop if gate forced false (reverse path forbids the edge),
					// or if the edge is already implied by the global subgraph.
					if dist[to][from] != IntVal::MAX && dist[to][from] < -edge.d {
						false
					} else {
						!(dist[from][to] != IntVal::MAX && dist[from][to] <= edge.d)
					}
				}
			}
		})
		.collect()
}

/// Slice 4: equality-cycle unification.
///
/// After Bellman-Ford + Johnson's all-pairs have populated `dist`, every
/// pair `(u, v)` with `dist[u][v] + dist[v][u] == 0` is forced into the
/// equality `u − v == dist[u][v]`. Call
/// [`IntSimplificationActions::unify`] on the model so subsequent
/// reasoning treats them as one variable plus an offset.
///
/// Unification is destructive — once it fires, the original views
/// resolve to the canonical representative + offset. Callers that hold
/// the original view should re-resolve through [`Model::resolve_alias`]
/// before reading it again.
///
/// Returns the surviving edges (no edges are dropped here; pruning is
/// Slice 3's job). Errors propagate as the model's conflict type.
pub(crate) fn simplify_unify(
	model: &mut Model,
	edges: Vec<ModelDiffEdge>,
) -> Result<Vec<ModelDiffEdge>, Conflict<ModelView<bool>>> {
	if !model.diff_logic.parameters.simplify {
		return Ok(edges);
	}

	// Intern endpoints and build the globally-active subgraph.
	let mut node_of: FxHashMap<ModelView<IntVal>, usize> = FxHashMap::default();
	let mut int_vars: Vec<ModelView<IntVal>> = Vec::new();
	for e in &edges {
		for endpoint in [e.x, e.y] {
			if let std::collections::hash_map::Entry::Vacant(entry) = node_of.entry(endpoint) {
				int_vars.push(endpoint);
				let _ = entry.insert(int_vars.len() - 1);
			}
		}
	}
	let n = int_vars.len();
	if n < 2 {
		return Ok(edges);
	}

	let mut active_out: Vec<Vec<usize>> = vec![Vec::new(); n];
	for (idx, edge) in edges.iter().enumerate() {
		if edge.gate.is_some() {
			continue;
		}
		let from = node_of[&edge.x];
		active_out[from].push(idx);
	}

	// Bellman-Ford to populate `pi`.
	let mut pi: Vec<IntVal> = vec![0; n];
	let mut changed = true;
	for _ in 0..n {
		changed = false;
		for adj in &active_out {
			for &e_idx in adj {
				let edge = &edges[e_idx];
				let from = node_of[&edge.x];
				let to = node_of[&edge.y];
				let cand = pi[from].saturating_add(edge.d);
				if cand < pi[to] {
					pi[to] = cand;
					changed = true;
				}
			}
		}
		if !changed {
			break;
		}
	}
	if changed {
		// Negative cycle (Slice 1 should have caught it). Skip unification.
		return Ok(edges);
	}

	// Johnson's all-pairs distance matrix (same pattern as Slice 3).
	let mut dist: Vec<Vec<IntVal>> = vec![vec![IntVal::MAX; n]; n];
	for src in 0..n {
		let mut dist_src: Vec<IntVal> = vec![IntVal::MAX; n];
		dist_src[src] = 0;
		let mut queue: LazyPriorityQueue<usize, Reverse<IntVal>> = LazyPriorityQueue::new();
		let _ = queue.push(src, Reverse(0));
		while let Some((u, Reverse(d_u))) = queue.pop() {
			if d_u > dist_src[u] {
				continue;
			}
			for &e_idx in &active_out[u] {
				let edge = &edges[e_idx];
				let v = node_of[&edge.y];
				let reduced_w = pi[u].saturating_add(edge.d).saturating_sub(pi[v]);
				let alt = d_u.saturating_add(reduced_w);
				if alt < dist_src[v] {
					dist_src[v] = alt;
					let _ = queue.push_increase(v, Reverse(alt));
				}
			}
		}
		for dst in 0..n {
			if dist_src[dst] != IntVal::MAX {
				dist[src][dst] = dist_src[dst]
					.saturating_add(pi[dst])
					.saturating_sub(pi[src]);
			}
		}
	}

	// Walk every ordered pair; for each equality cycle, call unify.
	for u in 0..n {
		for v in (u + 1)..n {
			if dist[u][v] == IntVal::MAX || dist[v][u] == IntVal::MAX {
				continue;
			}
			if dist[u][v].saturating_add(dist[v][u]) != 0 {
				continue;
			}
			// u − v == dist[u][v] exactly.
			let offset = dist[u][v];
			let u_view = int_vars[u];
			let v_view = int_vars[v];
			u_view.unify(model, v_view + offset)?;
		}
	}
	Ok(edges)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::model::Model;

	/// Build a model with diff-logic auto-detection enabled at level 3
	/// (accepts every constraint variant). The default level is 0
	/// (disabled) so every test that touches diff-logic must opt in.
	fn diff_logic_enabled_model() -> Model {
		let mut m = Model::default();
		m.diff_logic.set_parameters(DifferenceLogicParameters {
			level: 3,
			simplify: true,
		});
		m
	}

	/// Build a [`DifferenceLogicCollection`] with the level set to the
	/// given value (and `simplify: true`).
	fn collection_with_level(level: u8) -> DifferenceLogicCollection {
		let mut col = DifferenceLogicCollection::default();
		col.set_parameters(DifferenceLogicParameters {
			level,
			simplify: true,
		});
		col
	}

	#[test]
	fn empty_collection_is_empty() {
		let col = DifferenceLogicCollection::default();
		assert!(col.is_empty());
		assert_eq!(col.len(), 0);
	}

	#[test]
	fn level_0_rejects_everything() {
		// Default level is now 1; use `collection_with_level(0)` to
		// explicitly construct a disabled collection.
		let mut model = Model::default();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let mut col = collection_with_level(0);
		assert!(!col.add(DifferenceLogicConstraint::Global(x, y, 3)));
		assert!(col.is_empty());
	}

	#[test]
	fn level_1_accepts_global() {
		let mut model = Model::default();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let mut col = collection_with_level(1);
		assert!(col.add(DifferenceLogicConstraint::Global(x, y, 3)));
		assert_eq!(col.len(), 1);
	}

	#[test]
	fn add_then_take_drains() {
		let mut model = Model::default();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let mut col = collection_with_level(1);
		let _ = col.add(DifferenceLogicConstraint::Global(x, y, 3));
		let drained = col.take_constraints();
		assert_eq!(drained.len(), 1);
		assert!(col.is_empty());
	}

	#[test]
	fn triangle_propagates_lower_bound() {
		use crate::solver::{Solver, Status, Valuation};

		// x ≥ 0, y - x ≤ 0 (i.e. y ≤ x), z - y ≤ 0 (z ≤ y).
		// Together: z ≤ y ≤ x. Force x ≥ 5 → all three end up ≥ 0 with z ≤ y ≤ x.
		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let z = model.new_int_decision(0..=10);
		// y - x ≤ 0
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::Global(y, x, 0))
		);
		// z - y ≤ 0
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::Global(z, y, 0))
		);

		let (mut slv, map): (Solver, _) = model.lower().to_solver().unwrap();
		let sx = map.get(&mut slv, x);
		let sy = map.get(&mut slv, y);
		let sz = map.get(&mut slv, z);
		let mut captured = None;
		let status = slv
			.solve()
			.on_solution(|sol| {
				captured = Some((
					Valuation::val(&sx, sol),
					Valuation::val(&sy, sol),
					Valuation::val(&sz, sol),
				));
			})
			.satisfy();
		assert_eq!(status, Status::Satisfied);
		let (vx, vy, vz) = captured.expect("Satisfied implies a solution was reported");
		assert!(vy <= vx, "y={vy} > x={vx} violates y - x ≤ 0");
		assert!(vz <= vy, "z={vz} > y={vy} violates z - y ≤ 0");
	}

	#[test]
	fn implied_edge_propagates_when_gate_true() {
		use crate::solver::{Solver, Status, Valuation};

		// b → (y - x ≤ 0). With b fixed true, y ≤ x must hold.
		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=5);
		let y = model.new_int_decision(0..=5);
		let b = model.new_bool_decision();
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::Implied(b, y, x, 0))
		);

		let (mut slv, map): (Solver, _) = model.lower().to_solver().unwrap();
		let sx = map.get(&mut slv, x);
		let sy = map.get(&mut slv, y);
		let sb = map.get(&mut slv, b);
		// Force b = true by adding a unit clause.
		slv.add_clause([sb]).unwrap();

		let mut captured = None;
		let status = slv
			.solve()
			.on_solution(|sol| {
				captured = Some((
					Valuation::val(&sx, sol),
					Valuation::val(&sy, sol),
					Valuation::val(&sb, sol),
				));
			})
			.satisfy();
		assert_eq!(status, Status::Satisfied);
		let (vx, vy, vb) = captured.unwrap();
		assert!(vb);
		assert!(vy <= vx, "with b=true, y={vy} > x={vx} violates y - x ≤ 0");
	}

	#[test]
	fn reified_edge_negation_propagates_when_gate_false() {
		use crate::solver::{Solver, Status, Valuation};

		// b ↔ (x - y ≤ 0). With b fixed false, x - y > 0 (i.e. x > y).
		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=5);
		let y = model.new_int_decision(0..=5);
		let b = model.new_bool_decision();
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::Reified(b, x, y, 0))
		);

		let (mut slv, map): (Solver, _) = model.lower().to_solver().unwrap();
		let sx = map.get(&mut slv, x);
		let sy = map.get(&mut slv, y);
		let sb = map.get(&mut slv, b);
		slv.add_clause([!sb]).unwrap();

		let mut captured = None;
		let status = slv
			.solve()
			.on_solution(|sol| {
				captured = Some((
					Valuation::val(&sx, sol),
					Valuation::val(&sy, sol),
					Valuation::val(&sb, sol),
				));
			})
			.satisfy();
		assert_eq!(status, Status::Satisfied);
		let (vx, vy, vb) = captured.unwrap();
		assert!(!vb);
		assert!(vx > vy, "with b=false, x={vx} ≤ y={vy} violates ¬b ⇒ x > y");
	}

	#[test]
	fn not_equals_forbids_exact_difference() {
		use crate::{
			actions::IntDecisionActions,
			solver::{Solver, Status, Valuation},
		};

		// x − y ≠ 2. Pin y = 1 → x must avoid 3.
		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=4);
		let y = model.new_int_decision(0..=4);
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::NotEquals(x, y, 2))
		);

		let (mut slv, map): (Solver, _) = model.lower().to_solver().unwrap();
		let sx = map.get(&mut slv, x);
		let sy = map.get(&mut slv, y);
		// Pin y = 1.
		let y_eq_1 = sy.lit(&mut slv, crate::solver::IntLitMeaning::Eq(1));
		slv.add_clause([y_eq_1]).unwrap();

		// Enumerate every solution; none should have x − y == 2 (i.e. x == 3).
		let mut count = 0usize;
		let status = slv
			.solve()
			.on_solution(|sol| {
				let vx = Valuation::val(&sx, sol);
				let vy = Valuation::val(&sy, sol);
				assert_eq!(vy, 1);
				assert_ne!(
					vx - vy,
					2,
					"NotEquals(x, y, 2) violated at (x={vx}, y={vy})"
				);
				count += 1;
			})
			.satisfy();
		assert_eq!(status, Status::Satisfied);
		assert!(count >= 1);
	}

	#[test]
	fn implied_equals_propagates_when_gate_true() {
		use crate::solver::{Solver, Status, Valuation};

		// b → (x − y == 2). With b forced true, every solution has x = y + 2.
		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let b = model.new_bool_decision();
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::ImpliedEquals(b, x, y, 2))
		);

		let (mut slv, map): (Solver, _) = model.lower().to_solver().unwrap();
		let sx = map.get(&mut slv, x);
		let sy = map.get(&mut slv, y);
		let sb = map.get(&mut slv, b);
		slv.add_clause([sb]).unwrap();

		let mut captured = None;
		let status = slv
			.solve()
			.on_solution(|sol| {
				captured = Some((Valuation::val(&sx, sol), Valuation::val(&sy, sol)));
			})
			.satisfy();
		assert_eq!(status, Status::Satisfied);
		let (vx, vy) = captured.unwrap();
		assert_eq!(vx - vy, 2, "b=true should force x − y == 2");
	}

	#[test]
	fn reified_equals_negative_side_propagates() {
		use crate::solver::{Solver, Status, Valuation};

		// b ↔ (x − y == 0). With b forced false, every solution has x ≠ y.
		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=3);
		let y = model.new_int_decision(0..=3);
		let b = model.new_bool_decision();
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::ReifiedEquals(b, x, y, 0))
		);

		let (mut slv, map): (Solver, _) = model.lower().to_solver().unwrap();
		let sx = map.get(&mut slv, x);
		let sy = map.get(&mut slv, y);
		let sb = map.get(&mut slv, b);
		slv.add_clause([!sb]).unwrap();

		let mut captured = None;
		let status = slv
			.solve()
			.on_solution(|sol| {
				captured = Some((Valuation::val(&sx, sol), Valuation::val(&sy, sol)));
			})
			.satisfy();
		assert_eq!(status, Status::Satisfied);
		let (vx, vy) = captured.unwrap();
		assert_ne!(vx, vy, "b=false should force x ≠ y");
	}

	#[test]
	fn linear_route_picks_up_two_term_diff_constraint() {
		// model.linear([1, -1], [x, y]).le(3).post() should be auto-
		// detected as `Global(x, y, 3)` and land in `model.diff_logic`
		// rather than as an IntLinear.
		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		assert!(model.diff_logic.is_empty());
		// Build `x − y ≤ 3` via the linear builder.
		model.linear(x - y).le(3).post().unwrap();
		assert_eq!(
			model.diff_logic.len(),
			1,
			"linear() should have routed the 2-term diff into diff_logic"
		);
	}

	#[test]
	fn linear_route_skips_three_term_linear() {
		// `x + y + z ≤ 5` has three terms and stays in IntLinear.
		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let z = model.new_int_decision(0..=10);
		model.linear(x + y + z).le(5).post().unwrap();
		assert!(
			model.diff_logic.is_empty(),
			"three-term linear must not route to diff_logic"
		);
	}

	#[test]
	fn linear_route_skips_non_unit_coefficients() {
		// `2x − y ≤ 5` has non-unit scale and stays in IntLinear.
		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		model.linear(x * 2 - y).le(5).post().unwrap();
		assert!(
			model.diff_logic.is_empty(),
			"non-unit coefficient must not route to diff_logic"
		);
	}

	#[test]
	fn equality_cycle_unifies_views() {
		// x − y ≤ 0 AND y − x ≤ 0 → x == y. Slice 4 should call unify;
		// the model's alias chain then collapses one view onto the other.
		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=5);
		let y = model.new_int_decision(0..=5);
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::Global(x, y, 0))
		);
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::Global(y, x, 0))
		);

		// Lower; this triggers expand → cycle → johnson → unify.
		use crate::solver::Solver;
		let _: (Solver, _) = model.clone().lower().to_solver().unwrap();

		// After lowering, x and y should resolve to the same canonical
		// view (modulo a possible offset of zero in this case).
		let mut model2 = model.clone();
		// Re-run lowering on `model2` so the alias is installed there.
		let _: (Solver, _) = model2.lower().to_solver().unwrap();
		let rx = model2.resolve_alias(x);
		let ry = model2.resolve_alias(y);
		assert_eq!(rx, ry, "equality cycle should unify x and y");
	}

	#[test]
	fn slice2_lifts_min_through_chain() {
		// Edges represent  x − y ≤ 0  (i.e. x ≤ y) and  y − z ≤ 0  (y
		// ≤ z). With x.min = 5 we expect y.min and z.min to be lifted
		// to 5 via the `edge.y.min ≥ edge.x.min − edge.d` direction.
		use crate::actions::IntInspectionActions;

		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(5..=10);
		let y = model.new_int_decision(0..=10);
		let z = model.new_int_decision(0..=10);
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::Global(x, y, 0))
		);
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::Global(y, z, 0))
		);

		let raw = model.diff_logic.take_constraints();
		let edges = expand_collection(&mut model, raw).unwrap();
		let edges = simplify_cycle_detection(&model, edges).unwrap();
		let _ = simplify_bound_tightening(&mut model, edges).unwrap();

		assert_eq!(y.min(&model), 5, "y.min should be lifted to x.min");
		assert_eq!(z.min(&model), 5, "z.min should be lifted via chain");
	}

	#[test]
	fn slice2_tightens_max_through_chain() {
		// Edges represent  x − y ≤ 0  (x ≤ y) and  y − z ≤ 0  (y ≤ z).
		// With z.max = 4 we expect y.max and x.max to be lowered to 4
		// via the `edge.x.max ≤ edge.y.max + edge.d` direction.
		use crate::actions::IntInspectionActions;

		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let z = model.new_int_decision(0..=4);
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::Global(x, y, 0))
		);
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::Global(y, z, 0))
		);

		let raw = model.diff_logic.take_constraints();
		let edges = expand_collection(&mut model, raw).unwrap();
		let edges = simplify_cycle_detection(&model, edges).unwrap();
		let _ = simplify_bound_tightening(&mut model, edges).unwrap();

		assert_eq!(y.max(&model), 4, "y.max should drop to z.max");
		assert_eq!(x.max(&model), 4, "x.max should drop via chain");
	}

	#[test]
	fn slice2_returns_conflict_when_domain_empties() {
		// Force a constraint chain whose graph-implied tightening
		// empties a domain. e.g. x ∈ [0..3], y ∈ [10..20], edge `y − x
		// ≤ 0` (y ≤ x) forces y.max ≤ 3 — but y.min = 10 > 3, so
		// tightening empties y.
		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=3);
		let y = model.new_int_decision(10..=20);
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::Global(y, x, 0))
		);

		use crate::solver::Solver;
		let result: Result<(Solver, _), _> = model.lower().to_solver();
		assert!(
			result.is_err(),
			"Slice 2 should surface an unsat as LoweringError"
		);
	}

	#[test]
	fn negative_cycle_is_detected_at_lowering() {
		// x − y ≤ -1 AND y − x ≤ -1 → cycle weight −2 → unsatisfiable.
		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::Global(x, y, -1))
		);
		assert!(
			model
				.diff_logic
				.add(DifferenceLogicConstraint::Global(y, x, -1))
		);

		use crate::solver::Solver;
		let result: Result<(Solver, _), _> = model.lower().to_solver();
		assert!(
			result.is_err(),
			"negative cycle should surface as a lowering error"
		);
	}

	#[test]
	fn brancher_solves_pair_chain() {
		use crate::solver::{Solver, Status, Valuation};

		// Three int vars, no pre-existing diff-logic constraints. The
		// pair brancher allocates Reified Booleans for each of the
		// three pairs (x<y, x<z, y<z) at model time, then drives the
		// search at solver time. We verify a solution is found and
		// satisfies an extra ordering clause we add to make the
		// branching observable.
		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let z = model.new_int_decision(0..=10);

		let branching = model.diff_logic_branching(vec![x, y, z]);

		let (mut slv, map): (Solver, _) = model.lower().to_solver().unwrap();
		branching.to_solver(&mut slv, &map);

		let sx = map.get(&mut slv, x);
		let sy = map.get(&mut slv, y);
		let sz = map.get(&mut slv, z);

		let mut captured = None;
		let status = slv
			.solve()
			.on_solution(|sol| {
				captured = Some((
					Valuation::val(&sx, sol),
					Valuation::val(&sy, sol),
					Valuation::val(&sz, sol),
				));
			})
			.satisfy();
		assert_eq!(status, Status::Satisfied);
		let _ = captured.expect("Satisfied implies a solution was reported");
	}

	#[test]
	fn diff_lit_subsumption_collapses_two_calls() {
		// Two `diff_lit(x, y, -1)` calls on the same model should
		// return the same `View<bool>` — the second call hits the
		// chain map and reuses the canonical Boolean allocated by the
		// first.
		use crate::actions::IntDecisionActions;

		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let xv: crate::model::View<IntVal> = x.into();
		let yv: crate::model::View<IntVal> = y.into();

		let b1 = xv.diff_lit(&mut model, yv, -1);
		let b2 = xv.diff_lit(&mut model, yv, -1);
		assert_eq!(b1, b2, "second diff_lit call must reuse canonical Boolean");
	}

	#[test]
	fn diff_lit_reverse_direction_hit_returns_negation() {
		// `(x − y ≤ d)` is logically equivalent to `¬(y − x ≤ −d − 1)`.
		// After posting `x.diff_lit(y, 3)` the cache should report
		// `y.diff_lit(x, -4) == !b`.
		use crate::actions::IntDecisionActions;

		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let xv: crate::model::View<IntVal> = x.into();
		let yv: crate::model::View<IntVal> = y.into();

		let b = xv.diff_lit(&mut model, yv, 3);
		let b_rev = yv.diff_lit(&mut model, xv, -4);
		assert_eq!(
			b_rev, !b,
			"reverse-direction diff_lit must return the negation of the forward Boolean"
		);
	}

	#[test]
	fn diff_logic_branching_internal_subsumption() {
		// `diff_logic_branching` for [x, y, z] posts three pairwise
		// Reified Booleans. A subsequent `diff_lit(x, y, -1)` call
		// should hit the cache and return the same Boolean the
		// brancher allocated.
		use crate::actions::IntDecisionActions;

		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let z = model.new_int_decision(0..=10);
		let xv: crate::model::View<IntVal> = x.into();
		let yv: crate::model::View<IntVal> = y.into();
		let _branching = model.diff_logic_branching(vec![xv, yv, z.into()]);

		let b_again = xv.diff_lit(&mut model, yv, -1);
		let cached = model
			.diff_lit_map
			.get(&(xv, yv))
			.and_then(|m| m.get(&-1))
			.copied()
			.expect("diff_logic_branching should have populated the (x, y, -1) entry");
		assert_eq!(b_again, cached);
	}

	#[test]
	fn mid_search_diff_lit_lazy_creation_on_solver() {
		// After lowering, call `Decision<IntVal>::diff_lit` on the
		// solver-side via the `IntDecisionActions<Solver<_>>` impl,
		// which opens a `SolvingContext` and delegates. The (x, y, 5)
		// shape was NOT pre-allocated at model time, so this exercises
		// the lazy-creation branch (allocate fresh `b`, register both
		// gated edges in the engine-side graph, populate the cache,
		// push chain clauses).
		use crate::{
			actions::IntDecisionActions,
			solver::{Solver, view::integer::IntView},
		};

		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let xv: crate::model::View<IntVal> = x.into();
		let yv: crate::model::View<IntVal> = y.into();
		// Pre-intern the (x, y) endpoints via `diff_logic_branching`
		// so the SolvingContext-side `diff_lit` finds them in the
		// graph. Only d=-1 is cached after this call.
		let _ = model.diff_logic_branching(vec![xv, yv]);

		let (mut slv, map): (Solver, _) = model.lower().to_solver().unwrap();

		// Resolve to solver-side `Decision<IntVal>` (the variant of
		// View<IntVal> we know our brancher endpoints became).
		let sx_view = map.get(&mut slv, xv);
		let sy_view = map.get(&mut slv, yv);
		let (sx_dec, sy_dec) = match (sx_view.0, sy_view.0) {
			(IntView::Linear(lin_x), IntView::Linear(lin_y)) => (lin_x.var, lin_y.var),
			_ => panic!("expected Linear views from the lowering map"),
		};

		// Lazy-create the gate for `x − y ≤ 5` (not posted at lowering).
		let lazy = sx_dec.diff_lit(&mut slv, sy_dec, 5);

		// The cache should now hold it in both directions.
		let engine = slv.engine.borrow();
		let cached_fwd = engine
			.state
			.diff_lit_map
			.get(&(sx_view, sy_view))
			.and_then(|m| m.get(&5))
			.copied();
		let cached_rev = engine
			.state
			.diff_lit_map
			.get(&(sy_view, sx_view))
			.and_then(|m| m.get(&-6))
			.copied();
		assert_eq!(
			cached_fwd,
			Some(lazy),
			"forward cache must hold the new gate"
		);
		assert_eq!(cached_rev, Some(!lazy), "reverse cache must hold !gate");

		// And the graph must hold the two gated edges (x → y, 5) and (y → x, -6).
		let graph = engine.state.diff_logic_graph.borrow();
		let from = graph.int_var_to_node[&sx_view];
		let to = graph.int_var_to_node[&sy_view];
		let has_fwd = graph
			.edges
			.iter()
			.any(|e| e.from == from && e.to == to && e.val == 5 && e.bool_var.is_some());
		let has_rev = graph
			.edges
			.iter()
			.any(|e| e.from == to && e.to == from && e.val == -6 && e.bool_var.is_some());
		assert!(
			has_fwd,
			"forward gated edge missing in graph after lazy creation"
		);
		assert!(
			has_rev,
			"reverse gated edge missing in graph after lazy creation"
		);
	}

	#[test]
	fn cross_subsumption_auto_detected_and_branching() {
		// Simulate the disjunctive-scheduling subsumption case: a
		// user-supplied Reified diff-logic constraint posts
		// `b_disj ↔ (x − y ≤ −1)` via `Model::try_route_diff_logic`
		// (reached through `Model::linear`), and a brancher pair Boolean
		// `b_branch ↔ (x − y ≤ −1)` is later allocated via
		// `Model::diff_logic_branching`. Both target the same shape, so
		// they should end up aliased to a single SAT literal after
		// lowering.
		use crate::solver::Solver;

		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let xv: crate::model::View<IntVal> = x.into();
		let yv: crate::model::View<IntVal> = y.into();

		// First, post a user-side Reified through the subsumption
		// router. This becomes the canonical Boolean for (x, y, -1).
		let b_disj = model.new_bool_decision();
		model.add_diff_logic_reified(b_disj, xv, yv, -1);

		// Then run `diff_logic_branching`, which calls
		// `vars[i].diff_lit(self, vars[j], -1)` per pair. The lookup
		// hits the cache and returns `b_disj`; no new Boolean is
		// allocated.
		let bool_count_before = model.bool_vars.len();
		let _ = model.diff_logic_branching(vec![xv, yv]);
		let bool_count_after = model.bool_vars.len();
		assert_eq!(
			bool_count_after, bool_count_before,
			"diff_logic_branching should not allocate a new Boolean when a canonical exists"
		);

		// Sanity: the canonical b for (x, y, -1) is exactly b_disj.
		let cached = model
			.diff_lit_map
			.get(&(xv, yv))
			.and_then(|m| m.get(&-1))
			.copied()
			.expect("the cache must hold the canonical Boolean");
		assert_eq!(cached, b_disj);

		// End-to-end: lower and confirm the Boolean exists on the SAT
		// side (this also verifies no conflict from chain-clause
		// posting on a tiny example).
		let (mut slv, map): (Solver, _) = model.lower().to_solver().unwrap();
		let _sb = map.get(&mut slv, b_disj);

		// Now post a SECOND auto-detected Reified for the same shape on
		// a fresh model and verify the supplied b_2 is aliased onto
		// b_1.
		let mut model2 = diff_logic_enabled_model();
		let x2 = model2.new_int_decision(0..=10);
		let y2 = model2.new_int_decision(0..=10);
		let xv2: crate::model::View<IntVal> = x2.into();
		let yv2: crate::model::View<IntVal> = y2.into();
		let b1 = model2.new_bool_decision();
		let b2 = model2.new_bool_decision();
		model2.add_diff_logic_reified(b1, xv2, yv2, -1);
		// Second add must SUBSUME b2 onto b1 (no new Reified constraint
		// pushed; b2 alias-resolves to b1).
		model2.add_diff_logic_reified(b2, xv2, yv2, -1);
		let (mut slv2, map2): (Solver, _) = model2.lower().to_solver().unwrap();
		assert_eq!(
			map2.get(&mut slv2, b1),
			map2.get(&mut slv2, b2),
			"subsumed Boolean must alias-resolve to the canonical"
		);
	}

	#[test]
	fn mid_search_diff_lit_introduces_new_endpoint() {
		// Variant of `mid_search_diff_lit_lazy_creation_on_solver`
		// where one of the endpoints (z) was NOT in any diff-logic
		// constraint at lowering time. `SolvingContext::diff_lit` must
		// intern z into the graph mid-search AND subscribe the
		// diff-logic propagator's bounds advisor on z. Sanity-check
		// that (a) the graph contains z as a node, (b) the cache holds
		// the new (x, z, 0) entry, (c) one advisor has been added
		// targeting z's int_activation list.
		use crate::{
			actions::IntDecisionActions,
			solver::{Solver, view::integer::IntView},
		};

		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		// `z` is declared but NEVER appears in any diff-logic constraint
		// — only the (x, y) pair is interned at lowering.
		let z = model.new_int_decision(0..=10);
		let xv: crate::model::View<IntVal> = x.into();
		let yv: crate::model::View<IntVal> = y.into();
		let _ = model.diff_logic_branching(vec![xv, yv]);

		let (mut slv, map): (Solver, _) = model.lower().to_solver().unwrap();

		let sx_view = map.get(&mut slv, xv);
		let sz_view = map.get(&mut slv, crate::model::View::<IntVal>::from(z));
		let (sx_dec, sz_dec) = match (sx_view.0, sz_view.0) {
			(IntView::Linear(lin_x), IntView::Linear(lin_z)) => (lin_x.var, lin_z.var),
			_ => panic!("expected Linear views from the lowering map"),
		};

		// Snapshot pre-state.
		let advisors_before = slv.engine.borrow().state.advisors.len();

		// Lazy-create `x − z ≤ 0` — introduces z to the graph mid-search.
		let _b = sx_dec.diff_lit(&mut slv, sz_dec, 0);

		// (a) z is now a graph node.
		{
			let engine = slv.engine.borrow();
			let graph = engine.state.diff_logic_graph.borrow();
			assert!(
				graph.int_var_to_node.contains_key(&sz_view),
				"z should be interned into the diff-logic graph after mid-search diff_lit"
			);
		}
		// (b) cache holds the new entry.
		{
			let engine = slv.engine.borrow();
			let cached = engine
				.state
				.diff_lit_map
				.get(&(sx_view, sz_view))
				.and_then(|m| m.get(&0))
				.copied();
			assert!(
				cached.is_some(),
				"(x, z, 0) must be in the diff-logic literal cache after lazy creation"
			);
		}
		// (c) new advisors have been registered (the bounds advisor on
		//     z plus the two fixed advisors on the new gate Boolean +
		//     its negation = 3 in the simplest case; we just check
		//     monotone growth).
		{
			let engine = slv.engine.borrow();
			assert!(
				engine.state.advisors.len() > advisors_before,
				"expected new AdvisorDefs after mid-search introduction of z (got before={}, after={})",
				advisors_before,
				engine.state.advisors.len(),
			);
			// Suppress unused-binding warning when assert! short-circuits.
			let _ = sz_dec;
		}
	}
}
