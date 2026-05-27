//! The [`SolvingContext`] structure used to take actions during propagation
//! and solution checking.
//!
//! This structure contains the implementation of the action traits that are
//! exposed to propagators.

use std::fmt::{self, Debug, Formatter};

use pindakaas::solver::propagation::SolvingActions;
use tracing::trace;

use crate::{
	IntSet, IntVal,
	actions::{
		BoolInspectionActions, BoolPropagationActions, DecisionActions, IntDecisionActions,
		IntEvent, IntInspectionActions, IntPropCond, IntPropagationActions, PropagationActions,
		ReasoningContext, ReasoningEngine, Trailed, TrailingActions,
	},
	constraints::{Conflict, DeferredReason, Reason, ReasonBuilder},
	helpers::bytes::Bytes,
	solver::{
		BoxedPropagator, IntLitMeaning,
		activation_list::ActivationAction,
		decision::{Decision, integer::LazyLitDef},
		engine::{AdvRef, AdvisorDef, Engine, LitPropagation, PropRef, State, trace_new_lit},
		view::{View, boolean::BoolView, integer::IntView},
	},
};

/// Argument type for [`SolvingContext::propagate_int`] to communicate what
/// change to make to the integer decision variable.
///
/// Note that this enum is slightly different from [`IntLitMeaning`] in that it
/// represents the actual upper bound (less-eq), rather than
/// [`IntLitMeaning::Less`], which has to add `1` potentially causing overflow.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ChangeRequest {
	/// Set the lower bound of the integer decision variable to the given value.
	SetLowerBound(IntVal),
	/// Set the upper bound of the integer decision variable to the given value.
	SetUpperBound(IntVal),
	/// Set the value of the integer decision variable to the given value.
	SetValue(IntVal),
	/// Remove the given value from the domain of the integer decision variable.
	RemoveValue(IntVal),
}

/// Type used to communicate whether a change is redundant, conflicting, or new.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ChangeType {
	/// Change is redundant, no action needs to be taken.
	Redundant,
	/// Change is new and should be propagated.
	New,
	/// Change is conflicting, and a conflict should be raised.
	Conflicting,
}

/// Helper struct that temporarily captures a built reason to print it for
/// `tracing`.
struct ReasonTracePrint<'a>(&'a Result<Reason<Decision<bool>>, bool>);

/// Structure to hold the internal [`State`] of the propagation engine and the
/// [`SolvingActions`] exposed by the SAT solver.
///
/// This structure is used to run the propagators that have been scheduled.
///
/// Note that this structure is public to the user to allow the user to
/// construct [`BoxedPropagator`] and [`BoxedBrancher`], but it is not intended
/// to be constructed by the user. It should merely be seen as the
/// implementation of the [`PropagationActions`] trait.
pub struct SolvingContext<'a> {
	/// Actions to create new variables in the solver.
	pub(crate) slv: &'a mut dyn SolvingActions,
	/// Engine state object.
	pub(crate) state: &'a mut State,
	/// Current propagator being executed.
	pub(crate) current_prop: PropRef,
}

impl BoolInspectionActions<SolvingContext<'_>> for Decision<bool> {
	fn val(&self, ctx: &SolvingContext<'_>) -> Option<bool> {
		self.val(ctx.state)
	}
}

impl<'a> BoolPropagationActions<SolvingContext<'a>> for Decision<bool> {
	fn fix(
		&self,
		ctx: &mut SolvingContext<'a>,
		val: bool,
		reason: impl ReasonBuilder<SolvingContext<'a>>,
	) -> Result<(), Conflict<Decision<bool>>> {
		if val { *self } else { !(*self) }.require(ctx, reason)
	}

	fn require(
		&self,
		ctx: &mut SolvingContext<'a>,
		reason: impl ReasonBuilder<SolvingContext<'a>>,
	) -> Result<(), Conflict<Decision<bool>>> {
		match self.val(&ctx.state.trail) {
			Some(true) => Ok(()),
			Some(false) => Err(Conflict::new(ctx, Some(*self), reason)),
			None => {
				ctx.propagate_lit(*self, reason, None);
				Ok(())
			}
		}
	}
}

impl IntDecisionActions<SolvingContext<'_>> for Decision<IntVal> {
	fn lit(&self, ctx: &mut SolvingContext<'_>, meaning: IntLitMeaning) -> View<bool> {
		let var = &mut ctx.state.int_vars[self.idx()];
		let new_var = |def: LazyLitDef| {
			// Create new variable
			let v = ctx.slv.new_observed_var();
			ctx.state.statistics.lazy_literals += 1;
			ctx.state.trail.grow_to_boolvar(v);
			trace_new_lit!(*self, def, v);
			ctx.state.bool_to_int.insert_lazy(v, *self, def.meaning);
			// Add clauses to define the new variable
			for cl in def.meaning.defining_clauses(
				v.into(),
				def.prev.map(Into::into),
				def.next.map(Into::into),
			) {
				ctx.state.clauses.push_back(cl);
			}
			v
		};
		var.lit(meaning, new_var).0
	}

	/// Engine-side diff-logic literal lookup with mid-search lazy creation.
	///
	/// Cache (populated at lowering time from `Model::diff_lit_map` and
	/// extended on demand here) is keyed by solver-side `View<IntVal>`
	/// ordered pair with an inner `BTreeMap<IntVal, View<bool>>` chain.
	///
	/// On a cache miss this implementation:
	/// 1. Interns any brand-new endpoint into the diff-logic graph and
	///    subscribes the diff-logic propagator's bounds advisor on it so
	///    subsequent bound changes wake the propagator.
	/// 2. Allocates a fresh Reified gating Boolean via
	///    `ctx.slv.new_observed_var()` and subscribes a "fixed" advisor on it
	///    so SAT decisions on the gate wake the propagator.
	/// 3. Registers the forward + reverse gated edges (`x − y ≤ d` and `y − x ≤
	///    −d − 1`) in the graph.
	/// 4. Populates the cache in BOTH directions.
	/// 5. Pushes order-encoding chain implication clauses to `state.clauses`.
	///
	/// Precondition: the diff-logic propagator must already be
	/// initialised (`DiffLogicState::propagator_ref` set). For the
	/// model→engine lowering path this holds by construction — the first
	/// `Solver::add_diff_logic_edge` call auto-registers the propagator.
	fn diff_lit(&self, ctx: &mut SolvingContext<'_>, other: Self, d: IntVal) -> View<bool> {
		let x: View<IntVal> = (*self).into();
		let y: View<IntVal> = other.into();

		// 1. Cache hits, both directions.
		if let Some(b) = ctx.state.diff_lit_map.get(&(x, y)).and_then(|m| m.get(&d)) {
			return *b;
		}
		if let Some(b) = ctx
			.state
			.diff_lit_map
			.get(&(y, x))
			.and_then(|m| m.get(&(-d - 1)))
		{
			return !*b;
		}

		// 2. Probe forward chain neighbours BEFORE allocating.
		let prev_b = ctx
			.state
			.diff_lit_map
			.get(&(x, y))
			.and_then(|m| m.range(..d).next_back().map(|(_, &b)| b));
		let next_b = ctx
			.state
			.diff_lit_map
			.get(&(x, y))
			.and_then(|m| m.range((d + 1)..).next().map(|(_, &b)| b));

		// 3. Intern endpoints; subscribe bounds advisor on newly-interned endpoints so
		//    future bound changes wake the propagator. `Rc::clone` decouples the graph
		//    borrow from the simultaneous `&mut ctx.state.trail` borrow.
		let graph_rc = std::rc::Rc::clone(&ctx.state.diff_logic_graph);
		let (x_node, x_was_new, y_node, y_was_new) = {
			let mut graph = graph_rc.borrow_mut();
			let x_present = graph.int_var_to_node.contains_key(&x);
			let y_present = graph.int_var_to_node.contains_key(&y);
			let x_node = graph.intern_int(&mut ctx.state.trail, x);
			let y_node = graph.intern_int(&mut ctx.state.trail, y);
			(x_node, !x_present, y_node, !y_present)
		};
		if x_was_new {
			ctx.subscribe_diff_logic_int_bounds_advisor(x, x_node as u64);
		}
		if y_was_new {
			ctx.subscribe_diff_logic_int_bounds_advisor(y, y_node as u64);
		}

		// 4. Allocate a fresh SAT variable for the new gate.
		let raw_var = ctx.slv.new_observed_var();
		ctx.state.statistics.lazy_literals += 1;
		ctx.state.trail.grow_to_boolvar(raw_var);
		let new_lit: pindakaas::Lit = raw_var.into();
		let b: View<bool> = View(BoolView::Lit(Decision(new_lit)));

		// 5. Register both gated edges + intern the gate Boolean (and its negation)
		//    into the graph so the propagator can index them.
		let (gate_node, neg_node) = {
			let mut graph = graph_rc.borrow_mut();
			let gate_node = graph.intern_bool(&mut ctx.state.trail, b);
			let neg_node = graph.intern_bool(&mut ctx.state.trail, !b);
			let _ = graph.register_edge(&mut ctx.state.trail, x, y, d, Some(b));
			let _ = graph.register_edge(&mut ctx.state.trail, y, x, -d - 1, Some(!b));
			(gate_node, neg_node)
		};

		// 6. Subscribe a "fixed" advisor on both gate variants so SAT decisions on the
		//    new gate wake the propagator. The data payload matches the bool-node index
		//    the propagator passes to `advise_bool_fixed`.
		ctx.subscribe_diff_logic_bool_fixed_advisor(b, gate_node as u64);
		ctx.subscribe_diff_logic_bool_fixed_advisor(!b, neg_node as u64);

		// 7. Populate cache in BOTH directions.
		let _ = ctx
			.state
			.diff_lit_map
			.entry((x, y))
			.or_default()
			.insert(d, b);
		let _ = ctx
			.state
			.diff_lit_map
			.entry((y, x))
			.or_default()
			.insert(-d - 1, !b);

		// 8. Push order-encoding chain implication clauses to the SAT solver via
		//    `state.clauses`. `a → b` becomes `¬a ∨ b`.
		let as_raw = |v: View<bool>| -> pindakaas::Lit {
			match v.0 {
				BoolView::Lit(d) => d.0,
				BoolView::Const(_) => unreachable!(
					"chain neighbour can not be a constant: the cache only stores `Lit`-flavoured \
					 `View<bool>`s allocated via `new_observed_var`"
				),
			}
		};
		if let Some(bp) = prev_b {
			ctx.state.clauses.push_back(vec![!as_raw(bp), as_raw(b)]);
		}
		if let Some(bn) = next_b {
			ctx.state.clauses.push_back(vec![!as_raw(b), as_raw(bn)]);
		}

		b
	}
}

impl IntInspectionActions<SolvingContext<'_>> for Decision<IntVal> {
	fn bounds(&self, ctx: &SolvingContext<'_>) -> (IntVal, IntVal) {
		self.bounds(ctx.state)
	}

	fn domain(&self, ctx: &SolvingContext<'_>) -> IntSet {
		self.domain(ctx.state)
	}

	fn in_domain(&self, ctx: &SolvingContext<'_>, val: IntVal) -> bool {
		self.in_domain(ctx.state, val)
	}

	fn lit_meaning(&self, ctx: &SolvingContext<'_>, lit: View<bool>) -> Option<IntLitMeaning> {
		self.lit_meaning(ctx.state, lit)
	}

	fn max(&self, ctx: &SolvingContext<'_>) -> IntVal {
		self.max(ctx.state)
	}

	fn max_lit(&self, ctx: &SolvingContext<'_>) -> View<bool> {
		self.max_lit(ctx.state)
	}

	fn min(&self, ctx: &SolvingContext<'_>) -> IntVal {
		self.min(ctx.state)
	}

	fn min_lit(&self, ctx: &SolvingContext<'_>) -> View<bool> {
		self.min_lit(ctx.state)
	}

	fn try_lit(&self, ctx: &SolvingContext<'_>, meaning: IntLitMeaning) -> Option<View<bool>> {
		self.try_lit(ctx.state, meaning)
	}

	fn val(&self, ctx: &SolvingContext<'_>) -> Option<IntVal> {
		self.val(ctx.state)
	}
}

impl<'a> IntPropagationActions<SolvingContext<'a>> for Decision<IntVal> {
	fn fix(
		&self,
		ctx: &mut SolvingContext<'a>,
		val: IntVal,
		reason: impl ReasonBuilder<SolvingContext<'a>>,
	) -> Result<(), Conflict<Decision<bool>>> {
		ctx.propagate_int(*self, ChangeRequest::SetValue(val), reason)
	}

	fn remove_val(
		&self,
		ctx: &mut SolvingContext<'a>,
		val: IntVal,
		reason: impl ReasonBuilder<SolvingContext<'a>>,
	) -> Result<(), Conflict<Decision<bool>>> {
		ctx.propagate_int(*self, ChangeRequest::RemoveValue(val), reason)
	}

	fn tighten_max(
		&self,
		ctx: &mut SolvingContext<'a>,
		val: IntVal,
		reason: impl ReasonBuilder<SolvingContext<'a>>,
	) -> Result<(), Conflict<Decision<bool>>> {
		ctx.propagate_int(*self, ChangeRequest::SetUpperBound(val), reason)
	}

	fn tighten_min(
		&self,
		ctx: &mut SolvingContext<'a>,
		val: IntVal,
		reason: impl ReasonBuilder<SolvingContext<'a>>,
	) -> Result<(), Conflict<Decision<bool>>> {
		ctx.propagate_int(*self, ChangeRequest::SetLowerBound(val), reason)
	}

	fn tighten_difference(
		&self,
		ctx: &mut SolvingContext<'a>,
		other: Self,
		d: IntVal,
		reason: impl ReasonBuilder<SolvingContext<'a>>,
	) -> Result<(), Conflict<Decision<bool>>> {
		let b = self.diff_lit(ctx, other, d);
		b.fix(ctx, true, reason)
	}
}

impl Debug for ReasonTracePrint<'_> {
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
		match self.0 {
			Err(false) => write!(f, "false"),
			Err(true) => write!(f, "[]"),
			Ok(Reason::Eager(conj)) => conj
				.iter()
				.map(|&l| l.0.into())
				.collect::<Vec<i32>>()
				.fmt(f),
			Ok(Reason::Lazy(_)) => write!(f, "lazy"),
			&Ok(Reason::Simple(l)) => vec![i32::from(l.0)].fmt(f),
		}
	}
}

impl<'a> SolvingContext<'a> {
	/// Create a new SolvingContext given the solver actions exposed by the SAT
	/// solver and the engine state.
	pub(crate) fn new(slv: &'a mut dyn SolvingActions, state: &'a mut State) -> Self {
		Self {
			slv,
			state,
			current_prop: PropRef::INVALID,
		}
	}

	/// Mid-search analogue of
	/// [`InitializationContext::add_lit_advisor`]
	/// targeting the diff-logic propagator. Subscribes a fixed-event
	/// advisor on the given Boolean view so SAT decisions on it wake the
	/// propagator. No-op for `Const`-flavoured views and for literals
	/// already fixed on the trail.
	pub(crate) fn subscribe_diff_logic_bool_fixed_advisor(&mut self, view: View<bool>, data: u64) {
		let prop_ref = self.state.diff_logic_graph.borrow().propagator_ref.expect(
			"diff-logic propagator must be initialised before subscribing mid-search advisors",
		);
		let lit = match view.0 {
			BoolView::Lit(l) => l,
			BoolView::Const(_) => return,
		};
		if lit.val(&self.state.trail).is_some() {
			// already fixed — advisor never fires
			return;
		}
		self.state.advisors.push(AdvisorDef {
			bool2int: false,
			data,
			negated: false,
			propagator: prop_ref,
		});
		let adv = AdvRef::new(self.state.advisors.len() - 1);
		self.state
			.bool_activation
			.entry(lit.0.var())
			.or_default()
			.push(ActivationAction::<AdvRef, PropRef>::Advise(adv).into());
	}

	/// Mid-search analogue of the diff-logic propagator's
	/// initialisation-time `View<IntVal>::advise_when(ctx, Bounds, data)`.
	/// Subscribes a bounds advisor on the given int view so future
	/// bound changes wake the propagator. Constants are a no-op.
	pub(crate) fn subscribe_diff_logic_int_bounds_advisor(
		&mut self,
		view: View<IntVal>,
		data: u64,
	) {
		let prop_ref = self.state.diff_logic_graph.borrow().propagator_ref.expect(
			"diff-logic propagator must be initialised before subscribing mid-search advisors",
		);
		match view.0 {
			IntView::Linear(lin) => {
				let negated = lin.scale.is_negative();
				self.state.advisors.push(AdvisorDef {
					bool2int: false,
					data,
					negated,
					propagator: prop_ref,
				});
				let adv = AdvRef::new(self.state.advisors.len() - 1);
				self.state.int_activation[lin.var.idx()].add(
					ActivationAction::<AdvRef, PropRef>::Advise(adv),
					IntPropCond::Bounds,
				);
			}
			IntView::Const(_) => {
				// constant — no advisor needed
			}
			IntView::Bool(lin) => {
				// Bool-as-int: subscribe a fixed advisor on the underlying
				// literal with `bool2int: true` so the engine routes the
				// event through `advise_of_int_change`.
				if lin.var.val(&self.state.trail).is_some() {
					return;
				}
				self.state.advisors.push(AdvisorDef {
					bool2int: true,
					data,
					negated: false,
					propagator: prop_ref,
				});
				let adv = AdvRef::new(self.state.advisors.len() - 1);
				self.state
					.bool_activation
					.entry(lin.var.0.var())
					.or_default()
					.push(ActivationAction::<AdvRef, PropRef>::Advise(adv).into());
			}
		}
	}

	/// Internal method used to propagate an integer variable given a literal
	/// description to be enforced.
	#[inline]
	fn propagate_int(
		&mut self,
		iv: Decision<IntVal>,
		change_req: ChangeRequest,
		reason: impl ReasonBuilder<Self>,
	) -> Result<(), Conflict<Decision<bool>>> {
		let (lb, ub) = self.state.int_vars[iv.idx()].bounds(self);
		// Check whether a change is redundant, conflicting, or new with respect to
		// the bounds of an integer variable
		let check = match change_req {
			ChangeRequest::SetValue(i) if lb == i && ub == i => ChangeType::Redundant,
			ChangeRequest::SetValue(i) if i < lb || i > ub => ChangeType::Conflicting,
			ChangeRequest::RemoveValue(i) if i < lb || i > ub => ChangeType::Redundant,
			ChangeRequest::SetLowerBound(i) if i <= lb => ChangeType::Redundant,
			ChangeRequest::SetLowerBound(i) if i > ub => ChangeType::Conflicting,
			ChangeRequest::SetUpperBound(i) if i >= ub => ChangeType::Redundant,
			ChangeRequest::SetUpperBound(i) if i < lb => ChangeType::Conflicting,
			_ => ChangeType::New,
		};

		// Immediate return if there are no further changes
		if check == ChangeType::Redundant {
			return Ok(());
		}

		// Find the right literal, required whether we want to propagate, or raise a
		// conflict
		let new_var = |def: LazyLitDef| {
			// Create new variable
			let v = self.slv.new_observed_var();
			self.state.trail.grow_to_boolvar(v);
			trace_new_lit!(iv, def, v);
			self.state.bool_to_int.insert_lazy(v, iv, def.meaning);
			// Add clauses to define the new variable
			for cl in def.meaning.defining_clauses(
				v.into(),
				def.prev.map(Into::into),
				def.next.map(Into::into),
			) {
				self.state.clauses.push_back(cl);
			}
			v
		};
		let (bv, lit_req) = self.state.int_vars[iv.idx()].lit(
			match change_req {
				ChangeRequest::SetLowerBound(i) => IntLitMeaning::GreaterEq(i),
				ChangeRequest::SetUpperBound(i) => IntLitMeaning::Less(i + 1),
				ChangeRequest::SetValue(i) => IntLitMeaning::Eq(i),
				ChangeRequest::RemoveValue(i) => IntLitMeaning::NotEq(i),
			},
			new_var,
		);

		// Detect propagation conflicts:
		// 1. Always false (and immediate return if always true).
		let lit = match bv.0 {
			BoolView::Const(true) => return Ok(()),
			BoolView::Const(false) => return Err(Conflict::new(self, None, reason)),
			BoolView::Lit(lit) => lit,
		};
		// 2. Bounds check is known to be false.
		if check == ChangeType::Conflicting {
			return Err(Conflict::new(self, lit.into(), reason));
		}
		// 3. Literal is assigned false (and immediate return if assigned true).
		match lit.val(&self.state.trail) {
			Some(true) => return Ok(()),
			Some(false) => return Err(Conflict::new(self, lit.into(), reason)),
			None => {}
		}

		// Normal case:
		// Propagate the literal.
		let event = match lit_req {
			IntLitMeaning::Eq(_) => IntEvent::Fixed,
			IntLitMeaning::NotEq(_) => IntEvent::Domain,
			IntLitMeaning::GreaterEq(i) if i == ub => IntEvent::Fixed,
			IntLitMeaning::GreaterEq(_) => IntEvent::LowerBound,
			IntLitMeaning::Less(i) if i == lb + 1 => IntEvent::Fixed,
			IntLitMeaning::Less(_) => IntEvent::UpperBound,
		};
		self.propagate_lit(lit, reason, Some((iv, event)));
		// Make the domains match.
		match lit_req {
			IntLitMeaning::Eq(val) => {
				self.state.int_vars[iv.idx()].notify_lower_bound(&mut self.state.trail, val);
				self.state.int_vars[iv.idx()].notify_upper_bound(&mut self.state.trail, val);
			}
			IntLitMeaning::NotEq(_) => {}
			IntLitMeaning::GreaterEq(lb) => {
				self.state.int_vars[iv.idx()].notify_lower_bound(&mut self.state.trail, lb);
			}
			IntLitMeaning::Less(ub) => {
				self.state.int_vars[iv.idx()].notify_upper_bound(&mut self.state.trail, ub - 1);
			}
		};
		Ok(())
	}

	/// Internal method used to propagate a Boolean literal.
	///
	/// ## Warning
	///
	/// This method assumes that the literal has not already been assigned, not
	/// even to the same value.
	#[inline]
	fn propagate_lit(
		&mut self,
		lit: Decision<bool>,
		reason: impl ReasonBuilder<Self>,
		event: Option<(Decision<IntVal>, IntEvent)>,
	) {
		let reason = Reason::from_view(reason.build_reason(self));
		trace!(
			target: "solver",
			lit = i32::from(lit.0),
			reason = ?ReasonTracePrint(&reason),
			prop = self.current_prop.index(),
			"propagate"
		);
		self.state.propagation_queue.push_back(LitPropagation {
			lit: lit.0,
			reason,
			event,
		});
		let _prev = self.state.trail.assign_lit(lit.0);
		debug_assert_eq!(_prev, None);
	}

	/// Run the propagators in the queue until a propagator detects a conflict,
	/// returns literals to be propagated by the SAT solver, or the queue is
	/// empty.
	pub(crate) fn run_propagators(&mut self, propagators: &mut [BoxedPropagator]) {
		while let Some(p) = self.state.propagator_queue.pop() {
			debug_assert!(!self.state.failed);
			debug_assert!(self.state.conflict.is_none());
			self.current_prop = PropRef::from_raw(p);
			let prop = propagators[self.current_prop.index()].as_mut();
			let res = prop.propagate(self);
			self.state.statistics.propagations += 1;
			self.current_prop = PropRef::INVALID;
			if let Err(conflict) = res {
				trace!(
					target: "solver",
					lit = conflict
						.subject
						.map(|s| i32::from(s.0))
						.unwrap_or_default(),
					reason = ?ReasonTracePrint(&Ok(conflict.reason.clone())),
					"conflict detected"
				);
				debug_assert!(self.state.conflict.is_none());
				self.state.failed = true;
				self.state.conflict = Some(conflict);
			}
			if self.state.conflict.is_some() || !self.state.propagation_queue.is_empty() {
				return;
			}
		}
	}
}

impl Debug for SolvingContext<'_> {
	fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
		f.debug_struct("SolvingContext")
			.field("state", &self.state)
			.field("current_prop", &self.current_prop)
			.finish()
	}
}

impl DecisionActions for SolvingContext<'_> {
	fn num_conflicts(&self) -> u64 {
		self.state.statistics.conflicts
	}
}

impl PropagationActions for SolvingContext<'_> {
	fn declare_conflict(&mut self, reason: impl ReasonBuilder<Self>) -> Conflict<Decision<bool>> {
		Conflict::new(self, None, reason)
	}

	fn deferred_reason(&self, data: u64) -> DeferredReason {
		DeferredReason {
			propagator: self.current_prop.index() as u32,
			data,
		}
	}
}

impl ReasoningContext for SolvingContext<'_> {
	type Atom = <Engine as ReasoningEngine>::Atom;
	type Conflict = <Engine as ReasoningEngine>::Conflict;
}

impl TrailingActions for SolvingContext<'_> {
	fn set_trailed<T: Bytes>(&mut self, i: Trailed<T>, v: T) -> T {
		self.state.set_trailed(i, v)
	}

	fn trailed<T: Bytes>(&self, i: Trailed<T>) -> T {
		self.state.trailed(i)
	}
}

impl<'a> BoolPropagationActions<SolvingContext<'a>> for View<bool> {
	fn fix(
		&self,
		ctx: &mut SolvingContext<'a>,
		val: bool,
		reason: impl ReasonBuilder<SolvingContext<'a>>,
	) -> Result<(), Conflict<Decision<bool>>> {
		if val { *self } else { !(*self) }.require(ctx, reason)
	}

	fn require(
		&self,
		ctx: &mut SolvingContext<'a>,
		reason: impl ReasonBuilder<SolvingContext<'a>>,
	) -> Result<(), Conflict<Decision<bool>>> {
		match self.0 {
			BoolView::Lit(lit) => lit.require(ctx, reason),
			BoolView::Const(false) => Err(Conflict::new(ctx, None, reason)),
			BoolView::Const(true) => Ok(()),
		}
	}
}
