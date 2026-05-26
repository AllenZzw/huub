//! Main propagation engine of the solver.
//!
//! This module is split into several files:
//!
//! - [`state`]: the [`State`] struct that holds all engine-internal storage
//!   (trail, integer/Boolean variable data, queues, statistics).
//! - [`propagation`]: the [`pindakaas::solver::propagation::Propagator`]
//!   implementation that drives propagation, conflict analysis, and
//!   decision-making, together with the [`LitPropagation`] event record.
//! - [`advisor`]: the [`AdvRef`] / [`AdvisorDef`] types describing how
//!   propagators subscribe to variable change notifications.
//! - [`prop_ref`]: the [`PropRef`] index type for propagator storage.

/// Macro to output a trace message when a new literal is registered.
macro_rules! trace_new_lit {
	($iv:expr, $def:expr, $lit:expr) => {
		tracing::trace!(
			target: "literal",
			lit = i32::from($lit),
			int_var = $iv.ident(),
			is_eq = matches!($def.meaning, IntLitMeaning::Eq(_)),
			val = match $def.meaning {
				IntLitMeaning::Eq(val) => val,
				IntLitMeaning::Less(val) => val,
				_ => unreachable!(),
			},
			"register new literal"
		);
	};
}

pub(crate) mod advisor;
pub(crate) mod prop_ref;
pub(crate) mod propagation;
pub(crate) mod state;

use pindakaas::solver::propagation::{
	ClausePersistence, PropagatorDefinition as PropagatorExtensionDefinition,
};
pub(crate) use trace_new_lit;

pub(crate) use crate::solver::engine::{
	advisor::{AdvRef, AdvisorDef},
	prop_ref::PropRef,
	propagation::LitPropagation,
	state::State,
};
use crate::{
	actions::ReasoningEngine,
	constraints::{BoxedPropagator, Conflict},
	solver::{
		branchers::BoxedBrancher, decision::Decision,
		initialization_context::InitializationContext, solving_context::SolvingContext, view::View,
	},
};

/// A propagation engine implementing the
/// [`pindakaas::solver::propagation::Propagator`] trait.
#[derive(Clone, Debug, Default)]
pub struct Engine {
	/// Storage of the propagators.
	pub(crate) propagators: Vec<BoxedPropagator>,
	/// Storage of the branchers.
	pub(crate) branchers: Vec<BoxedBrancher>,
	/// Internal State representation of the propagation engine.
	pub(crate) state: State,
}

impl PropagatorExtensionDefinition for Engine {
	const CHECK_ONLY: bool = false;
	const REASON_PERSISTENCE: ClausePersistence = ClausePersistence::Forgettable;
}

impl ReasoningEngine for Engine {
	type Atom = View<bool>;
	type Conflict = Conflict<Decision<bool>>;

	type ExplanationContext<'a> = State;
	type InitializationContext<'a> = InitializationContext<'a>;
	type NotificationContext<'a> = State;
	type PropagationContext<'a> = SolvingContext<'a>;
}

#[cfg(test)]
mod tests {
	use pindakaas::solver::propagation::Propagator as ExternalPropagator;

	use crate::{
		IntVal,
		actions::{
			BoolPropagationActions, InitActions, IntDecisionActions, IntEvent, IntInitActions,
			IntPropCond, IntPropagationActions, ReasoningEngine,
		},
		constraints::Propagator,
		solver::{
			BoolView, Decision, IntLitMeaning, LiteralStrategy, Solver, View, engine::Engine,
		},
	};

	/// Regression test for losing an integer notification when a queued
	/// propagation is also implied by another propagated literal.
	///
	/// The propagator emits two consequences in order:
	/// - first `req_first`, then `ge_1_second >= 1`.
	/// - A clause also makes `req_first -> ge_1_second >= 1`.
	///
	/// After the engine returns `req_first` to the SAT solver, the lower-bound
	/// literal is still queued, but its effect is already reflected in the
	/// trailed integer state. When the SAT solver reports both assignments
	/// together, the lower-bound advisor still has to be notified exactly once.
	/// Before the fix, the queued event was purged and this notification was
	/// lost.
	#[test]
	fn queued_integer_event_survives_sat_assignment() {
		use std::{cell::RefCell, rc::Rc};

		#[derive(Clone, Debug)]
		struct ProducerAndListener {
			req_first: Decision<bool>,
			notifications: Rc<RefCell<usize>>,
			ge_1_second: View<IntVal>,
			done: bool,
		}

		impl Propagator<Engine> for ProducerAndListener {
			fn initialize(
				&mut self,
				ctx: &mut <Engine as ReasoningEngine>::InitializationContext<'_>,
			) {
				ctx.enqueue_now(true);
				self.ge_1_second
					.advise_when(ctx, IntPropCond::LowerBound, 0);
			}

			fn advise_of_int_change(
				&mut self,
				_: &mut <Engine as ReasoningEngine>::NotificationContext<'_>,
				data: u64,
				event: IntEvent,
			) -> bool {
				assert_eq!(data, 0);
				assert_eq!(event, IntEvent::LowerBound);
				*self.notifications.borrow_mut() += 1;
				false
			}

			fn propagate(
				&mut self,
				ctx: &mut <Engine as ReasoningEngine>::PropagationContext<'_>,
			) -> Result<(), <Engine as ReasoningEngine>::Conflict> {
				assert!(!self.done);
				self.done = true;
				self.req_first.require(ctx, [])?;
				self.ge_1_second.tighten_min(ctx, 1, [])?;
				Ok(())
			}
		}

		let mut slv: Solver = Solver::default();
		let notifications = Rc::new(RefCell::new(0));
		let imply = slv.new_bool_decision();
		let var = slv
			.new_int_decision(0..=2)
			.order_literals(LiteralStrategy::Eager)
			.view();
		slv.add_propagator(
			Box::new(ProducerAndListener {
				req_first: imply,
				notifications: Rc::clone(&notifications),
				ge_1_second: var,
				done: false,
			}),
			false,
		);
		let ge_view = var.lit(&mut slv, IntLitMeaning::GreaterEq(1));
		let BoolView::Lit(ge) = ge_view.0 else {
			unreachable!()
		};
		// The second consequence is also implied by the first one through SAT.
		slv.add_clause([(!imply).into(), ge_view]).unwrap();

		let (mut actions, mut engine) = slv.as_parts_mut();
		// Running propagate once communicates only the first consequence back to
		// SAT. The lower-bound propagation remains queued, but its bound update is
		// already visible in the integer trail.
		let propagated = ExternalPropagator::propagate(&mut *engine, &mut actions);
		assert_eq!(propagated, Some(imply.0));
		assert_eq!(engine.state.propagation_queue.len(), 1);
		assert_eq!(engine.state.propagation_queue[0].lit, ge.0);

		// SAT now reports both literals together. The queued lower-bound event must
		// survive this path so the advisor is still notified.
		ExternalPropagator::notify_assignments(&mut *engine, &[imply.0, ge.0]);
		assert_eq!(*notifications.borrow(), 1);

		let propagated = ExternalPropagator::propagate(&mut *engine, &mut actions);
		assert_eq!(propagated, None);

		assert_eq!(*notifications.borrow(), 1);
	}

	/// Regression test for the `Err(true)` reason path.
	///
	/// When a propagator's reason atoms are all `BoolView::Const(true)`
	/// — e.g. `view.lit(ctx, GreaterEq(v))` where `v` is at or below
	/// the view's original domain min — `Reason::from_view` filters
	/// them out and returns `Err(true)`. Semantically this is "the
	/// propagation is universally entailed; no antecedents needed."
	///
	/// Before the fix, `register_reason(_, Err(true))` removed the
	/// reason map entry; at non-zero decision levels the debug-only
	/// `debug_check_reason` then panicked even though `add_reason_clause`
	/// would have correctly returned a tautological unit clause.
	///
	/// The fix stores an explicit empty-conjunction `Reason::Eager` in
	/// the map so the invariant ("registered reason for propagated
	/// literals at non-zero levels") holds. The reason clause delivered
	/// to SAT is unchanged: `vec![propagated_lit]`.
	#[test]
	fn err_true_reason_does_not_panic_at_nonzero_level() {
		use std::{cell::RefCell, rc::Rc};

		use crate::actions::BoolInitActions;

		#[derive(Clone, Debug)]
		struct EmptyReasonOnTrigger {
			trigger: Decision<bool>,
			target: View<IntVal>,
			fired: Rc<RefCell<bool>>,
		}

		impl Propagator<Engine> for EmptyReasonOnTrigger {
			fn initialize(
				&mut self,
				ctx: &mut <Engine as ReasoningEngine>::InitializationContext<'_>,
			) {
				self.trigger.advise_when_fixed(ctx, 0);
			}

			fn advise_of_bool_change(
				&mut self,
				_: &mut <Engine as ReasoningEngine>::NotificationContext<'_>,
				_: u64,
			) -> bool {
				true
			}

			fn propagate(
				&mut self,
				ctx: &mut <Engine as ReasoningEngine>::PropagationContext<'_>,
			) -> Result<(), <Engine as ReasoningEngine>::Conflict> {
				if *self.fired.borrow() {
					return Ok(());
				}
				*self.fired.borrow_mut() = true;
				// `tighten_min` with an empty reason vector → `Err(true)`
				// reason — the exact path we want to exercise.
				self.target.tighten_min(ctx, 1, Vec::<View<bool>>::new())?;
				Ok(())
			}
		}

		let mut slv: Solver = Solver::default();
		let trigger = slv.new_bool_decision();
		let target = slv
			.new_int_decision(0..=2)
			.order_literals(LiteralStrategy::Eager)
			.view();
		let target_ge_1 = target.lit(&mut slv, IntLitMeaning::GreaterEq(1));
		let BoolView::Lit(target_ge_1_lit) = target_ge_1.0 else {
			unreachable!()
		};

		let fired = Rc::new(RefCell::new(false));
		slv.add_propagator(
			Box::new(EmptyReasonOnTrigger {
				trigger,
				target,
				fired: Rc::clone(&fired),
			}),
			false,
		);

		let (mut actions, mut engine) = slv.as_parts_mut();

		// Push to decision level 1, then assign the trigger boolean to
		// wake the propagator. With the fix, the engine's
		// `debug_check_reason` accepts the `Err(true)`-resulting
		// propagation without panicking.
		ExternalPropagator::notify_new_decision_level(&mut *engine);
		ExternalPropagator::notify_assignments(&mut *engine, &[trigger.0]);
		let propagated = ExternalPropagator::propagate(&mut *engine, &mut actions);
		assert_eq!(propagated, Some(target_ge_1_lit.0));
		assert!(*fired.borrow());

		// The reason clause for a universally-entailed propagation is the
		// tautological unit `[propagated_lit]`.
		let clause = ExternalPropagator::add_reason_clause(&mut *engine, target_ge_1_lit.0);
		assert_eq!(clause, vec![target_ge_1_lit.0]);
	}

	/// Practical regression for the same `Err(true)` reason path using a
	/// real propagator (`IntArrayMinimumBounds`).
	///
	/// Scenario: `min = array_min(a, b, c)` where `a ∈ 0..=100`,
	/// `b ∈ 0..=5`, `c ∈ 0..=50`, and `min ∈ 0..=200`. The propagator is
	/// posted with `from_model = true` so its level-0 fix-point pass is
	/// skipped (`simplify` / model lowering normally consume the path on
	/// `develop`; this matches the situation that arises in `radiation_i6_9`
	/// once diff-logic auto-detection lands and the propagator re-fires
	/// after a search decision instead of at level 0).
	///
	/// At decision level 1 we assign `c < 10`, which tightens `c`'s upper
	/// bound from 50 to 9 and wakes the bounds advisor. Inside
	/// `propagate`:
	///   min_ub      = min(100, 5, 9) = 5
	///   min_ub_var  = b   (still at its original upper bound 5)
	///   reason      = [b.max_lit(ctx)] = [BoolView::Const(true)]
	///
	/// The single-atom reason filters to empty, `Reason::from_view`
	/// returns `Err(true)`, and the engine registers an empty-eager
	/// reason. Before the fix, `debug_check_reason` panicked here; with
	/// the fix the explanation pipeline produces the tautological unit
	/// clause `[min ≤ 5]` and search continues.
	#[test]
	fn err_true_reason_in_int_array_minimum_at_nonzero_level() {
		use crate::constraints::int_array_minimum::IntArrayMinimumBounds;

		let mut slv: Solver = Solver::default();
		let a = slv
			.new_int_decision(0..=100)
			.order_literals(LiteralStrategy::Eager)
			.view();
		let b = slv
			.new_int_decision(0..=5)
			.order_literals(LiteralStrategy::Eager)
			.view();
		let c = slv
			.new_int_decision(0..=50)
			.order_literals(LiteralStrategy::Eager)
			.view();
		let min = slv
			.new_int_decision(0..=200)
			.order_literals(LiteralStrategy::Eager)
			.view();

		// Bound-tightening literals we need to drive the engine and to
		// match against the expected propagation result.
		let c_lt_10 = c.lit(&mut slv, IntLitMeaning::Less(10));
		let BoolView::Lit(c_lt_10_lit) = c_lt_10.0 else {
			unreachable!()
		};
		let min_lt_6 = min.lit(&mut slv, IntLitMeaning::Less(6));
		let BoolView::Lit(min_lt_6_lit) = min_lt_6.0 else {
			unreachable!()
		};

		// Post the propagator with `from_model = true` so it is *not*
		// enqueued at level 0 — emulating the situation where the
		// propagator's first opportunity to fire happens after a
		// decision has been made.
		slv.add_propagator(
			Box::new(IntArrayMinimumBounds {
				vars: vec![a, b, c],
				min,
			}),
			true,
		);

		let (mut actions, mut engine) = slv.as_parts_mut();

		// Level 0: propagate is a no-op because the propagator was
		// posted with `from_model = true`.
		assert_eq!(
			ExternalPropagator::propagate(&mut *engine, &mut actions),
			None,
		);

		// Push to decision level 1 and assign `c < 10`. This drops
		// `c`'s upper bound from 50 to 9 and fires the bounds advisor,
		// which re-enqueues `IntArrayMinimumBounds`.
		ExternalPropagator::notify_new_decision_level(&mut *engine);
		ExternalPropagator::notify_assignments(&mut *engine, &[c_lt_10_lit.0]);

		// Dequeue and run the propagator. With the engine fix it
		// produces `min < 6` (i.e. `min ≤ 5`); without the fix the
		// `debug_check_reason` assertion panics here.
		let propagated = ExternalPropagator::propagate(&mut *engine, &mut actions);
		assert_eq!(propagated, Some(min_lt_6_lit.0));

		// The reason clause is the tautological unit: SAT treats it as
		// a level-0 unit, matching the universal-entailment semantics.
		let clause = ExternalPropagator::add_reason_clause(&mut *engine, min_lt_6_lit.0);
		assert_eq!(clause, vec![min_lt_6_lit.0]);
	}
}
