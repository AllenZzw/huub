use pindakaas::{
	solver::{PropagatingSolver, PropagatorAccess, Solver as SolverTrait},
	Valuation as SatValuation,
};

use crate::{
	propagator::int_event::IntEvent,
	solver::{engine::PropRef, view::BoolViewInner, BoolView, IntView, SatSolver},
	Solver,
};

pub(crate) struct InitializationActions<'a, Sat: SatSolver>
where
	<Sat as SolverTrait>::ValueFn: PropagatorAccess,
{
	pub(crate) prop_ref: PropRef,
	pub(crate) slv: &'a mut Solver<Sat>,
}

impl<Sol: PropagatorAccess + SatValuation, Sat: SatSolver + SolverTrait<ValueFn = Sol>>
	InitializationActions<'_, Sat>
{
	#[allow(dead_code)] // TODO
	pub(crate) fn subscribe_bool(&mut self, var: BoolView, data: u32) {
		match var.0 {
			BoolViewInner::Lit(lit) => {
				<Sat as PropagatingSolver>::add_observed_var(&mut self.slv.core, lit.var());
				self.slv
					.engine_mut()
					.bool_subscribers
					.entry(lit.var())
					.or_default()
					.push((self.prop_ref, data))
			}
			BoolViewInner::Const(_) => {}
		}
	}

	pub(crate) fn subscribe_int(&mut self, var: IntView, event: IntEvent, data: u32) {
		use crate::solver::view::IntViewInner::*;

		match var.0 {
			VarRef(var) => self
				.slv
				.engine_mut()
				.int_subscribers
				.entry(var)
				.or_default()
				.push((self.prop_ref, event, data)),
			Const(_) => {}
		}
	}
}
