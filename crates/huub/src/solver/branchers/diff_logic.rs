//! Pair-based diff-logic brancher.
//!
//! Given an array of integer variables `[x_0, x_1, ..., x_{n-1}]`, for each
//! pair `(i, j)` with `i < j` the brancher performs two-way branching on
//! `x_i < x_j` vs `x_i ≥ x_j` via a reified Boolean `b_{ij} ↔ (x_i < x_j)`
//! (i.e. the diff-logic gate of `x_i − x_j ≤ −1`).
//!
//! The gates are allocated **lazily**: rather than posting all `n·(n−1)/2`
//! reified Booleans before search, the brancher creates each gate on demand
//! in [`Brancher::decide`] via [`IntDecisionActions::diff_lit`]
//! (get-or-create), the first time it reaches that pair while the pair is still
//! undecided. This avoids the up-front `O(n²)` Boolean/edge allocation when
//! propagation or an earlier decision already fixes most pair orders. The
//! global difference-logic propagator must be registered (the model sets
//! `has_diff_logic_emitter` when a diff-logic branching is posted) so that
//! fixing a gate actually enforces the corresponding order.
//!
//! `decide()` is a linear sweep from a trailed cursor over the pairs in
//! lexicographic order: get-or-create the gate for the current pair and branch
//! on it if unfixed, otherwise advance.

use crate::{
	IntVal,
	actions::{
		BoolInspectionActions, BrancherInitActions, DecisionActions, IntDecisionActions,
		ReasoningContext, Trailed,
	},
	solver::{
		branchers::{Brancher, Directive},
		view::View,
	},
};

/// Brancher that does pair-based two-way branching over a user-supplied
/// integer-variable array, allocating the pair gates lazily. See module
/// documentation.
#[derive(Clone, Debug)]
pub struct DiffLogicBrancher {
	/// Integer views to order. Only unit-scaled `Linear` views are kept (see
	/// [`Self::new_in`]) — they are the only valid `diff_lit` endpoints; Const,
	/// Boolean-backed, and scaled views are filtered out. Pairs are taken over
	/// this vector in lexicographic order.
	vars: Vec<View<IntVal>>,
	/// Trailed cursor: the lexicographic index of the next pair to consider.
	/// Pairs before this index have already been decided.
	next: Trailed<usize>,
}

impl DiffLogicBrancher {
	/// Construct a [`DiffLogicBrancher`] over `vars` and push it onto the
	/// solver's brancher queue.
	///
	/// Only **unit-scaled `Linear`** views are retained: `diff_lit` (used to
	/// create the pair gates) is defined only for unit-scaled `Linear×Linear`
	/// pairs, so Const views (fixed value, nothing to order), Boolean-backed
	/// integer views, and scaled views are dropped — silently ordering just the
	/// compatible subset rather than panicking. If fewer than two compatible
	/// views remain, no brancher is posted.
	pub fn new_in(solver: &mut impl BrancherInitActions, vars: Vec<View<IntVal>>) {
		let vars: Vec<_> = vars
			.into_iter()
			.filter(View::is_diff_lit_endpoint)
			.collect();
		if vars.len() < 2 {
			return;
		}
		for &v in &vars {
			solver.ensure_decidable(v);
		}
		let next = solver.new_trailed(0);
		solver.push_brancher(Box::new(DiffLogicBrancher { vars, next }));
	}

	/// Decode a lexicographic pair index `k` into `(i, j)` with `i < j` for an
	/// array of `n` elements. Caller must ensure `k < n·(n−1)/2`.
	fn pair_at(k: usize, n: usize) -> (usize, usize) {
		let mut i = 0;
		let mut rem = k;
		while rem >= n - 1 - i {
			rem -= n - 1 - i;
			i += 1;
		}
		(i, i + 1 + rem)
	}
}

impl<D> Brancher<D> for DiffLogicBrancher
where
	D: DecisionActions + ReasoningContext<Atom = View<bool>>,
	View<IntVal>: IntDecisionActions<D>,
	View<bool>: BoolInspectionActions<D>,
{
	fn decide(&mut self, ctx: &mut D) -> Directive {
		let n = self.vars.len();
		let total = n * (n - 1) / 2;
		let mut k = ctx.trailed(self.next);
		if k >= total {
			return Directive::Exhausted;
		}
		let (mut i, mut j) = Self::pair_at(k, n);
		while k < total {
			// Get-or-create the gate `b ↔ (x_i − x_j ≤ −1)` = `x_i < x_j`.
			let b = self.vars[i].diff_lit(ctx, self.vars[j], -1);
			if BoolInspectionActions::<D>::val(&b, ctx).is_none() {
				ctx.set_trailed(self.next, k);
				return Directive::Select(b);
			}
			k += 1;
			j += 1;
			if j >= n {
				i += 1;
				j = i + 1;
			}
		}
		ctx.set_trailed(self.next, total);
		Directive::Exhausted
	}
}

#[cfg(test)]
mod tests {
	use std::num::NonZero;

	use crate::{
		IntVal,
		solver::{
			Solver, Status,
			branchers::diff_logic::DiffLogicBrancher,
			view::{View, integer::IntView},
		},
	};

	/// `new_in` must keep only unit-scaled `Linear` views: scaled, Const, and
	/// Boolean-backed views are not valid `diff_lit` endpoints and would
	/// otherwise panic when the brancher creates a gate. Mixing such views into
	/// the array must filter them out (ordering just the compatible subset) and
	/// solve without panicking.
	#[test]
	fn new_in_filters_incompatible_views() {
		let mut slv: Solver = Solver::default();
		let a = slv.new_int_decision(0..=3).view();
		let b = slv.new_int_decision(0..=3).view();
		let scaled = a * NonZero::<IntVal>::new(2).unwrap();
		let konst: View<IntVal> = View(IntView::Const(1));
		// The brancher creates gates lazily via `diff_lit`, which needs the
		// global diff-logic propagator registered (lowering normally does this
		// when a diff-logic emitter is present).
		slv.ensure_diff_logic_propagator();
		// Only `a` and `b` are unit-scaled Linear; `scaled` and `konst` must be
		// dropped so the brancher never calls `diff_lit` on an unsupported pair
		// (which would `unimplemented!`-panic).
		DiffLogicBrancher::new_in(&mut slv, vec![scaled, konst, a, b]);
		assert_eq!(slv.solve().satisfy(), Status::Satisfied);
	}
}
