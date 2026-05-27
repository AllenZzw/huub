//! Pair-based diff-logic brancher.
//!
//! Given an array of integer variables `[x_0, x_1, ..., x_{n-1}]`, for
//! each pair `(i, j)` with `i < j` the brancher performs two-way
//! branching on `x_i < x_j` vs `x_i ≥ x_j` via a reified Boolean
//! `b_{ij} ↔ (x_i < x_j)`. The Booleans are allocated and posted as
//! `Reified` diff-logic constraints at model-construction time by
//! [`crate::model::Model::diff_logic_branching`]; the brancher receives
//! them as a flat `Vec<View<bool>>` indexed in lexicographic order.
//!
//! `decide()` is a simple linear sweep from a trailed cursor: pick the
//! first undecided pair-Boolean and branch on it.

use crate::{
	IntVal,
	actions::{BoolInspectionActions, BrancherInitActions, DecisionActions, Trailed},
	solver::{
		branchers::{Brancher, Directive},
		view::{View, boolean::BoolView, integer::IntView},
	},
};

/// Brancher that does pair-based two-way branching over a user-supplied
/// integer-variable array. See module documentation.
#[derive(Clone, Debug)]
pub struct DiffLogicBrancher {
	/// Per-pair reified Boolean: `pair_bools[k]` is the `View<bool>`
	/// that branches `x_i < x_j` (true) vs `x_i ≥ x_j` (false) where
	/// `k` is the lexicographic index of `(i, j)` with `i < j`.
	/// Length = `n * (n - 1) / 2` for a posted array of size `n`.
	pair_bools: Vec<View<bool>>,
	/// Trailed cursor into `pair_bools`: pairs before this index have
	/// already been decided.
	next: Trailed<usize>,
}

impl DiffLogicBrancher {
	/// Construct a [`DiffLogicBrancher`] and push it onto the solver's
	/// brancher queue. `pair_bools` must have length
	/// `vars.len() * (vars.len() - 1) / 2`, indexed lexicographically by
	/// `(i, j)` with `i < j`. If `vars` has fewer than 2 elements no
	/// brancher is posted.
	pub fn new_in(
		solver: &mut impl BrancherInitActions,
		vars: Vec<View<IntVal>>,
		pair_bools: Vec<View<bool>>,
	) {
		let n = vars.len();
		debug_assert_eq!(
			pair_bools.len(),
			n * n.saturating_sub(1) / 2,
			"pair_bools length must be n*(n-1)/2 (lex-indexed by (i, j) with i < j)"
		);
		if pair_bools.is_empty() {
			return; // 0 or 1 var: nothing to branch on.
		}

		for &v in &vars {
			if !matches!(v.0, IntView::Const(_)) {
				solver.ensure_decidable(v);
			}
		}
		for &b in &pair_bools {
			if let BoolView::Lit(_) = b.0 {
				solver.ensure_decidable::<bool>(b);
			}
		}

		let next = solver.new_trailed(0);
		solver.push_brancher(Box::new(DiffLogicBrancher { pair_bools, next }));
	}
}

impl<D> Brancher<D> for DiffLogicBrancher
where
	D: DecisionActions,
	View<bool>: BoolInspectionActions<D>,
{
	fn decide(&mut self, ctx: &mut D) -> Directive {
		let begin = ctx.trailed(self.next);
		if begin >= self.pair_bools.len() {
			return Directive::Exhausted;
		}

		for k in begin..self.pair_bools.len() {
			let b = self.pair_bools[k];
			if BoolInspectionActions::<D>::val(&b, ctx).is_none() {
				ctx.set_trailed(self.next, k);
				return Directive::Select(b);
			}
		}

		ctx.set_trailed(self.next, self.pair_bools.len());
		Directive::Exhausted
	}
}
