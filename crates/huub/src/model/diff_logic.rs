//! Model-level collection and simplification of difference logic
//! constraints `x − y ≤ d`.
//!
//! [`DifferenceLogicCollection`] lives on every [`Model`] and accumulates
//! raw constraints. [`crate::lower`] drains the collection at lowering
//! time: it expands the syntactic variants, runs [`simplify_cycle_detection`]
//! (Bellman-Ford negative-cycle check), and posts each surviving edge to
//! the engine via [`crate::solver::Solver::add_diff_logic_edge`].
//!
//! ## PR scope
//!
//! This file ships only the **Global** variant of
//! [`DifferenceLogicConstraint`] and **Slice 1** simplification (cycle
//! detection). Implied/Reified expansions, Johnson's pruning (Slice 3),
//! and equality unification (Slice 4) land in subsequent PRs.

use rustc_hash::FxHashMap;

use crate::{
	IntVal,
	constraints::{Conflict, Reason},
	model::{Model, View},
};

/// The syntactic variants of a difference constraint.
///
/// Only the [`Self::Global`] form is accepted in this PR; the remaining
/// variants are present in the enum so the wider plan keeps a stable
/// shape for subsequent PRs but are rejected by
/// [`DifferenceLogicCollection::add`] at the current level.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DifferenceLogicConstraint {
	/// A globally active difference constraint `x − y ≤ d`.
	Global(View<IntVal>, View<IntVal>, IntVal),
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

	/// Mutably set the parameters governing this collection (used by the
	/// [`crate::lower::Lowerer`] builder to thread CLI / programmatic
	/// configuration through).
	pub(crate) fn set_parameters(&mut self, parameters: DifferenceLogicParameters) {
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
			DifferenceLogicConstraint::Global(_, _, _) => self.parameters.level >= 1,
		};
		if accept {
			self.raw_constraints.push(constraint);
		}
		accept
	}

	/// Drain the raw constraints out of the collection. The
	/// [`crate::lower`] pipeline calls this once during lowering and
	/// expands the result into [`DiffEdge`]s.
	pub(crate) fn take_constraints(&mut self) -> Vec<DifferenceLogicConstraint> {
		std::mem::take(&mut self.raw_constraints)
	}
}

/// A flattened diff-logic edge in model-view terms. Produced by
/// [`expand_collection`] and consumed by the lowering pipeline.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DiffEdge {
	/// Source endpoint (the `x` in `x − y ≤ d`).
	pub x: View<IntVal>,
	/// Target endpoint.
	pub y: View<IntVal>,
	/// Edge weight.
	pub d: IntVal,
	/// Boolean gate. Always `None` for the Global variant.
	pub gate: Option<View<bool>>,
}

/// Expand the syntactic constraints into flat edges. For the Global
/// variant this is a 1:1 mapping; more complex variants (Reified,
/// NotEquals, ...) introduce auxiliary boolean variables and lower into
/// multiple edges.
pub(crate) fn expand_collection(
	_model: &mut Model,
	raw: Vec<DifferenceLogicConstraint>,
) -> Vec<DiffEdge> {
	let mut out = Vec::with_capacity(raw.len());
	for c in raw {
		match c {
			DifferenceLogicConstraint::Global(x, y, d) => {
				out.push(DiffEdge {
					x,
					y,
					d,
					gate: None,
				});
			}
		}
	}
	out
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
	edges: Vec<DiffEdge>,
) -> Result<Vec<DiffEdge>, Conflict<View<bool>>> {
	if !model.diff_logic.parameters.simplify {
		return Ok(edges);
	}

	// Intern endpoints into a stable node mapping.
	let mut node_of: FxHashMap<View<IntVal>, usize> = FxHashMap::default();
	let mut int_vars: Vec<View<IntVal>> = Vec::new();
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
	let active: Vec<&DiffEdge> = edges.iter().filter(|e| e.gate.is_none()).collect();

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

#[cfg(test)]
mod tests {
	use super::*;
	use crate::model::Model;

	#[test]
	fn empty_collection_is_empty() {
		let col = DifferenceLogicCollection::default();
		assert!(col.is_empty());
		assert_eq!(col.len(), 0);
	}

	#[test]
	fn level_1_accepts_global() {
		let mut model = Model::default();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let mut col = DifferenceLogicCollection::default();
		assert!(col.add(DifferenceLogicConstraint::Global(x, y, 3)));
		assert_eq!(col.len(), 1);
	}

	#[test]
	fn add_then_take_drains() {
		let mut model = Model::default();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let mut col = DifferenceLogicCollection::default();
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
		let mut model = Model::default();
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
				captured = Some((sx.val(sol), sy.val(sol), sz.val(sol)));
			})
			.satisfy();
		assert_eq!(status, Status::Satisfied);
		let (vx, vy, vz) = captured.expect("Satisfied implies a solution was reported");
		assert!(vy <= vx, "y={vy} > x={vx} violates y - x ≤ 0");
		assert!(vz <= vy, "z={vz} > y={vy} violates z - y ≤ 0");
	}

	#[test]
	fn negative_cycle_is_detected_at_lowering() {
		// x − y ≤ -1 AND y − x ≤ -1 → cycle weight −2 → unsatisfiable.
		let mut model = Model::default();
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
}
