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
//! Ships all seven [`DifferenceLogicConstraint`] variants and Slices 1,
//! 3, and 4 of model-stage simplification: Bellman-Ford cycle detection,
//! Johnson's all-pairs redundant-edge pruning, and equality-cycle
//! unification (via [`crate::actions::IntSimplificationActions::unify`]).

use pindakaas::propositional_logic::Formula;
use rustc_hash::FxHashMap;

use crate::{
	IntVal,
	actions::IntSimplificationActions,
	constraints::{Conflict, Reason},
	model::{Model, View},
};

/// The syntactic variants of a difference constraint.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DifferenceLogicConstraint {
	/// A globally active difference constraint `x − y ≤ d`.
	Global(View<IntVal>, View<IntVal>, IntVal),
	/// An implied difference constraint `b → (x − y ≤ d)`.
	Implied(View<bool>, View<IntVal>, View<IntVal>, IntVal),
	/// A reified difference constraint `b ↔ (x − y ≤ d)`.
	Reified(View<bool>, View<IntVal>, View<IntVal>, IntVal),
	/// An implied equality `b → (x − y == d)`.
	ImpliedEquals(View<bool>, View<IntVal>, View<IntVal>, IntVal),
	/// A disequality `x − y ≠ d`.
	NotEquals(View<IntVal>, View<IntVal>, IntVal),
	/// An implied disequality `b → (x − y ≠ d)`.
	ImpliedNotEquals(View<bool>, View<IntVal>, View<IntVal>, IntVal),
	/// A reified equality `b ↔ (x − y == d)`.
	ReifiedEquals(View<bool>, View<IntVal>, View<IntVal>, IntVal),
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
			level: 3,
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
) -> Result<Vec<DiffEdge>, Conflict<View<bool>>> {
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
			DifferenceLogicConstraint::Implied(b, x, y, d) => {
				out.push(DiffEdge {
					x,
					y,
					d,
					gate: Some(b),
				});
			}
			DifferenceLogicConstraint::Reified(b, x, y, d) => {
				out.push(DiffEdge {
					x,
					y,
					d,
					gate: Some(b),
				});
				out.push(DiffEdge {
					x: y,
					y: x,
					d: -d - 1,
					gate: Some(!b),
				});
			}
			DifferenceLogicConstraint::ImpliedEquals(b, x, y, d) => {
				out.push(DiffEdge {
					x,
					y,
					d,
					gate: Some(b),
				});
				out.push(DiffEdge {
					x: y,
					y: x,
					d: -d,
					gate: Some(b),
				});
			}
			DifferenceLogicConstraint::NotEquals(x, y, d) => {
				let c = model.new_bool_decision();
				out.push(DiffEdge {
					x,
					y,
					d: d - 1,
					gate: Some(c),
				});
				out.push(DiffEdge {
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
				out.push(DiffEdge {
					x,
					y,
					d,
					gate: Some(b),
				});
				out.push(DiffEdge {
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
	edges: &mut Vec<DiffEdge>,
	b: View<bool>,
	x: View<IntVal>,
	y: View<IntVal>,
	d: IntVal,
) -> Result<(), Conflict<View<bool>>> {
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
	edges.push(DiffEdge {
		x,
		y,
		d: d - 1,
		gate: Some(c1),
	});
	edges.push(DiffEdge {
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
pub(crate) fn simplify_johnson_pruning(model: &Model, edges: Vec<DiffEdge>) -> Vec<DiffEdge> {
	if !model.diff_logic.parameters.simplify {
		return edges;
	}

	// Intern endpoints and build node table.
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
		let mut queue: crate::helpers::priority_queue::LazyPriorityQueue<
			usize,
			std::cmp::Reverse<IntVal>,
		> = crate::helpers::priority_queue::LazyPriorityQueue::new();
		let _ = queue.push(src, std::cmp::Reverse(0));
		while let Some((u, std::cmp::Reverse(d_u))) = queue.pop() {
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
					let _ = queue.push_increase(v, std::cmp::Reverse(alt));
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
	edges: Vec<DiffEdge>,
) -> Result<Vec<DiffEdge>, Conflict<View<bool>>> {
	if !model.diff_logic.parameters.simplify {
		return Ok(edges);
	}

	// Intern endpoints and build the globally-active subgraph.
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
		let mut queue: crate::helpers::priority_queue::LazyPriorityQueue<
			usize,
			std::cmp::Reverse<IntVal>,
		> = crate::helpers::priority_queue::LazyPriorityQueue::new();
		let _ = queue.push(src, std::cmp::Reverse(0));
		while let Some((u, std::cmp::Reverse(d_u))) = queue.pop() {
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
					let _ = queue.push_increase(v, std::cmp::Reverse(alt));
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
	fn implied_edge_propagates_when_gate_true() {
		use crate::solver::{Solver, Status, Valuation};

		// b → (y - x ≤ 0). With b fixed true, y ≤ x must hold.
		let mut model = Model::default();
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
				captured = Some((sx.val(sol), sy.val(sol), sb.val(sol)));
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
		let mut model = Model::default();
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
				captured = Some((sx.val(sol), sy.val(sol), sb.val(sol)));
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
		let mut model = Model::default();
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
				let vx = sx.val(sol);
				let vy = sy.val(sol);
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
		let mut model = Model::default();
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
				captured = Some((sx.val(sol), sy.val(sol)));
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
		let mut model = Model::default();
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
				captured = Some((sx.val(sol), sy.val(sol)));
			})
			.satisfy();
		assert_eq!(status, Status::Satisfied);
		let (vx, vy) = captured.unwrap();
		assert_ne!(vx, vy, "b=false should force x ≠ y");
	}

	#[test]
	fn equality_cycle_unifies_views() {
		// x − y ≤ 0 AND y − x ≤ 0 → x == y. Slice 4 should call unify;
		// the model's alias chain then collapses one view onto the other.
		let mut model = Model::default();
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
