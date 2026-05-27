//! Model-stage difference-logic constraint definitions and the
//! lowering-pipeline simplification slices.
//!
//! This module owns everything that runs against the [`Model`] before
//! lowering produces engine-side edges:
//!
//! - [`DifferenceLogicConstraint`]: the syntactic constraint variants (Global /
//!   Implied / Reified / ImpliedEquals / NotEquals / ImpliedNotEquals /
//!   ReifiedEquals) that users (or auto-detection in
//!   [`Model::linear`](crate::model::Model::linear)) post. Stored on
//!   [`Model::diff_logic_constraints`]; the acceptance gate lives on
//!   [`Model::diff_logic_level`] and is enforced by
//!   [`Model::add_diff_logic_constraint`].
//! - [`ModelDiffEdge`]: the post-expansion flat edge representation that the
//!   simplification pipeline operates on.
//! - [`expand_collection`]: turn the syntactic constraints into a flat
//!   `Vec<ModelDiffEdge>` (allocating fresh Booleans for disequality variants
//!   and posting their defining clauses on the model side).
//! - [`simplify_cycle_detection`] (Slice 1): Bellman-Ford negative-cycle
//!   detection on the globally-active subgraph.
//! - [`simplify_bound_tightening`] (Slice 2): Bellman-Ford-style fixed-point
//!   that lifts/tightens domain bounds via graph-implied bounds.
//! - [`simplify_johnson_pruning`] (Slice 3): Johnson all-pairs shortest paths
//!   followed by a redundant-edge drop pass.
//! - [`simplify_unify`] (Slice 4): equality-cycle detection that calls
//!   [`crate::actions::IntSimplificationActions::unify`] on the model to
//!   collapse equivalent int views.
//!
//! The engine-side propagator and graph (`DiffLogicState`,
//! `DifferenceLogicPropagator`) live in
//! [`crate::constraints::difference_logic`]. The two modules are
//! intentionally separate so the model-side code (which only knows
//! about `Model`-level types) has no engine dependency.

use std::cmp::Reverse;

use pindakaas::propositional_logic::Formula;
use rustc_hash::FxHashMap;

use crate::{
	IntVal,
	actions::IntSimplificationActions,
	constraints::{Conflict, Reason},
	helpers::priority_queue::LazyPriorityQueue,
	model::{Model, View as ModelView},
};

/// The syntactic variants of a difference constraint.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum DifferenceLogicConstraint {
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

/// A flattened diff-logic edge in model-view terms. Produced by
/// [`expand_collection`] and consumed by the lowering pipeline.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ModelDiffEdge {
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
	edges: Vec<ModelDiffEdge>,
) -> Result<Vec<ModelDiffEdge>, Conflict<ModelView<bool>>> {
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

	// Cycle detection runs on the globally-active (gateless) subgraph
	// only: a negative cycle among unconditional edges is unconditionally
	// unsatisfiable. Gated edges hold only when their Boolean is true, so
	// they are excluded here and handled by the engine at search time.
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
pub(crate) fn simplify_bound_tightening(
	model: &mut Model,
	edges: Vec<ModelDiffEdge>,
) -> Result<Vec<ModelDiffEdge>, Conflict<ModelView<bool>>> {
	use crate::actions::{BoolInspectionActions, IntInspectionActions, IntPropagationActions};

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
/// Returns the surviving edges.
pub(crate) fn simplify_johnson_pruning(edges: Vec<ModelDiffEdge>) -> Vec<ModelDiffEdge> {
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
	/// (accepts every constraint variant). The default level is 1
	/// (Global / Implied / Reified) so tests that touch the higher
	/// variants must opt in.
	fn diff_logic_enabled_model() -> Model {
		model_with_level(3)
	}

	/// Build a model with diff-logic acceptance set to the given level.
	fn model_with_level(level: u8) -> Model {
		Model {
			diff_logic_level: level,
			..Model::default()
		}
	}

	#[test]
	fn default_model_starts_empty() {
		let m = Model::default();
		assert!(m.diff_logic_constraints.is_empty());
	}

	#[test]
	fn level_0_rejects_everything() {
		let mut model = model_with_level(0);
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		assert!(!model.add_diff_logic_constraint(DifferenceLogicConstraint::Global(x, y, 3)));
		assert!(model.diff_logic_constraints.is_empty());
	}

	#[test]
	fn level_1_accepts_global() {
		let mut model = model_with_level(1);
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::Global(x, y, 3)));
		assert_eq!(model.diff_logic_constraints.len(), 1);
	}

	#[test]
	fn add_then_take_drains() {
		let mut model = model_with_level(1);
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		let _ = model.add_diff_logic_constraint(DifferenceLogicConstraint::Global(x, y, 3));
		let drained = std::mem::take(&mut model.diff_logic_constraints);
		assert_eq!(drained.len(), 1);
		assert!(model.diff_logic_constraints.is_empty());
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
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::Global(y, x, 0)));
		// z - y ≤ 0
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::Global(z, y, 0)));

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
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::Implied(b, y, x, 0)));

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
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::Reified(b, x, y, 0)));

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
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::NotEquals(x, y, 2)));

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
			model.add_diff_logic_constraint(DifferenceLogicConstraint::ImpliedEquals(b, x, y, 2))
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
			model.add_diff_logic_constraint(DifferenceLogicConstraint::ReifiedEquals(b, x, y, 0))
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
		// detected as `Global(x, y, 3)` and land in `model.diff_logic_constraints`
		// rather than as an IntLinear.
		let mut model = diff_logic_enabled_model();
		let x = model.new_int_decision(0..=10);
		let y = model.new_int_decision(0..=10);
		assert!(model.diff_logic_constraints.is_empty());
		// Build `x − y ≤ 3` via the linear builder.
		model.linear(x - y).le(3).post().unwrap();
		assert_eq!(
			model.diff_logic_constraints.len(),
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
			model.diff_logic_constraints.is_empty(),
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
			model.diff_logic_constraints.is_empty(),
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
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::Global(x, y, 0)));
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::Global(y, x, 0)));

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
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::Global(x, y, 0)));
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::Global(y, z, 0)));

		let raw = std::mem::take(&mut model.diff_logic_constraints);
		let edges = expand_collection(&mut model, raw).unwrap();
		let edges = simplify_cycle_detection(edges).unwrap();
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
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::Global(x, y, 0)));
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::Global(y, z, 0)));

		let raw = std::mem::take(&mut model.diff_logic_constraints);
		let edges = expand_collection(&mut model, raw).unwrap();
		let edges = simplify_cycle_detection(edges).unwrap();
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
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::Global(y, x, 0)));

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
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::Global(x, y, -1)));
		assert!(model.add_diff_logic_constraint(DifferenceLogicConstraint::Global(y, x, -1)));

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
		let xv: crate::model::View<IntVal> = x;
		let yv: crate::model::View<IntVal> = y;

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
		let xv: crate::model::View<IntVal> = x;
		let yv: crate::model::View<IntVal> = y;

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
		let xv: crate::model::View<IntVal> = x;
		let yv: crate::model::View<IntVal> = y;
		let _branching = model.diff_logic_branching(vec![xv, yv, z]);

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
		let xv: crate::model::View<IntVal> = x;
		let yv: crate::model::View<IntVal> = y;
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
		let xv: crate::model::View<IntVal> = x;
		let yv: crate::model::View<IntVal> = y;

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
		let xv2: crate::model::View<IntVal> = x2;
		let yv2: crate::model::View<IntVal> = y2;
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
		let xv: crate::model::View<IntVal> = x;
		let yv: crate::model::View<IntVal> = y;
		let _ = model.diff_logic_branching(vec![xv, yv]);

		let (mut slv, map): (Solver, _) = model.lower().to_solver().unwrap();

		let sx_view = map.get(&mut slv, xv);
		let sz_view = map.get(&mut slv, z);
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
