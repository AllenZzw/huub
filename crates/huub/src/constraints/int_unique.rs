//! Structure and algorithms for the integer all different constraint, which
//! enforces that a list of integer variables each take a different value.

use std::cmp;

use itertools::{Either, Itertools};
use rustc_hash::FxHashSet;
use tracing::{info, warn};

use crate::{
	IntVal,
	actions::{
		InitActions, IntEvent, IntInspectionActions, IntPropCond, PostingActions,
		PropagationActions, ReasoningEngine,
	},
	constraints::{
		Constraint, IntModelActions, IntSolverActions, Propagator, SimplificationStatus,
	},
	lower::{LoweringContext, LoweringError},
	model::View,
	solver::{IntLitMeaning, engine::Engine, queue::PriorityLevel},
};

/// Representation of the integer `unique` constraint within a model.
///
/// This constraint enforces that all the given integer decisions take different
/// values.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct IntUnique {
	/// Instance of the [`IntUniqueBounds`] propagator.
	pub(crate) bounds_prop: IntUniqueBounds<View<IntVal>>,
	/// Instance of the [`IntUniqueValue`] propagator.
	pub(crate) value_prop: IntUniqueValue<View<IntVal>>,
	/// Whether to enable the bounds consistent propagator.
	///
	/// Defaults to `true`.
	pub(crate) bounds_propagation: Option<bool>,
	/// Whether to enable the value consistent propagator.
	///
	/// Defaults to `false`.
	pub(crate) value_propagation: Option<bool>,
	/// Whether to enable the cumulative-slack skip propagator (replaces the
	/// baseline `IntUniqueBounds` when true). Defaults to `false`.
	///
	/// Defaults to `false`.
	pub(crate) cumulative_slack_propagation: Option<bool>,
}

/// Bounds consistent propagator for the integer `unique` constraint.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct IntUniqueBounds<I> {
	/// List of integer variables that must take different values.
	pub(crate) var: Vec<I>,
	/// Struct to store information about variable
	var_info: Vec<UniqueVarMeta>,
	/// Cached lower bounds
	lb_cache: Vec<IntVal>,
	/// Cached upper bounds
	ub_cache: Vec<IntVal>,
	/// Index (from vars) of all variables sorted by min bound
	min_sorted: Vec<usize>,
	/// Index (from vars) of all variables sorted by max bound
	max_sorted: Vec<usize>,
	/// Number of different bounds
	num_bounds: usize,
	/// Ordered vector of all different max and min bounds with dummies
	bounds: Vec<IntVal>,
	/// The critical capacity pointers; that is, `predecessor[i]` points to the
	/// predecessor of i in the `bounds` list.
	predecessor: Vec<usize>,
	/// The diﬀerences between critical capacities; that is `diff[i]` is the
	/// diﬀerence of capacities between `bounds[i]` and its predecessor element
	/// in the list `bounds[predecessor[i]]`
	diff: Vec<IntVal>,
	/// The Hall interval pointers; that is, if `hall_interval[i] < i` then the
	/// half-open interval [`bounds[hall_interval[i]]`, `bounds[i]`) is
	/// contained in a Hall interval, and otherwise holds a pointer to the Hall
	/// interval it belongs to. This Hall interval is represented by a tree,
	/// with the root containing the value of its right end.
	hall_interval: Vec<usize>,
	/// Hall interval bucket transitions
	bucket: Vec<usize>,
}

/// Value consistent propagator for the integer `unique` constraint.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct IntUniqueValue<I> {
	/// List of integer variables that must take different values.
	vars: Vec<I>,
	/// List of (indexes of) variable signaled to be fixed.
	action_list: Vec<usize>,
}

/// Information that is tracked for each variable for the propagation of
/// [`IntUniqueBounds`]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct UniqueVarMeta {
	/// Transition for the variable's position in the Hall interval tree.
	next: usize,
	/// Minimum index in the [`IntUniqueBounds::bounds`] vector
	min_rank: usize,
	/// Maximum index in the [`IntUniqueBounds::bounds`] vector
	max_rank: usize,
}

impl IntUnique {
	/// Returns whether a bounds consistent propagator will be posted when
	/// creating a [`Solver`](crate::solver::Solver) object.
	pub fn bounds_propagation(&self) -> bool {
		self.bounds_propagation.unwrap_or(true)
	}

	/// Returns whether a value consistent propagator will be posted when
	/// creating a [`Solver`](crate::solver::Solver) object.
	pub fn value_propagation(&self) -> bool {
		self.value_propagation.unwrap_or(false)
	}

	/// Returns whether the cumulative-slack skip propagator will be posted
	/// when creating a [`Solver`](crate::solver::Solver) object. When true it
	/// replaces the baseline [`IntUniqueBounds`].
	pub fn cumulative_slack_propagation(&self) -> bool {
		self.cumulative_slack_propagation.unwrap_or(false)
	}
}

impl<E> Constraint<E> for IntUnique
where
	E: ReasoningEngine,
	View<IntVal>: IntModelActions<E>,
{
	fn simplify(
		&mut self,
		ctx: &mut E::PropagationContext<'_>,
	) -> Result<SimplificationStatus, E::Conflict> {
		self.propagate(ctx)?;
		Ok(SimplificationStatus::NoFixpoint)
	}

	fn to_solver(&self, slv: &mut LoweringContext<'_>) -> Result<(), LoweringError> {
		let (_vals, vars): (Vec<_>, Vec<_>) = self.bounds_prop.var.iter().partition_map(|&var| {
			let var = slv.solver_view(var);
			if let Some(val) = var.val(slv) {
				Either::Left(val)
			} else {
				Either::Right(var)
			}
		});
		// Propagation should have detected any duplicate fixed values and removed them
		// from the domains of other decision variables.
		debug_assert!(_vals.iter().unique().collect_vec().len() == _vals.len());
		debug_assert!(
			_vals
				.iter()
				.all(|&val| vars.iter().all(|var| !var.in_domain(slv, val)))
		);

		// If the number of non-fixed decision variables is less than or equal
		// to 1, there is no need to post any propagators.
		if vars.len() <= 1 {
			return Ok(());
		}

		let value_propagation = self.value_propagation();
		if value_propagation {
			IntUniqueValue::post(slv, vars.clone());
		}
		let cumulative_slack = self.cumulative_slack_propagation();
		let mut bounds_propagation = self.bounds_propagation();
		if !value_propagation && !bounds_propagation && !cumulative_slack {
			warn!(
				"all propagation algorithms are disabled for `int_unique` constraint, override with bounds propagation to ensure consistency"
			);
			bounds_propagation = true;
		}
		if cumulative_slack {
			// Cumulative-slack subsumes baseline bounds; do not post both.
			IntUniqueBoundsCumulativeSlack::post(slv, vars);
		} else if bounds_propagation {
			IntUniqueBounds::post(slv, vars);
		}
		Ok(())
	}
}

impl<E> Propagator<E> for IntUnique
where
	E: ReasoningEngine,
	View<IntVal>: IntSolverActions<E>,
{
	fn advise_of_backtrack(&mut self, ctx: &mut E::NotificationContext<'_>) {
		self.value_prop.advise_of_backtrack(ctx);
	}

	fn advise_of_int_change(
		&mut self,
		ctx: &mut E::NotificationContext<'_>,
		data: u64,
		event: IntEvent,
	) -> bool {
		match event {
			IntEvent::Bounds => self.bounds_prop.advise_of_int_change(ctx, data, event),
			IntEvent::Fixed => self.value_prop.advise_of_int_change(ctx, data, event),
			_ => unreachable!("IntUnique should only be advised of Bounds and Fixed events"),
		}
	}

	fn initialize(&mut self, ctx: &mut E::InitializationContext<'_>) {
		self.value_prop.initialize(ctx);
		self.bounds_prop.initialize(ctx);
	}

	fn propagate(&mut self, ctx: &mut E::PropagationContext<'_>) -> Result<(), E::Conflict> {
		if !self.value_prop.action_list.is_empty() {
			self.value_prop.propagate(ctx)?;
		}
		self.bounds_prop.propagate(ctx)
	}
}

impl<I> IntUniqueBounds<I> {
	/// Filter the lower bounds of the considered variables
	fn filter_lower<E>(&mut self, ctx: &mut E::PropagationContext<'_>) -> Result<(), E::Conflict>
	where
		E: ReasoningEngine,
		I: IntSolverActions<E>,
	{
		for i in 1..=self.num_bounds + 1 {
			self.hall_interval[i] = i - 1;
			self.predecessor[i] = i - 1;
			self.diff[i] = self.bounds[i] - self.bounds[i - 1];
			self.bucket[i] = usize::MAX;
		}

		for i in 0..self.var.len() {
			let max_rank = self.var_info[self.max_sorted[i]].max_rank;
			let min_rank = self.var_info[self.max_sorted[i]].min_rank;

			let mut z = Self::path_max(&self.predecessor, min_rank + 1);
			let j = self.predecessor[z];
			self.diff[z] -= 1;
			self.var_info[self.max_sorted[i]].next = self.bucket[z];
			self.bucket[z] = self.max_sorted[i];
			if self.diff[z] == 0 {
				self.predecessor[z] = z + 1;
				z = Self::path_max(&self.predecessor, self.predecessor[z]);
				self.predecessor[z] = j;
			};
			Self::path_set(&mut self.predecessor, min_rank + 1, z, z);

			if self.hall_interval[min_rank] > min_rank {
				let w = Self::path_max(&self.hall_interval, self.hall_interval[min_rank]);
				let hall_max = self.bounds[w];
				let mut hall_min = self.bounds[min_rank];
				let mut k = w;
				while self.bounds[k] > hall_min {
					let mut l = self.bucket[k];
					while l != usize::MAX {
						hall_min = cmp::min(hall_min, self.lb_cache[l]);
						l = self.var_info[l].next;
					}
					k -= 1;
				}

				let mut k = w;
				let mut reason = Vec::new();
				reason.push(
					self.var[self.max_sorted[i]].lit(ctx, IntLitMeaning::GreaterEq(hall_min)),
				);
				while self.bounds[k] > hall_min {
					let mut l = self.bucket[k];
					while l != usize::MAX {
						reason.push(self.var[l].lit(ctx, IntLitMeaning::GreaterEq(hall_min)));
						reason.push(self.var[l].lit(ctx, IntLitMeaning::Less(hall_max)));
						l = self.var_info[l].next;
					}
					k -= 1;
				}

				self.var[self.max_sorted[i]].tighten_min(ctx, hall_max, reason)?;
				self.lb_cache[self.max_sorted[i]] = hall_max;

				Self::path_set(&mut self.hall_interval, min_rank, w, w);
			}
			if self.diff[z] == self.bounds[z] - self.bounds[max_rank] {
				let h_max_rank = self.hall_interval[max_rank];
				// Save Hall interval
				Self::path_set(&mut self.hall_interval, h_max_rank, j - 1, max_rank);
				self.hall_interval[max_rank] = j - 1;
			}
		}
		Ok(())
	}

	/// Filter the upper bounds of the considered variables
	fn filter_upper<E>(&mut self, ctx: &mut E::PropagationContext<'_>) -> Result<(), E::Conflict>
	where
		E: ReasoningEngine,
		I: IntSolverActions<E>,
	{
		for i in 0..=self.num_bounds {
			self.hall_interval[i] = i + 1;
			self.predecessor[i] = i + 1;
			self.diff[i] = self.bounds[i + 1] - self.bounds[i];
			self.bucket[i] = usize::MAX;
		}

		for i in (0..self.var.len()).rev() {
			let max_rank = self.var_info[self.min_sorted[i]].max_rank;
			let min_rank = self.var_info[self.min_sorted[i]].min_rank;

			let mut z = Self::path_min(&self.predecessor, max_rank - 1);
			let j = self.predecessor[z];
			self.diff[z] -= 1;
			self.var_info[self.min_sorted[i]].next = self.bucket[z];
			self.bucket[z] = self.min_sorted[i];
			if self.diff[z] == 0 {
				self.predecessor[z] = z - 1;
				z = Self::path_min(&self.predecessor, self.predecessor[z]);
				self.predecessor[z] = j;
			}
			Self::path_set(&mut self.predecessor, max_rank - 1, z, z);

			if self.hall_interval[max_rank] < max_rank {
				let w = Self::path_min(&self.hall_interval, self.hall_interval[max_rank]);
				let hall_min = self.bounds[w];
				let mut hall_max = self.bounds[max_rank];
				let mut k = w;
				while self.bounds[k] < hall_max {
					let mut l = self.bucket[k];
					while l != usize::MAX {
						hall_max = cmp::max(hall_max, self.ub_cache[l] + 1);
						l = self.var_info[l].next;
					}
					k += 1;
				}

				let mut k = w;
				let mut reason = Vec::new();
				reason.push(self.var[self.min_sorted[i]].lit(ctx, IntLitMeaning::Less(hall_max)));
				while self.bounds[k] < hall_max {
					let mut l = self.bucket[k];
					while l != usize::MAX {
						reason.push(self.var[l].lit(ctx, IntLitMeaning::GreaterEq(hall_min)));
						reason.push(self.var[l].lit(ctx, IntLitMeaning::Less(hall_max)));
						l = self.var_info[l].next;
					}
					k += 1;
				}

				self.var[self.min_sorted[i]].tighten_max(ctx, hall_min - 1, reason)?;
				self.ub_cache[self.min_sorted[i]] = hall_min - 1;

				Self::path_set(&mut self.hall_interval, max_rank, w, w);
			}

			if self.diff[z] == self.bounds[min_rank] - self.bounds[z] {
				let h_min_rank = self.hall_interval[min_rank];
				// Save Hall interval
				Self::path_set(&mut self.hall_interval, h_min_rank, j + 1, min_rank);
				self.hall_interval[min_rank] = j + 1;
			}
		}
		Ok(())
	}

	/// Create a new [`IntUniqueBounds`] propagator.
	pub(crate) fn new(vars: Vec<I>) -> Self {
		let interval = vec![
			UniqueVarMeta {
				next: 0,
				min_rank: 0,
				max_rank: 0
			};
			vars.len()
		];
		let min_sorted: Vec<_> = (0..vars.len()).collect();
		let max_sorted: Vec<_> = (0..vars.len()).collect();

		let n = 2 * vars.len() + 2;
		Self {
			var: vars,
			var_info: interval,
			lb_cache: vec![0; n],
			ub_cache: vec![0; n],
			min_sorted,
			max_sorted,
			num_bounds: 0,
			bounds: vec![0; n],
			predecessor: vec![0; n],
			diff: vec![0; n],
			hall_interval: vec![0; n],
			bucket: vec![0; n],
		}
	}

	/// Follows path given by `transition` from `start` until we stop increasing
	fn path_max(transition: &[usize], mut start: usize) -> usize {
		while transition[start] > start {
			start = transition[start];
		}
		start
	}

	/// Follows path given by `transition` from `start` until we stop decreasing
	fn path_min(transition: &[usize], mut start: usize) -> usize {
		while transition[start] < start {
			start = transition[start];
		}
		start
	}

	/// Sets everything in the `transition` slice, between `start` and `end` to
	/// `to`
	///
	/// # Example
	///
	/// ```ignore
	/// # use huub::constraints::int_all_different::IntUniqueBounds;
	/// let mut transition = vec![4, 2, 0, 1, 3, 0]; // giving e.g. 0 -> 4 -> 3 -> 1 -> 2 -> 0
	/// IntUniqueBounds::path_set(&mut transition, 2, 3, 5);
	/// assert_eq!(transition, vec![5, 2, 5, 1, 5, 0]); // now gives // 0 -> 5 -> 0
	/// ```
	fn path_set(transition: &mut [usize], start: usize, end: usize, to: usize) {
		let mut last;
		let mut cur = start;
		while cur != end {
			last = cur;
			cur = transition[cur];
			transition[last] = to;
		}
	}

	/// Create a new [`IntUniqueBounds`] propagator and post it in the
	/// solver.
	pub fn post<E>(solver: &mut E, vars: Vec<I>)
	where
		E: PostingActions + ?Sized,
		I: IntSolverActions<Engine>,
	{
		solver.add_propagator(Box::new(Self::new(vars)));
	}

	/// Sorts max_sorted and min_sorted and sets the bounds vector
	fn sort<E>(&mut self, ctx: &mut E::PropagationContext<'_>)
	where
		E: ReasoningEngine,
		I: IntSolverActions<E>,
	{
		let size: usize = self.var.len();

		for (i, v) in self.var.iter().enumerate() {
			(self.lb_cache[i], self.ub_cache[i]) = v.bounds(ctx);
		}

		self.min_sorted.sort_by_key(|&i| self.lb_cache[i]);
		self.max_sorted.sort_by_key(|&i| self.ub_cache[i] + 1);

		let mut min: IntVal = self.lb_cache[self.min_sorted[0]];
		let mut max: IntVal = self.ub_cache[self.max_sorted[0]] + 1;
		let mut last: IntVal = min - 2;
		self.bounds[0] = last; // Dummy

		let mut i = 0;
		let mut j = 0;
		self.num_bounds = 0;
		loop {
			if i < size && min <= max {
				if min != last {
					self.num_bounds += 1;
					last = min;
					self.bounds[self.num_bounds] = min;
				}
				self.var_info[self.min_sorted[i]].min_rank = self.num_bounds;
				i += 1;
				if i < size {
					min = self.lb_cache[self.min_sorted[i]];
				}
			} else {
				if max != last {
					self.num_bounds += 1;
					last = max;
					self.bounds[self.num_bounds] = max;
				}
				self.var_info[self.max_sorted[j]].max_rank = self.num_bounds;
				j += 1;
				if j == size {
					break;
				}
				max = self.ub_cache[self.max_sorted[j]] + 1;
			}
		}
		self.bounds[self.num_bounds + 1] = self.bounds[self.num_bounds] + 2; // Dummy
	}
}

impl<E, I> Propagator<E> for IntUniqueBounds<I>
where
	E: ReasoningEngine,
	I: IntSolverActions<E>,
{
	fn initialize(&mut self, ctx: &mut <E as ReasoningEngine>::InitializationContext<'_>) {
		ctx.set_priority(PriorityLevel::Low);
		for v in &self.var {
			v.enqueue_when(ctx, IntPropCond::Bounds);
		}
	}

	#[tracing::instrument(
		name = "int_unique_bounds",
		target = "solver",
		level = "trace",
		skip(self, ctx)
	)]
	fn propagate(&mut self, ctx: &mut E::PropagationContext<'_>) -> Result<(), E::Conflict> {
		self.sort(ctx);
		self.filter_lower(ctx)?;
		self.filter_upper(ctx)?;
		Ok(())
	}
}

impl<I> IntUniqueValue<I> {
	/// Create a new [`IntUniqueValue`] propagator.
	pub(crate) fn new(vars: Vec<I>) -> Self {
		Self {
			vars,
			action_list: Vec::new(),
		}
	}

	/// Create a new [`IntUniqueBounds`] propagator and post it in the
	/// solver.
	pub fn post<E>(solver: &mut E, vars: Vec<I>)
	where
		E: PostingActions + ?Sized,
		I: IntSolverActions<Engine>,
	{
		solver.add_propagator(Box::new(Self::new(vars)));
	}
}

impl<E, I> Propagator<E> for IntUniqueValue<I>
where
	E: ReasoningEngine,
	I: IntSolverActions<E>,
{
	fn advise_of_backtrack(&mut self, _: &mut E::NotificationContext<'_>) {
		// We forget any previously remembered fixed decisions.
		self.action_list.clear();
	}

	fn advise_of_int_change(
		&mut self,
		_: &mut E::NotificationContext<'_>,
		data: u64,
		event: IntEvent,
	) -> bool {
		// We remember that the decision at index `data` has been fixed to a value.
		debug_assert_eq!(event, IntEvent::Fixed);
		self.action_list.push(data as usize);
		true
	}

	fn initialize(&mut self, ctx: &mut E::InitializationContext<'_>) {
		// Let the propagator be advised when each specific decision is fixed to a
		// value, with the index of the decision.
		for (i, v) in self.vars.iter().enumerate() {
			if self.vars[i].val(ctx).is_some() {
				// If the variable is already fixed, then add it to the action list immediately.
				self.action_list.push(i);
				ctx.enqueue_now(true);
			} else {
				v.advise_when(ctx, IntPropCond::Fixed, i as u64);
			}
		}
		// Advise the propagator of backtracking to clear the list of fixed decision
		// (indices).
		ctx.advise_on_backtrack();
	}

	#[tracing::instrument(
		name = "int_unique_value",
		target = "solver",
		level = "trace",
		skip(self, ctx)
	)]
	fn propagate(&mut self, ctx: &mut E::PropagationContext<'_>) -> Result<(), E::Conflict> {
		debug_assert!(!self.action_list.is_empty() && self.action_list.iter().all_unique());
		// We walk through all fixed decisions (indices).
		for &i in &self.action_list {
			// Retrieve the value and value literal for the fixed decision.
			let val = self.vars[i].val(ctx).unwrap();
			let reason = &[self.vars[i].val_lit(ctx).unwrap()];

			// We now enforce that all other decisions (at different indices) are not
			// equal to the fixed value.
			for (j, v) in self.vars.iter().enumerate() {
				if j != i {
					v.remove_val(ctx, val, reason)?;
				}
			}
		}
		// We clear the list of indices of fixed decisions.
		self.action_list.clear();
		Ok(())
	}
}

// ---------------------------------------------------------------------------
// cumulative-slack skip propagator (stand-alone)
//
// Owns its own Puget bounds-filter machinery so experimental skip logic can
// be modified without affecting the baseline `IntUniqueBounds`. Posting this
// propagator subsumes (replaces) the baseline bounds propagator.
//
// Slack table snapshotted at each propagation fixed point:
//     slack[l][u] = (bounds[u] - bounds[l])
//                 - |{i : lb_i >= bounds[l] AND ub_i < bounds[u]}|
// where `bounds` are the sorted distinct critical values lb_i and ub_i+1.
//
// Skip-check produces one of three outcomes per call:
//   - SKIP        — no Hall interval can change; any Cond-2-pushed bounds
//                   are tightened via `tighten_min` / `tighten_max` with a
//                   reason that enumerates the cached Hall intervals plus
//                   their enclosed variables.
//   - CANNOT_SKIP — slack==1 somewhere in the check range; a new Hall
//                   interval might form; run the full scan.
//   - FAILURE     — slack<=0 reached; an enclosing tight interval is
//                   overcrowded; defer to the full scan to build the conflict.
//
// Reference design: testing/alldiff/progress/2026-05-18.md
//                   ("Strategy F_cumulative" / "cumulative-slack skip").
// Naming taxonomy:  testing/alldiff/STRATEGIES.md.
// ---------------------------------------------------------------------------

/// Outcome of one F_CUMULATIVE_CHECK iteration across all updated variables.
/// (A FAILURE result is signalled via `Err(conflict)` rather than this enum.)
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum CumulativeOutcome {
	Skip,
	CannotSkip,
}

/// Stand-alone bounds-consistent `unique` propagator with a cumulative-slack
/// skip table. Owns the Puget filter machinery inline.
#[derive(Clone, Debug)]
pub struct IntUniqueBoundsCumulativeSlack<I> {
	// ---- Inline Puget bounds-filter state (mirrors IntUniqueBounds) ----
	var: Vec<I>,
	var_info: Vec<UniqueVarMeta>,
	lb_cache: Vec<IntVal>,
	ub_cache: Vec<IntVal>,
	min_sorted: Vec<usize>,
	max_sorted: Vec<usize>,
	num_bounds: usize,
	bounds: Vec<IntVal>,
	predecessor: Vec<usize>,
	diff: Vec<IntVal>,
	hall_interval: Vec<usize>,
	bucket: Vec<usize>,
	// ---- Cumulative-slack snapshot state ----
	/// Per-variable lb at last propagation fixed point.
	cached_lb: Vec<IntVal>,
	/// Per-variable ub at last propagation fixed point.
	cached_ub: Vec<IntVal>,
	/// Sorted distinct critical values (lb_i and ub_i+1) at last fixed point.
	s_bounds: Vec<IntVal>,
	s_bounds_len: usize,
	/// 2D slack table, indexed `slack[l * s_bounds_len + u]`.
	slack: Vec<IntVal>,
	/// Per-call working copy; supports multi-variable Phase-3 interaction.
	working: Vec<IntVal>,
	// ---- Bookkeeping ----
	/// Indices of variables changed since the last propagation call.
	var_updated: FxHashSet<usize>,
	pending_backtrack: bool,
	cache_valid: bool,
	// ---- Counters (Drop-emitted) ----
	prop_calls: u64,
	prop_full: u64,
	prop_skips: u64,
	/// Phase-1 (Cond-2) push tightenings committed on the SKIP path.
	prop_pushes: u64,
	prop_failures: u64,
	prop_check_failures: u64,
	prop_backtracks: u64,
}

impl<I> IntUniqueBoundsCumulativeSlack<I> {
	pub(crate) fn new(vars: Vec<I>) -> Self {
		let n_vars = vars.len();
		let interval = vec![
			UniqueVarMeta {
				next: 0,
				min_rank: 0,
				max_rank: 0
			};
			n_vars
		];
		let min_sorted: Vec<_> = (0..n_vars).collect();
		let max_sorted: Vec<_> = (0..n_vars).collect();
		let n = 2 * n_vars + 2;
		Self {
			var: vars,
			var_info: interval,
			lb_cache: vec![0; n],
			ub_cache: vec![0; n],
			min_sorted,
			max_sorted,
			num_bounds: 0,
			bounds: vec![0; n],
			predecessor: vec![0; n],
			diff: vec![0; n],
			hall_interval: vec![0; n],
			bucket: vec![0; n],
			cached_lb: vec![0; n_vars],
			cached_ub: vec![0; n_vars],
			s_bounds: Vec::new(),
			s_bounds_len: 0,
			slack: Vec::new(),
			working: Vec::new(),
			var_updated: FxHashSet::default(),
			pending_backtrack: false,
			cache_valid: false,
			prop_calls: 0,
			prop_full: 0,
			prop_skips: 0,
			prop_pushes: 0,
			prop_failures: 0,
			prop_check_failures: 0,
			prop_backtracks: 0,
		}
	}

	/// Create and post the propagator.
	pub fn post<E>(solver: &mut E, vars: Vec<I>)
	where
		E: PostingActions + ?Sized,
		I: IntSolverActions<Engine>,
	{
		solver.add_propagator(Box::new(Self::new(vars)));
	}

	// ------------------------------------------------------------------
	// Inlined Puget bounds-filter machinery (private; experimentation
	// happens here without touching `IntUniqueBounds`).
	// ------------------------------------------------------------------

	#[inline]
	fn path_max(transition: &[usize], mut start: usize) -> usize {
		while transition[start] > start {
			start = transition[start];
		}
		start
	}

	#[inline]
	fn path_min(transition: &[usize], mut start: usize) -> usize {
		while transition[start] < start {
			start = transition[start];
		}
		start
	}

	fn path_set(transition: &mut [usize], start: usize, end: usize, to: usize) {
		let mut last;
		let mut cur = start;
		while cur != end {
			last = cur;
			cur = transition[cur];
			transition[last] = to;
		}
	}

	fn sort<E>(&mut self, ctx: &mut E::PropagationContext<'_>)
	where
		E: ReasoningEngine,
		I: IntSolverActions<E>,
	{
		let size: usize = self.var.len();
		for (i, v) in self.var.iter().enumerate() {
			(self.lb_cache[i], self.ub_cache[i]) = v.bounds(ctx);
		}
		self.min_sorted.sort_by_key(|&i| self.lb_cache[i]);
		self.max_sorted.sort_by_key(|&i| self.ub_cache[i] + 1);
		let mut min: IntVal = self.lb_cache[self.min_sorted[0]];
		let mut max: IntVal = self.ub_cache[self.max_sorted[0]] + 1;
		let mut last: IntVal = min - 2;
		self.bounds[0] = last;
		let mut i = 0;
		let mut j = 0;
		self.num_bounds = 0;
		loop {
			if i < size && min <= max {
				if min != last {
					self.num_bounds += 1;
					last = min;
					self.bounds[self.num_bounds] = min;
				}
				self.var_info[self.min_sorted[i]].min_rank = self.num_bounds;
				i += 1;
				if i < size {
					min = self.lb_cache[self.min_sorted[i]];
				}
			} else {
				if max != last {
					self.num_bounds += 1;
					last = max;
					self.bounds[self.num_bounds] = max;
				}
				self.var_info[self.max_sorted[j]].max_rank = self.num_bounds;
				j += 1;
				if j == size {
					break;
				}
				max = self.ub_cache[self.max_sorted[j]] + 1;
			}
		}
		self.bounds[self.num_bounds + 1] = self.bounds[self.num_bounds] + 2;
	}

	fn filter_lower<E>(&mut self, ctx: &mut E::PropagationContext<'_>) -> Result<(), E::Conflict>
	where
		E: ReasoningEngine,
		I: IntSolverActions<E>,
	{
		for i in 1..=self.num_bounds + 1 {
			self.hall_interval[i] = i - 1;
			self.predecessor[i] = i - 1;
			self.diff[i] = self.bounds[i] - self.bounds[i - 1];
			self.bucket[i] = usize::MAX;
		}
		for i in 0..self.var.len() {
			let max_rank = self.var_info[self.max_sorted[i]].max_rank;
			let min_rank = self.var_info[self.max_sorted[i]].min_rank;
			let mut z = Self::path_max(&self.predecessor, min_rank + 1);
			let j = self.predecessor[z];
			self.diff[z] -= 1;
			self.var_info[self.max_sorted[i]].next = self.bucket[z];
			self.bucket[z] = self.max_sorted[i];
			if self.diff[z] == 0 {
				self.predecessor[z] = z + 1;
				z = Self::path_max(&self.predecessor, self.predecessor[z]);
				self.predecessor[z] = j;
			};
			Self::path_set(&mut self.predecessor, min_rank + 1, z, z);
			if self.hall_interval[min_rank] > min_rank {
				let w = Self::path_max(&self.hall_interval, self.hall_interval[min_rank]);
				let hall_max = self.bounds[w];
				let mut hall_min = self.bounds[min_rank];
				let mut k = w;
				while self.bounds[k] > hall_min {
					let mut l = self.bucket[k];
					while l != usize::MAX {
						hall_min = cmp::min(hall_min, self.lb_cache[l]);
						l = self.var_info[l].next;
					}
					k -= 1;
				}
				let mut k = w;
				let mut reason = Vec::new();
				reason.push(
					self.var[self.max_sorted[i]].lit(ctx, IntLitMeaning::GreaterEq(hall_min)),
				);
				while self.bounds[k] > hall_min {
					let mut l = self.bucket[k];
					while l != usize::MAX {
						reason.push(self.var[l].lit(ctx, IntLitMeaning::GreaterEq(hall_min)));
						reason.push(self.var[l].lit(ctx, IntLitMeaning::Less(hall_max)));
						l = self.var_info[l].next;
					}
					k -= 1;
				}
				self.var[self.max_sorted[i]].tighten_min(ctx, hall_max, reason)?;
				self.lb_cache[self.max_sorted[i]] = hall_max;
				Self::path_set(&mut self.hall_interval, min_rank, w, w);
			}
			if self.diff[z] == self.bounds[z] - self.bounds[max_rank] {
				let h_max_rank = self.hall_interval[max_rank];
				Self::path_set(&mut self.hall_interval, h_max_rank, j - 1, max_rank);
				self.hall_interval[max_rank] = j - 1;
			}
		}
		Ok(())
	}

	fn filter_upper<E>(&mut self, ctx: &mut E::PropagationContext<'_>) -> Result<(), E::Conflict>
	where
		E: ReasoningEngine,
		I: IntSolverActions<E>,
	{
		for i in 0..=self.num_bounds {
			self.hall_interval[i] = i + 1;
			self.predecessor[i] = i + 1;
			self.diff[i] = self.bounds[i + 1] - self.bounds[i];
			self.bucket[i] = usize::MAX;
		}
		for i in (0..self.var.len()).rev() {
			let max_rank = self.var_info[self.min_sorted[i]].max_rank;
			let min_rank = self.var_info[self.min_sorted[i]].min_rank;
			let mut z = Self::path_min(&self.predecessor, max_rank - 1);
			let j = self.predecessor[z];
			self.diff[z] -= 1;
			self.var_info[self.min_sorted[i]].next = self.bucket[z];
			self.bucket[z] = self.min_sorted[i];
			if self.diff[z] == 0 {
				self.predecessor[z] = z - 1;
				z = Self::path_min(&self.predecessor, self.predecessor[z]);
				self.predecessor[z] = j;
			}
			Self::path_set(&mut self.predecessor, max_rank - 1, z, z);
			if self.hall_interval[max_rank] < max_rank {
				let w = Self::path_min(&self.hall_interval, self.hall_interval[max_rank]);
				let hall_min = self.bounds[w];
				let mut hall_max = self.bounds[max_rank];
				let mut k = w;
				while self.bounds[k] < hall_max {
					let mut l = self.bucket[k];
					while l != usize::MAX {
						hall_max = cmp::max(hall_max, self.ub_cache[l] + 1);
						l = self.var_info[l].next;
					}
					k += 1;
				}
				let mut k = w;
				let mut reason = Vec::new();
				reason.push(self.var[self.min_sorted[i]].lit(ctx, IntLitMeaning::Less(hall_max)));
				while self.bounds[k] < hall_max {
					let mut l = self.bucket[k];
					while l != usize::MAX {
						reason.push(self.var[l].lit(ctx, IntLitMeaning::GreaterEq(hall_min)));
						reason.push(self.var[l].lit(ctx, IntLitMeaning::Less(hall_max)));
						l = self.var_info[l].next;
					}
					k += 1;
				}
				self.var[self.min_sorted[i]].tighten_max(ctx, hall_min - 1, reason)?;
				self.ub_cache[self.min_sorted[i]] = hall_min - 1;
				Self::path_set(&mut self.hall_interval, max_rank, w, w);
			}
			if self.diff[z] == self.bounds[min_rank] - self.bounds[z] {
				let h_min_rank = self.hall_interval[min_rank];
				Self::path_set(&mut self.hall_interval, h_min_rank, j + 1, min_rank);
				self.hall_interval[min_rank] = j + 1;
			}
		}
		Ok(())
	}

	fn full_propagate<E>(&mut self, ctx: &mut E::PropagationContext<'_>) -> Result<(), E::Conflict>
	where
		E: ReasoningEngine,
		I: IntSolverActions<E>,
	{
		self.sort::<E>(ctx);
		self.filter_lower::<E>(ctx)?;
		self.filter_upper::<E>(ctx)?;
		Ok(())
	}

	#[inline]
	fn working_cell(&self, l: usize, u: usize) -> IntVal {
		self.working[l * self.s_bounds_len + u]
	}

	#[inline]
	fn working_dec(&mut self, l: usize, u: usize) {
		self.working[l * self.s_bounds_len + u] -= 1;
	}

	/// Rank of `v` in `s_bounds[..s_bounds_len]`. Returns `s_bounds_len` if not
	/// found (i.e. `v > s_bounds[s_bounds_len - 1]`).
	#[inline]
	fn rank(&self, v: IntVal) -> Option<usize> {
		let m = self.s_bounds_len;
		let r = self.s_bounds[..m].partition_point(|&b| b < v);
		if r < m && self.s_bounds[r] == v {
			Some(r)
		} else {
			None
		}
	}

	/// Rebuild `s_bounds` and `slack` from `cached_lb`/`cached_ub`. O(n + m²).
	fn rebuild_slack(&mut self) {
		let n = self.cached_lb.len();
		// Collect sorted distinct critical values: every lb_i and ub_i+1.
		let mut critical: Vec<IntVal> = Vec::with_capacity(2 * n);
		for i in 0..n {
			critical.push(self.cached_lb[i]);
			critical.push(self.cached_ub[i] + 1);
		}
		critical.sort_unstable();
		critical.dedup();

		self.s_bounds_len = critical.len();
		self.s_bounds = critical;

		let m = self.s_bounds_len;
		self.slack.clear();
		self.slack.resize(m * m, 0);
		// slack[l][u] = (bounds[u] - bounds[l]) - |{i : lb_i >= bounds[l] AND ub_i < bounds[u]}|
		// We only populate cells with l < u (others stay 0 / unused).
		for l in 0..m {
			let bl = self.s_bounds[l];
			for u in (l + 1)..m {
				let bu = self.s_bounds[u];
				let mut enclosed = 0i64;
				for i in 0..n {
					if self.cached_lb[i] >= bl && self.cached_ub[i] < bu {
						enclosed += 1;
					}
				}
				self.slack[l * m + u] = (bu - bl) - enclosed;
			}
		}
	}

	/// Refresh `cached_lb`/`cached_ub` from the solver, then rebuild the
	/// slack table. Called after a full propagation succeeded.
	fn snapshot_after_full<E>(&mut self, ctx: &mut E::PropagationContext<'_>)
	where
		E: ReasoningEngine,
		I: IntSolverActions<E>,
	{
		for i in 0..self.var.len() {
			let (lb, ub) = self.var[i].bounds(ctx);
			self.cached_lb[i] = lb;
			self.cached_ub[i] = ub;
		}
		self.rebuild_slack();
		self.cache_valid = true;
	}

	// ------------------------------------------------------------------
	// Cumulative-slack skip check — broken into small helpers per phase
	// and per direction. Public entry is `cumulative_check`; everything
	// else is private.
	// ------------------------------------------------------------------

	/// Top-level skip check. Iterates updated variables and dispatches to
	/// the per-direction checker. Returns:
	///   * `Ok(Skip)`        — no Hall interval changed; cache stays valid.
	///   * `Ok(CannotSkip)`  — caller must run full propagation.
	///   * `Err(conflict)`   — overcrowded interval detected; the conflict
	///                         has already been declared on `ctx`.
	fn cumulative_check<E>(
		&mut self,
		ctx: &mut E::PropagationContext<'_>,
	) -> Result<CumulativeOutcome, E::Conflict>
	where
		E: ReasoningEngine,
		I: IntSolverActions<E>,
	{
		// Working copy of slack — Phase 3 decrements happen here so a later
		// variable's Phase 2 sees earlier variables' commitments.
		self.working.clear();
		self.working.extend_from_slice(&self.slack);

		// Snapshot the updated set so we can iterate it without holding a
		// borrow on `self.var_updated` while we mutate other parts of self.
		let updated: Vec<usize> = self.var_updated.iter().copied().collect();

		for &i in &updated {
			let (new_lb, new_ub) = self.var[i].bounds(ctx);
			let old_lb = self.cached_lb[i];
			let old_ub = self.cached_ub[i];

			// Domain expansion ⇒ stale snapshot (e.g. assumption removal
			// between propagation rounds). Defer to a full propagation.
			if new_lb < old_lb || new_ub > old_ub {
				return Ok(CumulativeOutcome::CannotSkip);
			}

			if new_lb > old_lb {
				match self.check_var_lb::<E>(ctx, i, new_lb, new_ub, old_lb)? {
					CumulativeOutcome::Skip => {}
					other => return Ok(other),
				}
			}
			if new_ub < old_ub {
				match self.check_var_ub::<E>(ctx, i, new_lb, new_ub, old_ub)? {
					CumulativeOutcome::Skip => {}
					other => return Ok(other),
				}
			}
		}
		Ok(CumulativeOutcome::Skip)
	}

	/// LB-direction check for one variable. Phase 1 push + Phase 2 scan +
	/// Phase 3 decrement; declares an overcrowded conflict directly if
	/// Phase 1 or Phase 2 spots one.
	fn check_var_lb<E>(
		&mut self,
		ctx: &mut E::PropagationContext<'_>,
		i: usize,
		new_lb: IntVal,
		new_ub: IntVal,
		old_lb: IntVal,
	) -> Result<CumulativeOutcome, E::Conflict>
	where
		E: ReasoningEngine,
		I: IntSolverActions<E>,
	{
		let j = match self.rank(new_ub + 1) {
			Some(j) => j,
			None => return Ok(CumulativeOutcome::CannotSkip),
		};
		let r_old = match self.rank(old_lb) {
			Some(r) => r,
			None => return Ok(CumulativeOutcome::CannotSkip),
		};

		let (effective_lb, hall_chain) = match self.phase1_push_lb(new_lb, new_ub, j) {
			Phase1Outcome::Done(eff, chain) => (eff, chain),
			Phase1Outcome::WipeOut(l, u) => {
				return Err(self.declare_overcrowded_conflict::<E>(ctx, l, u));
			}
			Phase1Outcome::BoundsMiss => return Ok(CumulativeOutcome::CannotSkip),
		};
		let r_eff = match self.rank(effective_lb) {
			Some(r) => r,
			None => return Ok(CumulativeOutcome::CannotSkip),
		};

		let new_hall = match self.phase2_scan(r_old + 1, r_eff + 1, j, self.s_bounds_len) {
			Phase2Outcome::Clean => false,
			Phase2Outcome::NewHallWouldForm => true,
			Phase2Outcome::Overcrowded(l, u) => {
				return Err(self.declare_overcrowded_conflict::<E>(ctx, l, u));
			}
		};
		self.phase3_decrement(r_old + 1, r_eff + 1, j, self.s_bounds_len);

		if new_hall {
			return Ok(CumulativeOutcome::CannotSkip);
		}
		if !hall_chain.is_empty() {
			let reason =
				self.build_push_reason::<E>(ctx, i, IntLitMeaning::GreaterEq(old_lb), &hall_chain);
			self.prop_pushes += 1;
			self.var[i].tighten_min(ctx, effective_lb, reason)?;
		}
		Ok(CumulativeOutcome::Skip)
	}

	/// UB-direction check for one variable. Symmetric to `check_var_lb`.
	fn check_var_ub<E>(
		&mut self,
		ctx: &mut E::PropagationContext<'_>,
		i: usize,
		new_lb: IntVal,
		new_ub: IntVal,
		old_ub: IntVal,
	) -> Result<CumulativeOutcome, E::Conflict>
	where
		E: ReasoningEngine,
		I: IntSolverActions<E>,
	{
		let r_lb = match self.rank(new_lb) {
			Some(r) => r,
			None => return Ok(CumulativeOutcome::CannotSkip),
		};
		let r_old_ub1 = match self.rank(old_ub + 1) {
			Some(r) => r,
			None => return Ok(CumulativeOutcome::CannotSkip),
		};

		let (effective_ub, hall_chain) = match self.phase1_push_ub(new_lb, new_ub, r_lb) {
			Phase1Outcome::Done(eff, chain) => (eff, chain),
			Phase1Outcome::WipeOut(l, u) => {
				return Err(self.declare_overcrowded_conflict::<E>(ctx, l, u));
			}
			Phase1Outcome::BoundsMiss => return Ok(CumulativeOutcome::CannotSkip),
		};
		let u_eff = match self.rank(effective_ub + 1) {
			Some(r) => r,
			None => return Ok(CumulativeOutcome::CannotSkip),
		};

		let new_hall = match self.phase2_scan(0, r_lb + 1, u_eff, r_old_ub1) {
			Phase2Outcome::Clean => false,
			Phase2Outcome::NewHallWouldForm => true,
			Phase2Outcome::Overcrowded(l, u) => {
				return Err(self.declare_overcrowded_conflict::<E>(ctx, l, u));
			}
		};
		self.phase3_decrement(0, r_lb + 1, u_eff, r_old_ub1);

		if new_hall {
			return Ok(CumulativeOutcome::CannotSkip);
		}
		if !hall_chain.is_empty() {
			let reason =
				self.build_push_reason::<E>(ctx, i, IntLitMeaning::Less(old_ub + 1), &hall_chain);
			self.prop_pushes += 1;
			self.var[i].tighten_max(ctx, effective_ub, reason)?;
		}
		Ok(CumulativeOutcome::Skip)
	}

	// ---- Phase helpers (private; per-direction Phase 1, shared Phase 2/3) ----

	/// Phase 1, lb direction. Walks up through chained cached Hall
	/// intervals, returning the effective lower bound plus the chain of
	/// `(l_h, u_h)` ranks traversed (empty if no push).
	///
	/// Soundness: the push is only valid when the cell `(r_eff_lo, u_h)`
	/// — anchored at the variable's CURRENT effective lower bound — has
	/// slack <= 0. A zero cell elsewhere in the rectangle would mean a
	/// Hall interval that var i could legitimately sit BELOW, so pushing
	/// would remove valid solutions.
	fn phase1_push_lb(&self, new_lb: IntVal, new_ub: IntVal, j: usize) -> Phase1Outcome {
		let mut effective = new_lb;
		let mut chain: Vec<(usize, usize)> = Vec::new();
		loop {
			let r_eff_lo = match self.rank(effective) {
				Some(r) => r,
				None => return Phase1Outcome::BoundsMiss,
			};
			let hit = self.find_zero_slack_row_lb(r_eff_lo, j);
			match hit {
				None => return Phase1Outcome::Done(effective, chain),
				Some((l_h, u_h)) => {
					chain.push((l_h, u_h));
					effective = self.s_bounds[u_h];
					if effective > new_ub {
						return Phase1Outcome::WipeOut(l_h, u_h);
					}
				}
			}
		}
	}

	/// Scan a single row `l = l_anchor` of the working slack table for the
	/// smallest `u` (in `(l_anchor, u_inclusive]`) whose cell is <= 0. Used
	/// by Phase 1 LB: returning the smallest such u gives the *tightest*
	/// sound push to `s_bounds[u]`.
	fn find_zero_slack_row_lb(
		&self,
		l_anchor: usize,
		u_inclusive: usize,
	) -> Option<(usize, usize)> {
		for u in (l_anchor + 1)..=u_inclusive {
			if self.working_cell(l_anchor, u) <= 0 {
				return Some((l_anchor, u));
			}
		}
		None
	}

	/// Phase 1, ub direction. Symmetric to `phase1_push_lb`: anchored at
	/// the variable's current effective upper bound (cell `(l_h, u_eff)`).
	fn phase1_push_ub(&self, new_lb: IntVal, new_ub: IntVal, r_lb: usize) -> Phase1Outcome {
		let mut effective = new_ub;
		let mut chain: Vec<(usize, usize)> = Vec::new();
		loop {
			let u_anchor = match self.rank(effective + 1) {
				Some(r) => r,
				None => return Phase1Outcome::BoundsMiss,
			};
			let hit = self.find_zero_slack_col_ub(r_lb + 1, u_anchor);
			match hit {
				None => return Phase1Outcome::Done(effective, chain),
				Some((l_h, u_h)) => {
					chain.push((l_h, u_h));
					effective = self.s_bounds[l_h] - 1;
					if effective < new_lb {
						return Phase1Outcome::WipeOut(l_h, u_h);
					}
				}
			}
		}
	}

	/// Scan a single column `u = u_anchor` of the working slack table for
	/// the largest `l` (in `[l_lo, u_anchor)`) whose cell is <= 0. The
	/// largest such l gives the tightest sound push to `s_bounds[l] - 1`.
	fn find_zero_slack_col_ub(&self, l_lo: usize, u_anchor: usize) -> Option<(usize, usize)> {
		for l in (l_lo..u_anchor).rev() {
			if self.working_cell(l, u_anchor) <= 0 {
				return Some((l, u_anchor));
			}
		}
		None
	}

	/// Phase 2: scan the half-open rectangle `[l_lo, l_hi) × [u_lo, u_hi)`
	/// of the working slack table.
	fn phase2_scan(&self, l_lo: usize, l_hi: usize, u_lo: usize, u_hi: usize) -> Phase2Outcome {
		let mut new_hall = false;
		for l in l_lo..l_hi {
			for u in u_lo..u_hi {
				if l >= u {
					continue;
				}
				let v = self.working_cell(l, u);
				if v <= 0 {
					return Phase2Outcome::Overcrowded(l, u);
				}
				if v == 1 {
					new_hall = true;
				}
			}
		}
		if new_hall {
			Phase2Outcome::NewHallWouldForm
		} else {
			Phase2Outcome::Clean
		}
	}

	/// Phase 3: decrement every working-slack cell in the rectangle so
	/// later variables in the same outer pass see the consumed slack.
	fn phase3_decrement(&mut self, l_lo: usize, l_hi: usize, u_lo: usize, u_hi: usize) {
		for l in l_lo..l_hi {
			for u in u_lo..u_hi {
				if l >= u {
					continue;
				}
				self.working_dec(l, u);
			}
		}
	}

	/// Build the reason for a Phase-1 push: var `i`'s original-bound literal
	/// plus, for every cached Hall interval in `chain`, the bound literals
	/// of every variable that was enclosed in it at snapshot time.
	fn build_push_reason<E>(
		&self,
		ctx: &mut E::PropagationContext<'_>,
		i: usize,
		anchor: IntLitMeaning,
		chain: &[(usize, usize)],
	) -> Vec<<E as ReasoningEngine>::Atom>
	where
		E: ReasoningEngine,
		I: IntSolverActions<E>,
	{
		let mut reason = Vec::new();
		reason.push(self.var[i].lit(ctx, anchor));
		for &(l_h, u_h) in chain {
			let bl = self.s_bounds[l_h];
			let bu = self.s_bounds[u_h];
			for k in 0..self.var.len() {
				if k == i {
					continue;
				}
				if self.cached_lb[k] >= bl && self.cached_ub[k] < bu {
					reason.push(self.var[k].lit(ctx, IntLitMeaning::GreaterEq(bl)));
					reason.push(self.var[k].lit(ctx, IntLitMeaning::Less(bu)));
				}
			}
		}
		reason
	}

	/// Declare an overcrowded-interval conflict on the half-open value
	/// range `[s_bounds[l], s_bounds[u])`. The reason is the bound
	/// literals of every variable whose current domain is contained in
	/// that interval (a Hall-set witness for the pigeon-hole violation).
	fn declare_overcrowded_conflict<E>(
		&self,
		ctx: &mut E::PropagationContext<'_>,
		l: usize,
		u: usize,
	) -> E::Conflict
	where
		E: ReasoningEngine,
		I: IntSolverActions<E>,
	{
		let bl = self.s_bounds[l];
		let bu = self.s_bounds[u];
		let mut reason = Vec::new();
		for k in 0..self.var.len() {
			let (lb, ub) = self.var[k].bounds(ctx);
			if lb >= bl && ub < bu {
				reason.push(self.var[k].lit(ctx, IntLitMeaning::GreaterEq(bl)));
				reason.push(self.var[k].lit(ctx, IntLitMeaning::Less(bu)));
			}
		}
		ctx.declare_conflict(reason)
	}
}

/// Result of the Phase-1 effective-bound push.
#[derive(Clone, Debug)]
enum Phase1Outcome {
	/// Push completed; final effective bound plus the chain of cached Hall
	/// intervals traversed (possibly empty).
	Done(IntVal, Vec<(usize, usize)>),
	/// The push would exceed the variable's residual domain. The witness
	/// Hall interval at ranks `(l, u)` is overcrowded.
	WipeOut(usize, usize),
	/// A fence value did not land on a cached bound — fall back to full prop.
	BoundsMiss,
}

/// Result of the Phase-2 rectangle scan.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Phase2Outcome {
	Clean,
	NewHallWouldForm,
	Overcrowded(usize, usize),
}

impl<I> Drop for IntUniqueBoundsCumulativeSlack<I> {
	fn drop(&mut self) {
		if self.prop_calls > 0 {
			info!(
				target: "int_unique_stats",
				n = self.var.len(),
				calls = self.prop_calls,
				full = self.prop_full,
				skips = self.prop_skips,
				pushes = self.prop_pushes,
				failures = self.prop_failures,
				check_failures = self.prop_check_failures,
				backtracks = self.prop_backtracks,
				skip_rate = self.prop_skips as f64 / self.prop_calls as f64,
				"IntUniqueBoundsCumulativeSlack stats"
			);
		}
	}
}

impl<E, I> Propagator<E> for IntUniqueBoundsCumulativeSlack<I>
where
	E: ReasoningEngine,
	I: IntSolverActions<E>,
{
	fn initialize(&mut self, ctx: &mut <E as ReasoningEngine>::InitializationContext<'_>) {
		ctx.set_priority(PriorityLevel::Low);
		ctx.advise_on_backtrack();
		for (i, v) in self.var.iter().enumerate() {
			v.advise_when(ctx, IntPropCond::Bounds, i as u64);
			v.enqueue_when(ctx, IntPropCond::Bounds);
		}
	}

	fn advise_of_backtrack(&mut self, _ctx: &mut E::NotificationContext<'_>) {
		self.pending_backtrack = true;
		self.var_updated.clear();
	}

	fn advise_of_int_change(
		&mut self,
		_ctx: &mut E::NotificationContext<'_>,
		data: u64,
		_event: IntEvent,
	) -> bool {
		let _ = self.var_updated.insert(data as usize);
		false
	}

	#[tracing::instrument(
		name = "int_unique_bounds_cumulative_slack",
		target = "solver",
		level = "trace",
		skip(self, ctx)
	)]
	fn propagate(&mut self, ctx: &mut E::PropagationContext<'_>) -> Result<(), E::Conflict> {
		self.prop_calls += 1;

		if self.pending_backtrack {
			self.prop_backtracks += 1;
			self.pending_backtrack = false;
			self.cache_valid = false;
			self.var_updated.clear();
		}

		if self.cache_valid {
			match self.cumulative_check::<E>(ctx) {
				Ok(CumulativeOutcome::Skip) => {
					self.prop_skips += 1;
					self.var_updated.clear();
					return Ok(());
				}
				Ok(CumulativeOutcome::CannotSkip) => {
					// Fall through to full propagation.
				}
				Err(c) => {
					// FAILURE — overcrowded interval; reason already declared.
					self.prop_check_failures += 1;
					self.prop_failures += 1;
					self.cache_valid = false;
					self.var_updated.clear();
					return Err(c);
				}
			}
		}

		// Full propagation path.
		self.prop_full += 1;
		let result = self.full_propagate::<E>(ctx);
		if let Err(c) = result {
			self.prop_failures += 1;
			self.cache_valid = false;
			self.var_updated.clear();
			return Err(c);
		}
		self.snapshot_after_full::<E>(ctx);
		self.var_updated.clear();
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use itertools::Itertools;
	use tracing_test::traced_test;

	use crate::{
		IntSet, IntVal,
		constraints::{
			int_linear::IntLinearLessEqBounds,
			int_unique::{IntUniqueBounds, IntUniqueBoundsCumulativeSlack, IntUniqueValue},
		},
		model::Model,
		solver::{LiteralStrategy, Solver, Status, Valuation},
	};

	#[test]
	#[traced_test]
	fn test_all_different_bounds_sat_1() {
		let mut slv = Solver::default();
		let a = slv
			.new_int_decision(1..=3)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let b = slv
			.new_int_decision(1..=3)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let c = slv
			.new_int_decision(1..=3)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		IntUniqueBounds::post(&mut slv, vec![a, b, c]);
		slv.assert_all_solutions(&[a, b, c], |sol| sol.iter().all_unique());
	}
	#[test]
	#[traced_test]
	fn test_all_different_bounds_sat_2() {
		let mut slv = Solver::default();
		let a = slv
			.new_int_decision(3..=4)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let b = slv
			.new_int_decision(2..=4)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let c = slv
			.new_int_decision(3..=4)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let d = slv
			.new_int_decision(2..=5)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let e = slv
			.new_int_decision(3..=6)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let f = slv
			.new_int_decision(1..=6)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();

		IntUniqueBounds::post(&mut slv, vec![a, b, c, d, e, f]);
		slv.assert_all_solutions(&[a, b, c, d, e, f], |sol| sol.iter().all_unique());
	}

	#[test]
	#[traced_test]
	fn test_all_different_bounds_sat_3() {
		let mut slv = Solver::default();
		let a = slv
			.new_int_decision(3..=6)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let b = slv
			.new_int_decision(3..=4)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let c = slv
			.new_int_decision(2..=5)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let d = slv
			.new_int_decision(2..=4)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let e = slv
			.new_int_decision(3..=4)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let f = slv
			.new_int_decision(1..=6)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();

		IntUniqueBounds::post(&mut slv, vec![a, b, c, d, e, f]);
		slv.assert_all_solutions(&[a, b, c, d, e, f], |sol| sol.iter().all_unique());
	}

	#[test]
	#[traced_test]
	fn test_all_different_bounds_unsat() {
		let mut slv = Solver::default();
		let a = slv
			.new_int_decision(1..=3)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let b = slv
			.new_int_decision(1..=3)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let c = slv
			.new_int_decision(1..=3)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();

		IntUniqueBounds::post(&mut slv, vec![a, b, c]);
		IntLinearLessEqBounds::post(&mut slv, vec![-a, -b, -c], -8);
		slv.assert_unsatisfiable();
	}

	#[test]
	#[traced_test]
	fn test_all_different_value_sat() {
		let mut slv = Solver::default();
		let a = slv
			.new_int_decision(1..=4)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let b = slv
			.new_int_decision(1..=4)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let c = slv
			.new_int_decision(1..=4)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();

		IntUniqueValue::post(&mut slv, vec![a, b, c]);

		slv.assert_all_solutions(&[a, b, c], |sol| sol.iter().all_unique());
	}

	#[test]
	#[traced_test]
	fn test_all_different_value_unsat() {
		let mut slv = Solver::default();
		let a = slv
			.new_int_decision(1..=2)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let b = slv
			.new_int_decision(1..=2)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let c = slv
			.new_int_decision(1..=2)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();

		IntUniqueValue::post(&mut slv, vec![a, b, c]);

		slv.assert_unsatisfiable();
	}

	#[test]
	#[traced_test]
	fn test_gapped_domain_regression() {
		let mut prb = Model::default();
		let prev: Vec<_> = [
			IntSet::from_iter([15..=15, 20..=20]),
			(20..=20).into(),
			IntSet::from_iter([15..=15, 20..=20]),
		]
		.into_iter()
		.map(|domain| prb.new_int_decision(domain))
		.collect();

		assert!(prb.unique(prev.iter().copied()).post().is_err());
	}

	fn test_sudoku(grid: &[&str], expected: Status) {
		debug_assert_eq!(grid.len(), 9);
		debug_assert!(grid.iter().all(|row| row.len() == 9));

		let mut slv: Solver = Solver::default();
		// create variables and add all different propagator for each row
		let all_vars: Vec<_> = grid
			.iter()
			.map(|row| {
				let vars: Vec<_> = row
					.chars()
					.map(|c| {
						if c.is_ascii_digit() {
							let num = IntVal::from(c.to_digit(10).unwrap());
							num.into()
						} else {
							slv.new_int_decision(1..=9)
								.order_literals(LiteralStrategy::Eager)
								.direct_literals(LiteralStrategy::Eager)
								.view()
						}
					})
					.collect();

				IntUniqueValue::post(&mut slv, vars.clone());
				vars
			})
			.collect();

		// add all different propagator for each column
		for (i, _) in grid.iter().enumerate() {
			let col_vars: Vec<_> = grid
				.iter()
				.enumerate()
				.map(|(j, _)| all_vars[j][i])
				.collect();

			IntUniqueValue::post(&mut slv, col_vars);
		}
		// add all different propagator for each 3 by 3 grid
		for i in 0..3 {
			for j in 0..3 {
				let mut block_vars: Vec<_> = Vec::with_capacity(grid.len());
				for x in 0..3 {
					for y in 0..3 {
						block_vars.push(all_vars[3 * i + x][3 * j + y]);
					}
				}

				IntUniqueValue::post(&mut slv, block_vars);
			}
		}
		assert_eq!(
			slv.solve()
				.on_solution(|sol| {
					(0..9).for_each(|r| {
						let row = all_vars[r].iter().map(|&v| v.val(sol)).collect_vec();
						assert!(
							row.iter().all_unique(),
							"Values in row {r} are not all different: {row:?}",
						);
					});
					(0..9).for_each(|c| {
						let col = all_vars.iter().map(|row| row[c].val(sol)).collect_vec();
						assert!(
							col.iter().all_unique(),
							"Values in column {c} are not all different: {col:?}",
						);
					});
					(0..3).for_each(|i| {
						(0..3).for_each(|j| {
							let block = (0..3)
								.flat_map(|x| (0..3).map(move |y| (x, y)))
								.map(|(x, y)| all_vars[3 * i + x][3 * j + y].val(sol))
								.collect_vec();
							assert!(
								block.iter().all_unique(),
								"Values in block ({i}, {j}) are not all different: {block:?}",
							);
						});
					});
				})
				.satisfy(),
			expected
		);
	}

	#[test]
	#[traced_test]
	fn test_sudoku_1() {
		test_sudoku(
			&[
				"2581.4.37",
				"936827514",
				"47153.28.",
				"7152.3.4.",
				"849675321",
				"36241..75",
				"1249..753",
				"593742168",
				"687351492",
			],
			Status::Satisfied,
		);
	}

	#[test]
	#[traced_test]
	fn test_sudoku_2() {
		test_sudoku(
			&[
				"...2.5...",
				".9....73.",
				"..2..9.6.",
				"2.....4.9",
				"....7....",
				"6.9.....1",
				".8.4..1..",
				".63....8.",
				"...6.8...",
			],
			Status::Satisfied,
		);
	}

	#[test]
	#[traced_test]
	fn test_sudoku_3() {
		test_sudoku(
			&[
				"3..9.4..1",
				"..2...4..",
				".61...79.",
				"6..247..5",
				".........",
				"2..836..4",
				".46...23.",
				"..9...6..",
				"5..3.9..8",
			],
			Status::Satisfied,
		);
	}

	#[test]
	#[traced_test]
	fn test_sudoku_4() {
		test_sudoku(
			&[
				"....1....",
				"3.14..86.",
				"9..5..2..",
				"7..16....",
				".2.8.5.1.",
				"....97..4",
				"..3..4..6",
				".48..69.7",
				"....8....",
			],
			Status::Satisfied,
		);
	}

	#[test]
	#[traced_test]
	fn test_sudoku_5() {
		test_sudoku(
			&[
				"..4..3.7.",
				".8..7....",
				".7...82.5",
				"4.....31.",
				"9.......8",
				".15.....4",
				"1.69...3.",
				"....2..6.",
				".2.4..5..",
			],
			Status::Satisfied,
		);
	}

	#[test]
	#[traced_test]
	fn test_sudoku_6() {
		test_sudoku(
			&[
				".43.8.25.",
				"6........",
				".....1.94",
				"9....4.7.",
				"...6.8...",
				".1.2....3",
				"82.5.....",
				"........5",
				".34.9.71.",
			],
			Status::Satisfied,
		);
	}

	// ------------------------------------------------------------------
	// cumulative-slack skip propagator — mirror of the IntUniqueBounds
	// tests above, posting IntUniqueBoundsCumulativeSlack instead.
	// ------------------------------------------------------------------

	#[test]
	#[traced_test]
	fn test_cumulative_slack_sat_1() {
		let mut slv = Solver::default();
		let a = slv
			.new_int_decision(1..=3)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let b = slv
			.new_int_decision(1..=3)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let c = slv
			.new_int_decision(1..=3)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		IntUniqueBoundsCumulativeSlack::post(&mut slv, vec![a, b, c]);
		slv.assert_all_solutions(&[a, b, c], |sol| sol.iter().all_unique());
	}

	#[test]
	#[traced_test]
	fn test_cumulative_slack_sat_2() {
		let mut slv = Solver::default();
		let a = slv
			.new_int_decision(3..=4)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let b = slv
			.new_int_decision(2..=4)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let c = slv
			.new_int_decision(3..=4)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let d = slv
			.new_int_decision(2..=5)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let e = slv
			.new_int_decision(3..=6)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let f = slv
			.new_int_decision(1..=6)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		IntUniqueBoundsCumulativeSlack::post(&mut slv, vec![a, b, c, d, e, f]);
		slv.assert_all_solutions(&[a, b, c, d, e, f], |sol| sol.iter().all_unique());
	}

	#[test]
	#[traced_test]
	fn test_cumulative_slack_unsat() {
		let mut slv = Solver::default();
		let a = slv
			.new_int_decision(1..=2)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let b = slv
			.new_int_decision(1..=2)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		let c = slv
			.new_int_decision(1..=2)
			.order_literals(LiteralStrategy::Eager)
			.direct_literals(LiteralStrategy::Eager)
			.view();
		IntUniqueBoundsCumulativeSlack::post(&mut slv, vec![a, b, c]);
		slv.assert_unsatisfiable();
	}
}
