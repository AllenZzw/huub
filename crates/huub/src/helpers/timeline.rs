//! The *time line* data structure of Fahimi & Quimper, *Linear-Time Filtering
//! Algorithms for the Disjunctive Constraint* (AAAI 2014 / Constraints 2018).
//!
//! The time line replaces Vilím's Θ-tree for algorithms that only ever *add*
//! tasks (never remove them): the overload check and the detectable-precedence
//! rule. It supports, in amortised constant time, scheduling a task at its
//! earliest start with preemption ([`TimeLine::schedule_task`]) and reading the
//! earliest completion time of the scheduled set
//! ([`TimeLine::earliest_completion_time`]).
//!
//! The structure keeps the sorted, de-duplicated earliest-start times of the
//! tasks together with a large sentinel as *time points* `t`. Each slot
//! `[t[a], t[a + 1])` has a remaining `capacity` `c[a]`, initially its width.
//! A [`UnionFind`] keeps `a` and `a + 1` in the same set iff `c[a] == 0`, so
//! [`UnionFind::find_greatest`] jumps over runs of saturated slots in amortised
//! constant time. Because tasks are never removed, capacities only decrease,
//! which is what bounds the amortised cost (see the paper's Theorem 2).

use crate::IntVal;

/// Disjoint-set structure specialised for the time line.
///
/// [`Self::union`] is only ever called on consecutive indices `(a, a + 1)`, and
/// the representative of a set is its *greatest* element, so
/// [`Self::find_greatest`] returns that representative directly. Each set is
/// therefore a contiguous range `[lo, hi]` and `find_greatest` returns `hi`.
#[derive(Clone, Debug)]
struct UnionFind {
	/// `parent[x]` points one step towards the (greatest) representative;
	/// `parent[x] == x` marks a root.
	parent: Vec<usize>,
}

impl UnionFind {
	/// Create `n` singleton sets `{0}, {1}, …, {n - 1}`.
	fn new(n: usize) -> Self {
		UnionFind {
			parent: (0..n).collect(),
		}
	}

	/// Return the greatest element of the set containing `a`, compressing the
	/// traversed path so later queries are cheaper.
	fn find_greatest(&mut self, a: usize) -> usize {
		let mut root = a;
		while self.parent[root] != root {
			root = self.parent[root];
		}
		// Path compression: point every node on the path straight at the root.
		let mut cur = a;
		while self.parent[cur] != root {
			let next = self.parent[cur];
			self.parent[cur] = root;
			cur = next;
		}
		root
	}

	/// Merge the sets containing `a` and `b`. Only called with `b > a` (in
	/// practice `b == a + 1`), so the root of `a`'s set is pointed at the
	/// greater root of `b`'s set, preserving the "root is greatest" invariant.
	fn union(&mut self, a: usize, b: usize) {
		let ra = self.find_greatest(a);
		let rb = self.find_greatest(b);
		debug_assert!(rb >= ra, "union expects b to be in the greater set");
		self.parent[ra] = rb;
	}
}

/// The time line over a fixed set of tasks.
///
/// Built once per propagation from the tasks' `est`, `lct`, and `p`; tasks are
/// then scheduled incrementally. See the module documentation for the meaning
/// of the fields.
#[derive(Clone, Debug)]
pub(crate) struct TimeLine {
	/// Sorted, de-duplicated time points: the distinct earliest-start times
	/// followed by one large sentinel. Length `|t|`.
	t: Vec<IntVal>,
	/// Remaining capacity of slot `[t[a], t[a + 1])`. Length `|t| - 1`.
	c: Vec<IntVal>,
	/// `m[i]` is the index into `t` of task `i`'s earliest start (`t[m[i]] ==
	/// est_i`).
	m: Vec<usize>,
	/// Index of the latest slot whose capacity was decremented, or `None`
	/// before any task has been scheduled.
	e: Option<usize>,
	/// Keeps `a` and `a + 1` merged iff `c[a] == 0`.
	uf: UnionFind,
}

impl TimeLine {
	/// Initialise the time line for the tasks whose parameters are given
	/// task-indexed in `est`, `lct`, and `p` (all the same length). Corresponds
	/// to the paper's `InitializeTimeline` (Algorithm 2).
	pub(crate) fn new(est: &[IntVal], lct: &[IntVal], p: &[IntVal]) -> Self {
		debug_assert_eq!(est.len(), lct.len());
		debug_assert_eq!(est.len(), p.len());

		// Time points: the distinct earliest starts, sorted, plus a sentinel
		// large enough that the whole scheduled set fits before it.
		let mut t: Vec<IntVal> = est.to_vec();
		t.sort_unstable();
		t.dedup();
		let sentinel = lct.iter().copied().max().unwrap_or(0) + p.iter().copied().sum::<IntVal>();
		t.push(sentinel);

		// Map each task to the index of its earliest start.
		let m = est
			.iter()
			.map(|&e| t.partition_point(|&tp| tp < e))
			.collect();

		let c = t.windows(2).map(|w| w[1] - w[0]).collect();
		let uf = UnionFind::new(t.len());

		TimeLine {
			t,
			c,
			m,
			e: None,
			uf,
		}
	}

	/// Schedule task `i` over the time line at its earliest start with
	/// preemption, consuming `p_i` units of capacity. Corresponds to the
	/// paper's `ScheduleTask` (Algorithm 3).
	pub(crate) fn schedule_task(&mut self, i: usize, p_i: IntVal) {
		let mut rho = p_i;
		let mut k = self.uf.find_greatest(self.m[i]);
		while rho > 0 {
			let delta = self.c[k].min(rho);
			rho -= delta;
			self.c[k] -= delta;
			if self.c[k] == 0 {
				self.uf.union(k, k + 1);
				k = self.uf.find_greatest(k);
			}
		}
		self.e = Some(self.e.map_or(k, |e| e.max(k)));
	}

	/// The earliest completion time of the set of tasks scheduled so far.
	/// Corresponds to the paper's `EarliestCompletionTime` (Algorithm 4).
	///
	/// Returns [`IntVal::MIN`] when no task has been scheduled (the paper's
	/// `ect_∅ = −∞`).
	pub(crate) fn earliest_completion_time(&self) -> IntVal {
		match self.e {
			Some(e) => self.t[e + 1] - self.c[e],
			None => IntVal::MIN,
		}
	}
}

#[cfg(test)]
mod tests {
	use crate::helpers::timeline::TimeLine;

	#[test]
	fn paper_example_1() {
		// Tasks (est, lct, p) = {(4, 15, 5), (1, 10, 6), (5, 8, 2)}.
		// Initialising produces time points {1, 4, 5, 28} with capacities
		// {3, 1, 23}; scheduling all three tasks yields ect = 28 − 14 = 14.
		let est = [4, 1, 5];
		let lct = [15, 10, 8];
		let p = [5, 6, 2];

		let mut tl = TimeLine::new(&est, &lct, &p);
		assert_eq!(tl.t, vec![1, 4, 5, 28]);
		assert_eq!(tl.c, vec![3, 1, 23]);

		for (i, &p_i) in p.iter().enumerate() {
			tl.schedule_task(i, p_i);
		}
		assert_eq!(tl.earliest_completion_time(), 14);
	}

	#[test]
	fn single_task_ect_is_est_plus_p() {
		let est = [4];
		let lct = [15];
		let p = [5];
		let mut tl = TimeLine::new(&est, &lct, &p);
		tl.schedule_task(0, p[0]);
		assert_eq!(tl.earliest_completion_time(), 9);
	}

	#[test]
	fn empty_schedule_is_negative_infinity() {
		let tl = TimeLine::new(&[1, 3], &[10, 12], &[2, 2]);
		assert_eq!(tl.earliest_completion_time(), i64::MIN);
	}
}
