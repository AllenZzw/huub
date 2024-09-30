use std::ops::{Add, Mul, Neg};

use rangelist::RangeList;

use crate::{helpers::div_ceil, IntVal, LitMeaning, NonZeroIntVal};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LinearTransform {
	pub(crate) scale: NonZeroIntVal,
	pub(crate) offset: IntVal,
}

impl LinearTransform {
	/// Creates a new linear transformation with the given scale and offset.
	pub fn new(scale: NonZeroIntVal, offset: IntVal) -> Self {
		Self { scale, offset }
	}
	/// Creates a new linear transformation with the given scale and no offset.
	pub fn scaled(scale: NonZeroIntVal) -> Self {
		Self { scale, offset: 0 }
	}
	/// Creates a new linear transformation with the given offset and no scale.
	pub fn offset(offset: IntVal) -> Self {
		Self {
			scale: NonZeroIntVal::new(1).unwrap(),
			offset,
		}
	}

	pub(crate) fn positive_scale(&self) -> bool {
		self.scale.get() > 0
	}

	/// Perform the linear transformation on a value.
	pub fn transform(&self, val: IntVal) -> IntVal {
		(val * self.scale.get()) + self.offset
	}

	/// Perform the linear tranformation for a `LitMeaning`.
	///
	/// There are multiple possible transformations for a single `LitMeaning` when self.scale.abs() > 1.
	/// For example, the literal `x ≥ 2` can be mapped to `2x - 3 ≥ 0` or `2x - 3 ≥ 1` under the mapping `2x - 3`.
	/// This function returns the weakest possible transformed `LitMeaning` that is equivalent to `lit`.
	pub fn transform_lit(&self, lit: LitMeaning) -> LitMeaning {
		match lit {
			LitMeaning::Eq(v) => LitMeaning::Eq(self.transform(v)),
			LitMeaning::NotEq(v) => LitMeaning::NotEq(self.transform(v)),
			LitMeaning::GreaterEq(v) => {
				if self.positive_scale() {
					LitMeaning::GreaterEq(self.transform(v - 1) + 1)
				} else {
					LitMeaning::Less(self.transform(v - 1))
				}
			}
			LitMeaning::Less(v) => {
				if self.positive_scale() {
					LitMeaning::Less(self.transform(v))
				} else {
					LitMeaning::GreaterEq(self.transform(v) + 1)
				}
			}
		}
	}

	/// Return the weakest `LitMeaning` that is equivalent to `lit` under the transformation.
	///
	/// For example, `2x - 3 ≥ 0` and `2x - 3 ≥ 1` are equivalent since they both refer to the literal `x ≥ 2`.
	pub fn relaxed_lit(&self, lit: LitMeaning) -> LitMeaning {
		if let LitMeaning::GreaterEq(_) | LitMeaning::Less(_) = lit {
			let rev_lit = self.rev_transform_lit(lit.clone());
			match (lit, rev_lit) {
				(LitMeaning::GreaterEq(_), Ok(LitMeaning::GreaterEq(v))) => {
					LitMeaning::GreaterEq(self.transform(v - 1) + 1)
				}
				(LitMeaning::GreaterEq(_), Ok(LitMeaning::Less(v))) => {
					LitMeaning::GreaterEq(self.transform(v) + 1)
				}
				(LitMeaning::Less(_), Ok(LitMeaning::Less(v))) => {
					LitMeaning::Less(self.transform(v))
				}
				(LitMeaning::Less(_), Ok(LitMeaning::GreaterEq(v))) => {
					LitMeaning::Less(self.transform(v - 1))
				}
				_ => unreachable!(
					"rev_transform_lit should always return a valid GreaterEq or Less litmeaning"
				),
			}
		} else {
			lit
		}
	}

	/// Perform the reverse linear transformation on a value.
	///
	/// Note that this should only be used when rev_remains_integer(val) is true.
	pub fn rev_transform(&self, val: IntVal) -> IntVal {
		debug_assert!(self.rev_remains_integer(val));
		(val - self.offset) / self.scale.get()
	}

	/// Returns whether a value remains an integer after reversing the transformation.
	pub fn rev_remains_integer(&self, val: IntVal) -> bool {
		(val - self.offset) % self.scale.get() == 0
	}

	/// Perform the reverse linear tranformation for a `LitMeaning`.
	///
	/// Note that this performs the correct rounding to maintain the meaning of
	/// the literal.
	///
	/// If equality literals are requested that cannot be correctly rounded, then
	/// a boolean `Err` is returned with wether the `LitMeaning` implicitly holds.
	pub fn rev_transform_lit(&self, mut lit: LitMeaning) -> Result<LitMeaning, bool> {
		let mut transformer = *self;
		if !self.positive_scale() {
			// Make positive by doing `*-1` on lit meaning and transformer
			(lit, transformer) = match lit {
				// -x >= i === x <= -i === x < -i + 1
				LitMeaning::GreaterEq(i) => (LitMeaning::Less(-i + 1), -transformer),
				// -x < i === x > -i === x >= -i + 1
				LitMeaning::Less(i) => (LitMeaning::GreaterEq(-i + 1), -transformer),
				_ => (lit, transformer),
			};
		}

		match lit {
			LitMeaning::Eq(i) => {
				if transformer.rev_remains_integer(i) {
					Ok(LitMeaning::Eq(transformer.rev_transform(i)))
				} else {
					Err(false)
				}
			}
			LitMeaning::NotEq(i) => {
				if transformer.rev_remains_integer(i) {
					Ok(LitMeaning::NotEq(transformer.rev_transform(i)))
				} else {
					Err(true)
				}
			}
			LitMeaning::GreaterEq(i) => Ok(LitMeaning::GreaterEq(div_ceil(
				i - transformer.offset,
				transformer.scale,
			))),
			LitMeaning::Less(i) => Ok(LitMeaning::Less(div_ceil(
				i - transformer.offset,
				transformer.scale,
			))),
		}
	}

	pub(crate) fn rev_transform_mask(&self, mask: &RangeList<IntVal>) -> RangeList<IntVal> {
		let get_val = |meaning| match meaning {
			LitMeaning::GreaterEq(i) => i,
			LitMeaning::Less(i) => i - 1,
			_ => unreachable!(),
		};

		mask.iter()
			.map(|r| {
				let a = get_val(
					self.rev_transform_lit(LitMeaning::GreaterEq(*r.start()))
						.unwrap(),
				);
				let b = get_val(
					self.rev_transform_lit(LitMeaning::Less(r.end() + 1))
						.unwrap(),
				);
				a.min(b)..=a.max(b)
			})
			.collect()
	}
}

impl Default for LinearTransform {
	fn default() -> Self {
		Self {
			scale: NonZeroIntVal::new(1).unwrap(),
			offset: 0,
		}
	}
}

impl Neg for LinearTransform {
	type Output = Self;
	fn neg(self) -> Self::Output {
		Self {
			scale: NonZeroIntVal::new(-self.scale.get()).unwrap(),
			offset: -self.offset,
		}
	}
}

impl Add<IntVal> for LinearTransform {
	type Output = Self;

	fn add(self, rhs: IntVal) -> Self::Output {
		LinearTransform {
			scale: self.scale,
			offset: self.offset + rhs,
		}
	}
}

impl Mul<NonZeroIntVal> for LinearTransform {
	type Output = Self;

	fn mul(self, rhs: NonZeroIntVal) -> Self::Output {
		LinearTransform {
			scale: NonZeroIntVal::new(self.scale.get() * rhs.get()).unwrap(),
			offset: self.offset * rhs.get(),
		}
	}
}

#[cfg(test)]
mod tests {
	use super::{LinearTransform, NonZeroIntVal};
	use crate::LitMeaning;
	use tracing_test::traced_test;

	#[test]
	#[traced_test]
	fn test_transform_strongest_lit() {
		let transformer = LinearTransform::new(NonZeroIntVal::new(2).unwrap(), -3);
		// x = 2 === 2x - 3 = 1
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::Eq(2)),
			LitMeaning::Eq(1)
		);
		// x = 1 === 2x - 3 = -1
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::Eq(1)),
			LitMeaning::Eq(-1)
		);
		// x ≠ 2 === 2x - 3 ≠ 1
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::NotEq(2)),
			LitMeaning::NotEq(1)
		);
		// x ≠ 1 === 2x - 3 ≠ -1
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::NotEq(1)),
			LitMeaning::NotEq(-1)
		);
		// x ≥ 2 === 2x - 3 ≥ 1
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::GreaterEq(2)),
			LitMeaning::GreaterEq(1)
		);
		// x ≥ 1 === 2x - 3 ≥ -1
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::GreaterEq(1)),
			LitMeaning::GreaterEq(-1)
		);
		// x ≥ 0 === 2x - 3 ≥ -3
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::GreaterEq(0)),
			LitMeaning::GreaterEq(-3)
		);
		// x < -1 === x ≤ -2 === 2x - 3 ≤ -7 === 2x - 3 < -6
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::Less(-1)),
			LitMeaning::Less(-6)
		);
		// x < 0 === x ≤ -1 === 2x - 3 ≤ -5 === 2x - 3 < -4
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::Less(0)),
			LitMeaning::Less(-4)
		);
		// x < 1 === x ≤ 0 === 2x - 3 ≤ -3 === 2x - 3 < -2
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::Less(1)),
			LitMeaning::Less(-2)
		);
		let transformer = LinearTransform::new(NonZeroIntVal::new(-3).unwrap(), 2);
		// x ≥ -1 === -3x + 2 ≤ 5 === -3x + 2 < 6
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::GreaterEq(-1)),
			LitMeaning::Less(6)
		);
		// x ≥ 0 === -3x + 2 ≤ 2 === -3x + 2 < 3
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::GreaterEq(0)),
			LitMeaning::Less(3)
		);
		// x ≥ 1 === -3x + 2 ≤ -1 === -3x + 2 < 0
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::GreaterEq(1)),
			LitMeaning::Less(0)
		);
		// x < -1 === x ≤ -2 === -3x + 2 ≥ 8
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::Less(-1)),
			LitMeaning::GreaterEq(8)
		);
		// x < 0 === x ≤ -1 === -3x + 2 ≥ 5
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::Less(0)),
			LitMeaning::GreaterEq(5)
		);
		// x < 1 === x ≤ 0 === -3x + 2 ≥ 2
		assert_eq!(
			!transformer.transform_lit(!LitMeaning::Less(1)),
			LitMeaning::GreaterEq(2)
		);
	}

	#[test]
	#[traced_test]
	fn test_weakest_transform_lit() {
		let transformer = LinearTransform::new(NonZeroIntVal::new(2).unwrap(), -3);
		// x = 2 === 2x - 3 = 1
		assert_eq!(
			transformer.transform_lit(LitMeaning::Eq(2)),
			LitMeaning::Eq(1)
		);
		// x = 1 === 2x - 3 = -1
		assert_eq!(
			transformer.transform_lit(LitMeaning::Eq(1)),
			LitMeaning::Eq(-1)
		);
		// x ≠ 2 === 2x - 3 ≠ 1
		assert_eq!(
			transformer.transform_lit(LitMeaning::NotEq(2)),
			LitMeaning::NotEq(1)
		);
		// x ≠ 1 === 2x - 3 ≠ -1
		assert_eq!(
			transformer.transform_lit(LitMeaning::NotEq(1)),
			LitMeaning::NotEq(-1)
		);
		// x ≥ 2 === x > 1 === 2x - 3 > -1 === 2x - 3 ≥ 0
		assert_eq!(
			transformer.transform_lit(LitMeaning::GreaterEq(2)),
			LitMeaning::GreaterEq(0)
		);
		// x ≥ 1 === x > 0 === 2x - 3 ≥ -3 === 2x - 3 ≥ -2
		assert_eq!(
			transformer.transform_lit(LitMeaning::GreaterEq(1)),
			LitMeaning::GreaterEq(-2)
		);
		// x ≥ 0 === x > -1 === 2x - 3 > -5 === 2x - 3 ≥ -4
		assert_eq!(
			transformer.transform_lit(LitMeaning::GreaterEq(0)),
			LitMeaning::GreaterEq(-4)
		);
		// x < -1 === 2x - 3 < -5
		assert_eq!(
			transformer.transform_lit(LitMeaning::Less(-1)),
			LitMeaning::Less(-5)
		);
		// x < 0 === 2x - 3 < -3
		assert_eq!(
			transformer.transform_lit(LitMeaning::Less(0)),
			LitMeaning::Less(-3)
		);
		// x < 1 === 2x - 3 < -1
		assert_eq!(
			transformer.transform_lit(LitMeaning::Less(1)),
			LitMeaning::Less(-1)
		);
		let transformer = LinearTransform::new(NonZeroIntVal::new(-3).unwrap(), 2);
		// x ≥ -1 === x > -2 === -3x + 2 < 8
		assert_eq!(
			transformer.transform_lit(LitMeaning::GreaterEq(-1)),
			LitMeaning::Less(8)
		);
		// x ≥ 0 === x > -1 === -3x + 2 < 5
		assert_eq!(
			transformer.transform_lit(LitMeaning::GreaterEq(0)),
			LitMeaning::Less(5)
		);
		// x ≥ 1 === x > 0 === -3x + 2 < 2
		assert_eq!(
			transformer.transform_lit(LitMeaning::GreaterEq(1)),
			LitMeaning::Less(2)
		);
		// x < -1 === -3x + 2 > 5 === -3x + 2 ≥ 6
		assert_eq!(
			transformer.transform_lit(LitMeaning::Less(-1)),
			LitMeaning::GreaterEq(6)
		);
		// x < 0 === -3x + 2 > 2 === -3x + 2 ≥ 3
		assert_eq!(
			transformer.transform_lit(LitMeaning::Less(0)),
			LitMeaning::GreaterEq(3)
		);
		// x < 1 === -3x + 2 > -1 === -3x + 2 ≥ 0
		assert_eq!(
			transformer.transform_lit(LitMeaning::Less(1)),
			LitMeaning::GreaterEq(0)
		);
	}

	#[test]
	#[traced_test]
	fn test_rev_transform_lit() {
		let transformer = LinearTransform::new(NonZeroIntVal::new(2).unwrap(), -3);
		// 2x - 3 = 2 === 2x = 5 -> constant false
		assert_eq!(transformer.rev_transform_lit(LitMeaning::Eq(2)), Err(false));
		// 2x - 3 = 1 === 2x = 4 -> x = 2
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::Eq(1)),
			Ok(LitMeaning::Eq(2))
		);
		// 2x - 3 ≠ 2 === 2x ≠ 5 -> constant true
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::NotEq(2)),
			Err(true)
		);
		// 2x - 3 = 1 === 2x ≠ 4 -> x ≠ 2
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::NotEq(1)),
			Ok(LitMeaning::NotEq(2))
		);
		// 2x - 3 ≥ 2 === 2x ≥ 5 == x ≥ ceil(5/2) == x ≥ 3
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::GreaterEq(2)),
			Ok(LitMeaning::GreaterEq(3))
		);
		// 2x - 3 ≥ 1 === 2x ≥ 4 == x ≥ ceil(4/2) == x ≥ 2
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::GreaterEq(1)),
			Ok(LitMeaning::GreaterEq(2))
		);
		// 2x - 3 ≥ 0 === 2x ≥ 3 == x ≥ ceil(3/2) == x ≥ 2
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::GreaterEq(0)),
			Ok(LitMeaning::GreaterEq(2))
		);
		// 2x - 3 ≥ -1 === 2x ≥ 2 == x ≥ ceil(2/2) == x ≥ 1
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::GreaterEq(-1)),
			Ok(LitMeaning::GreaterEq(1))
		);
		// 2x - 3 ≥ -2 === 2x ≥ 1 == x ≥ ceil(1/2) == x ≥ 1
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::GreaterEq(-2)),
			Ok(LitMeaning::GreaterEq(1))
		);
		// 2x - 3 < -2 === 2x < 1 === x < 1/2 === x ≤ floor(1/2) === x ≤ 0 === x < 1
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::Less(-2)),
			Ok(LitMeaning::Less(1))
		);
		// 2x - 3 < -1 === 2x < 2 === x < 2/2 === x < 1
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::Less(-1)),
			Ok(LitMeaning::Less(1))
		);
		// 2x - 3 < 0 === 2x < 3 === x < 3/2 === x ≤ floor(3/2) === x ≤ 1 === x < 2
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::Less(0)),
			Ok(LitMeaning::Less(2))
		);
		// 2x - 3 < 1 === 2x < 4 === x < 4/2 === x < 2
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::Less(1)),
			Ok(LitMeaning::Less(2))
		);
		// 2x - 3 < 2 === 2x < 5 === x < 5/2 === x ≤ floor(5/2) === x ≤ 2 === x < 3
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::Less(2)),
			Ok(LitMeaning::Less(3))
		);
		let transformer = LinearTransform::new(NonZeroIntVal::new(-2).unwrap(), 2);
		// -2x + 2 = 2 === 2x = 0 -> x = 0
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::Eq(2)),
			Ok(LitMeaning::Eq(0))
		);
		// -2x + 2 = 1 === 2x = -1 -> constant false
		assert_eq!(transformer.rev_transform_lit(LitMeaning::Eq(1)), Err(false));
		// -2x + 2 ≠ 2 === 2x ≠ 0 -> x ≠ 0
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::NotEq(2)),
			Ok(LitMeaning::NotEq(0))
		);
		// -2x + 2 ≠ 1 === 2x ≠ -1 -> constant true
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::NotEq(1)),
			Err(true)
		);
		// -2x + 2 ≥ -2 === -2x ≥ -4 === x ≤ 2 === x < 3
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::GreaterEq(-2)),
			Ok(LitMeaning::Less(3))
		);
		// -2x + 2 ≥ -1 === -2x ≥ -3 === x ≤ floor(-3/-2) === x ≤ 1 === x < 2
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::GreaterEq(-1)),
			Ok(LitMeaning::Less(2))
		);
		// -2x + 2 ≥ 0 === -2x ≥ -2 === x ≤ 1 === x < 2
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::GreaterEq(0)),
			Ok(LitMeaning::Less(2))
		);
		// -2x + 2 ≥ 1 === -2x ≥ -1 === x ≤ floor(-1/-2) === x ≤ 0 === x < 1
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::GreaterEq(1)),
			Ok(LitMeaning::Less(1))
		);
		// -2x + 2 ≥ 2 === -2x ≥ 0 === x ≤ 0 === x < 1
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::GreaterEq(1)),
			Ok(LitMeaning::Less(1))
		);
		// -2x + 2 < -2 === -2x < -4 === x > 2 === x ≥ 3
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::Less(-2)),
			Ok(LitMeaning::GreaterEq(3))
		);
		// -2x + 2 < -1 === -2x < -3 === 2x > 3 === x ≥ ceil(3/2) === x ≥ 2
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::Less(-1)),
			Ok(LitMeaning::GreaterEq(2))
		);
		// -2x + 2 < 0 === -2x < -2 === x > 1 === x ≥ 2
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::Less(0)),
			Ok(LitMeaning::GreaterEq(2))
		);
		// -2x + 2 < 1 === -2x < -1 === 2x > 1 === x ≥ ceil(1/2) === x ≥ 1
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::Less(1)),
			Ok(LitMeaning::GreaterEq(1))
		);
		// -2x + 2 < 2 === -2x < 0 === x > 0 === x ≥ 1
		assert_eq!(
			transformer.rev_transform_lit(LitMeaning::Less(2)),
			Ok(LitMeaning::GreaterEq(1))
		);
	}

	#[test]
	#[traced_test]
	fn test_relaxed_less_lit() {
		let transformer = LinearTransform::new(NonZeroIntVal::new(3).unwrap(), 1);
		// 3x + 1 < -3, reverse lit: x < -1, relaxed lit: 3x + 1 < -2
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::Less(-3)),
			LitMeaning::Less(-2)
		);
		// 3x + 1 < -2, reverse lit: x < -1, relaxed lit: 3x + 1 < -2
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::Less(-2)),
			LitMeaning::Less(-2)
		);
		// 3x + 1 < -1, reverse lit: x < 0, relaxed lit: 3x + 1 < 1
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::Less(-1)),
			LitMeaning::Less(1)
		);
		// 3x + 1 < 0, reverse lit: x < 0, relaxed lit: 3x + 1 < 1
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::Less(0)),
			LitMeaning::Less(1)
		);
		// 3x + 1 < 1, reverse lit: x < 0, relaxed lit: 3x + 1 < 1
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::Less(1)),
			LitMeaning::Less(1)
		);
		// 3x + 1 < 2, reverse lit: x < 1, relaxed lit: 3x + 1 < 4
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::Less(2)),
			LitMeaning::Less(4)
		);
		// 3x + 1 < 3, reverse lit: x < 1, relaxed lit: 3x + 1 < 4
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::Less(3)),
			LitMeaning::Less(4)
		);
		let transformer = LinearTransform::new(NonZeroIntVal::new(-3).unwrap(), 2);
		// -3x + 2 < -3, reverse lit: x ≥ 2 === x > 1, relaxed lit: -3x + 2 < -1
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::Less(-3)),
			LitMeaning::Less(-1)
		);
		// -3x + 2 < -2, reverse lit: x ≥ 2 === x > 1, relaxed lit: -3x + 2 < -1
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::Less(-2)),
			LitMeaning::Less(-1)
		);
		// -3x + 2 < -1, reverse lit: x ≥ 2 === x > 1, relaxed lit: -3x + 2 < -1
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::Less(-1)),
			LitMeaning::Less(-1)
		);
		// -3x + 2 < 0, reverse lit: x ≥ 1 === x > 0, relaxed lit: -3x + 2 < 2
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::Less(0)),
			LitMeaning::Less(2)
		);
		// -3x + 2 < 1, reverse lit: x ≥ 1 == x > 0, relaxed lit: -3x + 2 < 2
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::Less(1)),
			LitMeaning::Less(2)
		);
		// -3x + 2 < 2, reverse lit: x ≥ 1 === x > 0, relaxed lit: -3x + 2 < 2
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::Less(2)),
			LitMeaning::Less(2)
		);
		// -3x + 2 < 3, reverse lit: x ≥ 0 === x > -1, relaxed lit: -3x + 2 < 5
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::Less(3)),
			LitMeaning::Less(5)
		);
	}

	#[test]
	#[traced_test]
	fn test_relaxed_greater_or_equal_lit() {
		let transformer = LinearTransform::new(NonZeroIntVal::new(3).unwrap(), 1);
		// 3x + 1 ≥ -3, reverse lit: x ≥ -1, relaxed lit: 3x + 1 > -5 === 3x + 1 ≥ -4
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::GreaterEq(-3)),
			LitMeaning::GreaterEq(-4)
		);
		// 3x + 1 ≥ -2, reverse lit: x ≥ -1, relaxed lit: 3x + 1 > -5 === 3x + 1 ≥ -4
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::GreaterEq(-2)),
			LitMeaning::GreaterEq(-4)
		);
		// 3x + 1 ≥ -1, reverse lit: x ≥ 0, relaxed lit: 3x + 1 > -2 === 3x + 1 ≥ -1
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::GreaterEq(-1)),
			LitMeaning::GreaterEq(-1)
		);
		// 3x + 1 ≥ 0, reverse lit: x ≥ 0, relaxed lit: 3x + 1 > -2 === 3x + 1 ≥ -1
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::GreaterEq(0)),
			LitMeaning::GreaterEq(-1)
		);
		// 3x + 1 ≥ 1, reverse lit: x ≥ 0, relaxed lit: 3x + 1 > -2 === 3x + 1 ≥ -1
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::GreaterEq(1)),
			LitMeaning::GreaterEq(-1)
		);
		// 3x + 1 ≥ 1, reverse lit: x ≥ 1, relaxed lit: 3x + 1 > 1 === 3x + 1 ≥ 2
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::GreaterEq(2)),
			LitMeaning::GreaterEq(2)
		);
		// 3x + 1 ≥ 2, reverse lit: x ≥ 1, relaxed lit: 3x + 1 > 1 === 3x + 1 ≥ 2
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::GreaterEq(3)),
			LitMeaning::GreaterEq(2)
		);
		let transformer = LinearTransform::new(NonZeroIntVal::new(-3).unwrap(), 2);
		// -3x + 2 ≥ -2, reverse lit: x < 2, relaxed lit: -3x + 2 > -4 === -3x + 2 ≥ -3
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::GreaterEq(-3)),
			LitMeaning::GreaterEq(-3)
		);
		// -3x + 2 ≥ -2, reverse lit: x < 2, relaxed lit: -3x + 2 > -4 === -3x + 2 ≥ -3
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::GreaterEq(-2)),
			LitMeaning::GreaterEq(-3)
		);
		// -3x + 2 ≥ -1, reverse lit: x < 2, relaxed lit: -3x + 2 > -4 === -3x + 2 ≥ -3
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::GreaterEq(-1)),
			LitMeaning::GreaterEq(-3)
		);
		// -3x + 2 ≥ 0, reverse lit: x < 1, relaxed lit: -3x + 2 > -1 === -3x + 2 ≥ 0
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::GreaterEq(0)),
			LitMeaning::GreaterEq(0)
		);
		// -3x + 2 ≥ 1, reverse lit: x < 1, relaxed lit: -3x + 2 > -1 === -3x + 2 ≥ 0
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::GreaterEq(1)),
			LitMeaning::GreaterEq(0)
		);
		// -3x + 2 ≥ 2, reverse lit: x < 1, relaxed lit: -3x + 2 > -1 === -3x + 2 ≥ 0
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::GreaterEq(2)),
			LitMeaning::GreaterEq(0)
		);
		// -3x + 2 ≥ 3, reverse lit: x < 0, relaxed lit: -3x + 2 > 2 === -3x + 2 ≥ 3
		assert_eq!(
			transformer.relaxed_lit(LitMeaning::GreaterEq(3)),
			LitMeaning::GreaterEq(3)
		);
	}
}
