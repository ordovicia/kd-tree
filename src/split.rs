#[cfg(feature = "serialize")]
use serde_derive::{Deserialize, Serialize};

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Split<Axis: Clone> {
    pub dim: usize,
    pub thresh: Axis,
}

impl<Axis> Split<Axis>
where
    Axis: Clone + PartialOrd,
{
    #[inline]
    pub fn belongs_to_left<Point: AsRef<[Axis]>>(&self, point: &Point) -> bool {
        let point = point.as_ref();
        point[self.dim] < self.thresh
    }
}
