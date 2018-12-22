#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

use crate::split::Split;

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Cell<Axis: Clone> {
    min: Vec<Option<Axis>>,
    max: Vec<Option<Axis>>,
}

impl<Axis: Clone> Cell<Axis> {
    pub fn new(dim: usize) -> Self {
        Self {
            min: vec![None; dim],
            max: vec![None; dim],
        }
    }
}

impl<Axis> Cell<Axis>
where
    Axis: Clone + PartialOrd,
{
    pub fn split(&self, split: &Split<Axis>) -> (Self, Self) {
        let (mut left, mut right) = (self.clone(), self.clone());

        *left
            .max
            .get_mut(split.dim)
            .expect("unexpectedly mismatched dimensions in Cell::split()") =
            Some(split.thresh.clone());
        right.min[split.dim] = Some(split.thresh.clone());
        (left, right)
    }

    #[inline]
    pub fn dist_to_point(
        &self,
        point: &[Axis],
        dist_func: &dyn Fn(&[Axis], &[Axis]) -> Axis,
    ) -> Axis {
        let p2 = point
            .iter()
            .zip(self.min.iter().zip(self.max.iter()))
            .map(|(p, (min, max))| {
                let p_min = match min {
                    Some(min) if p < min => min,
                    _ => p,
                };

                match max {
                    Some(max) if p_min > max => max,
                    _ => p_min,
                }
            })
            .cloned()
            .collect::<Vec<_>>();

        dist_func(point, &p2[..])
    }
}
