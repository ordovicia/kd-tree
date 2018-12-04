use crate::split::Split;

#[derive(Debug, Clone)]
pub struct Cell<Axis: Clone> {
    min: Vec<Option<Axis>>,
    max: Vec<Option<Axis>>,
}

impl<Axis> Cell<Axis>
where
    Axis: Clone + PartialOrd,
{
    pub fn new(dim: usize) -> Self {
        Self {
            min: vec![None; dim],
            max: vec![None; dim],
        }
    }

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

    pub fn dist_to_point(&self, point: &[Axis], dist_func: &Fn(&[Axis], &[Axis]) -> Axis) -> Axis {
        let p2 = point
            .iter()
            .zip(self.min.iter().zip(self.max.iter()))
            .map(|(p, (min, max))| {
                let p_min = if let Some(min) = min {
                    if p < min {
                        min
                    } else {
                        p
                    }
                } else {
                    p
                };

                if let Some(max) = max {
                    if p_min > max {
                        max.clone()
                    } else {
                        p_min.clone()
                    }
                } else {
                    p_min.clone()
                }
            }).collect::<Vec<_>>();

        dist_func(point, &p2[..])
    }
}
