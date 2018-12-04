#[derive(Debug)]
pub struct Split<Axis> {
    pub dim: usize,
    pub thresh: Axis,
}

impl<Axis: PartialOrd> Split<Axis> {
    pub fn belongs_to_left<Point: AsRef<[Axis]>>(&self, point: &Point) -> bool {
        let point = point.as_ref();
        point[self.dim] < self.thresh
    }
}

impl<Axis: Clone> Clone for Split<Axis> {
    fn clone(&self) -> Self {
        Self {
            dim: self.dim,
            thresh: self.thresh.clone(),
        }
    }
}
