use std::cmp::Ordering;

pub(crate) struct DistOrderedPoint<Axis, Point, Value> {
    point: Point,
    value: Value,
    pub(crate) dist: Axis,
}

impl<Axis, Point, Value> DistOrderedPoint<Axis, Point, Value> {
    pub fn new(point: Point, value: Value, dist: Axis) -> Self {
        DistOrderedPoint { point, value, dist }
    }
}

impl<Axis, Point, Value> PartialOrd for DistOrderedPoint<Axis, Point, Value>
where
    Point: PartialEq,
    Axis: Ord,
{
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&rhs.dist)
    }
}

impl<Axis, Point, Value> Ord for DistOrderedPoint<Axis, Point, Value>
where
    Point: PartialEq,
    Axis: Ord,
{
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.dist.cmp(&rhs.dist)
    }
}

impl<Axis, Point: PartialEq, Value> PartialEq for DistOrderedPoint<Axis, Point, Value> {
    fn eq(&self, rhs: &Self) -> bool {
        self.point == rhs.point
    }
}

impl<Axis, Point: PartialEq, Value> Eq for DistOrderedPoint<Axis, Point, Value> {}

impl<Axis, Point, Value> Into<(Point, Value)> for DistOrderedPoint<Axis, Point, Value> {
    fn into(self) -> (Point, Value) {
        (self.point, self.value)
    }
}
