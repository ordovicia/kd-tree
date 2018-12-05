use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct PointDist<Axis, Point, Value> {
    pub point: Point,
    pub value: Value,
    pub dist: Axis,
}

impl<Axis, Point, Value> PointDist<Axis, Point, Value> {
    pub fn new(point: Point, value: Value, dist: Axis) -> Self {
        PointDist { point, value, dist }
    }
}

impl<Axis, Point, Value> PartialOrd for PointDist<Axis, Point, Value>
where
    Axis: PartialOrd,
    Point: PartialEq,
{
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&rhs.dist)
    }
}

impl<Axis, Point, Value> Ord for PointDist<Axis, Point, Value>
where
    Axis: Ord,
    Point: PartialEq,
{
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.dist.cmp(&rhs.dist)
    }
}

impl<Axis, Point: PartialEq, Value> PartialEq for PointDist<Axis, Point, Value> {
    fn eq(&self, rhs: &Self) -> bool {
        self.point == rhs.point
    }
}

impl<Axis, Point: PartialEq, Value> Eq for PointDist<Axis, Point, Value> {}
