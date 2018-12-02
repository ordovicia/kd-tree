use std::{collections::HashMap, hash::Hash};

use num_traits::Float;

use crate::{dist_ordered_point::DistOrderedPoint, error::ErrorKind, Result};

#[derive(Debug, Clone)]
pub struct KdTreeMap<Axis, Point, Value>
where
    Axis: Float,
    Point: AsRef<[Axis]> + PartialEq + Eq + Hash,
{
    dim: usize,
    node_capacity: usize,

    pub(crate) points: HashMap<Point, Value>,
    pub(crate) children: Option<Children<Axis, KdTreeMap<Axis, Point, Value>>>,

    bound: Bound<Axis>,
}

#[derive(Debug, Clone)]
pub(crate) struct Children<Axis, KdTree>
where
    Axis: Float,
{
    split: Split<Axis>,
    left: Box<KdTree>,
    right: Box<KdTree>,
}

#[derive(Debug, Clone)]
pub(crate) struct Split<Axis: Float> {
    dim: usize,
    thresh: Axis,
}

#[derive(Debug, Clone)]
pub(crate) struct Bound<Axis: Float> {
    min: Vec<Option<Axis>>,
    max: Vec<Option<Axis>>,
}

impl<Axis, Point, Value> KdTreeMap<Axis, Point, Value>
where
    Axis: Ord + Float,
    Point: AsRef<[Axis]> + PartialEq + Eq + Hash,
{
    /// Creates a kd-tree with `dim` dimensions.
    /// Every node in the tree can have up to `node_capacity` points.
    ///
    /// # Panic
    ///
    /// Panics if neither `dim` nor `node_capacity` is positive.
    pub fn new(dim: usize, node_capacity: usize) -> Self {
        assert!(dim > 0, "the number of dimensions must be positive");
        assert!(node_capacity > 0, "node capacity must be positive");

        Self {
            dim,
            node_capacity,
            points: HashMap::new(),
            children: None,
            bound: Bound {
                min: vec![None; dim],
                max: vec![None; dim],
            },
        }
    }

    /// Returns the number of dimensions of this kd-tree.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the node capacity of this kd-tree.
    pub fn node_capacity(&self) -> usize {
        self.node_capacity
    }

    /// Returns the number of points this kd-tree holds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate kdtree;
    /// # extern crate noisy_float;
    /// use kdtree::KdTreeMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// kdtree.insert([r64(1.0); 2], 1.0);
    /// assert_eq!(kdtree.size(), 1);
    ///
    /// kdtree.insert([r64(2.0); 2], 2.0);
    /// assert_eq!(kdtree.size(), 2);
    /// ```
    pub fn size(&self) -> usize {
        let mut size = self.points.len();

        if let Some(Children {
            ref left,
            ref right,
            ..
        }) = self.children
        {
            size += left.size();
            size += right.size();
        }

        size
    }

    /// Inserts a point to this kd-tree with a value.
    /// If the same point already exists in this kd-tree, the value will be overwritten with the
    /// new one.
    ///
    /// Returns `Err` when the number of dimension of the point does not match with that of this
    /// tree, or when the location of the point is not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate kdtree;
    /// # extern crate noisy_float;
    /// use kdtree::KdTreeMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
    ///
    /// let (p1, val): ([R64; 2], f64) = ([r64(1.0); 2], 1.0);
    /// kdtree.insert(p1, val);
    /// ```
    ///
    /// ```rust
    /// # extern crate kdtree;
    /// # extern crate noisy_float;
    /// use kdtree::KdTreeMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
    /// // The numbers of dimensions do not match
    /// assert!(kdtree.insert([r64(1.0); 1], 0.0).is_err());
    /// ```
    pub fn insert(&mut self, point: Point, value: Value) -> Result<()> {
        // #![allow(clippy::map_entry)]

        self.check_point(point.as_ref())?;

        if let Some(Children {
            ref split,
            ref mut left,
            ref mut right,
        }) = self.children
        {
            let Split { dim, thresh } = *split;
            if point.as_ref()[dim] < thresh {
                left.as_mut().insert(point, value)?;
            } else {
                right.as_mut().insert(point, value)?;
            }
        } else {
            self.points.insert(point, value);
            if self.points.len() > self.node_capacity {
                self.split();
            }
        }

        Ok(())
    }

    // TODO: support getting nearest N points
    //
    /// Returns the nearest point from the query point.
    /// The distance between two points are calculated with `dist_func` function.
    ///
    /// Returns `Ok(None)` when the kd-tree has no points.
    ///
    /// Returns `Err` when the number of dimension of the point does not match with that of this
    /// tree, or when the location of the point is not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate kdtree;
    /// # extern crate noisy_float;
    /// # extern crate num_traits;
    /// use kdtree::KdTreeMap;
    /// use noisy_float::prelude::*;
    /// use num_traits::{Float, Zero};
    ///
    /// let squared_euclidean = |p1: &[R64], p2: &[R64]| -> R64 {
    ///     p1.iter()
    ///         .zip(p2.iter())
    ///         .map(|(&p1, &p2)| (p1 - p2) * (p1 - p2))
    ///         .fold(R64::zero(), std::ops::Add::add)
    /// };
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
    ///
    /// let p1: [R64; 2] = [r64(1.0); 2];
    /// let p2: [R64; 2] = [r64(2.0); 2];
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&p1, &squared_euclidean).unwrap(),
    ///     None
    /// );
    ///
    /// kdtree.insert(p1, 1.0);
    /// kdtree.insert(p2, 2.0);
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&p1, &squared_euclidean).unwrap(),
    ///     Some((&p1, &1.0))
    /// );
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&[r64(3.0), r64(3.0)], &squared_euclidean).unwrap(),
    ///     Some((&p2, &2.0))
    /// );
    /// ```
    pub fn nearest(
        &self,
        query: &Point,
        dist_func: &Fn(&[Axis], &[Axis]) -> Axis,
    ) -> Result<Option<(&Point, &Value)>> {
        self.check_point(query.as_ref())?;

        if self.size() == 0 {
            return Ok(None);
        }

        let mut leaf = self;
        let mut nodes_other_side = vec![];
        while let Some(Children {
            split: Split { dim, thresh },
            left,
            right,
        }) = &leaf.children
        {
            if query.as_ref()[*dim] < *thresh {
                leaf = left;
                nodes_other_side.push(right);
            } else {
                leaf = right;
                nodes_other_side.push(left);
            }
        }

        let mut point_nearest = leaf.nearest_point_node(query.as_ref(), dist_func);
        while let Some(node_other_side) = nodes_other_side.pop() {
            if node_other_side.dist_to_point(query.as_ref(), dist_func) > point_nearest.dist {
                break;
            }

            let point_nearest_node = node_other_side.nearest_point_node(query.as_ref(), dist_func);
            if point_nearest_node < point_nearest {
                point_nearest = point_nearest_node;
            }
        }

        Ok(Some(point_nearest.into()))
    }

    fn with_points_bound(
        dim: usize,
        node_capacity: usize,
        points: HashMap<Point, Value>,
        bound: Bound<Axis>,
    ) -> Self {
        Self {
            dim,
            node_capacity,
            points,
            children: None,
            bound,
        }
    }

    pub(crate) fn check_point(&self, point: &[Axis]) -> Result<()> {
        if point.len() != self.dim {
            Err(ErrorKind::DimensionNotMatch {
                expected: self.dim,
                actual: point.len(),
            })?;
        }

        for dim in point.iter() {
            if !dim.is_finite() {
                Err(ErrorKind::LocationNotFinite)?;
            }
        }

        Ok(())
    }

    pub(crate) fn split(&mut self) {
        let Split { dim, thresh } = self.calc_split();

        let (mut bound_left, mut bound_right) = (self.bound.clone(), self.bound.clone());
        bound_left.max[dim] = Some(thresh);
        bound_right.min[dim] = Some(thresh);

        let (mut points_left, mut points_right) = (HashMap::new(), HashMap::new());
        for (point, values) in self.points.drain() {
            if point.as_ref()[dim] < thresh {
                points_left.insert(point, values);
            } else {
                points_right.insert(point, values);
            }
        }

        self.children = Some(Children {
            split: Split { dim, thresh },
            left: Box::new(KdTreeMap::with_points_bound(
                self.dim,
                self.node_capacity,
                points_left,
                bound_left,
            )),
            right: Box::new(KdTreeMap::with_points_bound(
                self.dim,
                self.node_capacity,
                points_right,
                bound_right,
            )),
        });
    }

    fn calc_split(&self) -> Split<Axis> {
        let (mut dim, mut max_width) = (0, Axis::zero());
        let mut thresh = Axis::zero();

        for (d, (width, median)) in (0..self.dim).map(|dim| (dim, self.bounding_width_median(dim)))
        {
            if width > max_width {
                dim = d;
                max_width = width;
                thresh = median;
            }
        }

        Split { dim, thresh }
    }

    fn bounding_width_median(&self, dim: usize) -> (Axis, Axis) {
        let points = self.points.keys().map(|p| p.as_ref()[dim]);

        let min = points
            .clone()
            .min()
            .expect("unexpectedly empty points in KdTreeMap:bounding_width_median()");
        let max = points.max().unwrap();

        let width = max - min;
        let median = min + (max - min) / Axis::from(2).unwrap();
        (width, median)
    }

    fn nearest_point_node(
        &self,
        query: &[Axis],
        dist_func: &Fn(&[Axis], &[Axis]) -> Axis,
    ) -> DistOrderedPoint<Axis, &Point, &Value> {
        self.points
            .iter()
            .map(|(p, v)| DistOrderedPoint::new(p, v, dist_func(query, p.as_ref())))
            .min()
            .expect("unexpectedly empty points in KdTreeMap::nearest_point_node()")
    }

    fn dist_to_point(&self, point: &[Axis], dist_func: &Fn(&[Axis], &[Axis]) -> Axis) -> Axis {
        use std::cmp;

        let p2 = point
            .iter()
            .zip(self.bound.min.iter().zip(self.bound.max.iter()))
            .map(|(&p, (min, max))| {
                let p_min = min.and_then(|m| Some(cmp::max(m, p))).unwrap_or(p);
                max.and_then(|m| Some(cmp::min(m, p_min))).unwrap_or(p_min)
            }).collect::<Vec<Axis>>();

        dist_func(point, &p2[..])
    }
}

impl<Axis, Point, Value> KdTreeMap<Axis, Point, Value>
where
    Axis: Ord + Float,
    Value: Clone,
    Point: AsRef<[Axis]> + Clone + PartialEq + Eq + Hash,
{
    /// Returns the all points this kd-tree holds.
    /// Points that descendant nodes have are all included.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate kdtree;
    /// # extern crate noisy_float;
    /// use kdtree::KdTreeMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1: [R64; 2] = [r64(1.0); 2];
    /// let p2: [R64; 2] = [r64(2.0); 2];
    ///
    /// kdtree.insert(p1, 1.0);
    /// kdtree.insert(p2, 2.0);
    ///
    /// assert_eq!(
    ///     kdtree.points_tree(),
    ///     [(p1, 1.0), (p2, 2.0)].iter().cloned().collect()
    /// );
    /// ```
    pub fn points_tree(&self) -> HashMap<Point, Value> {
        let mut points = self.points.clone();
        if let Some(Children {
            ref left,
            ref right,
            ..
        }) = self.children
        {
            for (p, v) in left.points_tree() {
                points.insert(p, v);
            }
            for (p, v) in right.points_tree() {
                points.insert(p, v);
            }
        }
        points
    }
}

#[cfg(test)]
mod tests {
    extern crate noisy_float;
    use self::noisy_float::prelude::*;

    use super::*;

    #[test]
    #[should_panic]
    fn panic_new_zero_dim() {
        let _: KdTreeMap<R64, [R64; 0], f64> = KdTreeMap::new(0, 1);
    }

    #[test]
    #[should_panic]
    fn panic_new_zero_capacity() {
        let _: KdTreeMap<R64, [R64; 2], f64> = KdTreeMap::new(2, 0);
    }
}
