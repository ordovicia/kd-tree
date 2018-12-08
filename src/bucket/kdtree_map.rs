use std::{collections::HashMap, hash::Hash};

use num_traits::Float;

use crate::{cell::Cell, error::ErrorKind, point_dist::PointDist, split::Split, Result};

#[derive(Clone, Debug)]
pub struct KdTreeMap<Axis, Point, Value>
where
    Axis: Ord + Float,
    Point: AsRef<[Axis]> + PartialEq + Eq + Hash,
{
    dim: usize,
    bucket_size: usize,

    points: HashMap<Point, Vec<Value>>,
    children: Option<Children<Axis, KdTreeMap<Axis, Point, Value>>>,

    cell: Cell<Axis>,
}

#[derive(Clone, Debug)]
struct Children<Axis: Clone, KdTree> {
    split: Split<Axis>,
    left: Box<KdTree>,
    right: Box<KdTree>,
}

impl<Axis, Point, Value> KdTreeMap<Axis, Point, Value>
where
    Axis: Ord + Float,
    Point: AsRef<[Axis]> + PartialEq + Eq + Hash,
{
    /// Creates a kd-tree with `dim` dimensions.
    /// Every node in the tree can have up to `bucket_size` points.
    ///
    /// # Panic
    ///
    /// Panics if neither `dim` nor `bucket_size` is positive.
    pub fn new(dim: usize, bucket_size: usize) -> Self {
        assert!(dim > 0, "the number of dimensions must be positive");
        assert!(bucket_size > 0, "bucket size must be positive");

        Self {
            dim,
            bucket_size,
            points: HashMap::new(),
            children: None,
            cell: Cell::new(dim),
        }
    }

    /// Returns the number of dimensions of this kd-tree.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the bucket size of this kd-tree.
    pub fn bucket_size(&self) -> usize {
        self.bucket_size
    }

    /// Returns the number of points this kd-tree holds.
    /// Multiple values on the same location are counted as many.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kd_tree::bucket::KdTreeMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1: [R64; 2] = [r64(1.0); 2];
    ///
    /// kdtree.append(p1, 1.0).unwrap();
    /// assert_eq!(kdtree.size(), 1);
    ///
    /// kdtree.append(p1, 2.0).unwrap();
    /// assert_eq!(kdtree.size(), 2);
    /// ```
    pub fn size(&self) -> usize {
        let mut size = self.points.values().map(|values| values.len()).sum();

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

    /// Returns the number of points this kd-tree holds.
    /// Multiple values on the same point are counted as a single point.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kd_tree::bucket::KdTreeMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1: [R64; 2] = [r64(1.0); 2];
    ///
    /// kdtree.append(p1, 1.0).unwrap();
    /// assert_eq!(kdtree.size_unique(), 1);
    ///
    /// kdtree.append(p1, 2.0).unwrap();
    /// assert_eq!(kdtree.size_unique(), 1);
    /// ```
    pub fn size_unique(&self) -> usize {
        let mut size = self.points.len();

        if let Some(Children {
            ref left,
            ref right,
            ..
        }) = self.children
        {
            size += left.size_unique();
            size += right.size_unique();
        }

        size
    }

    /// Appends a point to this kd-tree with a value.
    /// If the same point already exists in this kd-tree, the value is appended to the existing
    /// point.
    ///
    /// Returns `Err` when the number of dimension of the point does not match with that of this
    /// tree, or when the location of the point is not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kd_tree::bucket::KdTreeMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
    ///
    /// let p1 = [r64(1.0); 2];
    /// kdtree.append(p1, 1.0).unwrap();
    /// kdtree.append(p1, 2.0).unwrap();
    ///
    /// assert_eq!(kdtree.get(&p1).unwrap(), Some(&vec![1.0, 2.0]));
    /// ```
    pub fn append(&mut self, point: Point, value: Value) -> Result<()> {
        #![allow(clippy::map_entry)]

        self.check_point(point.as_ref())?;

        if self.points.contains_key(&point) {
            self.points.get_mut(&point).unwrap().push(value);
        } else if let Some(Children {
            ref split,
            ref mut left,
            ref mut right,
        }) = self.children
        {
            if split.belongs_to_left(&point) {
                left.as_mut().append(point, value)?;
            } else {
                right.as_mut().append(point, value)?;
            }
        } else {
            self.points.insert(point, vec![value]);
            if self.points.len() > self.bucket_size() {
                self.split();
            }
        }

        Ok(())
    }

    /// Inserts a point to this kd-tree with a value.
    /// If the same point already exists in this kd-tree, the values are overwritten with the new
    /// one.
    ///
    /// Returns `Err` when the number of dimension of the point does not match with that of this
    /// tree, or when the location of the point is not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kd_tree::bucket::KdTreeMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
    ///
    /// let p1 = [r64(1.0); 2];
    /// kdtree.insert(p1, 1.0).unwrap();
    /// kdtree.insert(p1, 2.0).unwrap();
    ///
    /// assert_eq!(kdtree.get(&p1).unwrap(), Some(&vec![2.0]));
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
            if split.belongs_to_left(&point) {
                left.as_mut().insert(point, value)?;
            } else {
                right.as_mut().insert(point, value)?;
            }
        } else {
            self.points.insert(point, vec![value]);
            if self.points.len() > self.bucket_size {
                self.split();
            }
        }

        Ok(())
    }

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
    /// use kd_tree::{bucket::KdTreeMap, PointDist};
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
    /// kdtree.append(p1, 1.0).unwrap();
    /// kdtree.append(p2, 2.0).unwrap();
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&p1, &squared_euclidean).unwrap(),
    ///     Some(PointDist { point: &p1, value: &vec![1.0], dist: r64(0.0) })
    /// );
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&[r64(3.0); 2], &squared_euclidean).unwrap(),
    ///     Some(PointDist { point: &p2, value: &vec![2.0], dist: r64(2.0) })
    /// );
    /// ```
    pub fn nearest(
        &self,
        query: &Point,
        dist_func: &Fn(&[Axis], &[Axis]) -> Axis,
    ) -> Result<Option<PointDist<Axis, &Point, &Vec<Value>>>> {
        self.check_point(query.as_ref())?;
        Ok(self.nearest_rec(query, None, dist_func))
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// Returns `Err` when the number of dimension of the point does not match with that of this
    /// tree, or when the location of the point is not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kd_tree::bucket::KdTreeMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
    /// let (p1, val): ([R64; 2], f64) = ([r64(1.0); 2], 1.0);
    ///
    /// kdtree.insert(p1, val).unwrap();
    /// assert_eq!(kdtree.get(&p1).unwrap(), Some(&vec![val]));
    ///
    /// assert_eq!(kdtree.get(&[r64(2.0); 2]).unwrap(), None);
    /// ```
    pub fn get(&self, query: &Point) -> Result<Option<&Vec<Value>>> {
        self.check_point(query.as_ref())?;

        let mut leaf = self;
        while let Some(Children { split, left, right }) = &leaf.children {
            if split.belongs_to_left(query) {
                leaf = left;
            } else {
                leaf = right;
            }
        }

        Ok(leaf.points.get(query))
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// Returns `Err` when the number of dimension of the point does not match with that of this
    /// tree, or when the location of the point is not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kd_tree::bucket::KdTreeMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
    /// let (p1, val): ([R64; 2], f64) = ([r64(1.0); 2], 1.0);
    ///
    /// kdtree.insert(p1, val).unwrap();
    ///
    /// *kdtree.get_mut(&p1).unwrap().unwrap() = vec![2.0];
    /// assert_eq!(kdtree.get(&p1).unwrap(), Some(&vec![2.0]));
    ///
    /// assert_eq!(kdtree.get_mut(&[r64(2.0); 2]).unwrap(), None);
    /// ```
    pub fn get_mut(&mut self, query: &Point) -> Result<Option<&mut Vec<Value>>> {
        self.check_point(query.as_ref())?;
        Ok(self.get_mut_unchecked(query))
    }

    fn check_point(&self, point: &[Axis]) -> Result<()> {
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

    fn nearest_rec(
        &self,
        query: &Point,
        mut dist_min: Option<Axis>,
        dist_func: &Fn(&[Axis], &[Axis]) -> Axis,
    ) -> Option<PointDist<Axis, &Point, &Vec<Value>>> {
        if self.size() == 0 {
            return None;
        }

        if let Some(dist_min) = dist_min {
            if self.cell.dist_to_point(query.as_ref(), dist_func) > dist_min {
                return None;
            }
        }

        let mut leaf = self;
        let mut other_side = vec![];
        while let Some(Children { split, left, right }) = &leaf.children {
            if split.belongs_to_left(query) {
                leaf = left;
                other_side.push(right);
            } else {
                leaf = right;
                other_side.push(left);
            }
        }

        let mut point_nearest = leaf.nearest_point_node(query.as_ref(), dist_func);
        if let Some(dm) = dist_min {
            dist_min = Some(std::cmp::min(dm, point_nearest.dist));
        } else {
            dist_min = Some(point_nearest.dist);
        }

        while let Some(other_side) = other_side.pop() {
            if let Some(point_other_side) = other_side.nearest_rec(query, dist_min, dist_func) {
                if point_other_side.dist < dist_min.unwrap() {
                    dist_min = Some(point_other_side.dist);
                    point_nearest = point_other_side;
                }
            }
        }

        Some(point_nearest)
    }

    fn nearest_point_node(
        &self,
        query: &[Axis],
        dist_func: &Fn(&[Axis], &[Axis]) -> Axis,
    ) -> PointDist<Axis, &Point, &Vec<Value>> {
        self.points
            .iter()
            .map(|(p, v)| PointDist::new(p, v, dist_func(query, p.as_ref())))
            .min()
            .expect("unexpectedly empty points in bucket::KdTreeMap::nearest_point_node()")
    }

    fn get_mut_unchecked(&mut self, query: &Point) -> Option<&mut Vec<Value>> {
        if let Some(Children {
            ref split,
            ref mut left,
            ref mut right,
        }) = self.children
        {
            if split.belongs_to_left(query) {
                left.get_mut_unchecked(query)
            } else {
                right.get_mut_unchecked(query)
            }
        } else {
            self.points.get_mut(query)
        }
    }

    fn split(&mut self) {
        let split = self.calc_split();

        let (cell_left, cell_right) = self.cell.split(&split);

        let (mut points_left, mut points_right) = (HashMap::new(), HashMap::new());
        for (point, values) in self.points.drain() {
            if split.belongs_to_left(&point) {
                points_left.insert(point, values);
            } else {
                points_right.insert(point, values);
            }
        }

        self.children = Some(Children {
            split,
            left: Box::new(KdTreeMap::with_points_cell(
                self.dim,
                self.bucket_size,
                points_left,
                cell_left,
            )),
            right: Box::new(KdTreeMap::with_points_cell(
                self.dim,
                self.bucket_size,
                points_right,
                cell_right,
            )),
        });
    }

    fn calc_split(&self) -> Split<Axis> {
        let (dim, _, thresh) = (0..self.dim)
            .map(|dim| {
                let (width, median) = self.bounding_width_median(dim);
                (dim, width, median)
            })
            .max_by_key(|(_, width, _)| *width)
            .expect("unexpectedly zero dimension in bucket::KdTreeMap::calc_split()");

        Split { dim, thresh }
    }

    fn bounding_width_median(&self, dim: usize) -> (Axis, Axis) {
        let points = self.points.keys().map(|p| p.as_ref()[dim]);

        let min = points
            .clone()
            .min()
            .expect("unexpectedly empty points in bucket::KdTreeMap:bounding_width_median()");
        let max = points.max().unwrap();

        let width = max - min;
        let median = min + (max - min) / Axis::from(2).unwrap();
        (width, median)
    }

    fn with_points_cell(
        dim: usize,
        bucket_size: usize,
        points: HashMap<Point, Vec<Value>>,
        cell: Cell<Axis>,
    ) -> Self {
        Self {
            dim,
            bucket_size,
            points,
            children: None,
            cell,
        }
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
    /// use kd_tree::bucket::KdTreeMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1: [R64; 2] = [r64(1.0); 2];
    /// let p2: [R64; 2] = [r64(2.0); 2];
    ///
    /// kdtree.append(p1, 1.0).unwrap();
    /// kdtree.append(p1, 2.0).unwrap();
    ///
    /// kdtree.append(p2, 3.0).unwrap();
    ///
    /// assert_eq!(
    ///     kdtree.points_tree(),
    ///     [(p1, vec![1.0, 2.0]), (p2, vec![3.0])].iter().cloned().collect()
    /// );
    /// ```
    pub fn points_tree(&self) -> HashMap<Point, Vec<Value>> {
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
    use super::*;
    use noisy_float::prelude::*;

    #[test]
    #[should_panic]
    fn test_new_panic_zero_dim() {
        let _: KdTreeMap<R64, [R64; 0], f64> = KdTreeMap::new(0, 1);
    }

    #[test]
    #[should_panic]
    fn test_new_panic_zero_bucket_size() {
        let _: KdTreeMap<R64, [R64; 2], f64> = KdTreeMap::new(2, 0);
    }

    #[test]
    fn test_append_err_dim() {
        let mut kdtree = KdTreeMap::new(2, 1);
        assert!(kdtree.append([r64(1.0); 1], 0.0).is_err());
    }

    #[test]
    fn test_insert_err_dim() {
        let mut kdtree = KdTreeMap::new(2, 1);
        assert!(kdtree.insert([r64(1.0); 1], 0.0).is_err());
    }
}
