use std::hash::Hash;

use num_traits::Float;

use crate::{bucket::kdtree_map::KdTreeMap, Result};

#[derive(Debug, Clone)]
pub struct KdTreeSet<Axis, Point>
where
    Axis: Ord + Float,
    Point: AsRef<[Axis]> + PartialEq + Eq + Hash,
{
    map: KdTreeMap<Axis, Point, ()>,
}

impl<Axis, Point> KdTreeSet<Axis, Point>
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
        Self {
            map: KdTreeMap::new(dim, bucket_size),
        }
    }

    /// Returns the number of dimensions of this kd-tree.
    pub fn dim(&self) -> usize {
        self.map.dim()
    }

    /// Returns the bucket size of this kd-tree.
    pub fn bucket_size(&self) -> usize {
        self.map.bucket_size()
    }

    /// Returns the number of points this kd-tree holds.
    /// Multiple same points are counted as many.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate kdtree;
    /// # extern crate noisy_float;
    /// use kdtree::bucket::KdTreeSet;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeSet::new(2, 1);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1: [R64; 2] = [r64(1.0); 2];
    ///
    /// kdtree.append(p1).unwrap();
    /// assert_eq!(kdtree.size(), 1);
    ///
    /// kdtree.append(p1).unwrap();
    /// assert_eq!(kdtree.size(), 2);
    /// ```
    pub fn size(&self) -> usize {
        self.map.size()
    }

    /// Returns the number of points this kd-tree holds.
    /// Multiple same point are counted as a single point.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate kdtree;
    /// # extern crate noisy_float;
    /// use kdtree::bucket::KdTreeSet;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeSet::new(2, 1);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1: [R64; 2] = [r64(1.0); 2];
    ///
    /// kdtree.append(p1).unwrap();
    /// assert_eq!(kdtree.size_unique(), 1);
    ///
    /// kdtree.append(p1).unwrap();
    /// assert_eq!(kdtree.size_unique(), 1);
    /// ```
    pub fn size_unique(&self) -> usize {
        self.map.size_unique()
    }

    /// Appends a point to this kd-tree.
    /// If the same point already exists in this kd-tree, the point is appended to the existing
    /// points.
    ///
    /// Returns `Err` when the number of dimension of the point does not match with that of this
    /// tree, or when the location of the point is not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate kdtree;
    /// # extern crate noisy_float;
    /// use kdtree::bucket::KdTreeSet;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeSet::new(2, 1);
    ///
    /// kdtree.append([r64(1.0); 2]).unwrap();
    /// ```
    pub fn append(&mut self, point: Point) -> Result<()> {
        self.map.append(point, ())
    }

    /// Returns the nearest point from the query point, and the number of the found point.
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
    /// use kdtree::bucket::KdTreeSet;
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
    /// let mut kdtree = KdTreeSet::new(2, 1);
    ///
    /// let p1: [R64; 2] = [r64(1.0); 2];
    /// let p2: [R64; 2] = [r64(2.0); 2];
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&p1, &squared_euclidean).unwrap(),
    ///     None
    /// );
    ///
    /// kdtree.append(p1).unwrap();
    /// kdtree.append(p2).unwrap();
    /// kdtree.append(p2).unwrap();
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&p1, &squared_euclidean).unwrap(),
    ///     Some((&p1, 1))
    /// );
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&[r64(3.0), r64(3.0)], &squared_euclidean).unwrap(),
    ///     Some((&p2, 2))
    /// );
    /// ```
    pub fn nearest(
        &self,
        query: &Point,
        dist_func: &Fn(&[Axis], &[Axis]) -> Axis,
    ) -> Result<Option<(&Point, usize)>> {
        self.map
            .nearest(query, dist_func)
            .map(|pv| pv.map(|(point, values)| (point, values.len())))
    }
}

impl<Axis, Point> KdTreeSet<Axis, Point>
where
    Axis: Ord + Float,
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
    /// use kdtree::bucket::KdTreeSet;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeSet::new(2, 1);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1: [R64; 2] = [r64(1.0); 2];
    /// let p2: [R64; 2] = [r64(2.0); 2];
    ///
    /// kdtree.append(p1).unwrap();
    /// kdtree.append(p1).unwrap();
    ///
    /// kdtree.append(p2).unwrap();
    ///
    /// assert!(
    ///     kdtree.points_tree() == vec![(p1, 2), (p2, 1)]
    ///     || kdtree.points_tree() == vec![(p2, 1), (p1, 2)]
    /// );
    /// ```
    pub fn points_tree(&self) -> Vec<(Point, usize)> {
        self.map
            .points_tree()
            .into_iter()
            .map(|(point, values)| (point, values.len()))
            .collect()
    }
}