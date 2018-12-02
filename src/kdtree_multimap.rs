use std::{collections::HashMap, hash::Hash};

use num_traits::Float;

use crate::{kdtree_map::*, Result};

#[derive(Debug, Clone)]
pub struct KdTreeMultiMap<Axis, Point, Value>
where
    Axis: Float,
    Point: AsRef<[Axis]> + PartialEq + Eq + Hash,
{
    map: KdTreeMap<Axis, Point, Vec<Value>>,
}

impl<Axis, Point, Value> KdTreeMultiMap<Axis, Point, Value>
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
        Self {
            map: KdTreeMap::new(dim, node_capacity),
        }
    }

    /// Returns the number of dimensions of this kd-tree.
    pub fn dim(&self) -> usize {
        self.map.dim()
    }

    /// Returns the node capacity of this kd-tree.
    pub fn node_capacity(&self) -> usize {
        self.map.node_capacity()
    }

    /// Returns the number of points this kd-tree holds.
    /// Multiple values on the same point are counted as a single point.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate kdtree;
    /// # extern crate noisy_float;
    /// use kdtree::KdTreeMultiMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMultiMap::new(2, 1);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1: [R64; 2] = [r64(1.0); 2];
    /// let p2: [R64; 2] = [r64(2.0); 2];
    ///
    /// kdtree.append(p1, 1.0);
    /// assert_eq!(kdtree.size(), 1);
    ///
    /// kdtree.append(p1, 2.0);
    /// assert_eq!(kdtree.size(), 1);
    ///
    /// kdtree.append(p2, 3.0);
    /// assert_eq!(kdtree.size(), 2);
    /// ```
    pub fn size(&self) -> usize {
        self.map.size()
    }

    /// Appends a point to this kd-tree with a value.
    /// If the same point already exists in this kd-tree, the value is appended to the point, not
    /// overwriting the existing values.
    ///
    /// Returns `Err` when the number of dimension of the point does not match with that of this
    /// tree, or when the location of the point is not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate kdtree;
    /// # extern crate noisy_float;
    /// use kdtree::KdTreeMultiMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMultiMap::new(2, 1);
    ///
    /// let (p1, val): ([R64; 2], f64) = ([r64(1.0); 2], 1.0);
    /// kdtree.append(p1, val);
    /// ```
    ///
    /// ```rust
    /// # extern crate kdtree;
    /// # extern crate noisy_float;
    /// use kdtree::KdTreeMultiMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMultiMap::new(2, 1);
    /// // The numbers of dimensions do not match
    /// assert!(kdtree.append([r64(1.0); 1], 0.0).is_err());
    /// ```
    pub fn append(&mut self, point: Point, value: Value) -> Result<()> {
        // #![allow(clippy::map_entry)]

        self.map.check_point(point.as_ref())?;

        if self.map.points.contains_key(&point) {
            self.map.points.get_mut(&point).unwrap().push(value);
        } else if let Some(Children {
            ref split,
            ref mut left,
            ref mut right,
        }) = self.map.children
        {
            let Split { dim, thresh } = *split;
            if point.as_ref()[dim] < thresh {
                left.as_mut().insert(point, value)?;
            } else {
                right.as_mut().insert(point, value)?;
            }
        } else {
            self.map.points.insert(point, vec![value]);
            if self.map.points.len() > self.node_capacity() {
                self.map.split();
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
    /// use kdtree::KdTreeMultiMap;
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
    /// let mut kdtree = KdTreeMultiMap::new(2, 1);
    ///
    /// let p1: [R64; 2] = [r64(1.0); 2];
    /// let p2: [R64; 2] = [r64(2.0); 2];
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&p1, &squared_euclidean).unwrap(),
    ///     None
    /// );
    ///
    /// kdtree.append(p1, 1.0);
    /// kdtree.append(p2, 2.0);
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&p1, &squared_euclidean).unwrap(),
    ///     Some((&p1, &vec![1.0]))
    /// );
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&[r64(3.0), r64(3.0)], &squared_euclidean).unwrap(),
    ///     Some((&p2, &vec![2.0]))
    /// );
    /// ```
    ///
    /// ```rust
    /// # extern crate kdtree;
    /// # extern crate noisy_float;
    /// # extern crate num_traits;
    /// use kdtree::KdTreeMultiMap;
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
    /// let mut kdtree = KdTreeMultiMap::new(3, 2);
    ///
    /// let p1: [R64; 3] = [r64(1.0); 3];
    /// let p2: [R64; 3] = [r64(2.0); 3];
    ///
    /// kdtree.append(p1, 1.0);
    /// kdtree.append(p1, 2.0);
    ///
    /// kdtree.append(p2, 3.0);
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&[r64(1.2); 3], &squared_euclidean).unwrap(),
    ///     Some((&p1, &vec![1.0, 2.0]))
    /// );
    /// ```
    pub fn nearest(
        &self,
        query: &Point,
        dist_func: &Fn(&[Axis], &[Axis]) -> Axis,
    ) -> Result<Option<(&Point, &Vec<Value>)>> {
        self.map.nearest(query, dist_func)
    }
}

impl<Axis, Point, Value> KdTreeMultiMap<Axis, Point, Value>
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
    /// use kdtree::KdTreeMultiMap;
    /// use noisy_float::prelude::*;
    ///
    /// let mut kdtree = KdTreeMultiMap::new(2, 1);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1: [R64; 2] = [r64(1.0); 2];
    /// let p2: [R64; 2] = [r64(2.0); 2];
    ///
    /// kdtree.append(p1, 1.0);
    /// kdtree.append(p1, 2.0);
    ///
    /// kdtree.append(p2, 3.0);
    ///
    /// assert_eq!(
    ///     kdtree.points_tree(),
    ///     [(p1, vec![1.0, 2.0]), (p2, vec![3.0])].iter().cloned().collect()
    /// );
    /// ```
    pub fn points_tree(&self) -> HashMap<Point, Vec<Value>> {
        let mut points = self.map.points.clone();
        if let Some(Children {
            ref left,
            ref right,
            ..
        }) = self.map.children
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
        let _: KdTreeMultiMap<R64, [R64; 0], f64> = KdTreeMultiMap::new(0, 1);
    }

    #[test]
    #[should_panic]
    fn panic_new_zero_capacity() {
        let _: KdTreeMultiMap<R64, [R64; 2], f64> = KdTreeMultiMap::new(2, 0);
    }
}
