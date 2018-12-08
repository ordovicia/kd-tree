use num_traits::Float;
#[cfg(feature = "serialize")]
use serde_derive::{Deserialize, Serialize};

use crate::{cell::Cell, error::ErrorKind, point_dist::PointDist, split::Split, Result};

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct KdTreeMap<Axis, Point, Value>
where
    Axis: Float,
    Point: AsRef<[Axis]>,
{
    dim: usize,
    depth: usize,

    point: Option<(Point, Vec<Value>)>,
    children: Option<Children<Axis, KdTreeMap<Axis, Point, Value>>>,

    cell: Cell<Axis>,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
struct Children<Axis: Clone, KdTree> {
    split: Split<Axis>,
    left: Box<KdTree>,
    right: Box<KdTree>,
}

impl<Axis, Point, Value> KdTreeMap<Axis, Point, Value>
where
    Axis: Float,
    Point: PartialEq + AsRef<[Axis]>,
{
    /// Creates a kd-tree with `dim` dimensions.
    ///
    /// # Panic
    ///
    /// Panics if `dim` is not positive.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "the number of dimensions must be positive");

        Self {
            dim,
            depth: 0,
            point: None,
            children: None,
            cell: Cell::new(dim),
        }
    }

    /// Returns the number of dimensions of this kd-tree.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of points this kd-tree holds.
    /// Multiple values on the same point are counted as many.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kd_tree::trie::KdTreeMap;
    ///
    /// let mut kdtree = KdTreeMap::new(2);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1 = [1.0; 2];
    ///
    /// kdtree.append(p1, 1.0).unwrap();
    /// assert_eq!(kdtree.size(), 1);
    ///
    /// kdtree.append(p1, 2.0).unwrap();
    /// assert_eq!(kdtree.size(), 2);
    /// ```
    pub fn size(&self) -> usize {
        let mut size = self
            .point
            .as_ref()
            .map(|(_, values)| values.len())
            .unwrap_or(0);

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
    /// use kd_tree::trie::KdTreeMap;
    ///
    /// let mut kdtree = KdTreeMap::new(2);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1 = [1.0; 2];
    ///
    /// kdtree.append(p1, 1.0).unwrap();
    /// assert_eq!(kdtree.size_unique(), 1);
    ///
    /// kdtree.append(p1, 2.0).unwrap();
    /// assert_eq!(kdtree.size_unique(), 1);
    /// ```
    pub fn size_unique(&self) -> usize {
        let mut size = if self.point.is_some() { 1 } else { 0 };

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
    /// use kd_tree::trie::KdTreeMap;
    ///
    /// let mut kdtree = KdTreeMap::new(2);
    ///
    /// let p1 = [1.0; 2];
    /// kdtree.append(p1, 1.0).unwrap();
    /// kdtree.append(p1, 2.0).unwrap();
    ///
    /// assert_eq!(kdtree.get(&p1).unwrap(), Some(&vec![1.0, 2.0]));
    /// ```
    pub fn append(&mut self, point: Point, value: Value) -> Result<()> {
        self.check_point(point.as_ref())?;

        if self.point.is_none() {
            self.point = Some((point, vec![value]));
        } else if self.point.as_ref().unwrap().0 == point {
            self.point.as_mut().unwrap().1.push(value);
        } else {
            if self.children.is_none() {
                self.split();
            }

            let Children { split, left, right } = self.children.as_mut().unwrap();
            if split.belongs_to_left(&point) {
                left.as_mut().append(point, value)?;
            } else {
                right.as_mut().append(point, value)?;
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
    /// use kd_tree::trie::KdTreeMap;
    ///
    /// let mut kdtree = KdTreeMap::new(2);
    ///
    /// let p1 = [1.0; 2];
    /// kdtree.insert(p1, 1.0).unwrap();
    /// kdtree.insert(p1, 2.0).unwrap();
    ///
    /// assert_eq!(kdtree.get(&p1).unwrap(), Some(&vec![2.0]));
    /// ```
    pub fn insert(&mut self, point: Point, value: Value) -> Result<()> {
        self.check_point(point.as_ref())?;

        if self.point.is_none() || self.point.as_ref().unwrap().0 == point {
            self.point = Some((point, vec![value]));
        } else {
            if self.children.is_none() {
                self.split();
            }

            let Children { split, left, right } = self.children.as_mut().unwrap();
            if split.belongs_to_left(&point) {
                left.as_mut().append(point, value)?;
            } else {
                right.as_mut().append(point, value)?;
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
    /// use kd_tree::{trie::KdTreeMap, PointDist};
    ///
    /// let squared_euclidean = |p1: &[f64], p2: &[f64]| -> f64 {
    ///     p1.iter()
    ///         .zip(p2.iter())
    ///         .map(|(&p1, &p2)| (p1 - p2) * (p1 - p2))
    ///         .sum()
    /// };
    ///
    /// let mut kdtree = KdTreeMap::new(2);
    ///
    /// let p1 = [1.0; 2];
    /// let p2 = [2.0; 2];
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
    ///     Some(PointDist { point: &p1, value: &vec![1.0], dist: 0.0 })
    /// );
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&[3.0; 2], &squared_euclidean).unwrap(),
    ///     Some(PointDist { point: &p2, value: &vec![2.0], dist: 2.0 })
    /// );
    /// ```
    pub fn nearest(
        &self,
        query: &Point,
        dist_func: &Fn(&[Axis], &[Axis]) -> Axis,
    ) -> Result<Option<PointDist<Axis, &Point, &Vec<Value>>>> {
        self.check_point(query.as_ref())?;
        Ok(self.nearest_unchecked(query, dist_func))
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// Returns `Err` when the number of dimension of the point does not match with that of this
    /// tree, or when the location of the point is not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kd_tree::trie::KdTreeMap;
    ///
    /// let mut kdtree = KdTreeMap::new(2);
    /// let (p1, val)  = ([1.0; 2], 1.0);
    ///
    /// kdtree.insert(p1, val).unwrap();
    /// assert_eq!(kdtree.get(&p1).unwrap(), Some(&vec![val]));
    ///
    /// assert_eq!(kdtree.get(&[2.0; 2]).unwrap(), None);
    /// ```
    pub fn get(&self, query: &Point) -> Result<Option<&Vec<Value>>> {
        self.check_point(query.as_ref())?;

        match &self.point {
            None => Ok(None),
            Some((point, values)) if point == query => Ok(Some(values)),
            Some(_) => {
                if let Some(Children {
                    ref split,
                    ref left,
                    ref right,
                }) = self.children
                {
                    if split.belongs_to_left(query) {
                        left.get(query)
                    } else {
                        right.get(query)
                    }
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// Returns `Err` when the number of dimension of the point does not match with that of this
    /// tree, or when the location of the point is not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kd_tree::trie::KdTreeMap;
    ///
    /// let mut kdtree = KdTreeMap::new(2);
    /// let (p1, val)  = ([1.0; 2], 1.0);
    ///
    /// kdtree.insert(p1, val).unwrap();
    ///
    /// *kdtree.get_mut(&p1).unwrap().unwrap() = vec![2.0];
    /// assert_eq!(kdtree.get(&p1).unwrap(), Some(&vec![2.0]));
    ///
    /// assert_eq!(kdtree.get_mut(&[2.0; 2]).unwrap(), None);
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

    fn nearest_unchecked(
        &self,
        query: &Point,
        dist_func: &Fn(&[Axis], &[Axis]) -> Axis,
    ) -> Option<PointDist<Axis, &Point, &Vec<Value>>> {
        if self.size() == 0 {
            return None;
        }

        let mut point_nearest = None;
        let mut leaf = self;
        let mut other_side = vec![];

        while let Some((point, value)) = &leaf.point {
            let pd = PointDist {
                point,
                value,
                dist: dist_func(query.as_ref(), point.as_ref()),
            };
            if point_nearest.as_ref().map(|pn| &pd < pn).unwrap_or(true) {
                point_nearest = Some(pd);
            }

            if let Some(Children { split, left, right }) = &leaf.children {
                if split.belongs_to_left(query) {
                    leaf = left;
                    other_side.push(right);
                } else {
                    leaf = right;
                    other_side.push(left);
                }
            } else {
                break;
            }
        }

        while let Some(other_side) = other_side.pop() {
            if point_nearest
                .as_ref()
                .map(|pn| other_side.cell.dist_to_point(query.as_ref(), dist_func) > pn.dist)
                .unwrap_or(false)
            {
                continue;
            }

            if let Some(point_other_side) = other_side.nearest_unchecked(query, dist_func) {
                if point_nearest
                    .as_ref()
                    .map(|pn| &point_other_side < pn)
                    .unwrap_or(true)
                {
                    point_nearest = Some(point_other_side);
                }
            }
        }

        point_nearest
    }

    fn get_mut_unchecked(&mut self, query: &Point) -> Option<&mut Vec<Value>> {
        match &mut self.point {
            None => None,
            Some((point, values)) if point == query => Some(values),
            Some(_) => {
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
                    None
                }
            }
        }
    }

    fn split(&mut self) {
        let split = self.calc_split();
        let (cell_left, cell_right) = self.cell.split(&split);

        self.children = Some(Children {
            split,
            left: Box::new(KdTreeMap::with_depth_cell(
                self.dim,
                self.depth + 1,
                cell_left,
            )),
            right: Box::new(KdTreeMap::with_depth_cell(
                self.dim,
                self.depth + 1,
                cell_right,
            )),
        });
    }

    fn calc_split(&self) -> Split<Axis> {
        let dim = self.depth % self.dim;
        let thresh = if let Some((point, _)) = &self.point {
            point.as_ref()[dim]
        } else {
            panic!("unexpectedly no point in KdTreeMap::calc_split")
        };

        Split { dim, thresh }
    }

    fn with_depth_cell(dim: usize, depth: usize, cell: Cell<Axis>) -> Self {
        Self {
            dim,
            depth,
            point: None,
            children: None,
            cell,
        }
    }
}

impl<Axis, Point, Value> KdTreeMap<Axis, Point, Value>
where
    Axis: Float,
    Value: Clone,
    Point: AsRef<[Axis]> + Clone,
{
    /// Returns the all points this kd-tree holds.
    /// Points that descendant nodes have are all included.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kd_tree::trie::KdTreeMap;
    ///
    /// let mut kdtree = KdTreeMap::new(2);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1 = [1.0; 2];
    /// let p2 = [2.0; 2];
    ///
    /// kdtree.append(p1, 1.0).unwrap();
    /// kdtree.append(p1, 2.0).unwrap();
    ///
    /// kdtree.append(p2, 3.0).unwrap();
    ///
    /// assert_eq!(
    ///     kdtree.points_tree(),
    ///     [(p1, vec![1.0, 2.0]), (p2, vec![3.0])].iter().cloned().collect::<Vec<_>>()
    /// );
    /// ```
    pub fn points_tree(&self) -> Vec<(Point, Vec<Value>)> {
        let mut points = vec![];
        if let Some((point, values)) = &self.point {
            points.push((point.clone(), values.clone()));
        };

        if let Some(Children {
            ref left,
            ref right,
            ..
        }) = self.children
        {
            points.append(&mut left.points_tree());
            points.append(&mut right.points_tree());
        }

        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_new_panic_zero_dim() {
        let _: KdTreeMap<f64, [f64; 0], f64> = KdTreeMap::new(0);
    }

    #[test]
    fn test_append_err_dim() {
        let mut kdtree = KdTreeMap::new(2);
        assert!(kdtree.append([1.0; 1], 0.0).is_err());
    }

    #[test]
    fn test_insert_err_dim() {
        let mut kdtree = KdTreeMap::new(2);
        assert!(kdtree.insert([1.0; 1], 0.0).is_err());
    }

    #[test]
    fn test_nearest_3d() {
        let squared_euclidean = |p1: &[f64], p2: &[f64]| -> f64 {
            p1.iter()
                .zip(p2.iter())
                .map(|(&p1, &p2)| (p1 - p2) * (p1 - p2))
                .sum()
        };

        let mut kdtree = KdTreeMap::new(3);

        let p1 = [1.0; 3];
        let p2 = [2.0; 3];

        kdtree.append(p1, 1.0).unwrap();
        kdtree.append(p1, 2.0).unwrap();

        kdtree.append(p2, 3.0).unwrap();

        assert_eq!(
            kdtree.nearest(&[1.2; 3], &squared_euclidean).unwrap(),
            Some(PointDist {
                point: &p1,
                value: &vec![1.0, 2.0],
                dist: 0.08,
            })
        );
    }
}
