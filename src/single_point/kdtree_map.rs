use num_traits::Float;

use crate::{
    cell::Cell, dist_ordered_point::DistOrderedPoint, error::ErrorKind, split::Split, Result,
};

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
struct Children<Axis, KdTree> {
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
    /// # extern crate kdtree;
    /// use kdtree::single_point::single_point::KdTreeMap;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
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
    /// # extern crate kdtree;
    /// use kdtree::single_point::KdTreeMap;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
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
    /// # extern crate kdtree;
    /// use kdtree::single_point::KdTreeMap;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
    ///
    /// let p1 = [1.0; 2];
    /// kdtree.append(p1, 1.0).unwrap();
    /// kdtree.append(p1, 2.0).unwrap();
    ///
    /// assert_eq!(kdtree.get(&p1).unwrap(), Some(&vec![1.0, 2.0]));
    /// ```
    pub fn append(&mut self, point: Point, value: Value) -> Result<()> {
        self.check_point(point.as_ref())?;

        match self.point {
            None => {
                self.point = Some((point, vec![value]));
            }
            Some((ref p, _)) if p == &point => {
                let values = self.point.as_mut().unwrap().1;
                values.push(value);
            }
            Some(_) => {
                if self.children.is_none() {
                    self.split();
                }

                let Children {
                    ref split,
                    ref mut left,
                    ref mut right,
                } = self.children.unwrap();
                if split.belongs_to_left(&point) {
                    left.as_mut().append(point, value)?;
                } else {
                    right.as_mut().append(point, value)?;
                }
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
    /// # extern crate kdtree;
    /// use kdtree::single_point::KdTreeMap;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
    ///
    /// let p1 = [1.0; 2];
    /// kdtree.insert(p1, 1.0).unwrap();
    /// kdtree.insert(p1, 2.0).unwrap();
    ///
    /// assert_eq!(kdtree.get(&p1).unwrap(), Some(&vec![2.0]));
    /// ```
    pub fn insert(&mut self, point: Point, value: Value) -> Result<()> {
        self.check_point(point.as_ref())?;

        match self.point {
            Some((ref p, _)) if p != &point => {
                if self.children.is_none() {
                    self.split();
                }

                let Children {
                    ref split,
                    ref mut left,
                    ref mut right,
                } = self.children.unwrap();
                if split.belongs_to_left(&point) {
                    left.as_mut().insert(point, value)?;
                } else {
                    right.as_mut().insert(point, value)?;
                }
            }
            _ => {
                self.point = Some((point, vec![value]));
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
    /// # extern crate kdtree;
    /// # extern crate num_traits;
    /// use kdtree::single_point::KdTreeMap;
    /// use num_traits::{Float, Zero};
    ///
    /// let squared_euclidean = |p1: f64, p2: &f64| -> f64 {
    ///     p1.iter()
    ///         .zip(p2.iter())
    ///         .map(|(&p1, &p2)| (p1 - p2) * (p1 - p2))
    ///         .fold(R64::zero(), std::ops::Add::add)
    /// };
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
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
    ///     Some((&p1, &vec![1.0]))
    /// );
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&[3.0; 2], &squared_euclidean).unwrap(),
    ///     Some((&p2, &vec![2.0]))
    /// );
    /// ```
    pub fn nearest(
        &self,
        query: &Point,
        dist_func: &Fn(&[Axis], &[Axis]) -> Axis,
    ) -> Result<Option<(&Point, &Vec<Value>)>> {
        self.check_point(query.as_ref())?;

        unimplemented!();
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// Returns `Err` when the number of dimension of the point does not match with that of this
    /// tree, or when the location of the point is not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate kdtree;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
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
    /// # extern crate kdtree;
    /// use kdtree::single_point::KdTreeMap;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
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
        let dim = self.dim % self.depth;
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
    /// # extern crate kdtree;
    /// use kdtree::single_point::KdTreeMap;
    ///
    /// let mut kdtree = KdTreeMap::new(2, 1);
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
    ///     [(p1, vec![1.0, 2.0]), (p2, vec![3.0])].iter().cloned().collect()
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
    fn test_new_panic_new_zero() {
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
            Some((&p1, &vec![1.0, 2.0]))
        );
    }
}
