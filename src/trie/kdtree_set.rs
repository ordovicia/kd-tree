use num_traits::Float;

use crate::{point_dist::PointDist, trie::kdtree_map::KdTreeMap, Result};

#[derive(Debug, Clone)]
pub struct KdTreeSet<Axis, Point>
where
    Axis: Float,
    Point: AsRef<[Axis]>,
{
    map: KdTreeMap<Axis, Point, ()>,
}

impl<Axis, Point> KdTreeSet<Axis, Point>
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
        Self {
            map: KdTreeMap::new(dim),
        }
    }

    /// Returns the number of dimensions of this kd-tree.
    pub fn dim(&self) -> usize {
        self.map.dim()
    }

    /// Returns the number of points this kd-tree holds.
    /// Multiple same points are counted as many.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate kd_tree;
    /// use kd_tree::trie::KdTreeSet;
    ///
    /// let mut kdtree = KdTreeSet::new(2);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1 = [1.0; 2];
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
    /// # extern crate kd_tree;
    /// use kd_tree::trie::KdTreeSet;
    ///
    /// let mut kdtree = KdTreeSet::new(2);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1 = [1.0; 2];
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
    /// # extern crate kd_tree;
    /// use kd_tree::trie::KdTreeSet;
    ///
    /// let mut kdtree = KdTreeSet::new(2);
    ///
    /// kdtree.append([1.0; 2]).unwrap();
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
    /// # extern crate kd_tree;
    /// # extern crate num_traits;
    /// use kd_tree::{trie::KdTreeSet, PointDist};
    ///
    /// let squared_euclidean = |p1: &[f64], p2: &[f64]| -> f64 {
    ///     p1.iter()
    ///         .zip(p2.iter())
    ///         .map(|(&p1, &p2)| (p1 - p2) * (p1 - p2))
    ///         .sum()
    /// };
    ///
    /// let mut kdtree = KdTreeSet::new(2);
    ///
    /// let p1 = [1.0; 2];
    /// let p2 = [2.0; 2];
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
    ///     Some(PointDist { point: &p1, value: 1, dist: 0.0 })
    /// );
    ///
    /// assert_eq!(
    ///     kdtree.nearest(&[3.0; 2], &squared_euclidean).unwrap(),
    ///     Some(PointDist { point: &p2, value: 2, dist: 2.0 })
    /// );
    /// ```
    pub fn nearest(
        &self,
        query: &Point,
        dist_func: &Fn(&[Axis], &[Axis]) -> Axis,
    ) -> Result<Option<PointDist<Axis, &Point, usize>>> {
        let pd = self.map.nearest(query, dist_func)?;
        let pd = pd.map(|PointDist { point, value, dist }| PointDist {
            point,
            value: value.len(),
            dist,
        });
        Ok(pd)
    }

    /// Returns the number of the query point in this kd-tree.
    ///
    /// Returns `Err` when the number of dimension of the point does not match with that of this
    /// tree, or when the location of the point is not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate kd_tree;
    /// use kd_tree::trie::KdTreeSet;
    ///
    /// let mut kdtree = KdTreeSet::new(2);
    /// let p1 = [1.0; 2];
    ///
    /// kdtree.append(p1).unwrap();
    /// kdtree.append(p1).unwrap();
    /// assert_eq!(kdtree.get_count(&p1).unwrap(), 2);
    ///
    /// assert_eq!(kdtree.get_count(&[2.0; 2]).unwrap(), 0);
    /// ```
    pub fn get_count(&self, query: &Point) -> Result<usize> {
        let values = self.map.get(query)?;
        let count = values.map(|values| values.len()).unwrap_or(0);
        Ok(count)
    }
}

impl<Axis, Point> KdTreeSet<Axis, Point>
where
    Axis: Float,
    Point: AsRef<[Axis]> + Clone,
{
    /// Returns the all points this kd-tree holds.
    /// Points that descendant nodes have are all included.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate kd_tree;
    /// use kd_tree::trie::KdTreeSet;
    ///
    /// let mut kdtree = KdTreeSet::new(2);
    /// assert_eq!(kdtree.size(), 0);
    ///
    /// let p1 = [1.0; 2];
    /// let p2 = [2.0; 2];
    ///
    /// kdtree.append(p1).unwrap();
    /// kdtree.append(p1).unwrap();
    ///
    /// kdtree.append(p2).unwrap();
    ///
    /// assert_eq!(
    ///     kdtree.points_tree(),
    ///     [(p1, 2), (p2, 1)].iter().cloned().collect::<Vec<_>>()
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
