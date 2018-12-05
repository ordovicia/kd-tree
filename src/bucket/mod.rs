//! KD-Tree implementation with buckets.
//!
//! Points are collected in leaf nodes until they overflow from the bucket.
//! When overflowing, the leaf is divided by the median value of the widest dimension of the
//! bounding region the point group.
//! The points are moved to the new child nodes according to their location, so only the leaf nodes
//! have points.

mod kdtree_map;
mod kdtree_set;

pub use self::kdtree_map::KdTreeMap;
pub use self::kdtree_set::KdTreeSet;
