//! KD-Tree implementation with trie.
//!
//! Each node has up to single point.
//! When adding a point, a leaf is devided by the plane passing through the existing point.
//! The dimension of the plane is determined by clclic order.

mod kdtree_map;
mod kdtree_set;

pub use self::kdtree_map::KdTreeMap;
pub use self::kdtree_set::KdTreeSet;
