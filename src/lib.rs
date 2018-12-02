// TODO: multi-set, single-map, single-set

//! ```rust
//! # extern crate kdtree;
//! # extern crate noisy_float;
//! # extern crate num_traits;
//! use kdtree::KdTreeMultiMap;
//! use noisy_float::prelude::*;
//! use num_traits::{Float, Zero};
//!
//! let squared_euclidean = |p1: &[R64], p2: &[R64]| -> R64 {
//!     p1.iter()
//!         .zip(p2.iter())
//!         .map(|(&p1, &p2)| (p1 - p2) * (p1 - p2))
//!         .fold(R64::zero(), std::ops::Add::add)
//! };
//!
//! let mut kdtree = KdTreeMultiMap::new(2, 4);
//!
//! let p1: [R64; 2] = [r64(1.0), r64(2.0)];
//! let p2: [R64; 2] = [r64(3.0), r64(1.0)];
//!
//! kdtree.insert(p1, 1.0);
//! kdtree.insert(p1, 2.0);
//!
//! kdtree.insert(p2, 3.0);
//!
//! assert_eq!(kdtree.size(), 2);
//!
//! assert_eq!(
//!     kdtree.nearest(&[r64(2.0); 2], &squared_euclidean).unwrap(),
//!     Some((&p1, &vec![1.0, 2.0]))
//! );
//!
//! assert_eq!(
//!     kdtree.points_tree(),
//!     [(p1, vec![1.0, 2.0]), (p2, vec![3.0])].iter().cloned().collect()
//! );
//! ```

extern crate failure;
extern crate num_traits;

mod dist_ordered_point;
mod error;
mod kdtree_map;
mod kdtree_multimap;

pub use error::{Error, ErrorKind};
pub use kdtree_map::KdTreeMap;
pub use kdtree_multimap::KdTreeMultiMap;

pub type Result<T> = std::result::Result<T, Error>;
