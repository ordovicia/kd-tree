// TODO
//  * support getting nearest k points
//  * opt out supporting multi values

//! # KD-tree
//!
//! ## Examples
//!
//! ```rust
//! # extern crate kdtree;
//! # extern crate noisy_float;
//! # extern crate num_traits;
//! use kdtree::bucket::{KdTreeMap, KdTreeSet};
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
//! // Map
//! let mut kdtree = KdTreeMap::new(2, 4);
//!
//! let p1: [R64; 2] = [r64(1.0), r64(2.0)];
//! let p2: [R64; 2] = [r64(3.0), r64(1.0)];
//!
//! kdtree.append(p1, 1.0).unwrap();
//! kdtree.append(p1, 2.0).unwrap(); // append a value to the existing point
//! kdtree.append(p2, 3.0).unwrap();
//!
//! assert_eq!(kdtree.size(), 3);
//! assert_eq!(kdtree.size_unique(), 2);
//!
//! assert_eq!(
//!     kdtree.nearest(&[r64(2.0); 2], &squared_euclidean).unwrap(),
//!     Some((&p1, &vec![1.0, 2.0]))
//! );
//!
//! kdtree.insert(p1, 4.0).unwrap(); // overwrite existing values
//! assert_eq!(
//!     kdtree.nearest(&[r64(2.0); 2], &squared_euclidean).unwrap(),
//!     Some((&p1, &vec![4.0]))
//! );
//!
//! // Set
//! let mut kdtree = KdTreeSet::new(2, 4);
//!
//! kdtree.append(p1).unwrap();
//! kdtree.append(p1).unwrap(); // append a point to the same location
//! kdtree.append(p2).unwrap();
//!
//! assert_eq!(kdtree.size(), 3);
//! assert_eq!(kdtree.size_unique(), 2);
//!
//! assert_eq!(
//!     kdtree.nearest(&[r64(2.0); 2], &squared_euclidean).unwrap(),
//!     Some((&p1, 2)) // the found points and their count
//! );
//! ```

extern crate failure;
extern crate num_traits;

pub mod bucket;
mod cell;
mod dist_ordered_point;
mod error;
mod split;

pub use error::{Error, ErrorKind};

pub type Result<T> = std::result::Result<T, Error>;
