# KD-Tree

Various KD-Tree implementations with Rust.

## Examples

### Bucket implementation

```rust
# extern crate kd_tree;
# extern crate noisy_float;
# extern crate num_traits;
use kd_tree::{bucket::{KdTreeMap, KdTreeSet}, PointDist};
use noisy_float::prelude::*;
use num_traits::{Float, Zero};

let squared_euclidean = |p1: &[R64], p2: &[R64]| -> R64 {
    p1.iter()
        .zip(p2.iter())
        .map(|(&p1, &p2)| (p1 - p2) * (p1 - p2))
        .fold(R64::zero(), std::ops::Add::add)
};

// Map
let mut kdtree = KdTreeMap::new(2, 4);

let p1: [R64; 2] = [r64(1.0), r64(2.0)];
let p2: [R64; 2] = [r64(4.0), r64(1.0)];

kdtree.append(p1, 1.0).unwrap();
kdtree.append(p1, 2.0).unwrap(); // append a value to the existing point
kdtree.append(p2, 3.0).unwrap();

assert_eq!(kdtree.size(), 3);
assert_eq!(kdtree.size_unique(), 2);

assert_eq!(
    kdtree.nearest(&[r64(2.0); 2], &squared_euclidean).unwrap(),
    Some(PointDist { point: &p1, value: &vec![1.0, 2.0], dist: r64(2.0) })
);

kdtree.insert(p1, 4.0).unwrap(); // overwrite existing values
assert_eq!(
    kdtree.nearest(&[r64(2.0); 2], &squared_euclidean).unwrap(),
    Some(PointDist { point: &p1, value: &vec![4.0], dist: r64(2.0) })
);

// Set
let mut kdtree = KdTreeSet::new(2, 4);

kdtree.append(p1).unwrap();
kdtree.append(p1).unwrap(); // append a point to the same location
kdtree.append(p2).unwrap();

assert_eq!(kdtree.size(), 3);
assert_eq!(kdtree.size_unique(), 2);

assert_eq!(
    kdtree.nearest(&[r64(2.0); 2], &squared_euclidean).unwrap(),
    Some(PointDist { point: &p1, value: /* count */ 2, dist: r64(2.0) })
);
```

### Trie implementation

```rust
# extern crate kd_tree;
use kd_tree::{trie::{KdTreeMap, KdTreeSet}, PointDist};

let squared_euclidean = |p1: &[f64], p2: &[f64]| -> f64 {
    p1.iter()
        .zip(p2.iter())
        .map(|(&p1, &p2)| (p1 - p2) * (p1 - p2))
        .sum()
};

// Map
let mut kdtree = KdTreeMap::new(2);

let p1 = [1.0, 2.0];
let p2 = [4.0, 1.0];

kdtree.append(p1, 1.0).unwrap();
kdtree.append(p1, 2.0).unwrap(); // append a value to the existing point
kdtree.append(p2, 3.0).unwrap();

assert_eq!(kdtree.size(), 3);
assert_eq!(kdtree.size_unique(), 2);

assert_eq!(
    kdtree.nearest(&[2.0; 2], &squared_euclidean).unwrap(),
    Some(PointDist { point: &p1, value: &vec![1.0, 2.0], dist: 2.0 })
);

kdtree.insert(p1, 4.0).unwrap(); // overwrite existing values
assert_eq!(
    kdtree.nearest(&[2.0; 2], &squared_euclidean).unwrap(),
    Some(PointDist { point: &p1, value: &vec![4.0], dist: 2.0 })
);

// Set
let mut kdtree = KdTreeSet::new(2);

kdtree.append(p1).unwrap();
kdtree.append(p1).unwrap(); // append a point to the same location
kdtree.append(p2).unwrap();

assert_eq!(kdtree.size(), 3);
assert_eq!(kdtree.size_unique(), 2);

assert_eq!(
    kdtree.nearest(&[2.0; 2], &squared_euclidean).unwrap(),
    Some(PointDist { point: &p1, value: /* count */ 2, dist: 2.0 })
);
```

## Related Project

The implementation is inspired by [mrhooray/kdtree](https://github.com/mrhooray/kdtree-rs).
