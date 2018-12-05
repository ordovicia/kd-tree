extern crate kd_tree;
extern crate rand;
extern crate serde_json;

use rand::{distributions::uniform::SampleUniform, Rng};

use kd_tree::trie;

const POINTS_NUM: usize = 8;

const DIM: usize = 3;
const VALUE: i32 = 42;

fn gen_rand<T: SampleUniform>(begin: T, end: T) -> T {
    rand::thread_rng().gen_range(begin, end)
}

#[test]
fn test_trie_map_serde() {
    let mut kdtree = trie::KdTreeMap::new(DIM);

    let points = (0..POINTS_NUM)
        .map(|_| {
            [
                f64::from(gen_rand(-9, 10)), // use i32 for precision
                f64::from(gen_rand(-9, 10)),
                f64::from(gen_rand(-9, 10)),
            ]
        }).collect::<Vec<[f64; DIM]>>();

    for p in points.clone() {
        kdtree.append(p, VALUE).unwrap();
    }

    let serialized = serde_json::to_string(&kdtree).unwrap();
    let deserialized: trie::KdTreeMap<f64, [f64; DIM], i32> =
        serde_json::from_str(&serialized).unwrap();

    assert_eq!(deserialized, kdtree);
}

#[test]
fn test_trie_set_serde() {
    let mut kdtree = trie::KdTreeSet::new(DIM);

    let points = (0..POINTS_NUM)
        .map(|_| {
            [
                f64::from(gen_rand(-9, 10)), // use i32 for precision
                f64::from(gen_rand(-9, 10)),
                f64::from(gen_rand(-9, 10)),
            ]
        }).collect::<Vec<[f64; DIM]>>();

    for p in points.clone() {
        kdtree.append(p).unwrap();
    }

    let serialized = serde_json::to_string(&kdtree).unwrap();
    let deserialized: trie::KdTreeSet<f64, [f64; DIM]> = serde_json::from_str(&serialized).unwrap();

    assert_eq!(deserialized, kdtree);
}
