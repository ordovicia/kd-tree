use noisy_float::prelude::*;
use num_traits::Float;
use rand::{distributions::uniform::SampleUniform, Rng};

use kd_tree::{bucket, trie};

fn squared_euclidean<T: Float>(p1: &[T], p2: &[T]) -> T {
    p1.iter()
        .zip(p2.iter())
        .map(|(&p1, &p2)| (p1 - p2) * (p1 - p2))
        .fold(T::zero(), std::ops::Add::add)
}

fn nearest_linear<'a, A, P>(query: &P, points: &'a [P], dist_func: &Fn(&[A], &[A]) -> A) -> &'a P
where
    A: Float,
    P: AsRef<[A]>,
{
    assert!(!points.is_empty());

    let query = query.as_ref();

    let mut nearest = &points[0];
    let mut min_dist = dist_func(nearest.as_ref(), query);

    for p in points {
        let dist = dist_func(p.as_ref(), query);
        if dist < min_dist {
            nearest = p;
            min_dist = dist;
        }
    }

    nearest
}

fn gen_rand<T: SampleUniform>(begin: T, end: T) -> T {
    rand::thread_rng().gen_range(begin, end)
}

fn gen_points(dim: usize, num: usize) -> Vec<Vec<f64>> {
    (0..num)
        .map(|_| (0..dim).map(|_| gen_rand(-1.0, 1.0)).collect())
        .collect()
}

fn gen_points_noisy(dim: usize, num: usize) -> Vec<Vec<R64>> {
    (0..num)
        .map(|_| (0..dim).map(|_| r64(gen_rand(-1.0, 1.0))).collect())
        .collect()
}

macro_rules! assert_nearest {
    ($kdtree: expr, $points: expr, $queries: expr) => {
        for q in &$queries {
            let nearest = $kdtree.nearest(q, &squared_euclidean).unwrap().unwrap();
            let linear = nearest_linear(q, &$points, &squared_euclidean);
            assert_eq!(nearest.point, linear);
        }
    };
}

fn assert_nearest_bucket_map(dim: usize, bucket_size: usize, points_num: usize) {
    let mut kdtree = bucket::KdTreeMap::new(dim, bucket_size);
    let points = gen_points_noisy(dim, points_num);

    for p in points.clone() {
        kdtree.append(p, gen_rand(0, 10)).unwrap();
    }

    assert_eq!(kdtree.size(), points_num);
    assert_nearest!(kdtree, points, gen_points_noisy(dim, points_num));
}

fn assert_nearest_bucket_set(dim: usize, bucket_size: usize, points_num: usize) {
    let mut kdtree = bucket::KdTreeSet::new(dim, bucket_size);
    let points = gen_points_noisy(dim, points_num);

    for p in points.clone() {
        kdtree.append(p).unwrap();
    }

    assert_eq!(kdtree.size(), points_num);
    assert_nearest!(kdtree, points, gen_points_noisy(dim, points_num));
}

fn assert_nearest_trie_map(dim: usize, points_num: usize) {
    let mut kdtree = trie::KdTreeMap::new(dim);
    let points = gen_points(dim, points_num);

    for p in points.clone() {
        kdtree.append(p, gen_rand(0, 10)).unwrap();
    }

    assert_eq!(kdtree.size(), points_num);
    assert_nearest!(kdtree, points, gen_points(dim, points_num));
}

fn assert_nearest_trie_set(dim: usize, points_num: usize) {
    let mut kdtree = trie::KdTreeSet::new(dim);
    let points = gen_points(dim, points_num);

    for p in points.clone() {
        kdtree.append(p).unwrap();
    }

    assert_eq!(kdtree.size(), points_num);
    assert_nearest!(kdtree, points, gen_points(dim, points_num));
}

#[test]
fn test_nearest_bucket_map_dim1_bucket1_points1024() {
    assert_nearest_bucket_map(1, 1, 1024);
}

#[test]
fn test_nearest_bucket_map_dim2_bucket4_points1024() {
    assert_nearest_bucket_map(2, 4, 1024);
}

#[test]
fn test_nearest_bucket_map_dim3_bucket16_points1024() {
    assert_nearest_bucket_map(3, 16, 1024);
}

#[test]
fn test_nearest_bucket_set_dim1_bucket1_points1024() {
    assert_nearest_bucket_set(1, 1, 1024)
}

#[test]
fn test_nearest_bucket_set_dim2_bucket4_points1024() {
    assert_nearest_bucket_set(2, 4, 1024)
}

#[test]
fn test_nearest_bucket_set_dim3_bucket16_points1024() {
    assert_nearest_bucket_set(3, 16, 1024)
}

#[test]
fn test_nearest_trie_map_dim1_points1024() {
    assert_nearest_trie_map(1, 1024);
}

#[test]
fn test_nearest_trie_map_dim2_points1024() {
    assert_nearest_trie_map(2, 1024);
}

#[test]
fn test_nearest_trie_map_dim3_points1024() {
    assert_nearest_trie_map(3, 1024);
}

#[test]
fn test_nearest_trie_set_dim1_points1024() {
    assert_nearest_trie_set(1, 1024);
}

#[test]
fn test_nearest_trie_set_dim2_points1024() {
    assert_nearest_trie_set(2, 1024);
}

#[test]
fn test_nearest_trie_set_dim3_points1024() {
    assert_nearest_trie_set(3, 1024);
}
