extern crate kd_tree;
extern crate noisy_float;
extern crate num_traits;
extern crate rand;

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

fn gen_rand<T: SampleUniform>(begin: T, end: T) -> T {
    rand::thread_rng().gen_range(begin, end)
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

#[test]
fn test_bucket_nearest_map_dim2_bucket1_points1024() {
    const POINTS_NUM: usize = 1024;

    const DIM: usize = 2;
    const BUCKET_SIZE: usize = 1;

    let mut kdtree = bucket::KdTreeMap::new(DIM, BUCKET_SIZE);
    let points = (0..POINTS_NUM)
        .map(|_| [r64(gen_rand(-1.0, 1.0)), r64(gen_rand(-1.0, 1.0))])
        .collect::<Vec<[R64; DIM]>>();

    for p in points.clone() {
        kdtree.append(p, gen_rand(0, 10)).unwrap();
    }

    assert_eq!(kdtree.size(), POINTS_NUM);

    for p in &points {
        let nearest = kdtree.nearest(p, &squared_euclidean).unwrap().unwrap();
        let linear = nearest_linear(p, &points, &squared_euclidean);

        assert_eq!(nearest.point, linear);
    }
}

#[test]
fn test_bucket_nearest_map_dim3_bucket4_points1024() {
    const POINTS_NUM: usize = 1024;

    const DIM: usize = 3;
    const BUCKET_SIZE: usize = 4;

    let mut kdtree = bucket::KdTreeMap::new(DIM, BUCKET_SIZE);
    let points = (0..POINTS_NUM)
        .map(|_| {
            [
                r64(gen_rand(-1.0, 1.0)),
                r64(gen_rand(-1.0, 1.0)),
                r64(gen_rand(-1.0, 1.0)),
            ]
        }).collect::<Vec<[R64; DIM]>>();

    for p in points.clone() {
        kdtree.append(p, gen_rand(0, 10)).unwrap();
    }

    assert_eq!(kdtree.size(), POINTS_NUM);

    for p in &points {
        let nearest = kdtree.nearest(p, &squared_euclidean).unwrap().unwrap();
        let linear = nearest_linear(p, &points, &squared_euclidean);

        assert_eq!(nearest.point, linear);
    }
}

#[test]
fn test_bucket_nearest_set_dim2_bucket1_points1024() {
    const POINTS_NUM: usize = 1024;

    const DIM: usize = 2;
    const BUCKET_SIZE: usize = 1;

    let mut kdtree = bucket::KdTreeSet::new(DIM, BUCKET_SIZE);
    let points = (0..POINTS_NUM)
        .map(|_| [r64(gen_rand(-1.0, 1.0)), r64(gen_rand(-1.0, 1.0))])
        .collect::<Vec<[R64; DIM]>>();

    for p in points.clone() {
        kdtree.append(p).unwrap();
    }

    assert_eq!(kdtree.size(), POINTS_NUM);

    for p in &points {
        let nearest = kdtree.nearest(p, &squared_euclidean).unwrap().unwrap();
        let linear = nearest_linear(p, &points, &squared_euclidean);

        assert_eq!(nearest.point, linear);
    }
}

#[test]
fn test_bucket_nearest_set_dim3_bucket4_points1024() {
    const POINTS_NUM: usize = 1024;

    const DIM: usize = 3;
    const BUCKET_SIZE: usize = 4;

    let mut kdtree = bucket::KdTreeSet::new(DIM, BUCKET_SIZE);
    let points = (0..POINTS_NUM)
        .map(|_| {
            [
                r64(gen_rand(-1.0, 1.0)),
                r64(gen_rand(-1.0, 1.0)),
                r64(gen_rand(-1.0, 1.0)),
            ]
        }).collect::<Vec<[R64; DIM]>>();

    for p in points.clone() {
        kdtree.append(p).unwrap();
    }

    assert_eq!(kdtree.size(), POINTS_NUM);

    for p in &points {
        let nearest = kdtree.nearest(p, &squared_euclidean).unwrap().unwrap();
        let linear = nearest_linear(p, &points, &squared_euclidean);

        assert_eq!(nearest.point, linear);
    }
}

#[test]
fn test_trie_nearest_map_dim2_points1024() {
    const POINTS_NUM: usize = 1024;
    const DIM: usize = 2;

    let mut kdtree = trie::KdTreeMap::new(DIM);
    let points = (0..POINTS_NUM)
        .map(|_| [r64(gen_rand(-1.0, 1.0)), r64(gen_rand(-1.0, 1.0))])
        .collect::<Vec<[R64; DIM]>>();

    for p in points.clone() {
        kdtree.append(p, gen_rand(0, 10)).unwrap();
    }

    assert_eq!(kdtree.size(), POINTS_NUM);

    for p in &points {
        let nearest = kdtree.nearest(p, &squared_euclidean).unwrap().unwrap();
        let linear = nearest_linear(p, &points, &squared_euclidean);

        assert_eq!(nearest.point, linear);
    }
}

#[test]
fn test_trie_nearest_map_dim3_points1024() {
    const POINTS_NUM: usize = 1024;
    const DIM: usize = 3;

    let mut kdtree = trie::KdTreeMap::new(DIM);
    let points = (0..POINTS_NUM)
        .map(|_| {
            [
                r64(gen_rand(-1.0, 1.0)),
                r64(gen_rand(-1.0, 1.0)),
                r64(gen_rand(-1.0, 1.0)),
            ]
        }).collect::<Vec<[R64; DIM]>>();

    for p in points.clone() {
        kdtree.append(p, gen_rand(0, 10)).unwrap();
    }

    assert_eq!(kdtree.size(), POINTS_NUM);

    for p in &points {
        let nearest = kdtree.nearest(p, &squared_euclidean).unwrap().unwrap();
        let linear = nearest_linear(p, &points, &squared_euclidean);

        assert_eq!(nearest.point, linear);
    }
}

#[test]
fn test_trie_nearest_set_dim2_points1024() {
    const POINTS_NUM: usize = 1024;
    const DIM: usize = 2;

    let mut kdtree = trie::KdTreeSet::new(DIM);
    let points = (0..POINTS_NUM)
        .map(|_| [r64(gen_rand(-1.0, 1.0)), r64(gen_rand(-1.0, 1.0))])
        .collect::<Vec<[R64; DIM]>>();

    for p in points.clone() {
        kdtree.append(p).unwrap();
    }

    assert_eq!(kdtree.size(), POINTS_NUM);

    for p in &points {
        let nearest = kdtree.nearest(p, &squared_euclidean).unwrap().unwrap();
        let linear = nearest_linear(p, &points, &squared_euclidean);

        assert_eq!(nearest.point, linear);
    }
}

#[test]
fn test_trie_nearest_set_dim3_points1024() {
    const POINTS_NUM: usize = 1024;
    const DIM: usize = 3;

    let mut kdtree = trie::KdTreeSet::new(DIM);
    let points = (0..POINTS_NUM)
        .map(|_| {
            [
                r64(gen_rand(-1.0, 1.0)),
                r64(gen_rand(-1.0, 1.0)),
                r64(gen_rand(-1.0, 1.0)),
            ]
        }).collect::<Vec<[R64; DIM]>>();

    for p in points.clone() {
        kdtree.append(p).unwrap();
    }

    assert_eq!(kdtree.size(), POINTS_NUM);

    for p in &points {
        let nearest = kdtree.nearest(p, &squared_euclidean).unwrap().unwrap();
        let linear = nearest_linear(p, &points, &squared_euclidean);

        assert_eq!(nearest.point, linear);
    }
}
