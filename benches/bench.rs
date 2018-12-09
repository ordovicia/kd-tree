#![feature(test)]

extern crate test;
use test::Bencher;

use noisy_float::prelude::*;
use num_traits::Float;
use rand::{distributions::uniform::SampleUniform, Rng};

use kd_tree::{bucket, trie};

#[inline]
fn squared_euclidean<T: Float>(p1: &[T], p2: &[T]) -> T {
    p1.iter()
        .zip(p2.iter())
        .map(|(&p1, &p2)| (p1 - p2) * (p1 - p2))
        .fold(T::zero(), std::ops::Add::add)
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

#[bench]
fn bench_append_bucket_map_dim3_bucket16_points1024(b: &mut Bencher) {
    const DIM: usize = 3;
    const BUCKET: usize = 16;
    const POINTS: usize = 1024;

    let mut kdtree = bucket::KdTreeMap::new(DIM, BUCKET);
    for p in gen_points_noisy(DIM, POINTS) {
        kdtree.append(p, gen_rand(0, 10)).unwrap();
    }

    b.iter(|| kdtree.append(vec![r64(0.0); 3], 0).unwrap());
}

#[bench]
fn bench_append_trie_map_dim3_points1024(b: &mut Bencher) {
    const DIM: usize = 3;
    const POINTS: usize = 1024;

    let mut kdtree = trie::KdTreeMap::new(DIM);
    for p in gen_points(DIM, POINTS) {
        kdtree.append(p, gen_rand(0, 10)).unwrap();
    }

    b.iter(|| kdtree.append(vec![0.0; 3], 0).unwrap());
}

#[bench]
fn bench_nearest_bucket_map_dim3_bucket16_points1024(b: &mut Bencher) {
    const DIM: usize = 3;
    const BUCKET: usize = 16;
    const POINTS: usize = 1024;

    let mut kdtree = bucket::KdTreeMap::new(DIM, BUCKET);
    for p in gen_points_noisy(DIM, POINTS) {
        kdtree.append(p, gen_rand(0, 10)).unwrap();
    }

    let p = vec![r64(0.0); 3];
    b.iter(|| kdtree.nearest(&p, &squared_euclidean).unwrap());
}

#[bench]
fn bench_nearest_bucket_map_dim3_bucket16_points65536(b: &mut Bencher) {
    const DIM: usize = 3;
    const BUCKET: usize = 16;
    const POINTS: usize = 65536;

    let mut kdtree = bucket::KdTreeMap::new(DIM, BUCKET);
    for p in gen_points_noisy(DIM, POINTS) {
        kdtree.append(p, gen_rand(0, 10)).unwrap();
    }

    let p = vec![r64(0.0); 3];
    b.iter(|| kdtree.nearest(&p, &squared_euclidean).unwrap());
}

#[bench]
fn bench_nearest_trie_map_dim3_points1024(b: &mut Bencher) {
    const DIM: usize = 3;
    const POINTS: usize = 1024;

    let mut kdtree = trie::KdTreeMap::new(DIM);
    for p in gen_points(DIM, POINTS) {
        kdtree.append(p, gen_rand(0, 10)).unwrap();
    }

    let p = vec![0.0; 3];
    b.iter(|| kdtree.nearest(&p, &squared_euclidean).unwrap());
}

#[bench]
fn bench_nearest_trie_map_dim3_points65536(b: &mut Bencher) {
    const DIM: usize = 3;
    const POINTS: usize = 65536;

    let mut kdtree = trie::KdTreeMap::new(DIM);
    for p in gen_points(DIM, POINTS) {
        kdtree.append(p, gen_rand(0, 10)).unwrap();
    }

    let p = vec![0.0; 3];
    b.iter(|| kdtree.nearest(&p, &squared_euclidean).unwrap());
}
