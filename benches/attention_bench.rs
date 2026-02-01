//! Benchmarks for attention mechanisms.
//!
//! Validates that attention computation scales correctly with sequence length
//! and that Flash Attention provides expected memory/compute tradeoffs.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tgi_gtc::attention::{
    flash_attention_simulated, multi_head_attention, scaled_dot_product_attention, AttentionConfig,
    FlashAttentionConfig, RotaryEmbedding,
};

/// Generate random test vectors.
fn random_vectors(seq_len: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..seq_len)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let x = ((i as u64 * 17 + j as u64 * 31 + seed) % 1000) as f32 / 1000.0;
                    x * 2.0 - 1.0 // Range [-1, 1]
                })
                .collect()
        })
        .collect()
}

/// Benchmark scaled dot-product attention with varying sequence lengths.
fn bench_sdpa(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/sdpa");
    let head_dim = 64;

    for seq_len in [32, 64, 128, 256] {
        group.throughput(Throughput::Elements((seq_len * seq_len) as u64));

        let q = random_vectors(seq_len, head_dim, 1);
        let k = random_vectors(seq_len, head_dim, 2);
        let v = random_vectors(seq_len, head_dim, 3);
        let scale = 1.0 / (head_dim as f32).sqrt();

        group.bench_with_input(BenchmarkId::new("causal", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                black_box(scaled_dot_product_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    scale,
                    true,
                ))
            });
        });
    }

    group.finish();
}

/// Benchmark Flash Attention vs standard attention.
fn bench_flash_vs_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/flash_vs_standard");
    let head_dim = 64;
    let seq_len = 128;

    let q = random_vectors(seq_len, head_dim, 1);
    let k = random_vectors(seq_len, head_dim, 2);
    let v = random_vectors(seq_len, head_dim, 3);
    let scale = 1.0 / (head_dim as f32).sqrt();

    group.bench_function("standard", |b| {
        b.iter(|| {
            black_box(scaled_dot_product_attention(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                scale,
                true,
            ))
        });
    });

    for block_size in [16, 32, 64] {
        let config = FlashAttentionConfig::new(block_size, block_size);

        group.bench_with_input(
            BenchmarkId::new("flash_block", block_size),
            &block_size,
            |b, _| {
                b.iter(|| {
                    black_box(flash_attention_simulated(
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        black_box(&config),
                        scale,
                    ))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark multi-head attention.
fn bench_multi_head_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/multi_head");
    let head_dim = 64;
    let num_heads = 8;

    for seq_len in [32, 64, 128] {
        let hidden_dim = head_dim * num_heads;
        group.throughput(Throughput::Elements((seq_len * hidden_dim) as u64));

        let q = random_vectors(seq_len, hidden_dim, 1);
        let k = random_vectors(seq_len, hidden_dim, 2);
        let v = random_vectors(seq_len, hidden_dim, 3);

        let config = AttentionConfig::new(num_heads, head_dim);

        group.bench_with_input(BenchmarkId::new("heads_8", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                black_box(multi_head_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    black_box(&config),
                ))
            });
        });
    }

    group.finish();
}

/// Benchmark RoPE embeddings.
fn bench_rotary_embeddings(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention/rope");

    for head_dim in [64, 128] {
        let rope = RotaryEmbedding::new(head_dim, 4096, 10000.0);

        group.bench_with_input(BenchmarkId::new("dim", head_dim), &head_dim, |b, &dim| {
            // Create batch of vectors to rotate
            let mut q: Vec<Vec<f32>> = (0..32).map(|_| vec![0.5f32; dim]).collect();

            b.iter(|| {
                rope.apply(&mut q, 0);
                black_box(q[0][0])
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sdpa,
    bench_flash_vs_standard,
    bench_multi_head_attention,
    bench_rotary_embeddings
);
criterion_main!(benches);
