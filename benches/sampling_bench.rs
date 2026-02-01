//! Benchmarks for token sampling strategies.
//!
//! Validates sampling overhead and performance characteristics of
//! temperature, top-k, top-p, and penalty-based sampling.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tgi_gtc::sampling::{argmax, softmax, top_k, Sampler, SamplingConfig};

/// Generate random logits with controlled distribution.
fn random_logits(vocab_size: usize, seed: u64) -> Vec<f32> {
    (0..vocab_size)
        .map(|i| {
            let x = ((i as u64 * 17 + seed) % 1000) as f32 / 100.0;
            x - 5.0 // Range roughly [-5, 5]
        })
        .collect()
}

/// Benchmark softmax with varying vocabulary sizes.
fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/softmax");

    for vocab_size in [1000, 32000, 128000] {
        group.throughput(Throughput::Elements(vocab_size as u64));

        let logits = random_logits(vocab_size, 1);

        group.bench_with_input(
            BenchmarkId::new("vocab", vocab_size),
            &vocab_size,
            |b, _| {
                b.iter(|| black_box(softmax(black_box(&logits))));
            },
        );
    }

    group.finish();
}

/// Benchmark argmax operation.
fn bench_argmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/argmax");

    for vocab_size in [1000, 32000, 128000] {
        group.throughput(Throughput::Elements(vocab_size as u64));

        let probs: Vec<f32> = softmax(&random_logits(vocab_size, 1));

        group.bench_with_input(
            BenchmarkId::new("vocab", vocab_size),
            &vocab_size,
            |b, _| {
                b.iter(|| black_box(argmax(black_box(&probs))));
            },
        );
    }

    group.finish();
}

/// Benchmark top-k filtering.
fn bench_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/top_k");
    let vocab_size = 32000;

    let probs: Vec<f32> = softmax(&random_logits(vocab_size, 1));

    for k in [10, 50, 100, 500] {
        group.bench_with_input(BenchmarkId::new("k", k), &k, |b, &k| {
            b.iter(|| black_box(top_k(black_box(&probs), k)));
        });
    }

    group.finish();
}

/// Benchmark full sampling pipeline with different configurations.
fn bench_sampling_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/pipeline");
    let vocab_size = 32000;
    let logits = random_logits(vocab_size, 1);

    // Greedy sampling
    group.bench_function("greedy", |b| {
        let config = SamplingConfig::greedy();
        let mut sampler = Sampler::new(config);

        b.iter(|| {
            let (token, prob) = sampler.sample(black_box(&logits));
            black_box((token, prob))
        });
    });

    // Temperature + top-p
    group.bench_function("temp_top_p", |b| {
        let config = SamplingConfig::default().temperature(0.8).top_p(0.9);
        let mut sampler = Sampler::new(config);

        b.iter(|| {
            let (token, prob) = sampler.sample(black_box(&logits));
            black_box((token, prob))
        });
    });

    // Full creative config
    group.bench_function("creative", |b| {
        let config = SamplingConfig::creative();
        let mut sampler = Sampler::new(config);

        b.iter(|| {
            let (token, prob) = sampler.sample(black_box(&logits));
            black_box((token, prob))
        });
    });

    // With penalties (simulates repetition)
    group.bench_function("with_penalties", |b| {
        let config = SamplingConfig::default()
            .temperature(0.7)
            .top_p(0.9)
            .repetition_penalty(1.1)
            .frequency_penalty(0.5);
        let mut sampler = Sampler::new(config);

        // Record some tokens to trigger penalties
        for i in 0..100 {
            sampler.record_token(i % 1000);
        }

        b.iter(|| {
            let (token, prob) = sampler.sample(black_box(&logits));
            black_box((token, prob))
        });
    });

    group.finish();
}

/// Benchmark sampling with token history tracking.
fn bench_token_recording(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling/recording");

    group.bench_function("record_1000_tokens", |b| {
        b.iter(|| {
            let config = SamplingConfig::default().repetition_penalty(1.1);
            let mut sampler = Sampler::new(config);

            for i in 0..1000 {
                sampler.record_token(black_box(i % 32000));
            }

            // Just consume the sampler by extracting something simple
            black_box(1000u32)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_softmax,
    bench_argmax,
    bench_top_k,
    bench_sampling_pipeline,
    bench_token_recording
);
criterion_main!(benches);
