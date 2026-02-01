//! Benchmarks for continuous batching patterns.
//!
//! Validates that batching overhead is minimal and throughput scales correctly.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tgi_gtc::batching::{BatchConfig, BatchRequest, ContinuousBatcher};

/// Benchmark batch formation with varying request counts.
fn bench_batch_formation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batching/formation");
    group.throughput(Throughput::Elements(1));

    for num_requests in [10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::new("requests", num_requests),
            &num_requests,
            |b, &n| {
                let config = BatchConfig::builder()
                    .max_batch_size(32)
                    .max_batch_tokens(4096)
                    .build();

                b.iter(|| {
                    let batcher = ContinuousBatcher::new(config.clone());

                    // Add requests
                    for _ in 0..n {
                        let req = BatchRequest::new(batcher.next_id(), 128, 256);
                        batcher.add(req);
                    }

                    // Form all batches
                    let mut batch_count = 0;
                    while let Some(batch) = batcher.force_batch() {
                        batch_count += black_box(batch.size());
                    }
                    batch_count
                });
            },
        );
    }

    group.finish();
}

/// Benchmark single request add operation.
fn bench_request_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("batching/add");

    let config = BatchConfig::builder()
        .max_batch_size(32)
        .max_batch_tokens(4096)
        .build();

    group.bench_function("single_add", |b| {
        let batcher = ContinuousBatcher::new(config.clone());

        b.iter(|| {
            let req = BatchRequest::new(batcher.next_id(), 128, 256);
            batcher.add(black_box(req));
        });
    });

    group.finish();
}

/// Benchmark batch token counting.
fn bench_token_counting(c: &mut Criterion) {
    let mut group = c.benchmark_group("batching/tokens");
    group.throughput(Throughput::Elements(32));

    group.bench_function("count_32_requests", |b| {
        let config = BatchConfig::builder()
            .max_batch_size(32)
            .max_batch_tokens(8192)
            .build();

        b.iter(|| {
            let batcher = ContinuousBatcher::new(config.clone());
            for _ in 0..32 {
                let req = BatchRequest::new(batcher.next_id(), 128, 256);
                batcher.add(req);
            }
            let batch = batcher.force_batch().unwrap();
            black_box(batch.total_input_tokens)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_batch_formation,
    bench_request_add,
    bench_token_counting
);
criterion_main!(benches);
