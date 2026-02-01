//! Benchmarks for KV cache block allocation.
//!
//! Validates block allocation, deallocation, and CoW fork performance.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tgi_gtc::kv_cache::{BlockAllocator, BlockAllocatorConfig, BlockId, BlockTable};

/// Benchmark block allocation with varying pool sizes.
fn bench_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache/allocation");

    for num_blocks in [256, 1024, 4096] {
        let config = BlockAllocatorConfig::with_blocks(num_blocks);

        group.bench_with_input(
            BenchmarkId::new("blocks", num_blocks),
            &num_blocks,
            |b, _| {
                b.iter(|| {
                    let mut allocator = BlockAllocator::new(config.clone());
                    let seq_id = 1;

                    // Allocate blocks until pool is exhausted or we have 100
                    let mut allocated = 0;
                    while allocator.can_allocate(1) && allocated < 100 {
                        if allocator.allocate(seq_id, 1).is_ok() {
                            allocated += 1;
                        } else {
                            break;
                        }
                    }

                    black_box(allocated)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark allocation/deallocation cycles.
fn bench_alloc_free_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache/alloc_free_cycle");
    group.throughput(Throughput::Elements(100));

    let config = BlockAllocatorConfig::with_blocks(1024);

    group.bench_function("100_cycles", |b| {
        b.iter(|| {
            let mut allocator = BlockAllocator::new(config.clone());

            for i in 0..100 {
                let seq_id = i as u64;
                if allocator.allocate(seq_id, 3).is_ok() {
                    allocator.free(seq_id);
                }
            }
        });
    });

    group.finish();
}

/// Benchmark CoW fork operation.
fn bench_fork(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache/fork");

    let config = BlockAllocatorConfig::with_blocks(4096);

    for num_parent_blocks in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("parent_blocks", num_parent_blocks),
            &num_parent_blocks,
            |b, &n| {
                b.iter(|| {
                    let mut allocator = BlockAllocator::new(config.clone());

                    // Create parent with n blocks
                    let parent_id = 1;
                    let _ = allocator.allocate(parent_id, n);

                    // Fork multiple children
                    for child_id in 2..=10u64 {
                        let _ = black_box(allocator.fork(parent_id, child_id));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory stats calculation.
fn bench_memory_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache/stats");

    let config = BlockAllocatorConfig::with_blocks(4096);

    group.bench_function("stats_with_1000_allocated", |b| {
        let mut allocator = BlockAllocator::new(config.clone());

        // Allocate blocks for many sequences
        for i in 0..200u64 {
            let _ = allocator.allocate(i, 5);
        }

        b.iter(|| black_box(allocator.memory_stats()));
    });

    group.finish();
}

/// Benchmark BlockTable operations.
fn bench_block_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache/block_table");

    for block_size in [16, 64, 256] {
        group.bench_with_input(
            BenchmarkId::new("block_size", block_size),
            &block_size,
            |b, &bs| {
                b.iter(|| {
                    let mut table = BlockTable::new(bs);

                    // Add 100 blocks
                    for i in 0..100u32 {
                        table.add_block(BlockId::new(i));
                    }

                    // Query positions
                    for pos in (0..5000).step_by(50) {
                        black_box(table.get_block_and_offset(pos));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark blocks_needed calculation.
fn bench_blocks_needed(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache/blocks_needed");

    for block_size in [16, 64, 256] {
        let table = BlockTable::new(block_size);

        group.bench_with_input(
            BenchmarkId::new("block_size", block_size),
            &block_size,
            |b, _| {
                b.iter(|| {
                    for tokens in [100, 500, 1000, 4000] {
                        black_box(table.blocks_needed(tokens));
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_allocation,
    bench_alloc_free_cycle,
    bench_fork,
    bench_memory_stats,
    bench_block_table,
    bench_blocks_needed
);
criterion_main!(benches);
