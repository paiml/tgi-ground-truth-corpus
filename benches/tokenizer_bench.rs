//! Benchmarks for tokenizer operations.
//!
//! Validates BPE encoding/decoding performance and vocabulary lookups.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tgi_gtc::tokenizer::{EncodedOutput, Tokenizer, TokenizerConfig};

/// Create a test tokenizer with default config.
fn create_test_tokenizer() -> Tokenizer {
    // Use the simple_ascii config which is already set up
    Tokenizer::new(TokenizerConfig::simple_ascii())
}

/// Benchmark encoding with varying text lengths.
fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenizer/encode");
    let tokenizer = create_test_tokenizer();

    let texts = [
        ("short", "The quick brown fox"),
        (
            "medium",
            "The quick brown fox jumps over the lazy dog. This is a sample sentence.",
        ),
        (
            "long",
            &"The quick brown fox jumps over the lazy dog. ".repeat(20),
        ),
    ];

    for (name, text) in texts {
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(BenchmarkId::new("length", name), &text, |b, &text| {
            b.iter(|| black_box(tokenizer.encode(black_box(text))));
        });
    }

    group.finish();
}

/// Benchmark decoding with varying token counts.
fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenizer/decode");
    let tokenizer = create_test_tokenizer();

    for num_tokens in [10, 50, 200] {
        // Create token IDs from known vocabulary (ASCII chars)
        let token_ids: Vec<u32> = (0..num_tokens).map(|i| (i % 26 + 97) as u32).collect();

        group.throughput(Throughput::Elements(num_tokens as u64));

        group.bench_with_input(
            BenchmarkId::new("tokens", num_tokens),
            &token_ids,
            |b, ids| {
                b.iter(|| black_box(tokenizer.decode(black_box(ids))));
            },
        );
    }

    group.finish();
}

/// Benchmark vocabulary lookups.
fn bench_vocab_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenizer/lookup");
    let tokenizer = create_test_tokenizer();

    group.bench_function("token_to_id", |b| {
        let tokens = ["a", "b", "c", "d", "e"];
        b.iter(|| {
            for token in &tokens {
                black_box(tokenizer.token_to_id(token));
            }
        });
    });

    group.bench_function("id_to_token", |b| {
        let ids = [97u32, 98, 99, 100, 101]; // a, b, c, d, e
        b.iter(|| {
            for &id in &ids {
                black_box(tokenizer.id_to_token(id));
            }
        });
    });

    group.finish();
}

/// Benchmark encoded output operations.
fn bench_encoded_output(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenizer/encoded_output");

    group.bench_function("pad_to_512", |b| {
        let ids: Vec<u32> = (0..100).collect();

        b.iter(|| {
            let mut out = EncodedOutput::from_ids(ids.clone());
            out.pad_to(512, 0);
            black_box(out)
        });
    });

    group.bench_function("truncate_to_128", |b| {
        let ids: Vec<u32> = (0..500).collect();

        b.iter(|| {
            let mut out = EncodedOutput::from_ids(ids.clone());
            out.truncate_to(128);
            black_box(out)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_vocab_lookup,
    bench_encoded_output
);
criterion_main!(benches);
