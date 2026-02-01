//! Profiling utilities for TGI patterns.
//!
//! Provides lightweight profiling primitives for examples and benchmarks.
//! Uses idiomatic Rust patterns with `std::hint::black_box` to prevent
//! optimization and accurate timing via `std::time::Instant`.
//!
//! # Sovereign AI Stack Equivalent
//!
//! Maps to `trueno::profiling` for GPU profiling with CUDA events.

use std::hint::black_box;
use std::time::{Duration, Instant};

/// A timer that measures wall-clock duration.
#[derive(Debug)]
pub struct Timer {
    name: String,
    start: Instant,
    iterations: u64,
}

impl Timer {
    /// Create a new timer with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
            iterations: 0,
        }
    }

    /// Record an iteration.
    pub fn record_iteration(&mut self) {
        self.iterations += 1;
    }

    /// Record N iterations.
    pub fn record_iterations(&mut self, n: u64) {
        self.iterations += n;
    }

    /// Get elapsed time.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Get iterations recorded.
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// Get name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Calculate throughput (iterations per second).
    pub fn throughput(&self) -> f64 {
        let secs = self.start.elapsed().as_secs_f64();
        if secs > 0.0 {
            self.iterations as f64 / secs
        } else {
            0.0
        }
    }

    /// Report timing to stdout.
    pub fn report(&self) {
        let elapsed = self.elapsed();
        let throughput = self.throughput();

        println!("  {}: {:?}", self.name, elapsed);
        if self.iterations > 0 {
            println!(
                "    {} iterations @ {:.2} iter/sec",
                self.iterations, throughput
            );
            if self.iterations > 1 {
                let per_iter = elapsed.as_nanos() as f64 / self.iterations as f64;
                println!("    {:.2} ns/iter", per_iter);
            }
        }
    }
}

/// Profile a function and return timing information.
pub fn profile<T, F: FnOnce() -> T>(name: &str, f: F) -> (T, Duration) {
    let start = Instant::now();
    let result = black_box(f());
    let elapsed = start.elapsed();

    println!("  {}: {:?}", name, elapsed);

    (result, elapsed)
}

/// Profile an iterated function.
pub fn profile_iterations<T, F: FnMut() -> T>(name: &str, iterations: u64, mut f: F) -> Duration {
    let start = Instant::now();

    for _ in 0..iterations {
        black_box(f());
    }

    let elapsed = start.elapsed();
    let per_iter = elapsed.as_nanos() as f64 / iterations as f64;

    println!(
        "  {}: {:?} ({} iters, {:.2} ns/iter)",
        name, elapsed, iterations, per_iter
    );

    elapsed
}

/// Performance metrics for a profiled operation.
#[derive(Debug, Clone)]
pub struct PerfMetrics {
    /// Operation name.
    pub name: String,
    /// Total duration.
    pub duration: Duration,
    /// Number of operations performed.
    pub operations: u64,
    /// Throughput in ops/sec.
    pub throughput: f64,
    /// Latency per operation.
    pub latency_ns: f64,
}

impl PerfMetrics {
    /// Create metrics from timing data.
    pub fn new(name: impl Into<String>, duration: Duration, operations: u64) -> Self {
        let secs = duration.as_secs_f64();
        let throughput = if secs > 0.0 {
            operations as f64 / secs
        } else {
            0.0
        };
        let latency_ns = if operations > 0 {
            duration.as_nanos() as f64 / operations as f64
        } else {
            0.0
        };

        Self {
            name: name.into(),
            duration,
            operations,
            throughput,
            latency_ns,
        }
    }

    /// Report metrics to stdout.
    pub fn report(&self) {
        println!("{}:", self.name);
        println!("  Duration:   {:?}", self.duration);
        println!("  Operations: {}", self.operations);
        println!("  Throughput: {:.2} ops/sec", self.throughput);
        println!("  Latency:    {:.2} ns/op", self.latency_ns);
    }

    /// Assert throughput is at least the given value.
    pub fn assert_throughput_min(&self, min_throughput: f64) {
        assert!(
            self.throughput >= min_throughput,
            "{}: throughput {:.2} ops/sec is below minimum {:.2}",
            self.name,
            self.throughput,
            min_throughput
        );
    }

    /// Assert latency is at most the given value.
    pub fn assert_latency_max(&self, max_latency_ns: f64) {
        assert!(
            self.latency_ns <= max_latency_ns,
            "{}: latency {:.2} ns is above maximum {:.2}",
            self.name,
            self.latency_ns,
            max_latency_ns
        );
    }
}

/// Memory profiling (simplified, uses allocator stats if available).
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Estimated bytes allocated.
    pub allocated: usize,
    /// Peak bytes allocated.
    pub peak: usize,
}

impl MemoryStats {
    /// Create new stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record allocation.
    pub fn record_alloc(&mut self, bytes: usize) {
        self.allocated += bytes;
        if self.allocated > self.peak {
            self.peak = self.allocated;
        }
    }

    /// Record deallocation.
    pub fn record_dealloc(&mut self, bytes: usize) {
        self.allocated = self.allocated.saturating_sub(bytes);
    }

    /// Report memory stats.
    pub fn report(&self) {
        println!("Memory:");
        println!(
            "  Allocated: {} bytes ({:.2} KB)",
            self.allocated,
            self.allocated as f64 / 1024.0
        );
        println!(
            "  Peak:      {} bytes ({:.2} KB)",
            self.peak,
            self.peak as f64 / 1024.0
        );
    }
}

/// Macro for timing a block of code.
#[macro_export]
macro_rules! timed {
    ($name:expr, $block:block) => {{
        let __start = std::time::Instant::now();
        let __result = { $block };
        let __elapsed = __start.elapsed();
        println!("  {}: {:?}", $name, __elapsed);
        (__result, __elapsed)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer_creation() {
        let timer = Timer::new("test");
        assert_eq!(timer.name(), "test");
        assert_eq!(timer.iterations(), 0);
    }

    #[test]
    fn test_timer_iterations() {
        let mut timer = Timer::new("test");
        timer.record_iteration();
        timer.record_iteration();
        timer.record_iterations(5);
        assert_eq!(timer.iterations(), 7);
    }

    #[test]
    fn test_perf_metrics() {
        let metrics = PerfMetrics::new("test", Duration::from_millis(100), 1000);
        assert!((metrics.throughput - 10000.0).abs() < 100.0); // ~10k ops/sec
        assert!((metrics.latency_ns - 100000.0).abs() < 1000.0); // ~100us/op
    }

    #[test]
    fn test_perf_metrics_zero_ops() {
        let metrics = PerfMetrics::new("test", Duration::from_millis(100), 0);
        assert_eq!(metrics.throughput, 0.0);
        assert_eq!(metrics.latency_ns, 0.0);
    }

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::new();
        stats.record_alloc(1024);
        stats.record_alloc(2048);
        assert_eq!(stats.allocated, 3072);
        assert_eq!(stats.peak, 3072);

        stats.record_dealloc(1024);
        assert_eq!(stats.allocated, 2048);
        assert_eq!(stats.peak, 3072); // Peak unchanged
    }

    #[test]
    fn test_profile() {
        let (result, duration) = profile("test_add", || 2 + 2);
        assert_eq!(result, 4);
        assert!(duration.as_nanos() < 1_000_000); // Should be sub-millisecond
    }
}
