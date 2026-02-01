//! Model Warmup
//!
//! Implements warmup procedures to ensure optimal performance before serving.
//!
//! # TGI Reference
//!
//! Based on TGI's warmup implementation.
//! See: <https://github.com/huggingface/text-generation-inference>
//!
//! # Why Warmup?
//!
//! - JIT compilation: First runs trigger kernel compilation
//! - Memory allocation: Pre-allocate KV cache and buffers
//! - CUDA graphs: Capture graphs for common batch sizes
//! - Cache warming: Fill CPU/GPU caches
//!
//! # Example
//!
//! ```rust
//! use tgi_gtc::warmup::{WarmupConfig, WarmupRunner, WarmupStats};
//!
//! let config = WarmupConfig::default();
//! let mut runner = WarmupRunner::new(config);
//!
//! // Run warmup
//! let stats = runner.run();
//! println!("Warmup completed in {:?}", stats.total_time);
//! ```

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Configuration for warmup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupConfig {
    /// Number of warmup iterations.
    pub num_iterations: usize,

    /// Batch sizes to warm up.
    pub batch_sizes: Vec<usize>,

    /// Sequence lengths to warm up.
    pub sequence_lengths: Vec<usize>,

    /// Maximum tokens to generate per warmup.
    pub max_new_tokens: usize,

    /// Whether to warm up prefill.
    pub warmup_prefill: bool,

    /// Whether to warm up decode.
    pub warmup_decode: bool,

    /// Whether to warm up with speculative decoding.
    pub warmup_speculation: bool,

    /// Whether to capture CUDA graphs.
    pub capture_cuda_graphs: bool,

    /// Timeout for warmup (seconds).
    pub timeout_secs: u64,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            num_iterations: 3,
            batch_sizes: vec![1, 2, 4, 8, 16, 32],
            sequence_lengths: vec![128, 256, 512, 1024],
            max_new_tokens: 32,
            warmup_prefill: true,
            warmup_decode: true,
            warmup_speculation: false,
            capture_cuda_graphs: false,
            timeout_secs: 300,
        }
    }
}

impl WarmupConfig {
    /// Minimal warmup (fast startup).
    pub fn minimal() -> Self {
        Self {
            num_iterations: 1,
            batch_sizes: vec![1],
            sequence_lengths: vec![128],
            max_new_tokens: 8,
            warmup_prefill: true,
            warmup_decode: true,
            warmup_speculation: false,
            capture_cuda_graphs: false,
            timeout_secs: 60,
        }
    }

    /// Full warmup (optimal performance).
    pub fn full() -> Self {
        Self {
            num_iterations: 5,
            batch_sizes: vec![1, 2, 4, 8, 16, 32, 64],
            sequence_lengths: vec![128, 256, 512, 1024, 2048],
            max_new_tokens: 64,
            warmup_prefill: true,
            warmup_decode: true,
            warmup_speculation: true,
            capture_cuda_graphs: true,
            timeout_secs: 600,
        }
    }

    /// Number of warmup configurations.
    pub fn num_configs(&self) -> usize {
        self.batch_sizes.len() * self.sequence_lengths.len() * self.num_iterations
    }
}

/// Statistics from a warmup run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupStats {
    /// Total warmup time.
    pub total_time: Duration,

    /// Time for prefill warmup.
    pub prefill_time: Duration,

    /// Time for decode warmup.
    pub decode_time: Duration,

    /// Number of successful runs.
    pub successful_runs: usize,

    /// Number of failed runs.
    pub failed_runs: usize,

    /// Configurations tested.
    pub configs_tested: usize,

    /// Peak memory usage (bytes).
    pub peak_memory_bytes: usize,

    /// Whether warmup completed successfully.
    pub success: bool,

    /// Error message if failed.
    pub error: Option<String>,
}

impl Default for WarmupStats {
    fn default() -> Self {
        Self {
            total_time: Duration::ZERO,
            prefill_time: Duration::ZERO,
            decode_time: Duration::ZERO,
            successful_runs: 0,
            failed_runs: 0,
            configs_tested: 0,
            peak_memory_bytes: 0,
            success: true,
            error: None,
        }
    }
}

impl WarmupStats {
    /// Success rate.
    pub fn success_rate(&self) -> f64 {
        let total = self.successful_runs + self.failed_runs;
        if total == 0 {
            1.0
        } else {
            self.successful_runs as f64 / total as f64
        }
    }

    /// Average time per configuration.
    pub fn avg_time_per_config(&self) -> Duration {
        if self.configs_tested == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.configs_tested as u32
        }
    }
}

/// A single warmup configuration.
#[derive(Debug, Clone)]
pub struct WarmupIteration {
    /// Batch size.
    pub batch_size: usize,

    /// Sequence length.
    pub sequence_length: usize,

    /// Number of tokens to generate.
    pub new_tokens: usize,

    /// Phase (prefill or decode).
    pub phase: WarmupPhase,
}

/// Warmup phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupPhase {
    /// Prefill phase.
    Prefill,
    /// Decode phase.
    Decode,
    /// Both phases.
    Both,
}

/// Warmup runner.
#[derive(Debug)]
pub struct WarmupRunner {
    config: WarmupConfig,
    iterations: Vec<WarmupIteration>,
    stats: WarmupStats,
}

impl WarmupRunner {
    /// Create a new warmup runner.
    pub fn new(config: WarmupConfig) -> Self {
        let iterations = Self::generate_iterations(&config);
        Self {
            config,
            iterations,
            stats: WarmupStats::default(),
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &WarmupConfig {
        &self.config
    }

    /// Get iterations.
    pub fn iterations(&self) -> &[WarmupIteration] {
        &self.iterations
    }

    /// Generate warmup iterations.
    fn generate_iterations(config: &WarmupConfig) -> Vec<WarmupIteration> {
        let mut iterations = Vec::new();

        for &batch_size in &config.batch_sizes {
            for &seq_len in &config.sequence_lengths {
                for _ in 0..config.num_iterations {
                    if config.warmup_prefill {
                        iterations.push(WarmupIteration {
                            batch_size,
                            sequence_length: seq_len,
                            new_tokens: 1,
                            phase: WarmupPhase::Prefill,
                        });
                    }

                    if config.warmup_decode {
                        iterations.push(WarmupIteration {
                            batch_size,
                            sequence_length: seq_len,
                            new_tokens: config.max_new_tokens,
                            phase: WarmupPhase::Decode,
                        });
                    }
                }
            }
        }

        iterations
    }

    /// Run warmup (simulated).
    pub fn run(&mut self) -> WarmupStats {
        let start = Instant::now();
        let timeout = Duration::from_secs(self.config.timeout_secs);

        let mut prefill_time = Duration::ZERO;
        let mut decode_time = Duration::ZERO;

        for iteration in &self.iterations {
            if start.elapsed() > timeout {
                self.stats.error = Some("Warmup timeout".to_string());
                self.stats.success = false;
                break;
            }

            let _iter_start = Instant::now();

            // Simulate warmup work
            let simulated_time = self.simulate_iteration(iteration);

            match iteration.phase {
                WarmupPhase::Prefill => prefill_time += simulated_time,
                WarmupPhase::Decode => decode_time += simulated_time,
                WarmupPhase::Both => {
                    prefill_time += simulated_time / 2;
                    decode_time += simulated_time / 2;
                }
            }

            self.stats.configs_tested += 1;
            self.stats.successful_runs += 1;

            // Simulate memory tracking
            let estimated_memory = self.estimate_memory(iteration);
            self.stats.peak_memory_bytes = self.stats.peak_memory_bytes.max(estimated_memory);
        }

        self.stats.total_time = start.elapsed();
        self.stats.prefill_time = prefill_time;
        self.stats.decode_time = decode_time;

        self.stats.clone()
    }

    /// Simulate an iteration (in practice, would run actual inference).
    fn simulate_iteration(&self, iteration: &WarmupIteration) -> Duration {
        // Simulate work based on batch size and sequence length
        let base_us = 1000; // 1ms base
        let per_token_us = 10;
        let per_batch_us = 100;

        let total_us = base_us
            + (iteration.sequence_length * per_token_us)
            + (iteration.batch_size * per_batch_us);

        Duration::from_micros(total_us as u64)
    }

    /// Estimate memory usage for an iteration.
    fn estimate_memory(&self, iteration: &WarmupIteration) -> usize {
        // Rough estimate: tokens * batch * hidden_dim * layers * 2 (K+V) * bytes_per_element
        let hidden_dim = 4096;
        let num_layers = 32;
        let bytes_per_element = 2; // FP16

        iteration.batch_size
            * iteration.sequence_length
            * hidden_dim
            * num_layers
            * 2
            * bytes_per_element
    }

    /// Get progress (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        if self.iterations.is_empty() {
            1.0
        } else {
            self.stats.configs_tested as f64 / self.iterations.len() as f64
        }
    }

    /// Check if warmup is complete.
    pub fn is_complete(&self) -> bool {
        self.stats.configs_tested >= self.iterations.len()
    }
}

/// Warmup result for a single batch size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchWarmupResult {
    /// Batch size.
    pub batch_size: usize,

    /// Prefill time per token (ns).
    pub prefill_time_per_token_ns: u64,

    /// Decode time per token (ns).
    pub decode_time_per_token_ns: u64,

    /// Memory used (bytes).
    pub memory_bytes: usize,

    /// CUDA graph captured.
    pub cuda_graph_captured: bool,
}

/// Collection of warmup results.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WarmupResults {
    /// Results per batch size.
    pub by_batch_size: Vec<BatchWarmupResult>,

    /// Overall statistics.
    pub stats: WarmupStats,

    /// Recommended batch size for throughput.
    pub recommended_batch_size: usize,

    /// Maximum supported batch size.
    pub max_batch_size: usize,
}

impl WarmupResults {
    /// Get result for a specific batch size.
    pub fn get(&self, batch_size: usize) -> Option<&BatchWarmupResult> {
        self.by_batch_size
            .iter()
            .find(|r| r.batch_size == batch_size)
    }

    /// Find optimal batch size for a given latency target.
    pub fn optimal_batch_for_latency(&self, max_latency_ms: f64) -> Option<usize> {
        let max_latency_ns = (max_latency_ms * 1_000_000.0) as u64;

        self.by_batch_size
            .iter()
            .filter(|r| r.decode_time_per_token_ns <= max_latency_ns)
            .max_by_key(|r| r.batch_size)
            .map(|r| r.batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_config_default() {
        let config = WarmupConfig::default();
        assert_eq!(config.num_iterations, 3);
        assert!(config.warmup_prefill);
        assert!(config.warmup_decode);
    }

    #[test]
    fn test_warmup_config_minimal() {
        let config = WarmupConfig::minimal();
        assert_eq!(config.num_iterations, 1);
        assert_eq!(config.batch_sizes.len(), 1);
    }

    #[test]
    fn test_warmup_config_num_configs() {
        let config = WarmupConfig {
            num_iterations: 2,
            batch_sizes: vec![1, 2],
            sequence_lengths: vec![128, 256],
            ..Default::default()
        };
        assert_eq!(config.num_configs(), 8); // 2 * 2 * 2
    }

    #[test]
    fn test_warmup_runner_new() {
        let config = WarmupConfig::minimal();
        let runner = WarmupRunner::new(config);

        assert!(!runner.iterations().is_empty());
    }

    #[test]
    fn test_warmup_runner_run() {
        let config = WarmupConfig::minimal();
        let mut runner = WarmupRunner::new(config);

        let stats = runner.run();

        assert!(stats.success);
        assert!(stats.total_time > Duration::ZERO);
        assert!(stats.successful_runs > 0);
    }

    #[test]
    fn test_warmup_stats_success_rate() {
        let mut stats = WarmupStats::default();
        stats.successful_runs = 8;
        stats.failed_runs = 2;

        assert!((stats.success_rate() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_warmup_stats_avg_time() {
        let mut stats = WarmupStats::default();
        stats.total_time = Duration::from_secs(10);
        stats.configs_tested = 5;

        assert_eq!(stats.avg_time_per_config(), Duration::from_secs(2));
    }

    #[test]
    fn test_warmup_runner_progress() {
        let config = WarmupConfig::minimal();
        let runner = WarmupRunner::new(config);

        assert_eq!(runner.progress(), 0.0);
        assert!(!runner.is_complete());
    }

    #[test]
    fn test_warmup_runner_complete() {
        let config = WarmupConfig::minimal();
        let mut runner = WarmupRunner::new(config);

        runner.run();

        assert!(runner.is_complete());
        assert_eq!(runner.progress(), 1.0);
    }

    #[test]
    fn test_batch_warmup_result() {
        let result = BatchWarmupResult {
            batch_size: 8,
            prefill_time_per_token_ns: 1000,
            decode_time_per_token_ns: 500,
            memory_bytes: 1024 * 1024,
            cuda_graph_captured: true,
        };

        assert_eq!(result.batch_size, 8);
        assert!(result.cuda_graph_captured);
    }

    #[test]
    fn test_warmup_results_optimal_batch() {
        let results = WarmupResults {
            by_batch_size: vec![
                BatchWarmupResult {
                    batch_size: 1,
                    prefill_time_per_token_ns: 50_000,
                    decode_time_per_token_ns: 50_000,
                    memory_bytes: 1000,
                    cuda_graph_captured: false,
                },
                BatchWarmupResult {
                    batch_size: 8,
                    prefill_time_per_token_ns: 100_000,
                    decode_time_per_token_ns: 100_000,
                    memory_bytes: 8000,
                    cuda_graph_captured: false,
                },
                BatchWarmupResult {
                    batch_size: 32,
                    prefill_time_per_token_ns: 200_000,
                    decode_time_per_token_ns: 200_000,
                    memory_bytes: 32000,
                    cuda_graph_captured: false,
                },
            ],
            stats: WarmupStats::default(),
            recommended_batch_size: 8,
            max_batch_size: 32,
        };

        // 0.15ms = 150_000 ns, should select batch_size=8 (100_000ns < 150_000ns, but 200_000ns > 150_000ns)
        assert_eq!(results.optimal_batch_for_latency(0.15), Some(8));

        // 0.25ms = 250_000 ns, should select batch_size=32 (200_000ns < 250_000ns)
        assert_eq!(results.optimal_batch_for_latency(0.25), Some(32));
    }
}
