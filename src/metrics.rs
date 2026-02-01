//! Metrics and Telemetry
//!
//! Implements Prometheus-style metrics for monitoring inference performance.
//!
//! # TGI Reference
//!
//! Based on TGI's metrics implementation.
//! See: <https://github.com/huggingface/text-generation-inference>
//!
//! # Example
//!
//! ```rust
//! use tgi_gtc::metrics::{Metrics, Counter, Histogram, Gauge};
//!
//! let mut metrics = Metrics::new();
//!
//! // Count requests
//! metrics.requests_total.inc();
//! metrics.requests_total.inc_by(5);
//!
//! // Record latency
//! metrics.request_duration.observe(0.125);
//!
//! // Set queue depth
//! metrics.queue_depth.set(42.0);
//!
//! // Export to Prometheus format
//! let output = metrics.export_prometheus();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// A counter metric (monotonically increasing).
#[derive(Debug, Default)]
pub struct Counter {
    value: AtomicU64,
    name: String,
    help: String,
    labels: HashMap<String, String>,
}

impl Counter {
    /// Create a new counter.
    pub fn new(name: &str, help: &str) -> Self {
        Self {
            value: AtomicU64::new(0),
            name: name.to_string(),
            help: help.to_string(),
            labels: HashMap::new(),
        }
    }

    /// Create with labels.
    pub fn with_labels(name: &str, help: &str, labels: HashMap<String, String>) -> Self {
        Self {
            value: AtomicU64::new(0),
            name: name.to_string(),
            help: help.to_string(),
            labels,
        }
    }

    /// Increment by 1.
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment by n.
    pub fn inc_by(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Get current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset to zero.
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }

    /// Export to Prometheus format.
    pub fn to_prometheus(&self) -> String {
        let labels_str = self.format_labels();
        format!(
            "# HELP {} {}\n# TYPE {} counter\n{}{} {}\n",
            self.name,
            self.help,
            self.name,
            self.name,
            labels_str,
            self.get()
        )
    }

    fn format_labels(&self) -> String {
        if self.labels.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = self
                .labels
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            format!("{{{}}}", pairs.join(","))
        }
    }
}

/// A gauge metric (can go up and down).
#[derive(Debug, Default)]
pub struct Gauge {
    value: AtomicU64, // Store as bits for f64
    name: String,
    help: String,
    labels: HashMap<String, String>,
}

impl Gauge {
    /// Create a new gauge.
    pub fn new(name: &str, help: &str) -> Self {
        Self {
            value: AtomicU64::new(0),
            name: name.to_string(),
            help: help.to_string(),
            labels: HashMap::new(),
        }
    }

    /// Set value.
    pub fn set(&self, v: f64) {
        self.value.store(v.to_bits(), Ordering::Relaxed);
    }

    /// Get current value.
    pub fn get(&self) -> f64 {
        f64::from_bits(self.value.load(Ordering::Relaxed))
    }

    /// Increment by n.
    pub fn inc(&self, n: f64) {
        let current = self.get();
        self.set(current + n);
    }

    /// Decrement by n.
    pub fn dec(&self, n: f64) {
        let current = self.get();
        self.set(current - n);
    }

    /// Export to Prometheus format.
    pub fn to_prometheus(&self) -> String {
        let labels_str = self.format_labels();
        format!(
            "# HELP {} {}\n# TYPE {} gauge\n{}{} {}\n",
            self.name,
            self.help,
            self.name,
            self.name,
            labels_str,
            self.get()
        )
    }

    fn format_labels(&self) -> String {
        if self.labels.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = self
                .labels
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            format!("{{{}}}", pairs.join(","))
        }
    }
}

/// A histogram metric with buckets.
#[derive(Debug)]
pub struct Histogram {
    name: String,
    help: String,
    buckets: Vec<f64>,
    counts: Vec<AtomicU64>,
    sum: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    /// Create a new histogram with default buckets.
    pub fn new(name: &str, help: &str) -> Self {
        Self::with_buckets(
            name,
            help,
            vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ],
        )
    }

    /// Create with custom buckets.
    pub fn with_buckets(name: &str, help: &str, buckets: Vec<f64>) -> Self {
        let counts = buckets.iter().map(|_| AtomicU64::new(0)).collect();
        Self {
            name: name.to_string(),
            help: help.to_string(),
            buckets,
            counts,
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Observe a value.
    pub fn observe(&self, v: f64) {
        // Update bucket counts
        for (i, &bucket) in self.buckets.iter().enumerate() {
            if v <= bucket {
                self.counts[i].fetch_add(1, Ordering::Relaxed);
            }
        }

        // Update sum and count
        let sum_bits = self.sum.load(Ordering::Relaxed);
        let current_sum = f64::from_bits(sum_bits);
        self.sum
            .store((current_sum + v).to_bits(), Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total count.
    pub fn get_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get sum.
    pub fn get_sum(&self) -> f64 {
        f64::from_bits(self.sum.load(Ordering::Relaxed))
    }

    /// Get mean.
    pub fn get_mean(&self) -> f64 {
        let count = self.get_count();
        if count == 0 {
            0.0
        } else {
            self.get_sum() / count as f64
        }
    }

    /// Export to Prometheus format.
    pub fn to_prometheus(&self) -> String {
        let mut output = format!(
            "# HELP {} {}\n# TYPE {} histogram\n",
            self.name, self.help, self.name
        );

        let mut cumulative = 0u64;
        for (i, &bucket) in self.buckets.iter().enumerate() {
            cumulative += self.counts[i].load(Ordering::Relaxed);
            output.push_str(&format!(
                "{}_bucket{{le=\"{}\"}} {}\n",
                self.name, bucket, cumulative
            ));
        }

        output.push_str(&format!(
            "{}_bucket{{le=\"+Inf\"}} {}\n",
            self.name,
            self.get_count()
        ));
        output.push_str(&format!("{}_sum {}\n", self.name, self.get_sum()));
        output.push_str(&format!("{}_count {}\n", self.name, self.get_count()));

        output
    }

    /// Reset all values.
    pub fn reset(&self) {
        for count in &self.counts {
            count.store(0, Ordering::Relaxed);
        }
        self.sum.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }
}

/// Collection of TGI-relevant metrics.
#[derive(Debug)]
pub struct Metrics {
    /// Total requests received.
    pub requests_total: Counter,

    /// Successful requests.
    pub requests_success: Counter,

    /// Failed requests.
    pub requests_failed: Counter,

    /// Request duration in seconds.
    pub request_duration: Histogram,

    /// Time to first token (TTFT) in seconds.
    pub time_to_first_token: Histogram,

    /// Time per output token in seconds.
    pub time_per_output_token: Histogram,

    /// Total tokens generated.
    pub tokens_generated: Counter,

    /// Total input tokens processed.
    pub tokens_input: Counter,

    /// Current queue depth.
    pub queue_depth: Gauge,

    /// Current batch size.
    pub batch_size: Gauge,

    /// GPU memory used in bytes.
    pub gpu_memory_used: Gauge,

    /// KV cache utilization (0-1).
    pub kv_cache_utilization: Gauge,

    /// Active requests.
    pub active_requests: Gauge,

    /// Prefill tokens per second.
    pub prefill_tokens_per_second: Gauge,

    /// Decode tokens per second.
    pub decode_tokens_per_second: Gauge,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// Create a new metrics collection.
    pub fn new() -> Self {
        Self {
            requests_total: Counter::new("tgi_requests_total", "Total number of requests received"),
            requests_success: Counter::new(
                "tgi_requests_success_total",
                "Total number of successful requests",
            ),
            requests_failed: Counter::new(
                "tgi_requests_failed_total",
                "Total number of failed requests",
            ),
            request_duration: Histogram::new(
                "tgi_request_duration_seconds",
                "Request duration in seconds",
            ),
            time_to_first_token: Histogram::with_buckets(
                "tgi_time_to_first_token_seconds",
                "Time to first token in seconds",
                vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
            ),
            time_per_output_token: Histogram::with_buckets(
                "tgi_time_per_output_token_seconds",
                "Time per output token in seconds",
                vec![0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
            ),
            tokens_generated: Counter::new(
                "tgi_tokens_generated_total",
                "Total number of tokens generated",
            ),
            tokens_input: Counter::new(
                "tgi_tokens_input_total",
                "Total number of input tokens processed",
            ),
            queue_depth: Gauge::new("tgi_queue_depth", "Current queue depth"),
            batch_size: Gauge::new("tgi_batch_size", "Current batch size"),
            gpu_memory_used: Gauge::new("tgi_gpu_memory_used_bytes", "GPU memory used in bytes"),
            kv_cache_utilization: Gauge::new(
                "tgi_kv_cache_utilization",
                "KV cache utilization (0-1)",
            ),
            active_requests: Gauge::new("tgi_active_requests", "Number of active requests"),
            prefill_tokens_per_second: Gauge::new(
                "tgi_prefill_tokens_per_second",
                "Prefill throughput in tokens per second",
            ),
            decode_tokens_per_second: Gauge::new(
                "tgi_decode_tokens_per_second",
                "Decode throughput in tokens per second",
            ),
        }
    }

    /// Export all metrics to Prometheus format.
    pub fn export_prometheus(&self) -> String {
        let mut output = String::new();

        output.push_str(&self.requests_total.to_prometheus());
        output.push_str(&self.requests_success.to_prometheus());
        output.push_str(&self.requests_failed.to_prometheus());
        output.push_str(&self.request_duration.to_prometheus());
        output.push_str(&self.time_to_first_token.to_prometheus());
        output.push_str(&self.time_per_output_token.to_prometheus());
        output.push_str(&self.tokens_generated.to_prometheus());
        output.push_str(&self.tokens_input.to_prometheus());
        output.push_str(&self.queue_depth.to_prometheus());
        output.push_str(&self.batch_size.to_prometheus());
        output.push_str(&self.gpu_memory_used.to_prometheus());
        output.push_str(&self.kv_cache_utilization.to_prometheus());
        output.push_str(&self.active_requests.to_prometheus());
        output.push_str(&self.prefill_tokens_per_second.to_prometheus());
        output.push_str(&self.decode_tokens_per_second.to_prometheus());

        output
    }

    /// Record a successful request.
    pub fn record_request_success(&self, duration_secs: f64, ttft_secs: f64, output_tokens: u64) {
        self.requests_total.inc();
        self.requests_success.inc();
        self.request_duration.observe(duration_secs);
        self.time_to_first_token.observe(ttft_secs);
        self.tokens_generated.inc_by(output_tokens);

        if output_tokens > 0 {
            let tpot = (duration_secs - ttft_secs) / output_tokens as f64;
            self.time_per_output_token.observe(tpot);
        }
    }

    /// Record a failed request.
    pub fn record_request_failure(&self) {
        self.requests_total.inc();
        self.requests_failed.inc();
    }

    /// Get success rate.
    pub fn success_rate(&self) -> f64 {
        let total = self.requests_total.get();
        if total == 0 {
            1.0
        } else {
            self.requests_success.get() as f64 / total as f64
        }
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        self.requests_total.reset();
        self.requests_success.reset();
        self.requests_failed.reset();
        self.request_duration.reset();
        self.time_to_first_token.reset();
        self.time_per_output_token.reset();
        self.tokens_generated.reset();
        self.tokens_input.reset();
    }
}

/// Summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    /// Total requests.
    pub total_requests: u64,
    /// Success rate (0-1).
    pub success_rate: f64,
    /// Mean request duration.
    pub mean_duration_ms: f64,
    /// Mean TTFT.
    pub mean_ttft_ms: f64,
    /// Tokens per second.
    pub tokens_per_second: f64,
    /// Current queue depth.
    pub queue_depth: f64,
    /// KV cache utilization.
    pub kv_cache_util: f64,
}

impl Metrics {
    /// Get summary statistics.
    pub fn summary(&self) -> MetricsSummary {
        let _duration_count = self.request_duration.get_count();
        let tokens = self.tokens_generated.get();
        let duration_sum = self.request_duration.get_sum();

        MetricsSummary {
            total_requests: self.requests_total.get(),
            success_rate: self.success_rate(),
            mean_duration_ms: self.request_duration.get_mean() * 1000.0,
            mean_ttft_ms: self.time_to_first_token.get_mean() * 1000.0,
            tokens_per_second: if duration_sum > 0.0 {
                tokens as f64 / duration_sum
            } else {
                0.0
            },
            queue_depth: self.queue_depth.get(),
            kv_cache_util: self.kv_cache_utilization.get(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let counter = Counter::new("test_counter", "A test counter");

        assert_eq!(counter.get(), 0);
        counter.inc();
        assert_eq!(counter.get(), 1);
        counter.inc_by(5);
        assert_eq!(counter.get(), 6);
    }

    #[test]
    fn test_counter_reset() {
        let counter = Counter::new("test", "test");
        counter.inc_by(100);
        counter.reset();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_gauge() {
        let gauge = Gauge::new("test_gauge", "A test gauge");

        gauge.set(42.5);
        assert_eq!(gauge.get(), 42.5);

        gauge.inc(7.5);
        assert_eq!(gauge.get(), 50.0);

        gauge.dec(10.0);
        assert_eq!(gauge.get(), 40.0);
    }

    #[test]
    fn test_histogram() {
        let hist = Histogram::new("test_hist", "A test histogram");

        hist.observe(0.05);
        hist.observe(0.15);
        hist.observe(0.5);

        assert_eq!(hist.get_count(), 3);
        assert!((hist.get_sum() - 0.7).abs() < 0.001);
        assert!((hist.get_mean() - 0.233).abs() < 0.01);
    }

    #[test]
    fn test_histogram_reset() {
        let hist = Histogram::new("test", "test");
        hist.observe(1.0);
        hist.observe(2.0);
        hist.reset();

        assert_eq!(hist.get_count(), 0);
        assert_eq!(hist.get_sum(), 0.0);
    }

    #[test]
    fn test_metrics_record_success() {
        let metrics = Metrics::new();

        metrics.record_request_success(1.0, 0.1, 100);

        assert_eq!(metrics.requests_total.get(), 1);
        assert_eq!(metrics.requests_success.get(), 1);
        assert_eq!(metrics.tokens_generated.get(), 100);
    }

    #[test]
    fn test_metrics_record_failure() {
        let metrics = Metrics::new();

        metrics.record_request_failure();

        assert_eq!(metrics.requests_total.get(), 1);
        assert_eq!(metrics.requests_failed.get(), 1);
    }

    #[test]
    fn test_metrics_success_rate() {
        let metrics = Metrics::new();

        metrics.record_request_success(1.0, 0.1, 10);
        metrics.record_request_success(1.0, 0.1, 10);
        metrics.record_request_failure();

        assert!((metrics.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_counter_prometheus_export() {
        let counter = Counter::new("test_requests", "Total requests");
        counter.inc_by(42);

        let output = counter.to_prometheus();
        assert!(output.contains("# TYPE test_requests counter"));
        assert!(output.contains("test_requests 42"));
    }

    #[test]
    fn test_histogram_prometheus_export() {
        let hist = Histogram::with_buckets("test_duration", "Duration", vec![0.1, 0.5, 1.0]);
        hist.observe(0.2);
        hist.observe(0.8);

        let output = hist.to_prometheus();
        assert!(output.contains("# TYPE test_duration histogram"));
        assert!(output.contains("test_duration_bucket"));
        assert!(output.contains("test_duration_sum"));
        assert!(output.contains("test_duration_count 2"));
    }

    #[test]
    fn test_metrics_summary() {
        let metrics = Metrics::new();

        metrics.record_request_success(1.0, 0.1, 100);
        metrics.queue_depth.set(5.0);
        metrics.kv_cache_utilization.set(0.75);

        let summary = metrics.summary();
        assert_eq!(summary.total_requests, 1);
        assert_eq!(summary.queue_depth, 5.0);
        assert_eq!(summary.kv_cache_util, 0.75);
    }

    #[test]
    fn test_counter_with_labels() {
        let mut labels = HashMap::new();
        labels.insert("method".to_string(), "POST".to_string());

        let counter = Counter::with_labels("requests", "Total requests", labels);
        counter.inc();

        let output = counter.to_prometheus();
        assert!(output.contains(r#"method="POST""#));
    }
}
