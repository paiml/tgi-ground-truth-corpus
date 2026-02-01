//! Continuous batching patterns.
//!
//! Production-ready batching patterns extracted from TGI's queue layer,
//! implementing continuous batching for optimal GPU utilization.
//!
//! # TGI Source
//!
//! Patterns derived from `backends/v3/src/queue.rs`:
//! - Continuous batching algorithm
//! - Prefill vs decode phase handling
//! - Dynamic batch formation
//! - Request prioritization
//!
//! # Sovereign AI Stack Equivalent
//!
//! Maps to `realizar::batch` for inference batching.
//!
//! # Key Concepts
//!
//! ## Continuous Batching
//!
//! Unlike static batching (wait for N requests, process together),
//! continuous batching:
//! 1. Starts inference immediately when requests arrive
//! 2. Dynamically adds new requests to running batches
//! 3. Removes completed sequences while others continue
//!
//! This achieves 2-4x higher throughput on GPU inference.
//!
//! ## Prefill vs Decode
//!
//! - **Prefill**: Process input tokens (compute-bound, parallelizable)
//! - **Decode**: Generate output tokens (memory-bound, sequential)
//!
//! TGI separates these phases for optimal scheduling.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Configuration for the continuous batcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchConfig {
    /// Maximum batch size.
    pub max_batch_size: usize,

    /// Maximum tokens per batch (prefill budget).
    pub max_batch_tokens: usize,

    /// Maximum waiting time before forcing batch.
    pub max_wait_ms: u64,

    /// Minimum batch size before processing.
    pub min_batch_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_batch_tokens: 4096,
            max_wait_ms: 50,
            min_batch_size: 1,
        }
    }
}

impl BatchConfig {
    /// Create a builder.
    pub fn builder() -> BatchConfigBuilder {
        BatchConfigBuilder::default()
    }
}

/// Builder for `BatchConfig`.
#[derive(Debug, Default)]
pub struct BatchConfigBuilder {
    max_batch_size: Option<usize>,
    max_batch_tokens: Option<usize>,
    max_wait_ms: Option<u64>,
    min_batch_size: Option<usize>,
}

impl BatchConfigBuilder {
    /// Set max batch size.
    pub const fn max_batch_size(mut self, value: usize) -> Self {
        self.max_batch_size = Some(value);
        self
    }

    /// Set max batch tokens.
    pub const fn max_batch_tokens(mut self, value: usize) -> Self {
        self.max_batch_tokens = Some(value);
        self
    }

    /// Set max wait time in milliseconds.
    pub const fn max_wait_ms(mut self, value: u64) -> Self {
        self.max_wait_ms = Some(value);
        self
    }

    /// Set minimum batch size.
    pub const fn min_batch_size(mut self, value: usize) -> Self {
        self.min_batch_size = Some(value);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> BatchConfig {
        let default = BatchConfig::default();
        BatchConfig {
            max_batch_size: self.max_batch_size.unwrap_or(default.max_batch_size),
            max_batch_tokens: self.max_batch_tokens.unwrap_or(default.max_batch_tokens),
            max_wait_ms: self.max_wait_ms.unwrap_or(default.max_wait_ms),
            min_batch_size: self.min_batch_size.unwrap_or(default.min_batch_size),
        }
    }
}

/// A request in the batch queue.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Unique request ID.
    pub id: u64,

    /// Input token count.
    pub input_tokens: usize,

    /// Maximum new tokens to generate.
    pub max_new_tokens: usize,

    /// When the request was queued.
    pub queued_at: Instant,

    /// Request priority (higher = more urgent).
    pub priority: u32,
}

impl BatchRequest {
    /// Create a new batch request.
    pub fn new(id: u64, input_tokens: usize, max_new_tokens: usize) -> Self {
        Self {
            id,
            input_tokens,
            max_new_tokens,
            queued_at: Instant::now(),
            priority: 0,
        }
    }

    /// Create with priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Time spent waiting in queue.
    pub fn wait_time(&self) -> Duration {
        self.queued_at.elapsed()
    }

    /// Total tokens (input + max output).
    pub const fn total_tokens(&self) -> usize {
        self.input_tokens + self.max_new_tokens
    }
}

/// A batch of requests ready for inference.
#[derive(Debug)]
pub struct Batch {
    /// Requests in this batch.
    pub requests: Vec<BatchRequest>,

    /// Total input tokens in batch.
    pub total_input_tokens: usize,

    /// When the batch was formed.
    pub formed_at: Instant,
}

impl Batch {
    /// Create a new batch from requests.
    pub fn new(requests: Vec<BatchRequest>) -> Self {
        let total_input_tokens = requests.iter().map(|r| r.input_tokens).sum();
        Self {
            requests,
            total_input_tokens,
            formed_at: Instant::now(),
        }
    }

    /// Number of requests in batch.
    pub fn size(&self) -> usize {
        self.requests.len()
    }

    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Average wait time for requests in batch.
    pub fn avg_wait_time(&self) -> Duration {
        if self.requests.is_empty() {
            return Duration::ZERO;
        }

        let total: Duration = self.requests.iter().map(|r| r.wait_time()).sum();
        total / self.requests.len() as u32
    }

    /// Maximum wait time in batch.
    pub fn max_wait_time(&self) -> Duration {
        self.requests
            .iter()
            .map(|r| r.wait_time())
            .max()
            .unwrap_or(Duration::ZERO)
    }
}

/// Continuous batcher for inference requests.
///
/// # TGI Source
///
/// Maps to `Queue` in `backends/v3/src/queue.rs`.
///
/// # Examples
///
/// ```rust
/// use tgi_gtc::batching::{ContinuousBatcher, BatchConfig, BatchRequest};
///
/// let batcher = ContinuousBatcher::new(BatchConfig::default());
///
/// // Add requests
/// batcher.add(BatchRequest::new(1, 100, 50));
/// batcher.add(BatchRequest::new(2, 200, 100));
///
/// // Form a batch
/// if let Some(batch) = batcher.try_form_batch() {
///     println!("Batch size: {}", batch.size());
/// }
/// ```
#[derive(Debug)]
pub struct ContinuousBatcher {
    config: BatchConfig,
    queue: std::sync::Mutex<VecDeque<BatchRequest>>,
    next_id: std::sync::atomic::AtomicU64,
}

impl ContinuousBatcher {
    /// Create a new batcher.
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            queue: std::sync::Mutex::new(VecDeque::new()),
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Get the configuration.
    pub const fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Generate next request ID.
    pub fn next_id(&self) -> u64 {
        self.next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    /// Add a request to the queue.
    pub fn add(&self, request: BatchRequest) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(request);
    }

    /// Get current queue length.
    pub fn queue_len(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Check if queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.lock().unwrap().is_empty()
    }

    /// Try to form a batch from queued requests.
    ///
    /// Returns `Some(Batch)` if enough requests are available,
    /// `None` if not enough requests or conditions not met.
    pub fn try_form_batch(&self) -> Option<Batch> {
        let mut queue = self.queue.lock().unwrap();

        if queue.is_empty() {
            return None;
        }

        // Check if we should form a batch
        let should_batch = queue.len() >= self.config.min_batch_size
            || queue
                .front()
                .map(|r| r.wait_time() >= Duration::from_millis(self.config.max_wait_ms))
                .unwrap_or(false);

        if !should_batch {
            return None;
        }

        // Collect requests up to limits
        let mut batch_requests = Vec::new();
        let mut total_tokens = 0;

        while let Some(request) = queue.front() {
            // Check batch size limit
            if batch_requests.len() >= self.config.max_batch_size {
                break;
            }

            // Check token budget
            if total_tokens + request.input_tokens > self.config.max_batch_tokens
                && !batch_requests.is_empty()
            {
                break;
            }

            let request = queue.pop_front().unwrap();
            total_tokens += request.input_tokens;
            batch_requests.push(request);
        }

        if batch_requests.is_empty() {
            None
        } else {
            Some(Batch::new(batch_requests))
        }
    }

    /// Force form a batch with all queued requests (up to limits).
    pub fn force_batch(&self) -> Option<Batch> {
        let mut queue = self.queue.lock().unwrap();

        if queue.is_empty() {
            return None;
        }

        let mut batch_requests = Vec::new();
        let mut total_tokens = 0;

        while let Some(request) = queue.front() {
            if batch_requests.len() >= self.config.max_batch_size {
                break;
            }

            if total_tokens + request.input_tokens > self.config.max_batch_tokens
                && !batch_requests.is_empty()
            {
                break;
            }

            let request = queue.pop_front().unwrap();
            total_tokens += request.input_tokens;
            batch_requests.push(request);
        }

        Some(Batch::new(batch_requests))
    }

    /// Clear all queued requests.
    ///
    /// Returns the cancelled requests.
    pub fn clear(&self) -> Vec<BatchRequest> {
        let mut queue = self.queue.lock().unwrap();
        queue.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.max_batch_tokens, 4096);
    }

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::builder()
            .max_batch_size(64)
            .max_batch_tokens(8192)
            .build();

        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.max_batch_tokens, 8192);
    }

    #[test]
    fn test_batch_request_creation() {
        let request = BatchRequest::new(1, 100, 50);
        assert_eq!(request.id, 1);
        assert_eq!(request.input_tokens, 100);
        assert_eq!(request.max_new_tokens, 50);
        assert_eq!(request.total_tokens(), 150);
    }

    #[test]
    fn test_batch_request_priority() {
        let request = BatchRequest::new(1, 100, 50).with_priority(10);
        assert_eq!(request.priority, 10);
    }

    #[test]
    fn test_batch_creation() {
        let requests = vec![
            BatchRequest::new(1, 100, 50),
            BatchRequest::new(2, 200, 100),
        ];

        let batch = Batch::new(requests);
        assert_eq!(batch.size(), 2);
        assert_eq!(batch.total_input_tokens, 300);
    }

    #[test]
    fn test_batcher_add_and_queue() {
        let batcher = ContinuousBatcher::new(BatchConfig::default());
        assert!(batcher.is_empty());

        batcher.add(BatchRequest::new(1, 100, 50));
        assert_eq!(batcher.queue_len(), 1);

        batcher.add(BatchRequest::new(2, 200, 100));
        assert_eq!(batcher.queue_len(), 2);
    }

    #[test]
    fn test_batcher_form_batch() {
        let config = BatchConfig::builder().min_batch_size(1).build();
        let batcher = ContinuousBatcher::new(config);

        batcher.add(BatchRequest::new(1, 100, 50));
        batcher.add(BatchRequest::new(2, 200, 100));

        let batch = batcher.try_form_batch().unwrap();
        assert_eq!(batch.size(), 2);
        assert!(batcher.is_empty());
    }

    #[test]
    fn test_batcher_respects_batch_size_limit() {
        let config = BatchConfig::builder()
            .max_batch_size(2)
            .min_batch_size(1)
            .build();
        let batcher = ContinuousBatcher::new(config);

        batcher.add(BatchRequest::new(1, 100, 50));
        batcher.add(BatchRequest::new(2, 100, 50));
        batcher.add(BatchRequest::new(3, 100, 50));

        let batch = batcher.try_form_batch().unwrap();
        assert_eq!(batch.size(), 2);
        assert_eq!(batcher.queue_len(), 1);
    }

    #[test]
    fn test_batcher_respects_token_limit() {
        let config = BatchConfig::builder()
            .max_batch_tokens(250)
            .min_batch_size(1)
            .build();
        let batcher = ContinuousBatcher::new(config);

        batcher.add(BatchRequest::new(1, 100, 50));
        batcher.add(BatchRequest::new(2, 100, 50));
        batcher.add(BatchRequest::new(3, 100, 50));

        let batch = batcher.try_form_batch().unwrap();
        assert_eq!(batch.size(), 2); // 100 + 100 = 200 <= 250
        assert_eq!(batcher.queue_len(), 1);
    }

    #[test]
    fn test_batcher_min_batch_size() {
        let config = BatchConfig::builder()
            .min_batch_size(3)
            .max_wait_ms(1000000) // Very long wait
            .build();
        let batcher = ContinuousBatcher::new(config);

        batcher.add(BatchRequest::new(1, 100, 50));
        batcher.add(BatchRequest::new(2, 100, 50));

        // Not enough requests
        assert!(batcher.try_form_batch().is_none());

        batcher.add(BatchRequest::new(3, 100, 50));

        // Now enough
        let batch = batcher.try_form_batch().unwrap();
        assert_eq!(batch.size(), 3);
    }

    #[test]
    fn test_batcher_force_batch() {
        let config = BatchConfig::builder()
            .min_batch_size(10) // High minimum
            .build();
        let batcher = ContinuousBatcher::new(config);

        batcher.add(BatchRequest::new(1, 100, 50));

        // Normal try should fail
        assert!(batcher.try_form_batch().is_none());

        // Force should work
        let batch = batcher.force_batch().unwrap();
        assert_eq!(batch.size(), 1);
    }

    #[test]
    fn test_batcher_clear() {
        let batcher = ContinuousBatcher::new(BatchConfig::default());

        batcher.add(BatchRequest::new(1, 100, 50));
        batcher.add(BatchRequest::new(2, 100, 50));

        let cancelled = batcher.clear();
        assert_eq!(cancelled.len(), 2);
        assert!(batcher.is_empty());
    }

    #[test]
    fn test_batcher_next_id() {
        let batcher = ContinuousBatcher::new(BatchConfig::default());

        assert_eq!(batcher.next_id(), 1);
        assert_eq!(batcher.next_id(), 2);
        assert_eq!(batcher.next_id(), 3);
    }

    #[test]
    fn test_batch_empty() {
        let batch = Batch::new(vec![]);
        assert!(batch.is_empty());
        assert_eq!(batch.size(), 0);
        assert_eq!(batch.total_input_tokens, 0);
        assert_eq!(batch.avg_wait_time(), std::time::Duration::ZERO);
        assert_eq!(batch.max_wait_time(), std::time::Duration::ZERO);
    }

    #[test]
    fn test_batch_wait_times() {
        let requests = vec![
            BatchRequest::new(1, 100, 50),
            BatchRequest::new(2, 200, 100),
        ];

        let batch = Batch::new(requests);
        // Wait times should be very small since just created
        assert!(batch.avg_wait_time() < std::time::Duration::from_millis(100));
        assert!(batch.max_wait_time() < std::time::Duration::from_millis(100));
    }

    #[test]
    fn test_batch_request_wait_time() {
        let request = BatchRequest::new(1, 100, 50);
        // Just created, wait time should be tiny
        assert!(request.wait_time() < std::time::Duration::from_millis(100));
    }

    #[test]
    fn test_batch_config_builder_defaults() {
        let config = BatchConfig::builder().build();
        let default = BatchConfig::default();

        assert_eq!(config.max_batch_size, default.max_batch_size);
        assert_eq!(config.max_batch_tokens, default.max_batch_tokens);
        assert_eq!(config.max_wait_ms, default.max_wait_ms);
        assert_eq!(config.min_batch_size, default.min_batch_size);
    }

    #[test]
    fn test_batch_config_builder_all_fields() {
        let config = BatchConfig::builder()
            .max_batch_size(16)
            .max_batch_tokens(2048)
            .max_wait_ms(100)
            .min_batch_size(4)
            .build();

        assert_eq!(config.max_batch_size, 16);
        assert_eq!(config.max_batch_tokens, 2048);
        assert_eq!(config.max_wait_ms, 100);
        assert_eq!(config.min_batch_size, 4);
    }

    #[test]
    fn test_batcher_config_access() {
        let config = BatchConfig::builder().max_batch_size(64).build();
        let batcher = ContinuousBatcher::new(config);

        assert_eq!(batcher.config().max_batch_size, 64);
    }

    #[test]
    fn test_batcher_empty_try_form_batch() {
        let batcher = ContinuousBatcher::new(BatchConfig::default());
        assert!(batcher.try_form_batch().is_none());
    }

    #[test]
    fn test_batcher_empty_force_batch() {
        let batcher = ContinuousBatcher::new(BatchConfig::default());
        assert!(batcher.force_batch().is_none());
    }

    #[test]
    fn test_force_batch_respects_size_limit() {
        let config = BatchConfig::builder()
            .max_batch_size(2)
            .min_batch_size(1)
            .build();
        let batcher = ContinuousBatcher::new(config);

        batcher.add(BatchRequest::new(1, 100, 50));
        batcher.add(BatchRequest::new(2, 100, 50));
        batcher.add(BatchRequest::new(3, 100, 50));

        let batch = batcher.force_batch().unwrap();
        assert_eq!(batch.size(), 2);
        assert_eq!(batcher.queue_len(), 1);
    }

    #[test]
    fn test_force_batch_respects_token_limit() {
        let config = BatchConfig::builder()
            .max_batch_tokens(150)
            .max_batch_size(100)
            .min_batch_size(1)
            .build();
        let batcher = ContinuousBatcher::new(config);

        batcher.add(BatchRequest::new(1, 100, 50));
        batcher.add(BatchRequest::new(2, 100, 50));

        let batch = batcher.force_batch().unwrap();
        assert_eq!(batch.size(), 1);
        assert_eq!(batcher.queue_len(), 1);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_batch_total_tokens(input in 1usize..1000, max_new in 1usize..1000) {
            let request = BatchRequest::new(1, input, max_new);
            prop_assert_eq!(request.total_tokens(), input + max_new);
        }

        #[test]
        fn prop_batch_size_respects_limit(max_size in 1usize..50, num_requests in 1usize..100) {
            let config = BatchConfig::builder()
                .max_batch_size(max_size)
                .max_batch_tokens(1000000) // High limit
                .min_batch_size(1)
                .build();
            let batcher = ContinuousBatcher::new(config);

            for i in 0..num_requests {
                batcher.add(BatchRequest::new(i as u64, 10, 10));
            }

            if let Some(batch) = batcher.try_form_batch() {
                prop_assert!(batch.size() <= max_size);
            }
        }

        #[test]
        fn prop_batch_tokens_respects_limit(max_tokens in 100usize..10000, num_requests in 1usize..50) {
            let config = BatchConfig::builder()
                .max_batch_size(1000) // High limit
                .max_batch_tokens(max_tokens)
                .min_batch_size(1)
                .build();
            let batcher = ContinuousBatcher::new(config);

            for i in 0..num_requests {
                batcher.add(BatchRequest::new(i as u64, 50, 50));
            }

            if let Some(batch) = batcher.try_form_batch() {
                prop_assert!(batch.total_input_tokens <= max_tokens);
            }
        }
    }
}
