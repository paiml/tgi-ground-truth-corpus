//! HTTP router patterns.
//!
//! Production-ready HTTP routing patterns extracted from TGI's router layer,
//! implementing OpenAI-compatible endpoints and health checks.
//!
//! # TGI Source
//!
//! Patterns derived from `router/src/server.rs`:
//! - Axum-based HTTP server
//! - OpenAI-compatible `/v1/chat/completions` endpoint
//! - Health and readiness probes
//! - Metrics endpoint
//!
//! # Sovereign AI Stack Equivalent
//!
//! Maps to `realizar::serve` for model serving infrastructure.
//!
//! # Examples
//!
//! ```rust
//! use tgi_gtc::router::{RouterConfig, Router};
//!
//! let config = RouterConfig::builder()
//!     .port(8080)
//!     .max_concurrent_requests(128)
//!     .build();
//!
//! let router = Router::new(config);
//! router.set_ready(true);
//! assert!(router.is_ready());
//! ```

use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Router configuration.
///
/// # TGI Source
///
/// Maps to router configuration in `router/src/main.rs` CLI args.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterConfig {
    /// HTTP port to listen on.
    pub port: u16,

    /// Maximum concurrent requests.
    pub max_concurrent_requests: usize,

    /// Maximum batch size for inference.
    pub max_batch_size: usize,

    /// Request timeout in seconds.
    pub timeout_secs: u64,

    /// Whether to enable OpenAI-compatible API.
    pub openai_compat: bool,

    /// Whether to enable metrics endpoint.
    pub enable_metrics: bool,

    /// Hostname for binding.
    pub hostname: String,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            max_concurrent_requests: 128,
            max_batch_size: 32,
            timeout_secs: 60,
            openai_compat: true,
            enable_metrics: true,
            hostname: "0.0.0.0".to_string(),
        }
    }
}

impl RouterConfig {
    /// Create a new builder.
    pub fn builder() -> RouterConfigBuilder {
        RouterConfigBuilder::default()
    }

    /// Get the bind address.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tgi_gtc::router::RouterConfig;
    ///
    /// let config = RouterConfig::default();
    /// assert_eq!(config.bind_address(), "0.0.0.0:8080");
    /// ```
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.hostname, self.port)
    }
}

/// Builder for `RouterConfig`.
#[derive(Debug, Default)]
pub struct RouterConfigBuilder {
    port: Option<u16>,
    max_concurrent_requests: Option<usize>,
    max_batch_size: Option<usize>,
    timeout_secs: Option<u64>,
    openai_compat: Option<bool>,
    enable_metrics: Option<bool>,
    hostname: Option<String>,
}

impl RouterConfigBuilder {
    /// Set the port.
    pub const fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Set max concurrent requests.
    pub const fn max_concurrent_requests(mut self, value: usize) -> Self {
        self.max_concurrent_requests = Some(value);
        self
    }

    /// Set max batch size.
    pub const fn max_batch_size(mut self, value: usize) -> Self {
        self.max_batch_size = Some(value);
        self
    }

    /// Set request timeout.
    pub const fn timeout_secs(mut self, value: u64) -> Self {
        self.timeout_secs = Some(value);
        self
    }

    /// Enable/disable OpenAI compatibility.
    pub const fn openai_compat(mut self, value: bool) -> Self {
        self.openai_compat = Some(value);
        self
    }

    /// Enable/disable metrics.
    pub const fn enable_metrics(mut self, value: bool) -> Self {
        self.enable_metrics = Some(value);
        self
    }

    /// Set hostname.
    pub fn hostname(mut self, value: impl Into<String>) -> Self {
        self.hostname = Some(value.into());
        self
    }

    /// Build the configuration.
    pub fn build(self) -> RouterConfig {
        let default = RouterConfig::default();
        RouterConfig {
            port: self.port.unwrap_or(default.port),
            max_concurrent_requests: self
                .max_concurrent_requests
                .unwrap_or(default.max_concurrent_requests),
            max_batch_size: self.max_batch_size.unwrap_or(default.max_batch_size),
            timeout_secs: self.timeout_secs.unwrap_or(default.timeout_secs),
            openai_compat: self.openai_compat.unwrap_or(default.openai_compat),
            enable_metrics: self.enable_metrics.unwrap_or(default.enable_metrics),
            hostname: self.hostname.unwrap_or(default.hostname),
        }
    }
}

/// Router state for tracking active requests.
///
/// # TGI Source
///
/// Maps to shared state in `router/src/server.rs`.
#[derive(Debug)]
pub struct RouterState {
    /// Whether the router is ready to accept requests.
    ready: AtomicBool,

    /// Current number of active requests.
    active_requests: AtomicUsize,

    /// Total requests received.
    total_requests: AtomicUsize,

    /// Total requests completed.
    completed_requests: AtomicUsize,

    /// Total requests failed.
    failed_requests: AtomicUsize,

    /// Maximum concurrent requests allowed.
    max_concurrent: usize,
}

impl RouterState {
    /// Create new router state.
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            ready: AtomicBool::new(false),
            active_requests: AtomicUsize::new(0),
            total_requests: AtomicUsize::new(0),
            completed_requests: AtomicUsize::new(0),
            failed_requests: AtomicUsize::new(0),
            max_concurrent,
        }
    }

    /// Mark the router as ready.
    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::SeqCst);
    }

    /// Check if router is ready.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    /// Check if router can accept more requests.
    pub fn can_accept(&self) -> bool {
        self.is_ready() && self.active_requests.load(Ordering::SeqCst) < self.max_concurrent
    }

    /// Try to acquire a request slot.
    ///
    /// Returns `Ok(RequestGuard)` if slot acquired, `Err` if at capacity.
    pub fn try_acquire(&self) -> Result<RequestGuard<'_>> {
        if !self.is_ready() {
            return Err(Error::scheduling("router not ready"));
        }

        let current = self.active_requests.fetch_add(1, Ordering::SeqCst);
        if current >= self.max_concurrent {
            self.active_requests.fetch_sub(1, Ordering::SeqCst);
            return Err(Error::resource_exhausted("max concurrent requests reached"));
        }

        self.total_requests.fetch_add(1, Ordering::SeqCst);
        Ok(RequestGuard { state: self })
    }

    /// Get current active request count.
    pub fn active_count(&self) -> usize {
        self.active_requests.load(Ordering::SeqCst)
    }

    /// Get total request count.
    pub fn total_count(&self) -> usize {
        self.total_requests.load(Ordering::SeqCst)
    }

    /// Get completed request count.
    pub fn completed_count(&self) -> usize {
        self.completed_requests.load(Ordering::SeqCst)
    }

    /// Get failed request count.
    pub fn failed_count(&self) -> usize {
        self.failed_requests.load(Ordering::SeqCst)
    }

    /// Record a completed request.
    fn record_completed(&self) {
        self.completed_requests.fetch_add(1, Ordering::SeqCst);
    }

    /// Record a failed request.
    fn record_failed(&self) {
        self.failed_requests.fetch_add(1, Ordering::SeqCst);
    }

    /// Release a request slot.
    fn release(&self) {
        self.active_requests.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Guard that releases request slot on drop.
///
/// Ensures request slots are properly released even on panic.
#[derive(Debug)]
pub struct RequestGuard<'a> {
    state: &'a RouterState,
}

impl<'a> RequestGuard<'a> {
    /// Mark request as completed successfully.
    pub fn complete(self) {
        self.state.record_completed();
        // Drop will release the slot
    }

    /// Mark request as failed.
    pub fn fail(self) {
        self.state.record_failed();
        // Drop will release the slot
    }
}

impl Drop for RequestGuard<'_> {
    fn drop(&mut self) {
        self.state.release();
    }
}

/// HTTP router for inference requests.
///
/// # TGI Source
///
/// Maps to the Axum router in `router/src/server.rs`.
///
/// # Examples
///
/// ```rust
/// use tgi_gtc::router::{Router, RouterConfig};
///
/// let router = Router::new(RouterConfig::default());
/// router.set_ready(true);
/// assert!(router.is_ready());
/// ```
#[derive(Debug)]
pub struct Router {
    config: RouterConfig,
    state: Arc<RouterState>,
}

impl Router {
    /// Create a new router.
    pub fn new(config: RouterConfig) -> Self {
        let state = Arc::new(RouterState::new(config.max_concurrent_requests));
        Self { config, state }
    }

    /// Get the configuration.
    pub const fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Get the state.
    pub fn state(&self) -> &RouterState {
        &self.state
    }

    /// Set router ready state.
    pub fn set_ready(&self, ready: bool) {
        self.state.set_ready(ready);
    }

    /// Check if router is ready.
    pub fn is_ready(&self) -> bool {
        self.state.is_ready()
    }

    /// Check if router can accept requests.
    pub fn can_accept(&self) -> bool {
        self.state.can_accept()
    }

    /// Try to acquire a request slot.
    pub fn try_acquire(&self) -> Result<RequestGuard<'_>> {
        self.state.try_acquire()
    }

    /// Get health status.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tgi_gtc::router::{Router, RouterConfig, HealthStatus};
    ///
    /// let router = Router::new(RouterConfig::default());
    /// let health = router.health();
    /// assert_eq!(health.status, "starting");
    ///
    /// router.set_ready(true);
    /// let health = router.health();
    /// assert_eq!(health.status, "healthy");
    /// ```
    pub fn health(&self) -> HealthStatus {
        HealthStatus {
            status: if self.is_ready() {
                "healthy"
            } else {
                "starting"
            }
            .to_string(),
            active_requests: self.state.active_count(),
            max_concurrent_requests: self.config.max_concurrent_requests,
        }
    }

    /// Get metrics.
    pub fn metrics(&self) -> RouterMetrics {
        RouterMetrics {
            total_requests: self.state.total_count(),
            active_requests: self.state.active_count(),
            completed_requests: self.state.completed_count(),
            failed_requests: self.state.failed_count(),
            max_concurrent_requests: self.config.max_concurrent_requests,
        }
    }
}

/// Health status response.
///
/// # TGI Source
///
/// Maps to health endpoint response in `router/src/server.rs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthStatus {
    /// Status string ("healthy", "starting", "unhealthy").
    pub status: String,

    /// Current active requests.
    pub active_requests: usize,

    /// Maximum concurrent requests.
    pub max_concurrent_requests: usize,
}

/// Router metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterMetrics {
    /// Total requests received.
    pub total_requests: usize,

    /// Currently active requests.
    pub active_requests: usize,

    /// Successfully completed requests.
    pub completed_requests: usize,

    /// Failed requests.
    pub failed_requests: usize,

    /// Maximum concurrent requests.
    pub max_concurrent_requests: usize,
}

impl RouterMetrics {
    /// Calculate success rate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tgi_gtc::router::RouterMetrics;
    ///
    /// let metrics = RouterMetrics {
    ///     total_requests: 100,
    ///     active_requests: 5,
    ///     completed_requests: 90,
    ///     failed_requests: 5,
    ///     max_concurrent_requests: 128,
    /// };
    /// assert!((metrics.success_rate() - 0.947).abs() < 0.01);
    /// ```
    pub fn success_rate(&self) -> f64 {
        let finished = self.completed_requests + self.failed_requests;
        if finished == 0 {
            1.0
        } else {
            self.completed_requests as f64 / finished as f64
        }
    }

    /// Calculate utilization.
    pub fn utilization(&self) -> f64 {
        self.active_requests as f64 / self.max_concurrent_requests as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_config_default() {
        let config = RouterConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.max_concurrent_requests, 128);
        assert!(config.openai_compat);
    }

    #[test]
    fn test_router_config_builder() {
        let config = RouterConfig::builder()
            .port(3000)
            .max_concurrent_requests(64)
            .openai_compat(false)
            .build();

        assert_eq!(config.port, 3000);
        assert_eq!(config.max_concurrent_requests, 64);
        assert!(!config.openai_compat);
    }

    #[test]
    fn test_router_config_bind_address() {
        let config = RouterConfig::builder()
            .hostname("127.0.0.1")
            .port(9000)
            .build();

        assert_eq!(config.bind_address(), "127.0.0.1:9000");
    }

    #[test]
    fn test_router_creation() {
        let router = Router::new(RouterConfig::default());
        assert!(!router.is_ready());
        assert!(!router.can_accept());
    }

    #[test]
    fn test_router_ready_state() {
        let router = Router::new(RouterConfig::default());
        assert!(!router.is_ready());

        router.set_ready(true);
        assert!(router.is_ready());
        assert!(router.can_accept());

        router.set_ready(false);
        assert!(!router.is_ready());
    }

    #[test]
    fn test_router_acquire_release() {
        let config = RouterConfig::builder().max_concurrent_requests(2).build();
        let router = Router::new(config);
        router.set_ready(true);

        // Acquire first slot
        let guard1 = router.try_acquire().unwrap();
        assert_eq!(router.state().active_count(), 1);

        // Acquire second slot
        let guard2 = router.try_acquire().unwrap();
        assert_eq!(router.state().active_count(), 2);

        // Third should fail
        assert!(router.try_acquire().is_err());

        // Release one
        drop(guard1);
        assert_eq!(router.state().active_count(), 1);

        // Now can acquire again
        let _guard3 = router.try_acquire().unwrap();
        assert_eq!(router.state().active_count(), 2);

        drop(guard2);
    }

    #[test]
    fn test_router_not_ready_rejects() {
        let router = Router::new(RouterConfig::default());
        // Not ready
        assert!(router.try_acquire().is_err());
    }

    #[test]
    fn test_request_guard_complete() {
        let config = RouterConfig::builder().max_concurrent_requests(10).build();
        let router = Router::new(config);
        router.set_ready(true);

        let guard = router.try_acquire().unwrap();
        guard.complete();

        assert_eq!(router.state().completed_count(), 1);
        assert_eq!(router.state().failed_count(), 0);
        assert_eq!(router.state().active_count(), 0);
    }

    #[test]
    fn test_request_guard_fail() {
        let config = RouterConfig::builder().max_concurrent_requests(10).build();
        let router = Router::new(config);
        router.set_ready(true);

        let guard = router.try_acquire().unwrap();
        guard.fail();

        assert_eq!(router.state().completed_count(), 0);
        assert_eq!(router.state().failed_count(), 1);
        assert_eq!(router.state().active_count(), 0);
    }

    #[test]
    fn test_health_status() {
        let router = Router::new(RouterConfig::default());

        let health = router.health();
        assert_eq!(health.status, "starting");

        router.set_ready(true);
        let health = router.health();
        assert_eq!(health.status, "healthy");
    }

    #[test]
    fn test_metrics() {
        let config = RouterConfig::builder().max_concurrent_requests(10).build();
        let router = Router::new(config);
        router.set_ready(true);

        // Complete some requests
        router.try_acquire().unwrap().complete();
        router.try_acquire().unwrap().complete();
        router.try_acquire().unwrap().fail();

        let metrics = router.metrics();
        assert_eq!(metrics.total_requests, 3);
        assert_eq!(metrics.completed_requests, 2);
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.active_requests, 0);
    }

    #[test]
    fn test_metrics_success_rate() {
        let metrics = RouterMetrics {
            total_requests: 100,
            active_requests: 0,
            completed_requests: 80,
            failed_requests: 20,
            max_concurrent_requests: 128,
        };

        assert!((metrics.success_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_metrics_utilization() {
        let metrics = RouterMetrics {
            total_requests: 100,
            active_requests: 32,
            completed_requests: 68,
            failed_requests: 0,
            max_concurrent_requests: 128,
        };

        assert!((metrics.utilization() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_router_config_builder_all_fields() {
        let config = RouterConfig::builder()
            .port(9000)
            .max_concurrent_requests(256)
            .max_batch_size(64)
            .timeout_secs(120)
            .openai_compat(false)
            .enable_metrics(false)
            .hostname("localhost")
            .build();

        assert_eq!(config.port, 9000);
        assert_eq!(config.max_concurrent_requests, 256);
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.timeout_secs, 120);
        assert!(!config.openai_compat);
        assert!(!config.enable_metrics);
        assert_eq!(config.hostname, "localhost");
    }

    #[test]
    fn test_router_state_total_count() {
        let state = RouterState::new(10);
        state.set_ready(true);

        // Simulate some requests
        let _guard1 = state.try_acquire().unwrap();
        let _guard2 = state.try_acquire().unwrap();

        assert_eq!(state.total_count(), 2);
    }

    #[test]
    fn test_metrics_success_rate_no_requests() {
        let metrics = RouterMetrics {
            total_requests: 0,
            active_requests: 0,
            completed_requests: 0,
            failed_requests: 0,
            max_concurrent_requests: 128,
        };

        // Should return 1.0 (100% success) when no requests
        assert!((metrics.success_rate() - 1.0).abs() < 0.001);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_config_builder_preserves_values(
            port in 1u16..65535,
            max_concurrent in 1usize..1000
        ) {
            let config = RouterConfig::builder()
                .port(port)
                .max_concurrent_requests(max_concurrent)
                .build();

            prop_assert_eq!(config.port, port);
            prop_assert_eq!(config.max_concurrent_requests, max_concurrent);
        }

        #[test]
        fn prop_acquire_respects_limit(max_concurrent in 1usize..100) {
            let config = RouterConfig::builder()
                .max_concurrent_requests(max_concurrent)
                .build();
            let router = Router::new(config);
            router.set_ready(true);

            let mut guards = Vec::new();
            for _ in 0..max_concurrent {
                guards.push(router.try_acquire().unwrap());
            }

            // Next acquire should fail
            prop_assert!(router.try_acquire().is_err());

            // After dropping one, should succeed
            drop(guards.pop());
            prop_assert!(router.try_acquire().is_ok());
        }

        #[test]
        fn prop_success_rate_bounded(completed in 0usize..1000, failed in 0usize..1000) {
            let metrics = RouterMetrics {
                total_requests: completed + failed,
                active_requests: 0,
                completed_requests: completed,
                failed_requests: failed,
                max_concurrent_requests: 128,
            };

            let rate = metrics.success_rate();
            prop_assert!(rate >= 0.0);
            prop_assert!(rate <= 1.0);
        }
    }
}
