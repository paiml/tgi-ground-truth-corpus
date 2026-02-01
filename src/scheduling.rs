//! Request scheduling patterns.
//!
//! Fair scheduling and queue management patterns.
//!
//! # TGI Source
//!
//! Patterns derived from `backends/v3/src/block_allocator.rs`:
//! - Block-based KV cache allocation
//! - Memory pool management
//! - Fair request scheduling
//!
//! # Sovereign AI Stack Equivalent
//!
//! Maps to `realizar::schedule` for inference scheduling.

use crate::{Error, Result};
use std::collections::VecDeque;

/// Scheduling priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Priority {
    /// Low priority (background tasks).
    Low,
    /// Normal priority (default).
    #[default]
    Normal,
    /// High priority (interactive).
    High,
    /// Critical priority (system).
    Critical,
}

impl Priority {
    /// Get numeric value for comparison.
    pub const fn value(&self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::High => 2,
            Self::Critical => 3,
        }
    }
}

/// A scheduled task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledTask {
    /// Unique task ID.
    pub id: u64,

    /// Task priority.
    pub priority: Priority,

    /// Estimated tokens to process.
    pub estimated_tokens: usize,

    /// Whether task is preemptible.
    pub preemptible: bool,
}

impl ScheduledTask {
    /// Create a new task.
    pub fn new(id: u64, estimated_tokens: usize) -> Self {
        Self {
            id,
            priority: Priority::Normal,
            estimated_tokens,
            preemptible: true,
        }
    }

    /// Set priority.
    pub const fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set preemptible flag.
    pub const fn preemptible(mut self, preemptible: bool) -> Self {
        self.preemptible = preemptible;
        self
    }
}

/// Scheduler configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerConfig {
    /// Maximum tasks in queue.
    pub max_queue_size: usize,

    /// Enable priority scheduling.
    pub priority_scheduling: bool,

    /// Enable preemption.
    pub enable_preemption: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 1000,
            priority_scheduling: true,
            enable_preemption: true,
        }
    }
}

impl SchedulerConfig {
    /// Create with max queue size.
    pub const fn with_max_queue(max_queue_size: usize) -> Self {
        Self {
            max_queue_size,
            priority_scheduling: true,
            enable_preemption: true,
        }
    }
}

/// Request scheduler.
///
/// # Examples
///
/// ```rust
/// use tgi_gtc::scheduling::{Scheduler, SchedulerConfig, ScheduledTask, Priority};
///
/// let scheduler = Scheduler::new(SchedulerConfig::default());
///
/// // Schedule tasks
/// let task = ScheduledTask::new(1, 100).with_priority(Priority::High);
/// scheduler.schedule(task).unwrap();
///
/// // Get next task
/// if let Some(task) = scheduler.next() {
///     assert_eq!(task.id, 1);
/// }
/// ```
#[derive(Debug)]
pub struct Scheduler {
    config: SchedulerConfig,
    queue: std::sync::Mutex<VecDeque<ScheduledTask>>,
    next_id: std::sync::atomic::AtomicU64,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            queue: std::sync::Mutex::new(VecDeque::new()),
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Get configuration.
    pub const fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Generate next task ID.
    pub fn next_id(&self) -> u64 {
        self.next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    /// Schedule a task.
    pub fn schedule(&self, task: ScheduledTask) -> Result<()> {
        let mut queue = self.queue.lock().unwrap();

        if queue.len() >= self.config.max_queue_size {
            return Err(Error::resource_exhausted("scheduler queue full"));
        }

        if self.config.priority_scheduling {
            // Insert by priority (higher priority first)
            let pos = queue
                .iter()
                .position(|t| t.priority < task.priority)
                .unwrap_or(queue.len());
            queue.insert(pos, task);
        } else {
            queue.push_back(task);
        }

        Ok(())
    }

    /// Get next task to execute.
    pub fn next(&self) -> Option<ScheduledTask> {
        self.queue.lock().unwrap().pop_front()
    }

    /// Peek at next task without removing.
    pub fn peek(&self) -> Option<ScheduledTask> {
        self.queue.lock().unwrap().front().cloned()
    }

    /// Get queue length.
    pub fn len(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Check if queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.lock().unwrap().is_empty()
    }

    /// Clear all scheduled tasks.
    pub fn clear(&self) -> Vec<ScheduledTask> {
        self.queue.lock().unwrap().drain(..).collect()
    }

    /// Cancel a specific task by ID.
    pub fn cancel(&self, task_id: u64) -> Option<ScheduledTask> {
        let mut queue = self.queue.lock().unwrap();
        if let Some(pos) = queue.iter().position(|t| t.id == task_id) {
            queue.remove(pos)
        } else {
            None
        }
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new(SchedulerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_priority_value() {
        assert_eq!(Priority::Low.value(), 0);
        assert_eq!(Priority::Normal.value(), 1);
        assert_eq!(Priority::High.value(), 2);
        assert_eq!(Priority::Critical.value(), 3);
    }

    #[test]
    fn test_scheduled_task_creation() {
        let task = ScheduledTask::new(1, 100);
        assert_eq!(task.id, 1);
        assert_eq!(task.priority, Priority::Normal);
        assert_eq!(task.estimated_tokens, 100);
        assert!(task.preemptible);
    }

    #[test]
    fn test_scheduled_task_builder() {
        let task = ScheduledTask::new(1, 100)
            .with_priority(Priority::High)
            .preemptible(false);

        assert_eq!(task.priority, Priority::High);
        assert!(!task.preemptible);
    }

    #[test]
    fn test_scheduler_config_default() {
        let config = SchedulerConfig::default();
        assert_eq!(config.max_queue_size, 1000);
        assert!(config.priority_scheduling);
        assert!(config.enable_preemption);
    }

    #[test]
    fn test_scheduler_config_custom() {
        let config = SchedulerConfig::with_max_queue(500);
        assert_eq!(config.max_queue_size, 500);
    }

    #[test]
    fn test_scheduler_schedule_and_next() {
        let scheduler = Scheduler::default();

        scheduler.schedule(ScheduledTask::new(1, 100)).unwrap();
        scheduler.schedule(ScheduledTask::new(2, 200)).unwrap();

        assert_eq!(scheduler.len(), 2);

        let task = scheduler.next().unwrap();
        assert_eq!(task.id, 1);

        assert_eq!(scheduler.len(), 1);
    }

    #[test]
    fn test_scheduler_priority_ordering() {
        let scheduler = Scheduler::default();

        scheduler
            .schedule(ScheduledTask::new(1, 100).with_priority(Priority::Low))
            .unwrap();
        scheduler
            .schedule(ScheduledTask::new(2, 100).with_priority(Priority::High))
            .unwrap();
        scheduler
            .schedule(ScheduledTask::new(3, 100).with_priority(Priority::Normal))
            .unwrap();

        // High priority should come first
        assert_eq!(scheduler.next().unwrap().id, 2);
        assert_eq!(scheduler.next().unwrap().id, 3);
        assert_eq!(scheduler.next().unwrap().id, 1);
    }

    #[test]
    fn test_scheduler_peek() {
        let scheduler = Scheduler::default();

        scheduler.schedule(ScheduledTask::new(1, 100)).unwrap();

        let peeked = scheduler.peek().unwrap();
        assert_eq!(peeked.id, 1);
        assert_eq!(scheduler.len(), 1); // Still in queue
    }

    #[test]
    fn test_scheduler_queue_full() {
        let config = SchedulerConfig::with_max_queue(2);
        let scheduler = Scheduler::new(config);

        scheduler.schedule(ScheduledTask::new(1, 100)).unwrap();
        scheduler.schedule(ScheduledTask::new(2, 100)).unwrap();

        let result = scheduler.schedule(ScheduledTask::new(3, 100));
        assert!(result.is_err());
    }

    #[test]
    fn test_scheduler_clear() {
        let scheduler = Scheduler::default();

        scheduler.schedule(ScheduledTask::new(1, 100)).unwrap();
        scheduler.schedule(ScheduledTask::new(2, 200)).unwrap();

        let cleared = scheduler.clear();
        assert_eq!(cleared.len(), 2);
        assert!(scheduler.is_empty());
    }

    #[test]
    fn test_scheduler_cancel() {
        let scheduler = Scheduler::default();

        scheduler.schedule(ScheduledTask::new(1, 100)).unwrap();
        scheduler.schedule(ScheduledTask::new(2, 200)).unwrap();
        scheduler.schedule(ScheduledTask::new(3, 300)).unwrap();

        let cancelled = scheduler.cancel(2);
        assert!(cancelled.is_some());
        assert_eq!(cancelled.unwrap().id, 2);
        assert_eq!(scheduler.len(), 2);

        // Cancel non-existent
        assert!(scheduler.cancel(99).is_none());
    }

    #[test]
    fn test_scheduler_empty() {
        let scheduler = Scheduler::default();
        assert!(scheduler.is_empty());
        assert!(scheduler.next().is_none());
        assert!(scheduler.peek().is_none());
    }

    #[test]
    fn test_scheduler_next_id() {
        let scheduler = Scheduler::default();
        assert_eq!(scheduler.next_id(), 1);
        assert_eq!(scheduler.next_id(), 2);
        assert_eq!(scheduler.next_id(), 3);
    }

    #[test]
    fn test_scheduler_no_priority() {
        let config = SchedulerConfig {
            priority_scheduling: false,
            ..Default::default()
        };
        let scheduler = Scheduler::new(config);

        scheduler
            .schedule(ScheduledTask::new(1, 100).with_priority(Priority::Low))
            .unwrap();
        scheduler
            .schedule(ScheduledTask::new(2, 100).with_priority(Priority::High))
            .unwrap();

        // Without priority scheduling, FIFO order
        assert_eq!(scheduler.next().unwrap().id, 1);
        assert_eq!(scheduler.next().unwrap().id, 2);
    }
}
