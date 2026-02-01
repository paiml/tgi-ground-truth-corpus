//! LoRA Adapters
//!
//! Implements Low-Rank Adaptation (LoRA) for efficient model fine-tuning
//! and runtime adapter loading/switching.
//!
//! # TGI Reference
//!
//! Based on TGI's LoRA adapter support.
//! See: <https://github.com/huggingface/text-generation-inference>
//!
//! # Algorithm
//!
//! LoRA decomposes weight updates as: W' = W + BA
//! - W: Original frozen weights (d × k)
//! - B: Low-rank matrix (d × r)
//! - A: Low-rank matrix (r × k)
//! - r: Rank (typically 8-64)
//!
//! # Example
//!
//! ```rust
//! use tgi_gtc::lora::{LoraAdapter, LoraConfig, AdapterManager};
//!
//! let config = LoraConfig::default();
//! let adapter = LoraAdapter::new("my-adapter", config);
//!
//! println!("Adapter rank: {}", adapter.config().rank);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for LoRA adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Rank of the low-rank matrices.
    pub rank: usize,

    /// Alpha scaling factor.
    pub alpha: f32,

    /// Dropout probability.
    pub dropout: f32,

    /// Target modules to apply LoRA.
    pub target_modules: Vec<String>,

    /// Whether to merge weights on load.
    pub merge_on_load: bool,

    /// Scaling factor (alpha / rank).
    pub scaling: f32,
}

impl Default for LoraConfig {
    fn default() -> Self {
        let rank = 8;
        let alpha = 16.0;
        Self {
            rank,
            alpha,
            dropout: 0.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            merge_on_load: false,
            scaling: alpha / rank as f32,
        }
    }
}

impl LoraConfig {
    /// Create a config with custom rank.
    pub fn with_rank(rank: usize) -> Self {
        let alpha = (rank * 2) as f32;
        Self {
            rank,
            alpha,
            scaling: alpha / rank as f32,
            ..Default::default()
        }
    }

    /// Create a config for QLoRA (quantized LoRA).
    pub fn qlora(rank: usize) -> Self {
        Self {
            rank,
            alpha: rank as f32,
            dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            scaling: 1.0,
            ..Default::default()
        }
    }

    /// Calculate the scaling factor.
    pub fn get_scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

/// Low-rank weight matrices for a single layer.
#[derive(Debug, Clone)]
pub struct LoraWeights {
    /// A matrix (r × k).
    pub a: Vec<Vec<f32>>,

    /// B matrix (d × r).
    pub b: Vec<Vec<f32>>,

    /// Scaling factor.
    pub scaling: f32,
}

impl LoraWeights {
    /// Create new LoRA weights.
    pub fn new(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, scaling: f32) -> Self {
        Self { a, b, scaling }
    }

    /// Create random LoRA weights for testing.
    pub fn random(d: usize, k: usize, r: usize, scaling: f32) -> Self {
        let a = (0..r)
            .map(|i| {
                (0..k)
                    .map(|j| ((i * k + j) % 100) as f32 / 1000.0)
                    .collect()
            })
            .collect();
        let b = (0..d)
            .map(|i| {
                (0..r)
                    .map(|j| ((i * r + j) % 100) as f32 / 1000.0)
                    .collect()
            })
            .collect();
        Self { a, b, scaling }
    }

    /// Get dimensions (d, k, r).
    pub fn dims(&self) -> (usize, usize, usize) {
        let d = self.b.len();
        let r = if self.b.is_empty() {
            0
        } else {
            self.b[0].len()
        };
        let k = if self.a.is_empty() {
            0
        } else {
            self.a[0].len()
        };
        (d, k, r)
    }

    /// Compute LoRA output: (B @ A) @ x * scaling.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let (d, k, r) = self.dims();

        if x.len() != k {
            return vec![0.0; d];
        }

        // Compute A @ x (r × 1)
        let mut ax = vec![0.0; r];
        for i in 0..r {
            for j in 0..k {
                ax[i] += self.a[i][j] * x[j];
            }
        }

        // Compute B @ ax (d × 1)
        let mut result = vec![0.0; d];
        for i in 0..d {
            for j in 0..r {
                result[i] += self.b[i][j] * ax[j];
            }
            result[i] *= self.scaling;
        }

        result
    }

    /// Merge LoRA weights into base weights: W' = W + B @ A * scaling.
    pub fn merge_into(&self, base: &mut [Vec<f32>]) {
        let (d, k, r) = self.dims();

        if base.len() != d || (d > 0 && base[0].len() != k) {
            return;
        }

        // Compute B @ A and add to base
        for i in 0..d {
            for j in 0..k {
                let mut delta = 0.0;
                for l in 0..r {
                    delta += self.b[i][l] * self.a[l][j];
                }
                base[i][j] += delta * self.scaling;
            }
        }
    }
}

/// A LoRA adapter containing weights for multiple layers.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Adapter name/ID.
    pub name: String,

    /// Configuration.
    pub config: LoraConfig,

    /// Weights per layer.
    pub layers: HashMap<String, LoraWeights>,

    /// Whether adapter is currently active.
    pub active: bool,
}

impl LoraAdapter {
    /// Create a new adapter.
    pub fn new(name: &str, config: LoraConfig) -> Self {
        Self {
            name: name.to_string(),
            config,
            layers: HashMap::new(),
            active: false,
        }
    }

    /// Get adapter configuration.
    pub fn config(&self) -> &LoraConfig {
        &self.config
    }

    /// Add weights for a layer.
    pub fn add_layer(&mut self, layer_name: &str, weights: LoraWeights) {
        self.layers.insert(layer_name.to_string(), weights);
    }

    /// Get weights for a layer.
    pub fn get_layer(&self, layer_name: &str) -> Option<&LoraWeights> {
        self.layers.get(layer_name)
    }

    /// Number of layers with LoRA weights.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.layers
            .values()
            .map(|w| {
                let (d, k, r) = w.dims();
                d * r + r * k
            })
            .sum()
    }

    /// Activate this adapter.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Deactivate this adapter.
    pub fn deactivate(&mut self) {
        self.active = false;
    }
}

/// Manager for multiple LoRA adapters.
#[derive(Debug, Default)]
pub struct AdapterManager {
    /// Registered adapters.
    adapters: HashMap<String, LoraAdapter>,

    /// Currently active adapter.
    active_adapter: Option<String>,

    /// Maximum number of adapters.
    max_adapters: usize,
}

impl AdapterManager {
    /// Create a new adapter manager.
    pub fn new(max_adapters: usize) -> Self {
        Self {
            adapters: HashMap::new(),
            active_adapter: None,
            max_adapters,
        }
    }

    /// Register an adapter.
    pub fn register(&mut self, adapter: LoraAdapter) -> Result<(), AdapterError> {
        if self.adapters.len() >= self.max_adapters {
            return Err(AdapterError::MaxAdaptersReached(self.max_adapters));
        }

        let name = adapter.name.clone();
        self.adapters.insert(name, adapter);
        Ok(())
    }

    /// Unregister an adapter.
    pub fn unregister(&mut self, name: &str) -> Option<LoraAdapter> {
        if self.active_adapter.as_deref() == Some(name) {
            self.active_adapter = None;
        }
        self.adapters.remove(name)
    }

    /// Get an adapter by name.
    pub fn get(&self, name: &str) -> Option<&LoraAdapter> {
        self.adapters.get(name)
    }

    /// Get mutable adapter by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut LoraAdapter> {
        self.adapters.get_mut(name)
    }

    /// Activate an adapter.
    pub fn activate(&mut self, name: &str) -> Result<(), AdapterError> {
        if !self.adapters.contains_key(name) {
            return Err(AdapterError::NotFound(name.to_string()));
        }

        // Deactivate current
        if let Some(current) = &self.active_adapter {
            if let Some(adapter) = self.adapters.get_mut(current) {
                adapter.deactivate();
            }
        }

        // Activate new
        if let Some(adapter) = self.adapters.get_mut(name) {
            adapter.activate();
        }
        self.active_adapter = Some(name.to_string());

        Ok(())
    }

    /// Deactivate current adapter.
    pub fn deactivate(&mut self) {
        if let Some(name) = &self.active_adapter {
            if let Some(adapter) = self.adapters.get_mut(name) {
                adapter.deactivate();
            }
        }
        self.active_adapter = None;
    }

    /// Get currently active adapter.
    pub fn active(&self) -> Option<&LoraAdapter> {
        self.active_adapter
            .as_ref()
            .and_then(|n| self.adapters.get(n))
    }

    /// List all registered adapters.
    pub fn list(&self) -> Vec<&str> {
        self.adapters.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered adapters.
    pub fn count(&self) -> usize {
        self.adapters.len()
    }
}

/// Error type for adapter operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum AdapterError {
    /// Adapter not found.
    #[error("Adapter not found: {0}")]
    NotFound(String),

    /// Maximum adapters reached.
    #[error("Maximum adapters reached: {0}")]
    MaxAdaptersReached(usize),

    /// Invalid adapter configuration.
    #[error("Invalid adapter configuration: {0}")]
    InvalidConfig(String),

    /// Weight loading failed.
    #[error("Failed to load weights: {0}")]
    LoadFailed(String),
}

/// Merged weights with LoRA applied.
#[derive(Debug, Clone)]
pub struct MergedWeights {
    /// Base weights.
    pub base: Vec<Vec<f32>>,

    /// Applied LoRA adapter name.
    pub adapter_name: Option<String>,

    /// Whether weights are merged.
    pub is_merged: bool,
}

impl MergedWeights {
    /// Create from base weights.
    pub fn new(base: Vec<Vec<f32>>) -> Self {
        Self {
            base,
            adapter_name: None,
            is_merged: false,
        }
    }

    /// Apply LoRA adapter.
    pub fn apply_lora(&mut self, adapter: &LoraAdapter, layer_name: &str) {
        if let Some(weights) = adapter.get_layer(layer_name) {
            weights.merge_into(&mut self.base);
            self.adapter_name = Some(adapter.name.clone());
            self.is_merged = true;
        }
    }

    /// Get dimensions.
    pub fn dims(&self) -> (usize, usize) {
        let d = self.base.len();
        let k = if d > 0 { self.base[0].len() } else { 0 };
        (d, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.get_scaling(), 2.0);
    }

    #[test]
    fn test_lora_config_with_rank() {
        let config = LoraConfig::with_rank(16);
        assert_eq!(config.rank, 16);
        assert_eq!(config.alpha, 32.0);
    }

    #[test]
    fn test_lora_weights_dims() {
        let weights = LoraWeights::random(128, 64, 8, 2.0);
        let (d, k, r) = weights.dims();
        assert_eq!(d, 128);
        assert_eq!(k, 64);
        assert_eq!(r, 8);
    }

    #[test]
    fn test_lora_weights_forward() {
        let weights = LoraWeights::random(4, 3, 2, 1.0);
        let x = vec![1.0, 1.0, 1.0];
        let y = weights.forward(&x);
        assert_eq!(y.len(), 4);
    }

    #[test]
    fn test_lora_adapter_new() {
        let config = LoraConfig::default();
        let adapter = LoraAdapter::new("test-adapter", config);
        assert_eq!(adapter.name, "test-adapter");
        assert!(!adapter.active);
        assert_eq!(adapter.num_layers(), 0);
    }

    #[test]
    fn test_lora_adapter_add_layer() {
        let config = LoraConfig::default();
        let mut adapter = LoraAdapter::new("test", config);

        let weights = LoraWeights::random(128, 64, 8, 2.0);
        adapter.add_layer("layer0.q_proj", weights);

        assert_eq!(adapter.num_layers(), 1);
        assert!(adapter.get_layer("layer0.q_proj").is_some());
    }

    #[test]
    fn test_adapter_manager() {
        let mut manager = AdapterManager::new(10);

        let adapter1 = LoraAdapter::new("adapter1", LoraConfig::default());
        let adapter2 = LoraAdapter::new("adapter2", LoraConfig::default());

        manager.register(adapter1).unwrap();
        manager.register(adapter2).unwrap();

        assert_eq!(manager.count(), 2);
        assert!(manager.get("adapter1").is_some());
    }

    #[test]
    fn test_adapter_manager_activate() {
        let mut manager = AdapterManager::new(10);
        let adapter = LoraAdapter::new("test", LoraConfig::default());
        manager.register(adapter).unwrap();

        manager.activate("test").unwrap();
        assert!(manager.active().is_some());
        assert!(manager.active().unwrap().active);
    }

    #[test]
    fn test_adapter_manager_deactivate() {
        let mut manager = AdapterManager::new(10);
        let adapter = LoraAdapter::new("test", LoraConfig::default());
        manager.register(adapter).unwrap();

        manager.activate("test").unwrap();
        manager.deactivate();

        assert!(manager.active().is_none());
    }

    #[test]
    fn test_adapter_manager_max_adapters() {
        let mut manager = AdapterManager::new(2);

        manager
            .register(LoraAdapter::new("a1", LoraConfig::default()))
            .unwrap();
        manager
            .register(LoraAdapter::new("a2", LoraConfig::default()))
            .unwrap();

        let result = manager.register(LoraAdapter::new("a3", LoraConfig::default()));
        assert!(matches!(result, Err(AdapterError::MaxAdaptersReached(2))));
    }

    #[test]
    fn test_merged_weights() {
        let base = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let merged = MergedWeights::new(base);

        assert!(!merged.is_merged);
        assert!(merged.adapter_name.is_none());
        assert_eq!(merged.dims(), (2, 2));
    }

    #[test]
    fn test_lora_num_parameters() {
        let config = LoraConfig::with_rank(8);
        let mut adapter = LoraAdapter::new("test", config);

        // d=128, k=64, r=8 -> 128*8 + 8*64 = 1024 + 512 = 1536
        let weights = LoraWeights::random(128, 64, 8, 2.0);
        adapter.add_layer("layer0", weights);

        assert_eq!(adapter.num_parameters(), 1536);
    }
}
