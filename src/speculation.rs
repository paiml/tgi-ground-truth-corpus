//! Speculative Decoding
//!
//! Implements speculative decoding (also known as assisted generation) where a
//! smaller draft model proposes candidate tokens that are verified by the main
//! model in a single forward pass.
//!
//! # TGI Reference
//!
//! Based on TGI's speculation implementation in `server/text_generation_server/`.
//! See: <https://github.com/huggingface/text-generation-inference>
//!
//! # Algorithm
//!
//! 1. Draft model generates K candidate tokens autoregressively
//! 2. Main model verifies all K tokens in a single forward pass
//! 3. Accept tokens until first mismatch, then sample from main model
//! 4. Repeat with new accepted prefix
//!
//! # Example
//!
//! ```rust
//! use tgi_gtc::speculation::{SpeculativeDecoder, SpeculationConfig, DraftProposal};
//!
//! let config = SpeculationConfig::default();
//! let mut decoder = SpeculativeDecoder::new(config);
//!
//! // Simulate draft model proposing tokens
//! let draft = DraftProposal::new(vec![100, 200, 300, 400], vec![0.9, 0.8, 0.7, 0.6]);
//!
//! // Simulate main model logits for verification
//! let main_logits: Vec<Vec<f32>> = vec![
//!     vec![0.0; 1000], // logits for position 0
//!     vec![0.0; 1000], // logits for position 1
//!     vec![0.0; 1000], // logits for position 2
//!     vec![0.0; 1000], // logits for position 3
//! ];
//!
//! let result = decoder.verify(&draft, &main_logits);
//! println!("Accepted {} tokens", result.accepted_count);
//! ```
//!
//! # Performance
//!
//! Speculative decoding can achieve 2-3x speedup when:
//! - Draft model is 10-20x smaller than main model
//! - Acceptance rate is >70%
//! - Draft tokens (K) is tuned (typically 4-8)

use serde::{Deserialize, Serialize};

/// Configuration for speculative decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationConfig {
    /// Number of tokens to speculate (K).
    pub num_speculative_tokens: usize,

    /// Minimum probability threshold for draft acceptance.
    pub acceptance_threshold: f32,

    /// Whether to use rejection sampling for verification.
    pub use_rejection_sampling: bool,

    /// Temperature for sampling (0 = greedy).
    pub temperature: f32,

    /// Maximum consecutive rejections before reducing speculation.
    pub max_rejections: usize,

    /// Whether to dynamically adjust num_speculative_tokens.
    pub dynamic_speculation: bool,
}

impl Default for SpeculationConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 4,
            acceptance_threshold: 0.0,
            use_rejection_sampling: true,
            temperature: 1.0,
            max_rejections: 3,
            dynamic_speculation: true,
        }
    }
}

impl SpeculationConfig {
    /// Create a greedy speculation config (deterministic).
    pub fn greedy(num_tokens: usize) -> Self {
        Self {
            num_speculative_tokens: num_tokens,
            temperature: 0.0,
            use_rejection_sampling: false,
            ..Default::default()
        }
    }

    /// Create config optimized for high acceptance rate.
    pub fn conservative() -> Self {
        Self {
            num_speculative_tokens: 2,
            acceptance_threshold: 0.1,
            ..Default::default()
        }
    }

    /// Create config optimized for maximum speculation.
    pub fn aggressive() -> Self {
        Self {
            num_speculative_tokens: 8,
            acceptance_threshold: 0.0,
            ..Default::default()
        }
    }
}

/// A proposal from the draft model.
#[derive(Debug, Clone)]
pub struct DraftProposal {
    /// Proposed token IDs.
    pub tokens: Vec<u32>,

    /// Draft model probabilities for each token.
    pub probabilities: Vec<f32>,

    /// Draft model logits (optional, for rejection sampling).
    pub logits: Option<Vec<Vec<f32>>>,
}

impl DraftProposal {
    /// Create a new draft proposal.
    pub fn new(tokens: Vec<u32>, probabilities: Vec<f32>) -> Self {
        assert_eq!(tokens.len(), probabilities.len());
        Self {
            tokens,
            probabilities,
            logits: None,
        }
    }

    /// Create a proposal with full logits for rejection sampling.
    pub fn with_logits(tokens: Vec<u32>, probabilities: Vec<f32>, logits: Vec<Vec<f32>>) -> Self {
        assert_eq!(tokens.len(), probabilities.len());
        assert_eq!(tokens.len(), logits.len());
        Self {
            tokens,
            probabilities,
            logits: Some(logits),
        }
    }

    /// Number of proposed tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if proposal is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Result of verification.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of accepted tokens (0 to K).
    pub accepted_count: usize,

    /// Accepted token IDs.
    pub accepted_tokens: Vec<u32>,

    /// The next token sampled from main model (after accepted tokens).
    pub next_token: u32,

    /// Probability of next token from main model.
    pub next_token_prob: f32,

    /// Whether all draft tokens were accepted.
    pub all_accepted: bool,

    /// Acceptance rate for this verification.
    pub acceptance_rate: f32,
}

impl VerificationResult {
    /// Total tokens generated (accepted + 1 new).
    pub fn total_tokens(&self) -> usize {
        self.accepted_count + 1
    }

    /// Speedup factor (tokens generated / forward passes).
    /// For speculation, we use 1 main model forward pass.
    pub fn speedup(&self) -> f32 {
        self.total_tokens() as f32
    }
}

/// Statistics for speculation performance.
#[derive(Debug, Clone, Default)]
pub struct SpeculationStats {
    /// Total verification rounds.
    pub total_rounds: u64,

    /// Total tokens proposed by draft model.
    pub total_proposed: u64,

    /// Total tokens accepted.
    pub total_accepted: u64,

    /// Total tokens generated (including sampled).
    pub total_generated: u64,

    /// Rounds where all tokens were accepted.
    pub full_acceptance_rounds: u64,

    /// Consecutive rejections (for dynamic adjustment).
    pub consecutive_rejections: usize,
}

impl SpeculationStats {
    /// Overall acceptance rate.
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_proposed == 0 {
            0.0
        } else {
            self.total_accepted as f32 / self.total_proposed as f32
        }
    }

    /// Average tokens per round.
    pub fn tokens_per_round(&self) -> f32 {
        if self.total_rounds == 0 {
            0.0
        } else {
            self.total_generated as f32 / self.total_rounds as f32
        }
    }

    /// Average speedup factor.
    pub fn average_speedup(&self) -> f32 {
        self.tokens_per_round()
    }

    /// Record a verification result.
    pub fn record(&mut self, result: &VerificationResult, proposed: usize) {
        self.total_rounds += 1;
        self.total_proposed += proposed as u64;
        self.total_accepted += result.accepted_count as u64;
        self.total_generated += result.total_tokens() as u64;

        if result.all_accepted {
            self.full_acceptance_rounds += 1;
            self.consecutive_rejections = 0;
        } else if result.accepted_count == 0 {
            self.consecutive_rejections += 1;
        } else {
            self.consecutive_rejections = 0;
        }
    }
}

/// Speculative decoder that verifies draft proposals.
#[derive(Debug)]
pub struct SpeculativeDecoder {
    config: SpeculationConfig,
    stats: SpeculationStats,
    current_speculation_depth: usize,
    rng_state: u64,
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder.
    pub fn new(config: SpeculationConfig) -> Self {
        let depth = config.num_speculative_tokens;
        Self {
            config,
            stats: SpeculationStats::default(),
            current_speculation_depth: depth,
            rng_state: 12345,
        }
    }

    /// Get current configuration.
    pub fn config(&self) -> &SpeculationConfig {
        &self.config
    }

    /// Get speculation statistics.
    pub fn stats(&self) -> &SpeculationStats {
        &self.stats
    }

    /// Get current speculation depth (may be adjusted dynamically).
    pub fn current_depth(&self) -> usize {
        self.current_speculation_depth
    }

    /// Verify a draft proposal against main model logits.
    ///
    /// # Arguments
    ///
    /// * `draft` - The draft model's proposal
    /// * `main_logits` - Logits from main model for each position
    ///
    /// # Returns
    ///
    /// Verification result with accepted tokens and next token.
    pub fn verify(
        &mut self,
        draft: &DraftProposal,
        main_logits: &[Vec<f32>],
    ) -> VerificationResult {
        assert!(!draft.is_empty());
        assert_eq!(draft.len(), main_logits.len());

        let mut accepted_count = 0;
        let mut accepted_tokens = Vec::new();

        // Verify each token
        for i in 0..draft.len() {
            let draft_token = draft.tokens[i];
            let draft_prob = draft.probabilities[i];
            let main_logits_i = &main_logits[i];

            // Get main model's probability for draft token
            let main_probs = self.softmax(main_logits_i);
            let main_prob = main_probs.get(draft_token as usize).copied().unwrap_or(0.0);

            // Acceptance criterion
            let accept = if self.config.use_rejection_sampling {
                // Rejection sampling: accept with probability min(1, p_main / p_draft)
                self.rejection_sample(main_prob, draft_prob)
            } else {
                // Greedy: accept if main model agrees
                let main_token = self.argmax(&main_probs);
                main_token == draft_token
            };

            if accept {
                accepted_count += 1;
                accepted_tokens.push(draft_token);
            } else {
                break;
            }
        }

        // Sample next token from main model at position after accepted tokens
        let sample_pos = accepted_count.min(main_logits.len() - 1);
        let sample_logits = &main_logits[sample_pos];
        let probs = self.softmax(sample_logits);
        let (next_token, next_token_prob) = self.sample(&probs);

        let all_accepted = accepted_count == draft.len();
        let acceptance_rate = accepted_count as f32 / draft.len() as f32;

        let result = VerificationResult {
            accepted_count,
            accepted_tokens,
            next_token,
            next_token_prob,
            all_accepted,
            acceptance_rate,
        };

        // Update stats
        self.stats.record(&result, draft.len());

        // Dynamic adjustment
        if self.config.dynamic_speculation {
            self.adjust_depth();
        }

        result
    }

    /// Verify with greedy decoding (no sampling).
    pub fn verify_greedy(
        &mut self,
        draft: &DraftProposal,
        main_logits: &[Vec<f32>],
    ) -> VerificationResult {
        assert!(!draft.is_empty());
        assert_eq!(draft.len(), main_logits.len());

        let mut accepted_count = 0;
        let mut accepted_tokens = Vec::new();

        for i in 0..draft.len() {
            let draft_token = draft.tokens[i];
            let main_logits_i = &main_logits[i];

            // Greedy: check if main model's argmax matches draft
            let main_token = self.argmax_logits(main_logits_i);

            if main_token == draft_token {
                accepted_count += 1;
                accepted_tokens.push(draft_token);
            } else {
                break;
            }
        }

        // Get next token from main model
        let sample_pos = accepted_count.min(main_logits.len() - 1);
        let main_token = self.argmax_logits(&main_logits[sample_pos]);
        let probs = self.softmax(&main_logits[sample_pos]);
        let next_token_prob = probs.get(main_token as usize).copied().unwrap_or(0.0);

        let all_accepted = accepted_count == draft.len();
        let acceptance_rate = accepted_count as f32 / draft.len() as f32;

        let result = VerificationResult {
            accepted_count,
            accepted_tokens,
            next_token: main_token,
            next_token_prob,
            all_accepted,
            acceptance_rate,
        };

        self.stats.record(&result, draft.len());

        if self.config.dynamic_speculation {
            self.adjust_depth();
        }

        result
    }

    /// Rejection sampling: accept with probability min(1, p_main / p_draft).
    fn rejection_sample(&mut self, main_prob: f32, draft_prob: f32) -> bool {
        if draft_prob <= 0.0 {
            return main_prob > 0.0;
        }

        let ratio = main_prob / draft_prob;
        if ratio >= 1.0 {
            true
        } else {
            let r = self.random_f32();
            r < ratio
        }
    }

    /// Dynamically adjust speculation depth based on acceptance rate.
    fn adjust_depth(&mut self) {
        if self.stats.consecutive_rejections >= self.config.max_rejections {
            // Reduce depth on repeated rejections
            self.current_speculation_depth = (self.current_speculation_depth / 2).max(1);
        } else if self.stats.total_rounds > 10 && self.stats.acceptance_rate() > 0.9 {
            // Increase depth on high acceptance
            self.current_speculation_depth =
                (self.current_speculation_depth + 1).min(self.config.num_speculative_tokens * 2);
        }
    }

    /// Compute softmax of logits.
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }

        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exp.iter().sum();

        if sum == 0.0 {
            vec![1.0 / logits.len() as f32; logits.len()]
        } else {
            exp.iter().map(|&x| x / sum).collect()
        }
    }

    /// Argmax of probabilities.
    fn argmax(&self, probs: &[f32]) -> u32 {
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    /// Argmax of logits (avoids softmax computation).
    fn argmax_logits(&self, logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    /// Sample from probability distribution.
    fn sample(&mut self, probs: &[f32]) -> (u32, f32) {
        if self.config.temperature == 0.0 {
            let idx = self.argmax(probs);
            let prob = probs.get(idx as usize).copied().unwrap_or(0.0);
            return (idx, prob);
        }

        let r = self.random_f32();
        let mut cumsum = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return (i as u32, p);
            }
        }

        // Fallback to last token
        let last = probs.len().saturating_sub(1);
        (last as u32, probs.get(last).copied().unwrap_or(0.0))
    }

    /// Simple LCG random number generator.
    fn random_f32(&mut self) -> f32 {
        const A: u64 = 6364136223846793005;
        const C: u64 = 1442695040888963407;
        self.rng_state = self.rng_state.wrapping_mul(A).wrapping_add(C);
        (self.rng_state >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SpeculationStats::default();
        self.current_speculation_depth = self.config.num_speculative_tokens;
    }
}

/// Tree-based speculation for multiple candidates.
///
/// Instead of a single sequence of K tokens, generate a tree of candidates
/// to increase acceptance probability.
#[derive(Debug, Clone)]
pub struct SpeculationTree {
    /// Root tokens (first level candidates).
    pub roots: Vec<u32>,

    /// Children for each path (indexed by path).
    pub children: Vec<Vec<u32>>,

    /// Probabilities for each token.
    pub probabilities: Vec<f32>,
}

impl SpeculationTree {
    /// Create a linear chain (equivalent to standard speculation).
    pub fn chain(tokens: Vec<u32>, probabilities: Vec<f32>) -> Self {
        Self {
            roots: if tokens.is_empty() {
                vec![]
            } else {
                vec![tokens[0]]
            },
            children: vec![tokens[1..].to_vec()],
            probabilities,
        }
    }

    /// Create a tree with multiple candidates at each level.
    pub fn new(roots: Vec<u32>, children: Vec<Vec<u32>>, probabilities: Vec<f32>) -> Self {
        Self {
            roots,
            children,
            probabilities,
        }
    }

    /// Total number of tokens in the tree.
    pub fn total_tokens(&self) -> usize {
        self.roots.len() + self.children.iter().map(|c| c.len()).sum::<usize>()
    }
}

/// Medusa-style parallel speculation with multiple heads.
///
/// Uses multiple prediction heads to generate candidates in parallel,
/// avoiding the sequential bottleneck of standard draft models.
#[derive(Debug, Clone)]
pub struct MedusaConfig {
    /// Number of Medusa heads.
    pub num_heads: usize,

    /// Candidates per head.
    pub candidates_per_head: usize,

    /// Top-k for candidate selection.
    pub top_k: usize,

    /// Tree attention pattern.
    pub use_tree_attention: bool,
}

impl Default for MedusaConfig {
    fn default() -> Self {
        Self {
            num_heads: 4,
            candidates_per_head: 5,
            top_k: 10,
            use_tree_attention: true,
        }
    }
}

/// Medusa speculation decoder.
#[derive(Debug)]
pub struct MedusaDecoder {
    config: MedusaConfig,
    stats: SpeculationStats,
}

impl MedusaDecoder {
    /// Create a new Medusa decoder.
    pub fn new(config: MedusaConfig) -> Self {
        Self {
            config,
            stats: SpeculationStats::default(),
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &MedusaConfig {
        &self.config
    }

    /// Get statistics.
    pub fn stats(&self) -> &SpeculationStats {
        &self.stats
    }

    /// Generate candidate tree from Medusa head outputs.
    ///
    /// Each head predicts tokens for positions 1, 2, ..., num_heads.
    pub fn generate_candidates(&self, head_logits: &[Vec<f32>]) -> SpeculationTree {
        assert_eq!(head_logits.len(), self.config.num_heads);

        let mut roots = Vec::new();
        let mut all_children = Vec::new();
        let mut all_probs = Vec::new();

        // Get top-k from first head as roots
        let first_topk = self.top_k_indices(&head_logits[0], self.config.top_k);
        for &(idx, prob) in &first_topk {
            roots.push(idx);
            all_probs.push(prob);
        }

        // Build children from subsequent heads
        for head in 1..self.config.num_heads {
            let topk = self.top_k_indices(&head_logits[head], self.config.candidates_per_head);
            let children: Vec<u32> = topk.iter().map(|&(idx, _)| idx).collect();
            let probs: Vec<f32> = topk.iter().map(|&(_, p)| p).collect();

            all_children.push(children);
            all_probs.extend(probs);
        }

        SpeculationTree::new(roots, all_children, all_probs)
    }

    /// Get top-k indices and probabilities from logits.
    fn top_k_indices(&self, logits: &[f32], k: usize) -> Vec<(u32, f32)> {
        let probs = self.softmax(logits);
        let mut indexed: Vec<(u32, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as u32, p))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);
        indexed
    }

    /// Compute softmax.
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }

        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exp.iter().sum();

        if sum == 0.0 {
            vec![1.0 / logits.len() as f32; logits.len()]
        } else {
            exp.iter().map(|&x| x / sum).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculation_config_default() {
        let config = SpeculationConfig::default();
        assert_eq!(config.num_speculative_tokens, 4);
        assert!(config.use_rejection_sampling);
    }

    #[test]
    fn test_speculation_config_greedy() {
        let config = SpeculationConfig::greedy(6);
        assert_eq!(config.num_speculative_tokens, 6);
        assert_eq!(config.temperature, 0.0);
        assert!(!config.use_rejection_sampling);
    }

    #[test]
    fn test_draft_proposal_new() {
        let draft = DraftProposal::new(vec![1, 2, 3], vec![0.9, 0.8, 0.7]);
        assert_eq!(draft.len(), 3);
        assert!(!draft.is_empty());
        assert!(draft.logits.is_none());
    }

    #[test]
    fn test_draft_proposal_with_logits() {
        let logits = vec![vec![0.0; 10]; 3];
        let draft = DraftProposal::with_logits(vec![1, 2, 3], vec![0.9, 0.8, 0.7], logits);
        assert!(draft.logits.is_some());
    }

    #[test]
    fn test_verification_greedy_all_accept() {
        let config = SpeculationConfig::greedy(4);
        let mut decoder = SpeculativeDecoder::new(config);

        // Draft proposes tokens 5, 6, 7, 8
        let draft = DraftProposal::new(vec![5, 6, 7, 8], vec![0.9, 0.9, 0.9, 0.9]);

        // Main model logits where argmax matches draft
        let mut main_logits = vec![vec![0.0f32; 100]; 4];
        main_logits[0][5] = 10.0; // argmax = 5
        main_logits[1][6] = 10.0; // argmax = 6
        main_logits[2][7] = 10.0; // argmax = 7
        main_logits[3][8] = 10.0; // argmax = 8

        let result = decoder.verify_greedy(&draft, &main_logits);

        assert_eq!(result.accepted_count, 4);
        assert!(result.all_accepted);
        assert_eq!(result.acceptance_rate, 1.0);
    }

    #[test]
    fn test_verification_greedy_partial_accept() {
        let config = SpeculationConfig::greedy(4);
        let mut decoder = SpeculativeDecoder::new(config);

        // Draft proposes tokens 5, 6, 7, 8
        let draft = DraftProposal::new(vec![5, 6, 7, 8], vec![0.9, 0.9, 0.9, 0.9]);

        // Main model disagrees at position 2
        let mut main_logits = vec![vec![0.0f32; 100]; 4];
        main_logits[0][5] = 10.0; // argmax = 5 ✓
        main_logits[1][6] = 10.0; // argmax = 6 ✓
        main_logits[2][99] = 10.0; // argmax = 99 ✗
        main_logits[3][8] = 10.0; // not reached

        let result = decoder.verify_greedy(&draft, &main_logits);

        assert_eq!(result.accepted_count, 2);
        assert!(!result.all_accepted);
        assert_eq!(result.accepted_tokens, vec![5, 6]);
        assert_eq!(result.next_token, 99);
    }

    #[test]
    fn test_verification_greedy_none_accept() {
        let config = SpeculationConfig::greedy(4);
        let mut decoder = SpeculativeDecoder::new(config);

        let draft = DraftProposal::new(vec![5, 6, 7, 8], vec![0.9, 0.9, 0.9, 0.9]);

        // Main model disagrees at first position
        let mut main_logits = vec![vec![0.0f32; 100]; 4];
        main_logits[0][99] = 10.0; // argmax = 99, not 5

        let result = decoder.verify_greedy(&draft, &main_logits);

        assert_eq!(result.accepted_count, 0);
        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.next_token, 99);
    }

    #[test]
    fn test_verification_result_total_tokens() {
        let result = VerificationResult {
            accepted_count: 3,
            accepted_tokens: vec![1, 2, 3],
            next_token: 4,
            next_token_prob: 0.8,
            all_accepted: false,
            acceptance_rate: 0.75,
        };

        assert_eq!(result.total_tokens(), 4);
        assert_eq!(result.speedup(), 4.0);
    }

    #[test]
    fn test_speculation_stats() {
        let mut stats = SpeculationStats::default();

        let result1 = VerificationResult {
            accepted_count: 4,
            accepted_tokens: vec![1, 2, 3, 4],
            next_token: 5,
            next_token_prob: 0.9,
            all_accepted: true,
            acceptance_rate: 1.0,
        };

        stats.record(&result1, 4);

        assert_eq!(stats.total_rounds, 1);
        assert_eq!(stats.total_proposed, 4);
        assert_eq!(stats.total_accepted, 4);
        assert_eq!(stats.total_generated, 5);
        assert_eq!(stats.full_acceptance_rounds, 1);
        assert_eq!(stats.acceptance_rate(), 1.0);
        assert_eq!(stats.tokens_per_round(), 5.0);
    }

    #[test]
    fn test_speculation_tree_chain() {
        let tree = SpeculationTree::chain(vec![1, 2, 3, 4], vec![0.9, 0.8, 0.7, 0.6]);

        assert_eq!(tree.roots, vec![1]);
        assert_eq!(tree.children.len(), 1);
        assert_eq!(tree.children[0], vec![2, 3, 4]);
        assert_eq!(tree.total_tokens(), 4);
    }

    #[test]
    fn test_medusa_config_default() {
        let config = MedusaConfig::default();
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.candidates_per_head, 5);
        assert!(config.use_tree_attention);
    }

    #[test]
    fn test_medusa_generate_candidates() {
        let config = MedusaConfig {
            num_heads: 2,
            candidates_per_head: 3,
            top_k: 3,
            use_tree_attention: false,
        };
        let decoder = MedusaDecoder::new(config);

        // Logits for 2 heads, vocab size 10
        let mut head_logits = vec![vec![0.0f32; 10]; 2];
        head_logits[0][5] = 10.0;
        head_logits[0][3] = 8.0;
        head_logits[0][7] = 6.0;
        head_logits[1][2] = 10.0;
        head_logits[1][4] = 8.0;
        head_logits[1][6] = 6.0;

        let tree = decoder.generate_candidates(&head_logits);

        assert_eq!(tree.roots.len(), 3);
        assert!(tree.roots.contains(&5));
        assert!(tree.roots.contains(&3));
        assert!(tree.roots.contains(&7));
    }

    #[test]
    fn test_dynamic_depth_adjustment() {
        let config = SpeculationConfig {
            num_speculative_tokens: 4,
            dynamic_speculation: true,
            max_rejections: 2,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecoder::new(config);

        assert_eq!(decoder.current_depth(), 4);

        // Simulate repeated rejections
        let draft = DraftProposal::new(vec![5], vec![0.9]);
        let mut main_logits = vec![vec![0.0f32; 100]; 1];
        main_logits[0][99] = 10.0; // Will reject

        // First rejection
        decoder.verify_greedy(&draft, &main_logits);
        // Second rejection
        decoder.verify_greedy(&draft, &main_logits);

        // After max_rejections, depth should be reduced
        assert!(decoder.current_depth() < 4);
    }
}
