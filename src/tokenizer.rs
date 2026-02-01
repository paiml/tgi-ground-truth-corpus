//! Tokenization patterns.
//!
//! BPE (Byte-Pair Encoding) tokenization for text processing.
//!
//! # TGI Source
//!
//! Patterns derived from tokenizer handling in TGI router:
//! - BPE tokenization
//! - Special token handling
//! - Token ID mapping
//!
//! # Sovereign AI Stack Equivalent
//!
//! Maps to `aprender::tokenize` for tokenization.
//!
//! # Key Concepts
//!
//! ## Byte-Pair Encoding (BPE)
//!
//! BPE builds vocabulary by iteratively merging frequent byte pairs:
//! 1. Start with individual bytes as tokens
//! 2. Find most frequent adjacent pair
//! 3. Merge into new token
//! 4. Repeat until vocabulary size reached
//!
//! This handles any input (including unknown words) by falling back to bytes.

use std::collections::HashMap;

/// A token ID.
pub type TokenId = u32;

/// Special tokens used in language models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialTokens {
    /// Beginning of sequence token.
    pub bos_token: Option<String>,
    /// End of sequence token.
    pub eos_token: Option<String>,
    /// Unknown token.
    pub unk_token: Option<String>,
    /// Padding token.
    pub pad_token: Option<String>,
    /// Separator token.
    pub sep_token: Option<String>,
    /// Mask token (for MLM).
    pub mask_token: Option<String>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            unk_token: Some("<unk>".to_string()),
            pad_token: Some("<pad>".to_string()),
            sep_token: None,
            mask_token: None,
        }
    }
}

impl SpecialTokens {
    /// Create empty special tokens.
    pub fn empty() -> Self {
        Self {
            bos_token: None,
            eos_token: None,
            unk_token: None,
            pad_token: None,
            sep_token: None,
            mask_token: None,
        }
    }

    /// GPT-2 style special tokens.
    pub fn gpt2() -> Self {
        Self {
            bos_token: Some("<|endoftext|>".to_string()),
            eos_token: Some("<|endoftext|>".to_string()),
            unk_token: Some("<|endoftext|>".to_string()),
            pad_token: Some("<|endoftext|>".to_string()),
            sep_token: None,
            mask_token: None,
        }
    }

    /// Llama style special tokens.
    pub fn llama() -> Self {
        Self {
            bos_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            unk_token: Some("<unk>".to_string()),
            pad_token: None,
            sep_token: None,
            mask_token: None,
        }
    }
}

/// A merge rule for BPE.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MergeRule {
    /// First token in the pair.
    pub first: String,
    /// Second token in the pair.
    pub second: String,
    /// Merged result.
    pub merged: String,
    /// Priority (lower = merge first).
    pub priority: u32,
}

impl MergeRule {
    /// Create a new merge rule.
    pub fn new(first: impl Into<String>, second: impl Into<String>, priority: u32) -> Self {
        let first = first.into();
        let second = second.into();
        let merged = format!("{}{}", first, second);
        Self {
            first,
            second,
            merged,
            priority,
        }
    }
}

/// Configuration for the tokenizer.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Vocabulary mapping token strings to IDs.
    pub vocab: HashMap<String, TokenId>,

    /// Reverse vocabulary mapping IDs to token strings.
    pub id_to_token: HashMap<TokenId, String>,

    /// BPE merge rules in priority order.
    pub merges: Vec<MergeRule>,

    /// Special tokens.
    pub special_tokens: SpecialTokens,

    /// Whether to add BOS token.
    pub add_bos: bool,

    /// Whether to add EOS token.
    pub add_eos: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self::simple_ascii()
    }
}

impl TokenizerConfig {
    /// Create a simple ASCII tokenizer for testing.
    pub fn simple_ascii() -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Add special tokens
        let special = SpecialTokens::default();
        let mut id = 0u32;

        if let Some(ref token) = special.bos_token {
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
            id += 1;
        }
        if let Some(ref token) = special.eos_token {
            if !vocab.contains_key(token) {
                vocab.insert(token.clone(), id);
                id_to_token.insert(id, token.clone());
                id += 1;
            }
        }
        if let Some(ref token) = special.unk_token {
            if !vocab.contains_key(token) {
                vocab.insert(token.clone(), id);
                id_to_token.insert(id, token.clone());
                id += 1;
            }
        }
        if let Some(ref token) = special.pad_token {
            if !vocab.contains_key(token) {
                vocab.insert(token.clone(), id);
                id_to_token.insert(id, token.clone());
                id += 1;
            }
        }

        // Add ASCII bytes
        for byte in 0u8..128 {
            let token = String::from(byte as char);
            if !vocab.contains_key(&token) {
                vocab.insert(token.clone(), id);
                id_to_token.insert(id, token);
                id += 1;
            }
        }

        // Add some common word tokens
        let common_words = ["the", "is", "a", "an", "of", "to", "in", "and", "or", "for"];
        for word in common_words {
            let token = format!(" {}", word);
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token);
            id += 1;
        }

        // Simple merges
        let merges = vec![
            MergeRule::new("t", "h", 0),
            MergeRule::new("th", "e", 1),
            MergeRule::new(" ", "the", 2),
            MergeRule::new("i", "s", 3),
            MergeRule::new(" ", "is", 4),
            MergeRule::new("a", "n", 5),
            MergeRule::new(" ", "an", 6),
        ];

        Self {
            vocab,
            id_to_token,
            merges,
            special_tokens: special,
            add_bos: true,
            add_eos: false,
        }
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Check if a token is special.
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special_tokens.bos_token.as_deref() == Some(token)
            || self.special_tokens.eos_token.as_deref() == Some(token)
            || self.special_tokens.unk_token.as_deref() == Some(token)
            || self.special_tokens.pad_token.as_deref() == Some(token)
            || self.special_tokens.sep_token.as_deref() == Some(token)
            || self.special_tokens.mask_token.as_deref() == Some(token)
    }
}

/// BPE Tokenizer.
///
/// # TGI Source
///
/// Maps to tokenizer handling in `router/src/lib.rs`.
///
/// # Examples
///
/// ```rust
/// use tgi_gtc::tokenizer::{Tokenizer, TokenizerConfig};
///
/// let config = TokenizerConfig::simple_ascii();
/// let tokenizer = Tokenizer::new(config);
///
/// let tokens = tokenizer.encode("Hello world");
/// let text = tokenizer.decode(&tokens);
/// ```
#[derive(Debug, Clone)]
pub struct Tokenizer {
    config: TokenizerConfig,
    /// Merge lookup for O(1) pair checking.
    merge_lookup: HashMap<(String, String), String>,
}

impl Tokenizer {
    /// Create a new tokenizer.
    pub fn new(config: TokenizerConfig) -> Self {
        let mut merge_lookup = HashMap::new();
        for rule in &config.merges {
            merge_lookup.insert(
                (rule.first.clone(), rule.second.clone()),
                rule.merged.clone(),
            );
        }

        Self {
            config,
            merge_lookup,
        }
    }

    /// Get configuration.
    pub const fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size()
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        let mut ids = Vec::new();

        // Add BOS if configured
        if self.config.add_bos {
            if let Some(ref bos) = self.config.special_tokens.bos_token {
                if let Some(&id) = self.config.vocab.get(bos) {
                    ids.push(id);
                }
            }
        }

        // Tokenize the text
        let tokens = self.tokenize(text);
        for token in tokens {
            if let Some(&id) = self.config.vocab.get(&token) {
                ids.push(id);
            } else if let Some(ref unk) = self.config.special_tokens.unk_token {
                if let Some(&id) = self.config.vocab.get(unk) {
                    ids.push(id);
                }
            }
        }

        // Add EOS if configured
        if self.config.add_eos {
            if let Some(ref eos) = self.config.special_tokens.eos_token {
                if let Some(&id) = self.config.vocab.get(eos) {
                    ids.push(id);
                }
            }
        }

        ids
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[TokenId]) -> String {
        let mut text = String::new();

        for &id in ids {
            if let Some(token) = self.config.id_to_token.get(&id) {
                // Skip special tokens in output
                if !self.config.is_special_token(token) {
                    text.push_str(token);
                }
            }
        }

        text
    }

    /// Decode a single token ID.
    pub fn decode_token(&self, id: TokenId) -> Option<&str> {
        self.config.id_to_token.get(&id).map(String::as_str)
    }

    /// Get token ID for a string.
    pub fn token_to_id(&self, token: &str) -> Option<TokenId> {
        self.config.vocab.get(token).copied()
    }

    /// Get string for a token ID.
    pub fn id_to_token(&self, id: TokenId) -> Option<&str> {
        self.config.id_to_token.get(&id).map(String::as_str)
    }

    /// Tokenize text into string tokens using BPE.
    fn tokenize(&self, text: &str) -> Vec<String> {
        // Start with characters
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Apply BPE merges
        loop {
            let mut best_merge: Option<(usize, String)> = None;
            let mut best_priority = u32::MAX;

            // Find the highest priority merge
            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(merged) = self.merge_lookup.get(&pair) {
                    // Find priority
                    if let Some(rule) = self
                        .config
                        .merges
                        .iter()
                        .find(|r| r.first == pair.0 && r.second == pair.1)
                    {
                        if rule.priority < best_priority {
                            best_priority = rule.priority;
                            best_merge = Some((i, merged.clone()));
                        }
                    }
                }
            }

            // Apply the best merge
            if let Some((idx, merged)) = best_merge {
                tokens[idx] = merged;
                tokens.remove(idx + 1);
            } else {
                break;
            }
        }

        tokens
    }

    /// Get the BOS token ID.
    pub fn bos_token_id(&self) -> Option<TokenId> {
        self.config
            .special_tokens
            .bos_token
            .as_ref()
            .and_then(|t| self.config.vocab.get(t).copied())
    }

    /// Get the EOS token ID.
    pub fn eos_token_id(&self) -> Option<TokenId> {
        self.config
            .special_tokens
            .eos_token
            .as_ref()
            .and_then(|t| self.config.vocab.get(t).copied())
    }

    /// Get the PAD token ID.
    pub fn pad_token_id(&self) -> Option<TokenId> {
        self.config
            .special_tokens
            .pad_token
            .as_ref()
            .and_then(|t| self.config.vocab.get(t).copied())
    }

    /// Get the UNK token ID.
    pub fn unk_token_id(&self) -> Option<TokenId> {
        self.config
            .special_tokens
            .unk_token
            .as_ref()
            .and_then(|t| self.config.vocab.get(t).copied())
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new(TokenizerConfig::default())
    }
}

/// Encoded output with metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedOutput {
    /// Token IDs.
    pub ids: Vec<TokenId>,

    /// Attention mask (1 for real tokens, 0 for padding).
    pub attention_mask: Vec<u8>,

    /// Token type IDs (for sentence pair tasks).
    pub token_type_ids: Vec<u8>,
}

impl EncodedOutput {
    /// Create from token IDs.
    pub fn from_ids(ids: Vec<TokenId>) -> Self {
        let len = ids.len();
        Self {
            ids,
            attention_mask: vec![1; len],
            token_type_ids: vec![0; len],
        }
    }

    /// Pad to a specific length.
    pub fn pad_to(&mut self, length: usize, pad_id: TokenId) {
        while self.ids.len() < length {
            self.ids.push(pad_id);
            self.attention_mask.push(0);
            self.token_type_ids.push(0);
        }
    }

    /// Truncate to a specific length.
    pub fn truncate_to(&mut self, length: usize) {
        self.ids.truncate(length);
        self.attention_mask.truncate(length);
        self.token_type_ids.truncate(length);
    }

    /// Get length.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens_default() {
        let special = SpecialTokens::default();
        assert_eq!(special.bos_token, Some("<s>".to_string()));
        assert_eq!(special.eos_token, Some("</s>".to_string()));
    }

    #[test]
    fn test_special_tokens_gpt2() {
        let special = SpecialTokens::gpt2();
        assert_eq!(special.bos_token, Some("<|endoftext|>".to_string()));
        assert_eq!(special.eos_token, Some("<|endoftext|>".to_string()));
    }

    #[test]
    fn test_special_tokens_llama() {
        let special = SpecialTokens::llama();
        assert_eq!(special.bos_token, Some("<s>".to_string()));
        assert_eq!(special.eos_token, Some("</s>".to_string()));
        assert!(special.pad_token.is_none());
    }

    #[test]
    fn test_special_tokens_empty() {
        let special = SpecialTokens::empty();
        assert!(special.bos_token.is_none());
        assert!(special.eos_token.is_none());
    }

    #[test]
    fn test_merge_rule() {
        let rule = MergeRule::new("a", "b", 0);
        assert_eq!(rule.first, "a");
        assert_eq!(rule.second, "b");
        assert_eq!(rule.merged, "ab");
        assert_eq!(rule.priority, 0);
    }

    #[test]
    fn test_tokenizer_config_simple() {
        let config = TokenizerConfig::simple_ascii();
        assert!(config.vocab_size() > 100); // At least ASCII + special
        assert!(config.add_bos);
        assert!(!config.add_eos);
    }

    #[test]
    fn test_tokenizer_config_is_special() {
        let config = TokenizerConfig::simple_ascii();
        assert!(config.is_special_token("<s>"));
        assert!(config.is_special_token("</s>"));
        assert!(!config.is_special_token("hello"));
    }

    #[test]
    fn test_tokenizer_encode_decode() {
        let tokenizer = Tokenizer::default();

        let text = "hello";
        let ids = tokenizer.encode(text);

        // Should have BOS + characters
        assert!(!ids.is_empty());

        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_tokenizer_vocab_lookup() {
        let tokenizer = Tokenizer::default();

        // Check that we can look up tokens
        let bos_id = tokenizer.bos_token_id();
        assert!(bos_id.is_some());

        let eos_id = tokenizer.eos_token_id();
        assert!(eos_id.is_some());
    }

    #[test]
    fn test_tokenizer_decode_token() {
        let tokenizer = Tokenizer::default();

        if let Some(bos_id) = tokenizer.bos_token_id() {
            let token = tokenizer.decode_token(bos_id);
            assert_eq!(token, Some("<s>"));
        }
    }

    #[test]
    fn test_tokenizer_token_to_id() {
        let tokenizer = Tokenizer::default();

        let id = tokenizer.token_to_id("<s>");
        assert!(id.is_some());

        let id = tokenizer.token_to_id("nonexistent_token_xyz");
        assert!(id.is_none());
    }

    #[test]
    fn test_tokenizer_id_to_token() {
        let tokenizer = Tokenizer::default();

        if let Some(bos_id) = tokenizer.bos_token_id() {
            let token = tokenizer.id_to_token(bos_id);
            assert_eq!(token, Some("<s>"));
        }

        // Non-existent ID
        let token = tokenizer.id_to_token(999999);
        assert!(token.is_none());
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let tokenizer = Tokenizer::default();
        assert!(tokenizer.vocab_size() > 0);
        assert_eq!(tokenizer.vocab_size(), tokenizer.config().vocab_size());
    }

    #[test]
    fn test_encoded_output_from_ids() {
        let ids = vec![1, 2, 3, 4, 5];
        let output = EncodedOutput::from_ids(ids.clone());

        assert_eq!(output.ids, ids);
        assert_eq!(output.attention_mask, vec![1, 1, 1, 1, 1]);
        assert_eq!(output.token_type_ids, vec![0, 0, 0, 0, 0]);
        assert_eq!(output.len(), 5);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_encoded_output_pad() {
        let mut output = EncodedOutput::from_ids(vec![1, 2, 3]);
        output.pad_to(5, 0);

        assert_eq!(output.ids, vec![1, 2, 3, 0, 0]);
        assert_eq!(output.attention_mask, vec![1, 1, 1, 0, 0]);
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_encoded_output_truncate() {
        let mut output = EncodedOutput::from_ids(vec![1, 2, 3, 4, 5]);
        output.truncate_to(3);

        assert_eq!(output.ids, vec![1, 2, 3]);
        assert_eq!(output.attention_mask, vec![1, 1, 1]);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_encoded_output_empty() {
        let output = EncodedOutput::from_ids(vec![]);
        assert!(output.is_empty());
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn test_tokenizer_multiple_encode_decode() {
        let tokenizer = Tokenizer::default();

        let texts = ["hello", "world", "test", "a b c"];
        for text in texts {
            let ids = tokenizer.encode(text);
            let decoded = tokenizer.decode(&ids);
            assert_eq!(decoded, text);
        }
    }

    #[test]
    fn test_tokenizer_empty_string() {
        let tokenizer = Tokenizer::default();

        let ids = tokenizer.encode("");
        // Should only have BOS token
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], tokenizer.bos_token_id().unwrap());

        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_tokenizer_special_token_ids() {
        let tokenizer = Tokenizer::default();

        assert!(tokenizer.bos_token_id().is_some());
        assert!(tokenizer.eos_token_id().is_some());
        assert!(tokenizer.unk_token_id().is_some());
        assert!(tokenizer.pad_token_id().is_some());
    }
}
