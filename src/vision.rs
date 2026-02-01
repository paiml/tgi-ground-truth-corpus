//! Vision/Multimodal Support
//!
//! Implements image input handling for Vision-Language Models (VLMs).
//!
//! # TGI Reference
//!
//! Based on TGI's multimodal support for models like LLaVA, Qwen-VL.
//! See: <https://github.com/huggingface/text-generation-inference>
//!
//! # Example
//!
//! ```rust
//! use tgi_gtc::vision::{ImageInput, ImageProcessor, MultimodalRequest};
//!
//! // Create an image input
//! let image = ImageInput::from_base64("iVBORw0KGgo...".to_string());
//!
//! // Process for model input
//! let processor = ImageProcessor::default();
//! let processed = processor.process(&image);
//!
//! println!("Image patches: {}", processed.num_patches);
//! ```

use serde::{Deserialize, Serialize};

/// Image input data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInput {
    /// Image source type.
    pub source: ImageSource,

    /// Original dimensions (width, height).
    pub original_size: Option<(u32, u32)>,

    /// Content type (e.g., "image/png").
    pub content_type: Option<String>,
}

/// Source of image data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageSource {
    /// Base64-encoded image data.
    Base64(String),
    /// URL to image.
    Url(String),
    /// Raw pixel data (RGB, flattened).
    Pixels {
        data: Vec<u8>,
        width: u32,
        height: u32,
        channels: u8,
    },
}

impl ImageInput {
    /// Create from base64 string.
    pub fn from_base64(data: String) -> Self {
        Self {
            source: ImageSource::Base64(data),
            original_size: None,
            content_type: None,
        }
    }

    /// Create from URL.
    pub fn from_url(url: String) -> Self {
        Self {
            source: ImageSource::Url(url),
            original_size: None,
            content_type: None,
        }
    }

    /// Create from raw pixels.
    pub fn from_pixels(data: Vec<u8>, width: u32, height: u32, channels: u8) -> Self {
        Self {
            source: ImageSource::Pixels {
                data,
                width,
                height,
                channels,
            },
            original_size: Some((width, height)),
            content_type: Some("image/raw".to_string()),
        }
    }

    /// Set content type.
    pub fn with_content_type(mut self, ct: &str) -> Self {
        self.content_type = Some(ct.to_string());
        self
    }

    /// Set original size.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.original_size = Some((width, height));
        self
    }

    /// Check if image is base64.
    pub fn is_base64(&self) -> bool {
        matches!(self.source, ImageSource::Base64(_))
    }

    /// Check if image is URL.
    pub fn is_url(&self) -> bool {
        matches!(self.source, ImageSource::Url(_))
    }

    /// Estimate size in bytes.
    pub fn estimated_bytes(&self) -> usize {
        match &self.source {
            ImageSource::Base64(s) => s.len() * 3 / 4, // Base64 overhead
            ImageSource::Url(_) => 0,
            ImageSource::Pixels { data, .. } => data.len(),
        }
    }
}

/// Configuration for image processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProcessorConfig {
    /// Target image size (width, height).
    pub target_size: (u32, u32),

    /// Patch size for ViT-style processing.
    pub patch_size: u32,

    /// Mean for normalization (RGB).
    pub mean: [f32; 3],

    /// Std for normalization (RGB).
    pub std: [f32; 3],

    /// Whether to resize maintaining aspect ratio.
    pub keep_aspect_ratio: bool,

    /// Maximum number of image patches.
    pub max_patches: usize,

    /// Interpolation method.
    pub interpolation: Interpolation,
}

/// Interpolation method for resizing.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum Interpolation {
    /// Nearest neighbor.
    Nearest,
    /// Bilinear interpolation.
    #[default]
    Bilinear,
    /// Bicubic interpolation.
    Bicubic,
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        Self {
            target_size: (224, 224),
            patch_size: 14,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            keep_aspect_ratio: true,
            max_patches: 256,
            interpolation: Interpolation::Bilinear,
        }
    }
}

impl ImageProcessorConfig {
    /// Config for CLIP-style models.
    pub fn clip() -> Self {
        Self {
            target_size: (224, 224),
            patch_size: 14,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.26130258, 0.27577711],
            ..Default::default()
        }
    }

    /// Config for SigLIP-style models.
    pub fn siglip() -> Self {
        Self {
            target_size: (384, 384),
            patch_size: 14,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            ..Default::default()
        }
    }

    /// Config for high-resolution models.
    pub fn high_res() -> Self {
        Self {
            target_size: (448, 448),
            patch_size: 14,
            max_patches: 1024,
            ..Default::default()
        }
    }

    /// Calculate number of patches.
    pub fn num_patches(&self) -> usize {
        let (w, h) = self.target_size;
        let pw = w / self.patch_size;
        let ph = h / self.patch_size;
        (pw * ph) as usize
    }
}

/// Processed image ready for model input.
#[derive(Debug, Clone)]
pub struct ProcessedImage {
    /// Normalized pixel values (CHW format).
    pub pixels: Vec<f32>,

    /// Image dimensions (channels, height, width).
    pub shape: (usize, usize, usize),

    /// Number of patches.
    pub num_patches: usize,

    /// Original image size.
    pub original_size: (u32, u32),

    /// Processed size.
    pub processed_size: (u32, u32),
}

impl ProcessedImage {
    /// Get total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.0 * self.shape.1 * self.shape.2
    }

    /// Get pixels as CHW tensor shape.
    pub fn as_tensor_shape(&self) -> [usize; 3] {
        [self.shape.0, self.shape.1, self.shape.2]
    }
}

/// Image processor for VLMs.
#[derive(Debug)]
pub struct ImageProcessor {
    config: ImageProcessorConfig,
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self::new(ImageProcessorConfig::default())
    }
}

impl ImageProcessor {
    /// Create a new processor.
    pub fn new(config: ImageProcessorConfig) -> Self {
        Self { config }
    }

    /// Create for CLIP models.
    pub fn clip() -> Self {
        Self::new(ImageProcessorConfig::clip())
    }

    /// Get configuration.
    pub fn config(&self) -> &ImageProcessorConfig {
        &self.config
    }

    /// Process an image input.
    pub fn process(&self, input: &ImageInput) -> ProcessedImage {
        let (width, height) = input.original_size.unwrap_or(self.config.target_size);

        // Simulate processing - in practice, would decode, resize, normalize
        let (target_w, target_h) = self.compute_target_size(width, height);
        let channels = 3;
        let num_pixels = (channels * target_h * target_w) as usize;

        // Create normalized pixels (simulated)
        let pixels = vec![0.0f32; num_pixels];

        ProcessedImage {
            pixels,
            shape: (channels as usize, target_h as usize, target_w as usize),
            num_patches: self.config.num_patches(),
            original_size: (width, height),
            processed_size: (target_w, target_h),
        }
    }

    /// Compute target size maintaining aspect ratio if configured.
    fn compute_target_size(&self, width: u32, height: u32) -> (u32, u32) {
        let (target_w, target_h) = self.config.target_size;

        if !self.config.keep_aspect_ratio {
            return (target_w, target_h);
        }

        let scale_w = target_w as f32 / width as f32;
        let scale_h = target_h as f32 / height as f32;
        let scale = scale_w.min(scale_h);

        let new_w = (width as f32 * scale) as u32;
        let new_h = (height as f32 * scale) as u32;

        // Round to patch size
        let new_w = (new_w / self.config.patch_size) * self.config.patch_size;
        let new_h = (new_h / self.config.patch_size) * self.config.patch_size;

        (
            new_w.max(self.config.patch_size),
            new_h.max(self.config.patch_size),
        )
    }

    /// Process raw pixel data.
    pub fn process_pixels(&self, data: &[u8], width: u32, height: u32) -> ProcessedImage {
        let input = ImageInput::from_pixels(data.to_vec(), width, height, 3);
        self.process(&input)
    }

    /// Batch process multiple images.
    pub fn process_batch(&self, inputs: &[ImageInput]) -> Vec<ProcessedImage> {
        inputs.iter().map(|i| self.process(i)).collect()
    }
}

/// A multimodal request with text and images.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalRequest {
    /// Text prompt.
    pub text: String,

    /// Image inputs.
    pub images: Vec<ImageInput>,

    /// Image positions in text (index of <image> tokens).
    pub image_positions: Vec<usize>,
}

impl MultimodalRequest {
    /// Create a new request with text only.
    pub fn text_only(text: &str) -> Self {
        Self {
            text: text.to_string(),
            images: Vec::new(),
            image_positions: Vec::new(),
        }
    }

    /// Create a request with a single image.
    pub fn with_image(text: &str, image: ImageInput) -> Self {
        Self {
            text: text.to_string(),
            images: vec![image],
            image_positions: vec![0],
        }
    }

    /// Add an image.
    pub fn add_image(mut self, image: ImageInput, position: usize) -> Self {
        self.images.push(image);
        self.image_positions.push(position);
        self
    }

    /// Number of images.
    pub fn num_images(&self) -> usize {
        self.images.len()
    }

    /// Check if request has images.
    pub fn has_images(&self) -> bool {
        !self.images.is_empty()
    }

    /// Validate request.
    pub fn validate(&self) -> Result<(), VisionError> {
        if self.images.len() != self.image_positions.len() {
            return Err(VisionError::MismatchedImagePositions);
        }

        for (i, pos) in self.image_positions.iter().enumerate() {
            if *pos > self.text.len() {
                return Err(VisionError::InvalidImagePosition {
                    image_index: i,
                    position: *pos,
                });
            }
        }

        Ok(())
    }
}

/// Error type for vision operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum VisionError {
    /// Mismatched image positions.
    #[error("Number of images doesn't match number of positions")]
    MismatchedImagePositions,

    /// Invalid image position.
    #[error("Invalid position {position} for image {image_index}")]
    InvalidImagePosition { image_index: usize, position: usize },

    /// Failed to decode image.
    #[error("Failed to decode image: {0}")]
    DecodeError(String),

    /// Image too large.
    #[error("Image exceeds maximum size: {size} > {max}")]
    ImageTooLarge { size: usize, max: usize },

    /// Unsupported format.
    #[error("Unsupported image format: {0}")]
    UnsupportedFormat(String),
}

/// Vision encoder output.
#[derive(Debug, Clone)]
pub struct VisionEmbedding {
    /// Embedding vectors (num_patches × hidden_dim).
    pub embeddings: Vec<Vec<f32>>,

    /// Number of patches.
    pub num_patches: usize,

    /// Hidden dimension.
    pub hidden_dim: usize,
}

impl VisionEmbedding {
    /// Create new embedding.
    pub fn new(embeddings: Vec<Vec<f32>>) -> Self {
        let num_patches = embeddings.len();
        let hidden_dim = embeddings.first().map(|e| e.len()).unwrap_or(0);
        Self {
            embeddings,
            num_patches,
            hidden_dim,
        }
    }

    /// Get embedding for a specific patch.
    pub fn get_patch(&self, idx: usize) -> Option<&[f32]> {
        self.embeddings.get(idx).map(|v| v.as_slice())
    }

    /// Flatten to 1D vector.
    pub fn flatten(&self) -> Vec<f32> {
        self.embeddings.iter().flatten().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_input_base64() {
        let img = ImageInput::from_base64("abc123".to_string());
        assert!(img.is_base64());
        assert!(!img.is_url());
    }

    #[test]
    fn test_image_input_url() {
        let img = ImageInput::from_url("https://example.com/image.png".to_string());
        assert!(img.is_url());
        assert!(!img.is_base64());
    }

    #[test]
    fn test_image_input_pixels() {
        let data = vec![0u8; 224 * 224 * 3];
        let img = ImageInput::from_pixels(data, 224, 224, 3);
        assert_eq!(img.original_size, Some((224, 224)));
    }

    #[test]
    fn test_image_processor_config_default() {
        let config = ImageProcessorConfig::default();
        assert_eq!(config.target_size, (224, 224));
        assert_eq!(config.patch_size, 14);
    }

    #[test]
    fn test_image_processor_config_num_patches() {
        let config = ImageProcessorConfig {
            target_size: (224, 224),
            patch_size: 14,
            ..Default::default()
        };
        assert_eq!(config.num_patches(), 256); // (224/14)^2 = 16^2 = 256
    }

    #[test]
    fn test_image_processor_process() {
        let processor = ImageProcessor::default();
        let input = ImageInput::from_base64("test".to_string()).with_size(640, 480);

        let processed = processor.process(&input);
        assert_eq!(processed.original_size, (640, 480));
        assert_eq!(processed.shape.0, 3); // RGB channels
    }

    #[test]
    fn test_multimodal_request_text_only() {
        let req = MultimodalRequest::text_only("Hello, world!");
        assert!(!req.has_images());
        assert_eq!(req.num_images(), 0);
    }

    #[test]
    fn test_multimodal_request_with_image() {
        let img = ImageInput::from_base64("test".to_string());
        let req = MultimodalRequest::with_image("Describe this image:", img);

        assert!(req.has_images());
        assert_eq!(req.num_images(), 1);
    }

    #[test]
    fn test_multimodal_request_validate() {
        let img = ImageInput::from_base64("test".to_string());
        let req = MultimodalRequest::with_image("Describe this:", img);

        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_multimodal_request_validate_invalid() {
        let req = MultimodalRequest {
            text: "test".to_string(),
            images: vec![ImageInput::from_base64("a".to_string())],
            image_positions: vec![0, 1], // Mismatch
        };

        assert!(req.validate().is_err());
    }

    #[test]
    fn test_vision_embedding() {
        let embeddings = vec![vec![1.0, 2.0, 3.0]; 16];
        let emb = VisionEmbedding::new(embeddings);

        assert_eq!(emb.num_patches, 16);
        assert_eq!(emb.hidden_dim, 3);
        assert!(emb.get_patch(0).is_some());
        assert!(emb.get_patch(20).is_none());
    }

    #[test]
    fn test_vision_embedding_flatten() {
        let embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let emb = VisionEmbedding::new(embeddings);

        assert_eq!(emb.flatten(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_image_input_estimated_bytes() {
        let b64 = ImageInput::from_base64("AAAA".to_string()); // 4 chars = 3 bytes
        assert_eq!(b64.estimated_bytes(), 3);

        let url = ImageInput::from_url("https://example.com".to_string());
        assert_eq!(url.estimated_bytes(), 0);
    }
}
