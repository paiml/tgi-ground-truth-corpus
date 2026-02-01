//! Quantization Example
//!
//! Demonstrates TGI's quantization types and their
//! compression characteristics.
//!
//! # Run
//!
//! ```bash
//! cargo run --example quantization
//! ```

use tgi_gtc::quantization::QuantType;

fn main() {
    println!("=== Quantization Types Demo ===\n");

    println!("GGUF Quantization Formats:");
    println!("{}", "=".repeat(60));
    println!(
        "{:<10} {:>15} {:>20} {:>12}",
        "Type", "Bits/Weight", "Compression Ratio", "Memory (7B)"
    );
    println!("{}", "-".repeat(60));

    let quant_types = [
        QuantType::Q4_0,
        QuantType::Q4_K,
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::Q8_0,
        QuantType::F16,
        QuantType::F32,
    ];

    // 7B model has ~7 billion parameters
    let params_7b: f64 = 7_000_000_000.0;

    for qtype in quant_types {
        let bits = qtype.bits_per_weight();
        let ratio = qtype.compression_ratio();
        let memory_gb = (params_7b * bits as f64 / 8.0) / 1_000_000_000.0;

        println!(
            "{:<10} {:>15.1} {:>19.1}x {:>10.1} GB",
            format!("{:?}", qtype),
            bits,
            ratio,
            memory_gb
        );
    }

    println!("{}", "=".repeat(60));

    // Show memory savings
    println!("\nMemory Savings for 7B Model:");
    println!("{}", "-".repeat(40));

    let f32_memory = (params_7b * 32.0 / 8.0) / 1_000_000_000.0;
    println!("  F32 baseline: {:.1} GB", f32_memory);

    for qtype in [QuantType::Q4_K, QuantType::Q8_0, QuantType::F16] {
        let memory = (params_7b * qtype.bits_per_weight() as f64 / 8.0) / 1_000_000_000.0;
        let savings = (1.0 - memory / f32_memory) * 100.0;
        println!("  {:?}: {:.1} GB ({:.0}% smaller)", qtype, memory, savings);
    }

    // Recommendations
    println!("\nRecommendations:");
    println!("{}", "-".repeat(40));
    println!("  • Q4_K: Best for consumer GPUs (RTX 3080/4080)");
    println!("  • Q5_K: Balanced quality/size for prosumer GPUs");
    println!("  • Q8_0: High quality, needs more VRAM");
    println!("  • F16: Full precision, production servers");

    println!("\n=== Demo Complete ===");
}
