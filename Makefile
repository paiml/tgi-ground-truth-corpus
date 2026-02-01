# TGI Ground Truth Corpus - Quality Gates
# PMAT A+ Compliance: 95% coverage, 80% mutation score

.PHONY: all build test lint fmt check coverage mutants clean tier1 tier2 tier3 tier4

# Default target
all: tier2

# =============================================================================
# Build Targets
# =============================================================================

build:
	cargo build

build-release:
	cargo build --release

# =============================================================================
# Quality Tiers (Certeza Methodology)
# =============================================================================

# Tier 1: On-save (<1s)
tier1: fmt-check check clippy

# Tier 2: Pre-commit (<5s)
tier2: tier1 test-lib

# Tier 3: Pre-push (1-5min)
tier3: tier2 test coverage-check

# Tier 4: CI/CD (full)
tier4: tier3 test-release mutants doc

# =============================================================================
# Testing
# =============================================================================

test:
	cargo nextest run --all-features

test-lib:
	cargo nextest run --lib

test-release:
	cargo nextest run --release --all-features

test-doc:
	cargo test --doc

# =============================================================================
# Linting & Formatting
# =============================================================================

lint: clippy

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

fmt:
	cargo fmt

fmt-check:
	cargo fmt -- --check

check:
	cargo check --all-targets --all-features

# =============================================================================
# Coverage (95% target)
# =============================================================================

coverage:
	cargo llvm-cov --all-features --html
	@echo "Report: target/llvm-cov/html/index.html"

coverage-check:
	@cargo llvm-cov --all-features --fail-under-lines 95

coverage-lcov:
	cargo llvm-cov --all-features --lcov --output-path lcov.info

# =============================================================================
# Mutation Testing (80% target)
# =============================================================================

mutants:
	cargo mutants --timeout 60 -- --all-features

mutants-fast:
	cargo mutants --timeout 30 --jobs 4 -- --lib

mutants-file:
	cargo mutants --timeout 60 --file $(FILE) -- --all-features

# =============================================================================
# Documentation
# =============================================================================

doc:
	cargo doc --no-deps --all-features

doc-open:
	cargo doc --no-deps --all-features --open

book:
	mdbook build book

book-serve:
	mdbook serve book --open

# =============================================================================
# Benchmarks
# =============================================================================

bench:
	cargo bench

bench-save:
	cargo bench -- --save-baseline main

bench-compare:
	cargo bench -- --baseline main

# =============================================================================
# Utilities
# =============================================================================

clean:
	cargo clean
	rm -rf target/llvm-cov

tree:
	@find src -name "*.rs" | head -50

loc:
	@tokei src tests

deps:
	cargo tree --depth 1

outdated:
	cargo outdated

audit:
	cargo audit

# =============================================================================
# Development Helpers
# =============================================================================

watch:
	cargo watch -x "check --all-features"

watch-test:
	cargo watch -x "nextest run --lib"

# =============================================================================
# CI Simulation
# =============================================================================

ci: tier4
	@echo "✅ All CI checks passed"
