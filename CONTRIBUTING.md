# Contributing to WaveML-JAX

Thank you for your interest in contributing to **WaveML-JAX** (Lightborne Intelligence).

This repository is a **reference implementation** of **Error Rectification by Alignment (ERA)** and wave-native learning primitives. We welcome contributions that improve correctness, clarity, reproducibility, and research utility—while keeping the project focused, auditable, and stable.

---

## Code of Conduct

Be respectful. Be precise. Assume good intent.  
Harassment, discrimination, or hostile conduct will not be tolerated.

---

## Scope: What We Accept

### ✅ Good contributions
- Bug fixes and numerical stability improvements
- Performance improvements that preserve correctness and reproducibility
- Improved tests and validation suites
- Documentation improvements (docstrings, explanations, figures, usage examples)
- New **synthetic** benchmarks that reveal collapse vs graceful degradation
- Baselines implemented fairly (parameter-matched where possible)
- Small, well-justified features that align with ERA governance principles

### ❌ Out of scope
- Production deployment tooling, data pipelines, or infra scaffolding
- Domain-calibrated “magic numbers,” private operating envelopes, or tuned profiles
- Large model releases or heavyweight training recipes presented as “SOTA”
- Unreviewable refactors that reduce interpretability
- Changes that weaken ERA invariants or obscure state semantics

If you’re unsure whether something is in scope, open an issue first.

---

## Guiding Principles

1. **Correctness over cleverness**  
   This is a governance library. A small correct implementation is better than a complex one.

2. **Reproducibility is non-negotiable**  
   Deterministic experiments, fixed seeds, and stable metrics are preferred.

3. **Governance first**  
   ERA invariants are not optional. If you propose a change that alters invariant behavior, you must justify it with tests and clear reasoning.

4. **Fair comparisons**  
   Baselines must be parameter-matched or capacity-matched, with comparable training setups.

---

## How to Contribute

### 1) Open an Issue
For bugs, feature ideas, or research proposals:
- Describe the problem clearly
- Include minimal reproduction steps
- Provide expected vs observed behavior
- Attach logs/plots if relevant

### 2) Submit a Pull Request (PR)
Your PR should include:
- A clear description of the change and why it matters
- Tests that cover the change
- Notes on reproducibility (seed, deterministic settings)
- Benchmarks if performance or stability claims are made

---

## Branching & PR Rules

- Create a feature branch from `main`
- Keep PRs small and focused
- One logical change per PR
- Avoid drive-by refactors (format-only, rename-only) unless requested
- If a PR touches core invariants, add a short “Invariant Impact” note

---

## Coding Standards

- Python 3.10+ (or project default)
- JAX-first style: pure functions, explicit PRNG keys, minimal side-effects
- Type hints where practical
- Avoid hidden state; prefer explicit `NamedTuple`/dataclasses for states
- No silent shape changes—validate shapes in tests
- Prefer simple readable math over micro-optimizations

---

## Reproducibility Requirements

If your change affects training or evaluation:

- Use fixed random seeds
- Use deterministic JAX settings where applicable
- Provide a script or command that reproduces the result
- Report metrics with:
  - dataset/task description
  - model size / parameter count
  - runtime notes if relevant

For sweep experiments (noise, delay, SNR), include:
- the sweep range
- the aggregation method
- the exact metric definition

---

## Testing

All PRs should:
- Pass existing tests
- Add new tests for new behavior
- Include stability checks where relevant (no NaNs, bounded norms)

Recommended test types:
- Unit tests: invariant enforcement, shape correctness
- Property tests: bounded energy, phase wrapping stability
- Regression tests: previously failing cases

---

## Documentation

Documentation improvements are strongly encouraged.

When adding new concepts, include:
- a short explanation of “why”
- the mathematical form (when appropriate)
- a minimal runnable example

---

## Security & Responsible Disclosure

If you believe you’ve found a security issue or vulnerability:
- Do **not** open a public issue.
- Email the maintainers (contact information in the repository).

---

## Licensing & Contributor Terms

WaveML-JAX is licensed under **Apache License 2.0**.

By submitting a contribution, you agree that:
- Your contribution is licensed under Apache-2.0
- You have the right to submit the work
- You grant the standard Apache-2.0 patent license for your contribution (as applicable)

This helps ensure the project remains usable for research and industry evaluation.

---

## Maintainer Review Criteria

We prioritize:
- correctness and clarity
- reproducibility
- principled governance alignment
- minimal, auditable changes
- fair baselines and honest metrics

We may request changes or decline contributions that:
- introduce ambiguity in invariants
- reduce interpretability
- expand scope beyond a reference implementation
- cannot be reproduced reliably

---

## Thank You

If you contribute thoughtfully, you strengthen the integrity of the work.

**Memorization collapses. Alignment endures.**
