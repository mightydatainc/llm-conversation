# llm-conversation

`llm-conversation` is a cross-language toolkit for building reliable LLM-powered features.

The project exists to remove repeated integration work from application teams. Instead of re-implementing conversation state, structured-output handling, and reliability controls in every codebase, this repository provides shared building blocks with provider-agnostic design.

## Purpose: Simplify Multi-Stage LLM Conversation State Management and Structured Output Queries

- Application code needs deterministic behavior at boundaries where model output is probabilistic.
- Conversation state handling, role sequencing, and request shaping are easy to get subtly wrong.
- Structured-output pipelines often degrade into ad hoc parsing and prompt-only guardrails.
- Teams maintaining multiple language stacks need equivalent semantics and failure behavior.

## Core Capabilities

This toolkit provides reusable components for the failure-prone parts of LLM integration:

- Stateful conversation containers for multi-step prompting and revision workflows.
- Stateless submit utilities for one-shot tasks and service-layer integration.
- Structured JSON response modes, including both advisory and schema-constrained workflows.
- Schema shorthand helpers to reduce boilerplate when defining output structure.
- Reliability controls such as retries, backoff, and optional multi-worker shotgunning.
- Cross-language parity so Python and TypeScript implementations align on behavior.

## Typical Use Cases

- Normalizing external-source data (client, vendor, or user-provided fields) into canonical internal structures.
- Building backend workflows that require predictable machine-readable LLM output.
- Running multi-turn transformations where intermediate context and role management matter.
- Replacing hand-rolled parsing and brittle prompt-only output control.

## Scope

This repository focuses on repeated integration concerns:

- Managing multi-message conversations over time.
- Working with structured JSON outputs safely.
- Keeping shared semantics aligned across implementations.

This repository is intentionally _not_:

- An agent framework or orchestration platform.
- A full application starter or workflow engine.
- A replacement for domain-specific validation/business rules.

## Repository Layout

- `packages/python-llm-conversation`: Python package implementation.
- `packages/typescript-llm-conversation`: TypeScript package implementation.

Both packages follow the same product intent and are developed together to maintain consistency.

## Design Principles

- Minimal, composable abstractions.
- Explicit behavior and stable contracts.
- Production-first defaults.
- Parity where shared behavior matters most.
- Provider-agnostic design with clear adaptation points.

## Engineering Posture

- Prefer predictable interfaces over magic abstractions.
- Keep failure modes visible and controllable.
- Encode cross-language behavior with tests rather than convention.

## Documentation

The root README describes intent and project-level context.

For package usage, API reference, and language-specific setup, see:

- `packages/python-llm-conversation/README.md`
- `packages/typescript-llm-conversation/README.md`

## Quality and Reliability

The project prioritizes practical correctness over synthetic demos. Package-level test suites are designed to protect real integration behavior, including conversation-state management and structured-output workflows.

Python unit tests are run with `unittest` (see the Testing section in `packages/python-llm-conversation/README.md`).

## Releases

Python and TypeScript packages are versioned and released from this monorepo using repository automation. Release mechanics are kept with each package and workflow configuration.

## License

MIT. See `LICENSE`.
