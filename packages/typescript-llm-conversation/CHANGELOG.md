# Changelog

All notable changes to this package will be documented in this file.

## [1.0.9] - 2026-09-02

### Expanded

- Added DeepSeek support. It's so similar to OpenAI that I made it simply re-use the OpenAI pathway.


## [1.0.8] - 2026-08-05

### Fixed

- Corrected a naming-convention error: the `token_usage` field (Pythonic snake_case) introduced in 1.0.7 has been renamed to `tokenUsage` (camelCase) to follow TypeScript conventions. This is a breaking change for any consumer that accessed `client.token_usage` directly.

## [1.0.7] - 2026-08-04

### Added

- Added tracking of token usage counts. The client objects now monkey-patch a top-level field called `token_usage`, which tallies totals about how many tokens have been spent on LLM submit calls during that client object's lifetime.

## [1.0.6] - 2026-03-28

### Fixed

- Fixed package consumption in CommonJS backends by adding a CommonJS build output and wiring the `require` condition in package exports.
- Fixed root package export compatibility for consumer resolvers by adding robust root export conditions.

### Changed

- Build now emits dual module formats (ESM and CommonJS) for broader downstream compatibility.

## [1.0.5] - 2026-03-12

### Changed

- Bumped package version after updating npmjs OIDC credentials for release publishing.

## [1.0.4] - 2026-03-12

### Changed

- Bumped package version to republish while fixing the TypeScript CI/CD workflow.

## [1.0.3] - 2026-03-12

### Fixed

- Fixed the TypeScript CI/CD pipeline: removed `registry-url` from the `setup-node` action, which was writing an `.npmrc` that conflicted with npm Trusted Publisher OIDC auth and caused `npm publish` to fail with a 404.
- Fixed test files being included in the published npm tarball by removing `tests/**/*` from `tsconfig.json` include paths.

## [1.0.2] - 2026-03-12

### Changed

- Made the GPT shotgun reliability integration test more efficient by reducing redundant work across attempts.
- Made the GPT shotgun reliability integration test less flaky by improving handling of stochastic live-model responses.

## [1.0.1] - 2026-03-12

### Added

- Added JSONSchemaFormat enum shorthand support for:
	- `[String, ["alpha", "beta", ...]]`
	- `["string", ["alpha", "beta", ...]]`
	- `[String, "description", ["alpha", "beta", ...]]`
	- `["string", "description", ["alpha", "beta", ...]]`

### Fixed

- Preserved tuple disambiguation behavior so `["string", "description"]` remains a type+description tuple, not an enum.
- Preserved existing tuple validation behavior for non-numeric and numeric tuple branches while adding string enum shorthands.

## [1.0.0] - 2026-03-12

### Added

- Initial public release.

