- Start Date: 2026-03-03
- Authors: Joe Isaacs
- Tracking Issue: TBD

## Summary

A backward compatibility testing framework for the Vortex file format, consisting of a **generator** that writes fixture `.vortex` files and a **reader** that validates them. Both are maintained on `develop` and backported to selected release branches so that each version can produce fixtures with its writer and verify fixtures from all earlier versions with its reader. Fixtures are stored in a public S3 bucket and validated in a weekly CI job.

## Motivation

Vortex guarantees backward compatibility from release 0.36.0, but there are no tests validating this. Format-level changes can silently break old-file compatibility, and without automated checks we won't know until a user hits it in production.

## Design

### Overview

We maintain one set of fixture `.vortex` files per release, from 0.36.0 through to the latest. Generation is manual (triggered per release or backfilled), so some intermediate versions may be skipped. The fixture sets are stored in a public S3 bucket, and a weekly CI job validates that the current reader can still open all of them.

Two binaries in a standalone crate (`vortex-test/compat-gen/`), not a workspace member. The crate uses path deps to workspace crates, so it compiles against whatever version is checked out.

```
  v0.36.0                  v0.58.0                  HEAD
  ┌──────────┐             ┌──────────┐             ┌──────────┐
  │compat-gen│──upload──>  │compat-gen│──upload──>  │compat-gen│──upload──>
  └──────────┘     │       └──────────┘     │       └──────────┘     │
                   v                        v                        v
              S3: v0.36.0/             S3: v0.58.0/             S3: vHEAD/
                   │                        │                        │
                   └────────────┬───────────┘────────────────────────┘
                                v
                          ┌────────────┐
                          │compat-test │  (at any version: reads ALL
                          │            │   fixtures from <= that version)
                          └────────────┘
```

| Binary        | Purpose                                                                         |
| ------------- | ------------------------------------------------------------------------------- |
| `compat-gen`  | Write fixture `.vortex` files + a `manifest.json` listing them                  |
| `compat-test` | Fetch fixtures from S3, read them, rebuild expected arrays, `assert_arrays_eq!` |

When cherry-picked onto an old release branch the only thing that changes is a thin API adapter layer (~20 lines that call the version's write/read API). Everything else — fixture definitions, correctness checks — stays identical.

### Fixture Suite

**Synthetic fixtures** (deterministic, hardcoded values):

| File                   | Schema                                          | Data                                   | Purpose                    |
| ---------------------- | ----------------------------------------------- | -------------------------------------- | -------------------------- |
| `primitives.vortex`    | `Struct{u8, u16, u32, u64, i32, i64, f32, f64}` | Boundary values (0, min, max) per type | Primitive type round-trip  |
| `strings.vortex`       | `Struct{Utf8}`                                  | `["", "hello", "こんにちは", "🦀"]`    | String encoding round-trip |
| `booleans.vortex`      | `Struct{Bool}`                                  | `[true, false, true, true, false]`     | Bool round-trip            |
| `nullable.vortex`      | `Struct{Nullable<i32>, Nullable<Utf8>}`         | Mix of values and nulls                | Null handling              |
| `struct_nested.vortex` | `Struct{Struct{i32, Utf8}, f64}`                | Nested struct                          | Nested type round-trip     |
| `chunked.vortex`       | Chunked `Struct{u32}`                           | 3 chunks of 1000 rows each             | Multi-chunk files          |

Every stable array encoding should also contribute a fixture file — a struct with multiple columns, each using a different encoding of that array type. This ensures that encoding-specific read paths are exercised across versions.

**Realistic fixtures** (real-world schemas and data distributions):

| File                        | Source                               | Rows | Purpose                                     |
| --------------------------- | ------------------------------------ | ---- | ------------------------------------------- |
| `tpch_lineitem.vortex`      | TPC-H SF 0.01, `lineitem` table      | ~60K | Real-world numeric + string schema          |
| `tpch_orders.vortex`        | TPC-H SF 0.01, `orders` table        | ~15K | Date + decimal types                        |
| `clickbench_hits_1k.vortex` | First 1000 rows of ClickBench `hits` | 1000 | Wide table (105 columns), deep nested types |

SF 0.01 is used instead of 0.1 to keep fixture file sizes small (~few MB) so downloads in tests are fast.

### Fixture Trait

Each fixture implements a common trait that the generator and tester both use:

```rust
trait Fixture {
    /// The filename for this fixture (e.g., "primitives.vortex").
    fn name(&self) -> &str;

    /// Build the expected array. Must be deterministic.
    fn build(&self) -> ArrayRef;
}
```

A single `Fixture` impl is sufficient for both generation and validation:

- `compat-gen` calls `build()` and writes the result to disk
- `compat-test` calls the same `build()` to produce the expected array and compares it against what was read from the old file via `assert_arrays_eq!`

All fixture types — synthetic, TPC-H, ClickBench — implement the same trait. The registry is just a `Vec<Box<dyn Fixture>>`.

```rust
// Synthetic: hardcoded values
struct PrimitivesFixture;
impl Fixture for PrimitivesFixture {
    fn name(&self) -> &str { "primitives.vortex" }
    fn build(&self) -> ArrayRef {
        StructArray::from_fields(&[
            ("u8",  vec![0u8, 128, 255].into_array()),
            ("u16", vec![0u16, 32768, 65535].into_array()),
            // ...
        ]).into_array()
    }
}

// TPC-H: deterministic via tpchgen
struct TpchLineitemFixture;
impl Fixture for TpchLineitemFixture {
    fn name(&self) -> &str { "tpch_lineitem.vortex" }
    fn build(&self) -> ArrayRef {
        // generate via tpchgen-arrow at SF 0.01
    }
}
```

### Correctness Strategy

Correctness is validated by **comparing arrays in memory** — no checksums or spot-checks needed.

For every fixture in every version:

1. Download the old `.vortex` file from S3 (written by an older Vortex version)
2. Read it into an array with the current reader
3. Call `fixture.build()` to produce the expected array at the current version
4. `assert_arrays_eq!(actual, expected)`

This works because all fixture builders are deterministic: synthetic fixtures use hardcoded values, TPC-H uses `tpchgen` (deterministic per SF), and ClickBench uses an immutable public parquet file.

### Manifest Format

Each version's fixture set includes a `manifest.json` sidecar that lists the fixtures available for that version. This allows `compat-test` to discover what to download and handles the case where newer versions add new fixture types.

```json
{
  "version": "0.36.0",
  "generated_at": "2025-01-15T10:30:00Z",
  "fixtures": [
    "primitives.vortex",
    "strings.vortex",
    "booleans.vortex",
    "nullable.vortex",
    "struct_nested.vortex",
    "chunked.vortex",
    "tpch_lineitem.vortex",
    "tpch_orders.vortex",
    "clickbench_hits_1k.vortex"
  ]
}
```

### API Adapter Layer

The only part that changes per version. When cherry-picking onto an old branch, you adapt this module (~20 lines).

```rust
// ---- adapter.rs (current API, HEAD) ----
use vortex::VortexSession;

pub fn write_file(path: &Path, stream: impl ArrayStream) -> Result<()> {
    let session = VortexSession::default();
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let mut file = tokio::fs::File::create(path).await?;
        session.write_options().write(&mut file, stream).await?;
        Ok(())
    })
}

pub fn read_file(bytes: Bytes) -> Result<VortexFile> {
    let session = VortexSession::default();
    session.open_options().open_buffer(bytes)
}
```

```rust
// ---- adapter.rs (0.36.0 API) ----
pub fn write_file(path: &Path, stream: impl ArrayStream) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let mut file = tokio::fs::File::create(path).await?;
        VortexWriteOptions::default().write(&mut file, stream).await?;
        Ok(())
    })
}

pub fn read_file(bytes: Bytes) -> Result<VortexFile> {
    VortexOpenOptions::in_memory().open(bytes)
}
```

### S3 Layout (Public Bucket)

Fixtures are stored in a **public S3 bucket** so that anyone can run `compat-test` locally without credentials, and CI doesn't need special S3 auth for reads. Only uploads (from `compat-gen`) require write credentials.

```
s3://vortex-compat-fixtures/    (public read)
  v0.36.0/
    manifest.json
    primitives.vortex
    strings.vortex
    ...
  v0.58.0/
    manifest.json
    ...
```

Fixtures are also accessible via plain HTTPS (`https://vortex-compat-fixtures.s3.amazonaws.com/v0.36.0/primitives.vortex`), so `compat-test` can use either anonymous S3 access or plain HTTP — no AWS SDK configuration required.

### Adding New Fixtures in Future Releases

When a future release adds support for a new type or feature (e.g., list arrays, extension types), we want to add a fixture that exercises it.

The manifest handles this naturally. Each version's `manifest.json` lists exactly which fixtures exist. `compat-test` only validates what's listed:

```
v0.36.0/manifest.json  →  ["primitives.vortex", "strings.vortex", ...]
v0.65.0/manifest.json  →  ["primitives.vortex", "strings.vortex", ..., "list.vortex"]
```

Adding a new fixture:

1. Add the builder function in `fixtures/` (e.g., `build_list_array()`)
2. Register it in `fixtures/mod.rs` so `compat-gen` includes it
3. Tag a release — the pre-release CI job generates fixtures including the new one
4. Old versions are untouched — their manifests don't mention the new fixture

The `FIXTURE_REGISTRY` maps fixture names to builder functions. If a fixture name from an old manifest isn't in the current registry (e.g., a fixture was retired), it's skipped with a warning rather than failing.

```rust
for version in discover_versions_from_s3() {
    let manifest = fetch_manifest(version);
    for fixture_name in manifest.fixtures {
        if let Some(builder) = FIXTURE_REGISTRY.get(fixture_name) {
            let old_bytes = fetch_fixture(version, fixture_name);
            let old_array = read_file(old_bytes);
            let expected = builder();
            assert_arrays_eq!(old_array, expected);
        } else {
            warn!("Unknown fixture {fixture_name} in {version}, skipping");
        }
    }
}
```

### CI Workflow

**Pre-release upload** (`compat-gen-upload.yml`): Triggered automatically when a version tag is pushed, or manually via `workflow_dispatch` with a tag input. Generates fixtures at that version and uploads to the public S3 bucket, replacing any existing files under that version's prefix only (other versions are untouched).

```yaml
name: Compat Fixture Upload
on:
  push:
    tags: ["[0-9]+.[0-9]+.[0-9]+"]
  workflow_dispatch:
    inputs:
      tag:
        description: "Git tag to generate fixtures for (e.g. 0.58.0)"
        required: true

jobs:
  upload-fixtures:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.inputs.tag || github.ref_name }}

      - uses: dtolnay/rust-toolchain@stable

      - name: Generate fixtures
        run: |
          VERSION=${{ github.event.inputs.tag || github.ref_name }}
          cargo run --manifest-path vortex-test/compat-gen/Cargo.toml \
            --bin compat-gen -- --version "$VERSION" --output /tmp/fixtures/

      - name: Upload to S3
        run: |
          VERSION=${{ github.event.inputs.tag || github.ref_name }}
          aws s3 cp /tmp/fixtures/ \
            s3://vortex-compat-fixtures/v${VERSION}/ --recursive
```

For backfilling old versions (0.36.0, etc.) that predate the framework, use `workflow_dispatch` manually — the cherry-picked `adapter.rs` handles the old API.

**Weekly compat check** (`compat-test-weekly.yml`): Runs weekly and on-demand. Downloads all fixture versions from S3 (public, no credentials needed) and validates them against the current reader at HEAD.

```yaml
name: Compat Test
on:
  schedule:
    - cron: "0 6 * * 1"
  workflow_dispatch: {}

jobs:
  compat-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Run compat tests
        run: |
          cargo run --manifest-path vortex-test/compat-gen/Cargo.toml \
            --bin compat-test
```

### Crate Layout

```
vortex-test/compat-gen/
  Cargo.toml          # standalone binary crate, path deps to workspace
  src/
    main.rs           # CLI entry point (--bin compat-gen)
    adapter.rs        # version-specific write/read API (~20 lines to adapt)
    fixtures/
      mod.rs          # fixture registry — maps name → builder function
      synthetic.rs    # build_primitives(), build_strings(), etc.
      tpch.rs         # build_tpch_lineitem(), build_tpch_orders()
      clickbench.rs   # build_clickbench_hits_1k()
    manifest.rs       # manifest.json serde (just a list of fixture names)
    test_main.rs      # --bin compat-test entry point
    validate.rs       # fetch from S3 + assert_arrays_eq! logic
```

The `fixtures/` module is the shared core: `compat-gen` calls each builder and writes to disk; `compat-test` calls the same builders to produce expected arrays and compares them against what was read from old files.

The `Cargo.toml` is not listed in workspace members, so it doesn't affect the main build:

```toml
[package]
name = "vortex-compat"
version = "0.1.0"

[[bin]]
name = "compat-gen"
path = "src/main.rs"

[[bin]]
name = "compat-test"
path = "src/test_main.rs"

[dependencies]
vortex = { path = "../../vortex" }
vortex-array = { path = "../../vortex-array" }
vortex-file = { path = "../../vortex-file" }
vortex-buffer = { path = "../../vortex-buffer" }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
object_store = { version = "0.11", features = ["aws", "http"] }
clap = { version = "4", features = ["derive"] }
tpchgen = "2"
tpchgen-arrow = "2"
arrow = "57"
```

## Compatibility

This RFC does not change the file format, wire format, or any public APIs. It is purely additive testing infrastructure.

The `compat-gen` crate is standalone and not a workspace member, so it has no impact on the build or dependency graph of the main project.

The only operational requirement is a public S3 bucket for fixture storage. Read access is anonymous; write access is restricted to CI with OIDC credentials.

## Drawbacks

- **S3 dependency**: Tests require network access to fetch fixtures. If S3 is unreachable, the weekly check skips rather than fails, but this means a full week could pass without validation.
- **Cherry-pick maintenance**: Backporting to old releases requires adapting `adapter.rs` to each version's write/read API. This is a small one-time cost per version (~20 lines) but does require someone to do it manually for versions that predate the framework.
- **Fixture storage cost**: Each version adds ~10–20 MB of fixtures to S3. At one version per release, this grows slowly, but over many years it accumulates.
- **`tpchgen` determinism assumption**: If the `tpchgen` crate changes its output for the same scale factor in a future version, the TPC-H comparison will fail. This is mitigable by pinning the crate version or regenerating fixtures.

## Prior Art

- **Apache Parquet**: The `parquet-testing` repo stores fixture files in git. Works because Parquet fixtures are small, but doesn't scale well. The Parquet project also has a formal compatibility test suite that validates readers against writers from different language implementations.
- **Apache Arrow IPC**: The `arrow-integration` project generates IPC files from each language implementation and cross-validates them. Similar to our approach but tests cross-language compat rather than cross-version.
- **Protocol Buffers**: Google maintains a `conformance` test suite that validates proto2/proto3 encoding across versions. The test runner is a separate binary, similar to our `compat-test`.
- **SQLite**: Maintains a set of test databases going back to very early versions. Their `sqldiff` tool can compare databases for equality.

## Related RFCs

This RFC depends on or is closely related to several topics that warrant their own RFCs:

- **Stable array encodings**: A separate RFC should define what it means for an array encoding to be "stable" — i.e., the encoding's serialized format is frozen and the reader must support it across versions. This includes criteria for promoting an encoding to stable, the process for deprecating one, and what guarantees stable implies (e.g., bit-level format stability, metadata schema stability).
- **File format versioning**: How does the file format itself evolve? If the footer layout, segment format, or layout metadata changes, how do we version that and maintain backward compat? This RFC tests the outcome but doesn't define the versioning mechanism.
- **Encoding registry and discovery**: When the reader encounters an encoding ID it doesn't recognize (e.g., a file written by a newer version with a new encoding), what happens? Should it fail, skip, or fall back? This affects how we handle forward compatibility.

## Unresolved Questions

- **Bucket name and region**: The exact S3 bucket name (`vortex-compat-fixtures`) and region need to be decided. It should be in `us-east-1` for lowest latency from GitHub Actions runners.
- **Which versions to backfill**: We need to decide which historical versions to generate fixtures for. At minimum 0.36.0 (the first stable version) and the latest release, but intermediate versions (0.45.0, 0.50.0, 0.58.0) would increase coverage.

## Future Possibilities

- **Automated release pipeline**: When cutting a new release, the CI pipeline could automatically run `compat-gen` and upload fixtures, removing the manual step entirely.
- **Cross-language compat**: Once the Python and Java bindings have file readers, extend the framework to validate that Python/Java can read files written by the Rust writer (and vice versa).
