# Vortex RFC Site

Static site generator for Vortex RFC proposals built with Bun.

## Project Structure

```
index.ts      - Main build script and dev server
styles.css    - Site styling (light/dark themes)
proposed/     - RFC markdown files in proposed state
accepted/     - RFC markdown files in accepted state
completed/    - RFC markdown files in completed state
dist/         - Build output (gitignored)
```

RFC filenames follow the format `NNNN-slug.md` (e.g., `0001-galp-patches.md`).
Numbering is global across all states - no duplicates allowed.

## Commands

```sh
bun run build    # Build static site to ./dist/
bun run dev      # Dev server with live reload on localhost:3000
bun run clean    # Remove dist/
```

## How the Build Works

1. Scans `proposed/`, `accepted/`, `completed/` for RFC files
2. Parses RFC number from filename (e.g., `0002-foo.md` → RFC 0002)
3. Determines state from containing folder
4. Extracts title from first `# ` heading
5. Converts markdown to HTML using `Bun.markdown.html()`
6. Generates `dist/index.html` (table of contents with filter UI)
7. Generates `dist/rfc/{number}.html` for each RFC

## Dev Server

- Uses `Bun.serve()` to serve static files from `dist/`
- Watches `proposed/`, `accepted/`, `completed/`, and `styles.css` for changes
- SSE endpoint at `/__reload` for live reload

## RFC States

RFCs progress through three states by moving files between folders:

- **proposed**: New RFCs under discussion
- **accepted**: Approved RFCs ready for implementation
- **completed**: Fully implemented RFCs

The index page shows a state pill for each RFC and supports filtering by state.

## Styling

- CSS custom properties for theming (`--bg`, `--fg`, `--link`, etc.)
- System preference detection via `prefers-color-scheme`
- Two-state toggle: dark ↔ light
- Theme persisted to localStorage

## Adding Features

- Keep dependencies minimal (only `@types/bun` currently)
- Use Bun built-ins: `Bun.file`, `Bun.Glob`, `Bun.markdown`, `Bun.serve`
