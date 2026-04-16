# Vortex RFC Site

Static site generator for Vortex RFC proposals built with Bun.

## Project Structure

```
index.ts      - Main build script and dev server
styles.css    - Site styling (light/dark themes)
rfcs/         - RFC markdown files (merged to develop = accepted)
dist/         - Build output (gitignored)
```

RFC filenames follow the format `NNNN-slug.md` (e.g., `0027-patches-format.md`).
The RFC number must match the PR number used to propose it. No duplicate numbers allowed.

## Commands

```sh
bun run build    # Build static site to ./dist/
bun run dev      # Dev server with live reload on localhost:3000
bun run clean    # Remove dist/
```

## How the Build Works

1. Scans `rfcs/` for RFC markdown files
2. Parses RFC number from filename (e.g., `0002-foo.md` → RFC 0002)
3. Extracts title from first `# ` heading
4. Converts markdown to HTML using `Bun.markdown.html()`
5. Generates `dist/index.html` (table of contents)
6. Generates `dist/rfc/{number}.html` for each RFC

## Dev Server

- Uses `Bun.serve()` to serve static files from `dist/`
- Watches `rfcs/` and `styles.css` for changes
- SSE endpoint at `/__reload` for live reload

## RFC Workflow

1. Open a PR with a new file `rfcs/NNNN-slug.md` where NNNN matches the PR number
2. PR builds a preview artifact for reviewers
3. Merging the PR to `develop` accepts the RFC

## Styling

- CSS custom properties for theming (`--bg`, `--fg`, `--link`, etc.)
- System preference detection via `prefers-color-scheme`
- Two-state toggle: dark ↔ light
- Theme persisted to localStorage

## Adding Features

- Keep dependencies minimal (only `@types/bun` currently)
- Use Bun built-ins: `Bun.file`, `Bun.Glob`, `Bun.markdown`, `Bun.serve`
