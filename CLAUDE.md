# Vortex RFC Site

Static site generator for Vortex RFC proposals built with Bun.

## Project Structure

```
index.ts      - Main build script and dev server
styles.css    - Site styling (light/dark themes)
proposals/    - RFC markdown files (format: NNNN-slug.md)
dist/         - Build output (gitignored)
```

## Commands

```sh
bun run build    # Build static site to ./dist/
bun run dev      # Dev server with live reload on localhost:3000
bun run clean    # Remove dist/
```

## How the Build Works

1. Scans `proposals/*.md` for RFC files
2. Parses RFC number from filename (e.g., `0002-foo.md` → RFC 0002)
3. Extracts title from first `# ` heading
4. Converts markdown to HTML using `Bun.markdown.html()`
5. Generates `dist/index.html` (table of contents)
6. Generates `dist/rfc/{number}.html` for each RFC

## Dev Server

- Uses `Bun.serve()` to serve static files from `dist/`
- Watches `proposals/` and `styles.css` for changes
- SSE endpoint at `/__reload` for live reload

## Styling

- CSS custom properties for theming (`--bg`, `--fg`, `--link`, etc.)
- System preference detection via `prefers-color-scheme`
- Two-state toggle: dark ↔ light
- Theme persisted to localStorage

## Adding Features

- Keep dependencies minimal (only `@types/bun` currently)
- Use Bun built-ins: `Bun.file`, `Bun.Glob`, `Bun.markdown`, `Bun.serve`
