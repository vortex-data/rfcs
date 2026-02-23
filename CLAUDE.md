# Vortex RFC Site

Static site generator for Vortex RFC proposals.

## Project Structure

```
index.ts      - Main build script and dev server
styles.css    - Site styling (light/dark themes, monospace aesthetic)
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
- Live reload script only injected in dev mode

## Styling

- CSS custom properties for theming (`--bg`, `--fg`, `--link`, etc.)
- System preference detection via `prefers-color-scheme`
- Three-state toggle: auto → dark → light → auto
- Theme persisted to localStorage

## Adding Features

- Keep dependencies minimal (only `@types/bun` currently)
- Use Bun built-ins: `Bun.file`, `Bun.Glob`, `Bun.markdown`, `Bun.serve`
- Maintain the retro monospace aesthetic

---

Default to using Bun instead of Node.js.

- Use `bun <file>` instead of `node <file>` or `ts-node <file>`
- Use `bun test` instead of `jest` or `vitest`
- Use `bun build <file.html|file.ts|file.css>` instead of `webpack` or `esbuild`
- Use `bun install` instead of `npm install` or `yarn install` or `pnpm install`
- Use `bun run <script>` instead of `npm run <script>` or `yarn run <script>` or `pnpm run <script>`
- Use `bunx <package> <command>` instead of `npx <package> <command>`
- Bun automatically loads .env, so don't use dotenv.

## APIs

- `Bun.serve()` supports WebSockets, HTTPS, and routes. Don't use `express`.
- `bun:sqlite` for SQLite. Don't use `better-sqlite3`.
- `Bun.redis` for Redis. Don't use `ioredis`.
- `Bun.sql` for Postgres. Don't use `pg` or `postgres.js`.
- `WebSocket` is built-in. Don't use `ws`.
- Prefer `Bun.file` over `node:fs`'s readFile/writeFile
- Bun.$`ls` instead of execa.

## Testing

Use `bun test` to run tests.

```ts#index.test.ts
import { test, expect } from "bun:test";

test("hello world", () => {
  expect(1).toBe(1);
});
```

## Frontend

Use HTML imports with `Bun.serve()`. Don't use `vite`. HTML imports fully support React, CSS, Tailwind.

Server:

```ts#index.ts
import index from "./index.html"

Bun.serve({
  routes: {
    "/": index,
    "/api/users/:id": {
      GET: (req) => {
        return new Response(JSON.stringify({ id: req.params.id }));
      },
    },
  },
  // optional websocket support
  websocket: {
    open: (ws) => {
      ws.send("Hello, world!");
    },
    message: (ws, message) => {
      ws.send(message);
    },
    close: (ws) => {
      // handle close
    }
  },
  development: {
    hmr: true,
    console: true,
  }
})
```

HTML files can import .tsx, .jsx or .js files directly and Bun's bundler will transpile & bundle automatically. `<link>` tags can point to stylesheets and Bun's CSS bundler will bundle.

```html#index.html
<html>
  <body>
    <h1>Hello, world!</h1>
    <script type="module" src="./frontend.tsx"></script>
  </body>
</html>
```

With the following `frontend.tsx`:

```tsx#frontend.tsx
import React from "react";
import { createRoot } from "react-dom/client";

// import .css files directly and it works
import './index.css';

const root = createRoot(document.body);

export default function Frontend() {
  return <h1>Hello, world!</h1>;
}

root.render(<Frontend />);
```

Then, run index.ts

```sh
bun --hot ./index.ts
```

For more information, read the Bun API docs in `node_modules/bun-types/docs/**.mdx`.
