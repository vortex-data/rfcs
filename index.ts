import { $, type Server } from "bun";
import { watch } from "fs";
import { createHighlighter, type Highlighter } from "shiki";

const isDev = process.argv.includes("--dev");
const isPreview = process.argv.includes("--preview");
const PORT = 3000;

interface GitCommit {
  hash: string;
  date: Date;
}

interface GitHubAuthor {
  login: string;
  avatarUrl: string;
  profileUrl: string;
}

interface RFCGitInfo {
  accepted: GitCommit | null;
  lastUpdated: GitCommit | null; // null if same as accepted or no git history
  author: GitHubAuthor | null;
}

interface RFC {
  number: string;
  title: string;
  filename: string;
  html: string;
  git: RFCGitInfo;
}

interface ProposedRFC {
  number: string;
  title: string;
  prNumber: number;
  prUrl: string;
  author: GitHubAuthor | null;
}

const THEME_SCRIPT = `
(function() {
  const saved = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
})();
`;

const ICON_SUN = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>`;

const ICON_MOON = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>`;

const ICON_EXTERNAL = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 7h10v10"/><path d="M7 17 17 7"/></svg>`;

const ICON_GITHUB = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>`;

const TOGGLE_SCRIPT = `
function toggleTheme() {
  const root = document.documentElement;
  const current = root.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  updateToggleIcon();
}

function updateToggleIcon() {
  const btn = document.querySelector('.theme-toggle');
  if (!btn) return;
  const current = document.documentElement.getAttribute('data-theme');
  btn.innerHTML = current === 'dark' ? '${ICON_SUN}' : '${ICON_MOON}';
  btn.setAttribute('aria-label', current === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
}

document.addEventListener('DOMContentLoaded', updateToggleIcon);
`;

const LIVE_RELOAD_SCRIPT = `
(function() {
  const evtSource = new EventSource('/__reload');
  evtSource.onmessage = function() {
    location.reload();
  };
})();
`;

function baseHTML(
  title: string,
  content: string,
  cssPath: string = "styles.css",
  liveReload: boolean = false,
  repoUrl: string | null = null,
): string {
  const basePath = cssPath === "styles.css" ? "./" : "../";
  const githubLink = repoUrl
    ? `<a href="${repoUrl}" class="github-link" aria-label="View on GitHub" target="_blank" rel="noopener">${ICON_GITHUB}</a>`
    : "";
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHTML(title)}</title>
  <link rel="icon" type="image/svg+xml" href="${basePath}vortex_logo.svg">
  <link rel="stylesheet" href="${cssPath}">
  <script>${THEME_SCRIPT}</script>
</head>
<body>
  <div class="container">
    <header>
      <a href="${basePath}" class="header-brand">
        <img src="${basePath}vortex_logo.svg" alt="Vortex" class="header-logo">
        <h1>Vortex RFCs</h1>
      </a>
      <div class="header-actions">
        ${githubLink}
        <button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle theme"></button>
      </div>
    </header>
    <main>
${content}
    </main>
    <footer>
      Vortex RFC Archive
    </footer>
  </div>
  <script>${TOGGLE_SCRIPT}</script>${liveReload ? `\n  <script>${LIVE_RELOAD_SCRIPT}</script>` : ""}
</body>
</html>`;
}

function escapeHTML(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function indexPage(
  rfcs: RFC[],
  proposed: ProposedRFC[],
  repoUrl: string | null,
  liveReload: boolean = false,
): string {
  // Sort in reverse numeric order (newest first)
  const sorted = [...rfcs].sort((a, b) => b.number.localeCompare(a.number));

  const list = sorted
    .map((rfc) => {
      const dateStr = rfc.git.accepted ? formatDate(rfc.git.accepted.date) : "";

      let authorHTML = "";
      if (rfc.git.author && rfc.git.accepted) {
        const commitUrl = repoUrl
          ? `${repoUrl}/commit/${rfc.git.accepted.hash}`
          : `https://github.com/${rfc.git.author.login}`;
        authorHTML = `
          <a href="${commitUrl}" class="rfc-author-link" title="${rfc.git.author.login}">
            <img src="${rfc.git.author.avatarUrl}" alt="${rfc.git.author.login}" class="rfc-author-avatar">
            <span class="rfc-author-name">${rfc.git.author.login}</span>
          </a>`;
      }

      return `
      <li>
        <a href="rfc/${rfc.number}.html" class="rfc-item">
          <span class="rfc-number">RFC ${rfc.number}</span>
          <span class="rfc-title">${escapeHTML(rfc.title)}</span>
          <span class="rfc-date">${dateStr}</span>
        </a>${authorHTML}
      </li>`;
    })
    .join("\n");

  let proposedSection = "";
  if (proposed.length > 0) {
    const proposedList = proposed
      .map((rfc) => {
        let authorHTML = "";
        if (rfc.author) {
          authorHTML = `
          <a href="${rfc.author.profileUrl}" class="rfc-author-link" title="${rfc.author.login}">
            <img src="${rfc.author.avatarUrl}" alt="${rfc.author.login}" class="rfc-author-avatar">
            <span class="rfc-author-name">${rfc.author.login}</span>
          </a>`;
        }

        return `
      <li>
        <a href="${rfc.prUrl}" class="rfc-item" target="_blank" rel="noopener">
          <span class="rfc-number">RFC ${rfc.number}</span>
          <span class="rfc-title">${escapeHTML(rfc.title)}</span>
          <span class="rfc-date">PR #${rfc.prNumber}</span>
        </a>${authorHTML}
      </li>`;
      })
      .join("\n");

    proposedSection = `
      <h2>Proposed</h2>
      <ul class="rfc-list rfc-list-proposed">
${proposedList}
      </ul>`;
  }

  const content = `
      <h1>Request for Comments</h1>
      <p>Technical proposals for the Vortex file format.</p>
${proposedSection}
      <h2>Accepted</h2>
      <ul class="rfc-list">
${list}
      </ul>`;

  return baseHTML("Vortex RFCs", content, "styles.css", liveReload, repoUrl);
}

function formatDate(date: Date): string {
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function rfcPage(
  rfc: RFC,
  repoUrl: string | null,
  liveReload: boolean = false,
): string {
  let gitHeader = `
      <div class="rfc-meta-header">
        <div class="rfc-meta-item">
          <span class="rfc-status-pill status-accepted">Accepted</span>
        </div>`;

  if (rfc.git.accepted || rfc.git.author) {
    // Author section
    if (rfc.git.author) {
      gitHeader += `
        <div class="rfc-meta-item rfc-author">
          <a href="${rfc.git.author.profileUrl}" class="author-link">
            <img src="${rfc.git.author.avatarUrl}" alt="${rfc.git.author.login}" class="author-avatar">
            <span class="author-name">${rfc.git.author.login}</span>
          </a>
        </div>`;
    }

    // Accepted date
    if (rfc.git.accepted) {
      const commitLink = repoUrl
        ? ` <a href="${repoUrl}/commit/${rfc.git.accepted.hash}" class="commit-link" title="View commit">${ICON_EXTERNAL}</a>`
        : "";

      gitHeader += `
        <div class="rfc-meta-item">
          <span class="rfc-meta-label">Accepted:</span>
          <span class="rfc-meta-value">${formatDate(rfc.git.accepted.date)}${commitLink}</span>
        </div>`;
    }

    // Last updated date
    if (rfc.git.lastUpdated) {
      const commitLink = repoUrl
        ? ` <a href="${repoUrl}/commit/${rfc.git.lastUpdated.hash}" class="commit-link" title="View commit">${ICON_EXTERNAL}</a>`
        : "";

      gitHeader += `
        <div class="rfc-meta-item">
          <span class="rfc-meta-label">Last updated:</span>
          <span class="rfc-meta-value">${formatDate(rfc.git.lastUpdated.date)}${commitLink}</span>
        </div>`;
    }
  }

  gitHeader += `
      </div>`;

  const content = `
      <a href="../" class="back-link">&larr; Back to index</a>${gitHeader}
      <article class="rfc-content">
        ${rfc.html}
      </article>`;

  return baseHTML(
    `RFC ${rfc.number} - ${rfc.title}`,
    content,
    "../styles.css",
    liveReload,
    repoUrl,
  );
}

function parseRFCNumber(filename: string): string {
  // Extract number from filename like "0002-patches-galp.md"
  const match = filename.match(/^(\d+)/);
  return match?.[1] ?? "0000";
}

async function getGitHubRepoUrl(): Promise<string | null> {
  try {
    const result = await $`git remote get-url origin`.quiet();
    const url = result.stdout.toString().trim();
    // Convert git@github.com:user/repo.git to https://github.com/user/repo
    if (url.startsWith("git@github.com:")) {
      return "https://github.com/" + url.slice(15).replace(/\.git$/, "");
    }
    // Convert https://github.com/user/repo.git to https://github.com/user/repo
    if (url.startsWith("https://github.com/")) {
      return url.replace(/\.git$/, "");
    }
    return null;
  } catch {
    return null;
  }
}

async function getGitHubAuthor(
  repoPath: string,
  commitHash: string,
): Promise<GitHubAuthor | null> {
  try {
    // Use gh CLI to fetch commit info from GitHub API
    const result =
      await $`gh api repos/${repoPath}/commits/${commitHash} --jq '.author.login, .author.avatar_url, .author.html_url'`.quiet();
    const lines = result.stdout.toString().trim().split("\n");

    if (lines.length >= 3 && lines[0] && lines[1] && lines[2]) {
      return {
        login: lines[0],
        avatarUrl: lines[1],
        profileUrl: lines[2],
      };
    }
    return null;
  } catch {
    return null;
  }
}

async function getGitHistory(
  filepath: string,
  repoPath: string | null,
): Promise<RFCGitInfo> {
  try {
    const result =
      await $`git log --follow --format=%H\ %aI -- ${filepath}`.quiet();
    const lines = result.stdout.toString().trim().split("\n").filter(Boolean);

    if (lines.length === 0) {
      return { accepted: null, lastUpdated: null, author: null };
    }

    const parseCommit = (line: string): GitCommit => {
      const parts = line.split(" ");
      const hash = parts[0] ?? "";
      const dateStr = parts[1] ?? "";
      return { hash, date: new Date(dateStr) };
    };

    const mostRecent = parseCommit(lines[0]!);
    const oldest = parseCommit(lines[lines.length - 1]!);

    // Fetch author info from the first commit
    const author = repoPath
      ? await getGitHubAuthor(repoPath, oldest.hash)
      : null;

    // If only one commit, or same commit, don't show lastUpdated
    if (lines.length === 1 || mostRecent.hash === oldest.hash) {
      return { accepted: oldest, lastUpdated: null, author };
    }

    return { accepted: oldest, lastUpdated: mostRecent, author };
  } catch {
    return { accepted: null, lastUpdated: null, author: null };
  }
}

interface ValidationError {
  filename: string;
  message: string;
}

async function validateProposals(): Promise<ValidationError[]> {
  const errors: ValidationError[] = [];
  const glob = new Bun.Glob("*");
  const seenNumbers = new Map<string, string>();

  for await (const filename of glob.scan("./rfcs")) {
    // Check filename format: NNNN-slug.md
    if (!filename.match(/^\d{4}-[a-zA-Z0-9_-]+\.md$/)) {
      errors.push({
        filename: `rfcs/${filename}`,
        message: `Invalid filename format. Expected: NNNN-name.md (e.g., 0007-my-proposal.md)`,
      });
      continue;
    }

    // Check for duplicate RFC numbers
    const number = filename.slice(0, 4);
    const existing = seenNumbers.get(number);
    if (existing) {
      errors.push({
        filename: `rfcs/${filename}`,
        message: `Duplicate RFC number ${number} (also used by ${existing})`,
      });
    } else {
      seenNumbers.set(number, `rfcs/${filename}`);
    }
  }

  return errors;
}

function parseTitle(markdown: string, filename: string): string {
  // Try to extract title from first # heading
  const match = markdown.match(/^#\s+(.+)$/m);
  if (match?.[1]) {
    // Clean up "RFC XXX - " prefix if present
    return match[1].replace(/^RFC\s+\d+\s*[-:]\s*/i, "").trim();
  }
  // Fallback to filename
  return filename.replace(/^\d+-/, "").replace(/\.md$/, "").replace(/-/g, " ");
}

// Syntax highlighting with Shiki
let highlighter: Highlighter | null = null;

async function getHighlighter(): Promise<Highlighter> {
  if (!highlighter) {
    highlighter = await createHighlighter({
      themes: ["github-light", "github-dark"],
      langs: ["rust", "python", "markdown", "cpp", "c"],
    });
  }
  return highlighter;
}

async function highlightCodeBlocks(html: string): Promise<string> {
  const hl = await getHighlighter();

  // Match <pre><code class="language-X">...</code></pre> blocks
  const codeBlockRegex =
    /<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g;

  const matches = [...html.matchAll(codeBlockRegex)];
  let result = html;

  for (const match of matches) {
    const [fullMatch, lang, code] = match;
    if (!lang || !code) continue;

    // Decode HTML entities back to raw code
    const rawCode = code
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&amp;/g, "&")
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'");

    // Check if this language is supported
    const loadedLangs = hl.getLoadedLanguages();
    if (!loadedLangs.includes(lang)) {
      // Skip unsupported languages, leave original markup
      continue;
    }

    // Generate highlighted HTML with both themes using CSS variables
    const highlighted = hl.codeToHtml(rawCode.trim(), {
      lang,
      themes: {
        light: "github-light",
        dark: "github-dark",
      },
      defaultColor: false,
    });

    result = result.replace(fullMatch, highlighted);
  }

  return result;
}

async function getProposedRFCs(
  repoPath: string | null,
): Promise<ProposedRFC[]> {
  if (!repoPath) return [];
  try {
    const result =
      await $`gh pr list --repo ${repoPath} --state open --json number,title,url,files,author`.quiet();
    const prs = JSON.parse(result.stdout.toString());
    const proposed: ProposedRFC[] = [];

    for (const pr of prs) {
      // Find RFC files in the PR's changed files (match rfcs/, proposed/, proposals/)
      const rfcFile = pr.files?.find(
        (f: { path: string }) =>
          f.path.match(
            /^(rfcs|proposed|proposals)\/\d{4}-[a-zA-Z0-9_-]+\.md$/,
          ) && !f.path.endsWith("/0000-template.md"),
      );
      if (!rfcFile) continue;

      const rfcNumber = rfcFile.path.match(/(\d{4})-/)?.[1];
      if (!rfcNumber) continue;

      proposed.push({
        number: rfcNumber,
        title: pr.title,
        prNumber: pr.number,
        prUrl: pr.url,
        author: pr.author
          ? {
              login: pr.author.login,
              avatarUrl: `https://github.com/${pr.author.login}.png?size=48`,
              profileUrl: `https://github.com/${pr.author.login}`,
            }
          : null,
      });
    }

    return proposed.sort((a, b) => b.number.localeCompare(a.number));
  } catch {
    return [];
  }
}

async function build(liveReload: boolean = false): Promise<number> {
  console.log("Building Vortex RFC site...\n");

  // Validate proposals first
  const validationErrors = await validateProposals();
  if (validationErrors.length > 0) {
    console.error("Validation errors found:\n");
    for (const error of validationErrors) {
      console.error(`  ${error.filename}: ${error.message}`);
    }
    console.error("");
    process.exit(1);
  }

  // Get GitHub repo URL for commit links
  const repoUrl = await getGitHubRepoUrl();
  // Extract repo path (e.g., "vortex-data/rfcs") for API calls
  const repoPath = repoUrl ? repoUrl.replace("https://github.com/", "") : null;

  const glob = new Bun.Glob("*.md");
  const rfcs: RFC[] = [];

  for await (const filename of glob.scan("./rfcs")) {
    console.log(`Processing rfcs/${filename}...`);

    const path = `./rfcs/${filename}`;
    const content = await Bun.file(path).text();
    const rawHtml = Bun.markdown.html(content, { autolinks: true });
    const html = await highlightCodeBlocks(rawHtml);
    const number = parseRFCNumber(filename);
    const title = parseTitle(content, filename);
    const git = await getGitHistory(path, repoPath);

    rfcs.push({ number, title, filename, html, git });
  }

  if (rfcs.length === 0) {
    console.log("No RFC files found in ./rfcs/");
    return 0;
  }

  // Clean and create dist directory
  await $`rm -rf dist`.quiet();
  await $`mkdir -p dist/rfc`.quiet();

  // Copy CSS
  await Bun.write("dist/styles.css", await Bun.file("styles.css").text());

  // Copy static assets
  const logo = Bun.file("static/vortex_logo.svg");
  if (await logo.exists()) {
    await Bun.write("dist/vortex_logo.svg", await logo.text());
  }

  // Copy all static assets to dist/static/
  await $`mkdir -p dist/static`.quiet();
  const staticGlob = new Bun.Glob("*");
  for await (const filename of staticGlob.scan("./static")) {
    const src = Bun.file(`static/${filename}`);
    const dest = `dist/static/${filename}`;
    await Bun.write(dest, src);
    console.log(`Copied static/${filename} -> ${dest}`);
  }

  // Fetch proposed RFCs from open PRs
  const proposed = await getProposedRFCs(repoPath);
  if (proposed.length > 0) {
    console.log(`Found ${proposed.length} proposed RFC(s) from open PRs`);
  }

  // Generate index page
  const indexHTML = indexPage(rfcs, proposed, repoUrl, liveReload);
  await Bun.write("dist/index.html", indexHTML);
  console.log("Generated dist/index.html");

  // Generate individual RFC pages
  for (const rfc of rfcs) {
    const html = rfcPage(rfc, repoUrl, liveReload);
    const outPath = `dist/rfc/${rfc.number}.html`;
    await Bun.write(outPath, html);
    console.log(`Generated ${outPath}`);
  }

  console.log(`\nBuild complete! ${rfcs.length} RFC(s) processed.`);
  return rfcs.length;
}

// Track SSE clients for live reload
const reloadClients = new Set<ReadableStreamDefaultController>();

function notifyReload() {
  for (const controller of reloadClients) {
    try {
      controller.enqueue("data: reload\n\n");
    } catch {
      reloadClients.delete(controller);
    }
  }
}

async function startDevServer() {
  // Initial build with live reload enabled
  await build(true);
  console.log(`\nStarting dev server at http://localhost:${PORT}`);
  console.log("Watching for changes in ./rfcs/ and ./styles.css\n");

  // Debounce rebuilds
  let rebuildTimeout: Timer | null = null;
  const scheduleRebuild = () => {
    if (rebuildTimeout) clearTimeout(rebuildTimeout);
    rebuildTimeout = setTimeout(async () => {
      console.log("\nFile change detected, rebuilding...");
      await build(true);
      notifyReload();
    }, 100);
  };

  watch("./rfcs", { recursive: true }, (_event, filename) => {
    if (filename?.endsWith(".md")) {
      scheduleRebuild();
    }
  });

  // Watch styles.css
  watch("./styles.css", () => {
    scheduleRebuild();
  });

  // Start server
  Bun.serve({
    port: PORT,
    async fetch(req) {
      const url = new URL(req.url);
      let pathname = url.pathname;

      // SSE endpoint for live reload
      if (pathname === "/__reload") {
        const stream = new ReadableStream({
          start(controller) {
            reloadClients.add(controller);
          },
          cancel(controller) {
            reloadClients.delete(controller);
          },
        });

        return new Response(stream, {
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
          },
        });
      }

      // Serve static files from dist/
      if (pathname === "/") pathname = "/index.html";
      if (pathname.endsWith("/")) pathname += "index.html";

      const filePath = `./dist${pathname}`;
      const file = Bun.file(filePath);

      if (await file.exists()) {
        return new Response(file);
      }

      return new Response("Not Found", { status: 404 });
    },
  });
}

async function buildPreview(): Promise<void> {
  console.log("Building RFC preview...\n");

  const repoUrl = await getGitHubRepoUrl();
  const repoPath = repoUrl ? repoUrl.replace("https://github.com/", "") : null;
  const css = await Bun.file("styles.css").text();

  // Find new/changed RFC files compared to the base branch
  const glob = new Bun.Glob("*.md");
  const rfcs: RFC[] = [];

  for await (const filename of glob.scan("./rfcs")) {
    const path = `./rfcs/${filename}`;
    const content = await Bun.file(path).text();
    const rawHtml = Bun.markdown.html(content, { autolinks: true });
    const html = await highlightCodeBlocks(rawHtml);
    const number = parseRFCNumber(filename);
    const title = parseTitle(content, filename);
    const git = await getGitHistory(path, repoPath);

    rfcs.push({ number, title, filename, html, git });
  }

  // Build a single self-contained index page with all RFCs and inlined CSS
  const proposed = await getProposedRFCs(repoPath);
  const sorted = [...rfcs].sort((a, b) => b.number.localeCompare(a.number));

  let rfcPages = "";
  for (const rfc of sorted) {
    rfcPages += `
    <article class="rfc-content" id="rfc-${rfc.number}">
      <hr>
      ${rfc.html}
    </article>`;
  }

  const list = sorted
    .map(
      (rfc) => `
      <li>
        <a href="#rfc-${rfc.number}" class="rfc-item">
          <span class="rfc-number">RFC ${rfc.number}</span>
          <span class="rfc-title">${escapeHTML(rfc.title)}</span>
        </a>
      </li>`,
    )
    .join("\n");

  const content = `
      <h1>Request for Comments</h1>
      <p>Technical proposals for the Vortex file format.</p>
      <h2>Accepted</h2>
      <ul class="rfc-list">
${list}
      </ul>
${rfcPages}`;

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Vortex RFCs — Preview</title>
  <style>${css}</style>
  <script>${THEME_SCRIPT}</script>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-brand">
        <h1>Vortex RFCs — Preview</h1>
      </div>
      <div class="header-actions">
        <button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle theme"></button>
      </div>
    </header>
    <main>
${content}
    </main>
  </div>
  <script>${TOGGLE_SCRIPT}</script>
</body>
</html>`;

  await $`mkdir -p dist`.quiet();
  await Bun.write("dist/preview.html", html);
  console.log("Generated dist/preview.html");
}

if (isDev) {
  startDevServer().catch(console.error);
} else if (isPreview) {
  buildPreview().catch(console.error);
} else {
  build()
    .then((count) => {
      if (count > 0) {
        console.log("Output directory: ./dist/");
      }
    })
    .catch(console.error);
}
