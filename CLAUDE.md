# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-topic research notes repository with a Hugo-based blog site. Research covers diffusion models, robot manipulation (UMI/VLA), autonomous driving (world models, RL), and 3D vision. Notes are written in `topics/` and published via a Hugo site in `site/`.

## Repository Structure

- `topics/` — Research notes and paper PDFs, organized by topic (01–04). Each topic has `papers/` and `notes/` subdirectories. Files prefixed with `_` are raw materials excluded from sync.
- `site/` — Hugo static site (theme: PaperMod via git submodule). Content is generated from `topics/` via sync script or written directly in `site/content/`.
- `SITE.md` — Comprehensive site usage documentation (deployment, config, troubleshooting).

## Commands

### Local Development (Hugo site)

```bash
# Dev server with hot reload (Docker, no local Hugo needed)
cd site && docker compose --profile dev up
# → http://localhost:1313

# Production build preview (nginx)
cd site && docker compose up --build
# → http://localhost:8080

# Local Hugo (if installed)
cd site && hugo server -D --disableFastRender
```

### Sync Notes to Site

```bash
cd site && ./scripts/sync-notes.sh
```

Copies `topics/*/notes/*.md` → `site/content/blog/<slug>/`, injecting Hugo front matter if missing. Sync is one-way (topics → site). Mapping defined in `TOPIC_MAP` inside the script.

### Build for Deployment

```bash
cd site && docker run --rm -v $(pwd):/src hugomods/hugo:exts-0.140.0 hugo --minify
```

Output goes to `site/public/`.

### Submodule Init (if theme is empty)

```bash
git submodule update --init --depth=1
```

## Key Architecture Decisions

- **Language**: Chinese (zh-cn) is the primary content language. Notes and blog posts are written in Chinese.
- **Math rendering**: KaTeX via Hugo's passthrough extension. Use `$...$` for inline, `$$...$$` for block. Add a space between `$` and adjacent Chinese characters.
- **Note sync flow**: `topics/` is the source of truth for research notes. The sync script flattens nested directories under `notes/` into a single content directory per topic. Don't edit synced files in `content/blog/<slug>/` directly.
- **Paper PDFs**: Served via nginx volume mount at `/papers/` in production mode (mapped from `topics/` directory).
- **Hugo version**: 0.140.0 (extended), pinned in Docker images.
- **Paper cards**: Use the `paper-card` shortcode in content files for structured paper references (supports `title`, `authors`, `year`, `arxiv`, `venue`, `description` params).
