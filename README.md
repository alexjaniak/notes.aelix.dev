# notes.aelix.dev

Alex Janiak's local-first Obsidian vault, published as a public Quartz notes site.

## Structure

- `Research/Notes/` — working notes and reference material.
- `Research/Blog/` — polished, standalone posts.
- `Research/Attachments/` — images and diagrams referenced from notes.

## Operating Model

- **Vault** (this repo) is the source of truth — plain Markdown, edited in
  Obsidian or via Claude Code.
- **Git** is the version history.
- **Quartz + GitHub Pages** publish the vault at https://notes.aelix.dev on every
  push to `main`.
- **Claude Code / Codex** are the ingest and curation layer — see `AGENTS.md`.

## Publishing

    npm run publish -- "notes: <what changed>"

## Local Development

    npm i
    npx quartz plugin install --from-config
    npm run serve   # http://localhost:8080
