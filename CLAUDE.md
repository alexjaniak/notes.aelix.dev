# AGENTS.md

This repo is Alex Janiak's public, local-first Obsidian vault and Quartz notes site.

## Mission

Act as a lightweight capture and curation layer for `Research/`. Help write,
organize, and clean up notes and blog posts, then commit and push so the public
site (https://notes.aelix.dev) stays current.

## Folder Structure

- `Research/Notes/` — working notes, in-progress thinking, reference material.
- `Research/Blog/` — polished posts meant to be read standalone.
- `Research/Attachments/` — images/diagrams referenced via `![[wikilink]]` syntax.

## Search Before Create

Before creating a new note, search `Research/` for one that already covers the
topic. Update it instead of creating a near-duplicate. Only create a new note
when the topic is genuinely distinct from anything that already exists.

## Publishing

- Commit and push to publish — GitHub Actions rebuilds and deploys on every push
  to `main`.
- Prefer `npm run publish -- "<message>"` to stage, commit, build (sanity check),
  and push in one step.
- Use short, plain commit messages describing what changed.

## Privacy

This repo is public. Anything committed and pushed is visible on GitHub, whether
or not Quartz renders it. Never paste secrets, API keys, tokens, or private/
business content — that belongs in the separate private vault, not here.

## Prohibited Without Approval

- Broad reorganization of the folder structure.
- Deleting notes or attachments.
- Changing publishing scope, visibility, or git remote configuration.
- Storing secrets or credentials of any kind.
