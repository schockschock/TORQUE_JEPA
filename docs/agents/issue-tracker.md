# Issue Tracker: Local Markdown

Issues and PRDs live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- PRD: `.scratch/<feature-slug>/PRD.md`
- Tickets: `.scratch/<feature-slug>/issues/<NN>-<slug>.md`
- Number tickets from `01` in dependency order
- Record triage state as a `Status:` line near top
- Append discussion under `## Comments`

## Publish

When a skill says "publish to issue tracker," create files under the relevant `.scratch/<feature-slug>/` directory.

## Fetch

Read referenced local file. User normally supplies path or ticket number.
