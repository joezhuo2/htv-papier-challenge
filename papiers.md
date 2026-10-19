# Papiers findings

Docs: wiki.papiers.ai (full text at wiki.papiers.ai/llms-full.txt). Papiers is optional for the challenge. The CLI reads, and from 0.1.6 it can also create highlights.

## CLI

- Needs Node 22 or newer. Install with `npm install -g papiers`, check with `papiers version`.
- Create a token in Papiers under Settings, API tokens (shown once). Run `papiers auth login`, or set `PAPIERS_TOKEN` in the gitignored `.env`. `papiers auth status` and `papiers doctor` check the setup.
- Retrieval: `papiers search QUERY --json` (flags `--limit` 1 to 50, `--kind passage|conversation|equation`, `--document-id`, `--passage-ids`, `--highlight-ids`, `--doc-ids`, `--filter FIELD.OP=VALUE`), `papiers list --kind highlight|note|conversation --since 30d --json`, `papiers read ID... --format metadata|markdown|plain --json`, `papiers browse srch_...`. IDs start with `doc_`, `pas_`, `hl_`, `cnv_`, `eq_`, `nte_`, `srch_`.
- Pipe example: `papiers search "accumulation" --passage-ids | papiers read @- --format plain --json`.
- Filters worth using: `annotated.eq=true`, engaged, `publishedYear.gte=2024`.
- The npm package ships `skills/papiers-research/SKILL.md`, which weights highlights, notes and conversations above saved but unread items. Point the agent at it.
- Limits: 200,000 characters per read item, 100 IDs per request, 25 filters. Exit codes 0 success, 1 failure, 2 invalid usage.

## Write-back

From CLI 0.1.6, `papiers highlight <doc_id> --pages N --start "first words" [--end "last words"] [--note TEXT] [--color HEX]` creates a highlight (no `--pages` for a webpage) with a write-permission token. The war room uses it only for final verdicts a human approves in the Write to Papiers preview, with an `[RWR]` note, and later sessions read those notes back as prior verdicts that are never evidence, and refuse to repeat an experiment a note's fingerprint (params, base config, bounds, rounds) says was already run (see `docs/architecture.md`). The Markdown export below still runs. The notes below are from before 0.1.6.

Before 0.1.6 the CLI could not import, edit, highlight, tag or publish. The challenge page says agents can write findings back, but no path exists in the docs, and the workshop repo does not use Papiers (it keeps `results.tsv` and `runs/` locally). Plan:

- Read context through the CLI before each hypothesis and cite the note id in the hypothesis.
- Export branch summaries and the ledger as files in a local export folder (PDF or Markdown). Joe imports them manually (Ctrl+N, drop a PDF or folder) or pastes them into a Note.
- Label the Markdown files "exported". Say "written" only for a write-back item the CLI reported as successful. Do not claim automatic write-back in the pitch.
- Ask the Papiers team at the hackathon whether a write path exists.

## Adding papers (desktop app, Ctrl+N or +)

- arXiv links: paste the page link and it imports as a PDF. Links: arxiv.org/abs/2205.14135 (FlashAttention), 1710.03740 (Mixed Precision Training), 1901.06588 (Accumulation Bit-Width Scaling), 1606.08415 (GELUs).
- Triton paper (Tillet, Kung, Cox, MAPL 2019, not on arXiv): download the PDF from eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf and drop it in.
- Triton matmul tutorial: paste triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html as an article. Highlight web articles with the Chrome extension's Highlight mode.
- Limits: 200 MB and 7,000 pages per PDF. The Chrome extension saves pages behind logins.

## Highlights and notes

- Select text to highlight (Ctrl+Shift+H for a PDF selection), then add an annotation to the highlight. Click Note in the PDF toolbar for a Markdown note, where the first line becomes the title.
- For each paper make one or two highlights with a short annotation in Joe's own words and one note. Write only what Joe believes. Do not copy the mock script's note about FP16 accumulate doubling throughput, since it is a placeholder.
