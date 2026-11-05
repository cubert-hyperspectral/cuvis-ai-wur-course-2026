# cuvis-ai-wur-course

The WUR 2026 course notebooks ("AI for spectral and hyperspectral data processing", Liza Genke and
Puneet Mishra) rebuilt as a cuvis-ai plugin: tested nodes, pipeline YAMLs, trainrun YAMLs and three
teaching notebooks that end in a reproduction table against the course numbers. README.md names the
sources and licences; CHANGELOG.md tracks the work.

## Layout

- `cuvis_ai_wur_course/node/`: one module per node family (regression, metrics, losses, pca,
  pretreatment, fusion, kiwi, features, models, unet, maps, source).
- `cuvis_ai_wur_course/data/`: the `wur_course_npz` data module; `prepare_data.py` is the
  `prepare-course-data` CLI (Click).
- `cuvis_ai_wur_course/upstream/`: vendored study and course code, never edited here.
- `plugins/cuvis_ai_wur_course.yaml`: the manifest. After a node or port change, regenerate its
  metadata with `uv run --with ruamel.yaml python -m scripts.emit_metadata --manifest plugins/cuvis_ai_wur_course.yaml`.
- `pipelines/`, `trainruns/`: the YAMLs the notebooks mirror. `tools/build_pipelines.py` regenerates
  every pipeline YAML and PNG; edit the builder, not the YAML.
- `notebooks/`: the three notebooks and `utils.py` (plotting and tables only). Re-bake with
  `uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/<name>.ipynb`
  from the repository root; a failed cell leaves the file untouched.
- `tests/`: pytest with synthetic tensors and no downloads; `-m "not slow"` skips the pipelines that
  load SAM3, YOLO, ResNet-18 or the U-Net weights.
- `data/`, `outputs/`: local and ignored. `tools/make_synthetic_course_data.py` writes a stand-in
  with the course shapes when the course folder is absent.

## Build and test

- `uv sync --all-extras` (uv, never pip; the `plugins` extra brings sam3 and ultralytics).
- `uv run pytest -q -m "not slow"`, `uv run ruff format`, `uv run ruff check --fix`; line length 100.
- Plugin catalogs for every `restore-pipeline` or `restore-trainrun` call:
  `--plugins-dir plugins --plugins-dir .venv/Lib/site-packages/cuvis_ai/configs/plugins`.
  A trainrun resolves `pipeline` and `splits_path` relative to its file and `universe_csv` relative
  to the working directory, so run from the repository root.

## Conventions

- Nodes inherit `cuvis_ai_core.node.Node`, declare `INPUT_SPECS` and `OUTPUT_SPECS`, forward only
  their own constructor names to `super().__init__`, and omit the `Node` suffix. Unbacked symbolic
  dims are `-1`: a string dim resolves as a node attribute.
- The pipeline fills any unconnected input port named like a batch key (`mask`, `wavelengths`,
  `target`) from the batch. Connect optional ports explicitly or leave them out.
- The SAM3 node emits its mask and object ids on the cpu whatever device the pipeline runs on;
  nodes that combine that mask with pipeline tensors move the small input onto the big one's device.
- Test first on every node (red test, pure tensors, then the code); notebooks, YAML and docs are
  exempt.
- No ticket ids, phase names or AI mentions in code, comments, commits or docs.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **cuvis-ai-wur-course-2026** (20 symbols, 15 relationships, 0 execution flows).

> Index stale? Run `node .gitnexus/run.cjs analyze --index-only` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? Bootstrap with `npx`, `bunx`, or `pnpm dlx` — e.g. `bunx gitnexus@latest analyze` (npm 11 npx crash; #1939).

## Always Do

- **MUST run impact before editing.** Use `impact({target: "symbolName", direction: "upstream"})` or `node .gitnexus/run.cjs impact "symbolName" --direction upstream --repo .`; report callers, processes, and risk. Never substitute grep for graph analysis.
- **MUST analyze graph changes before committing.** Use `detect_changes({scope: "all"})` (MCP) or `node .gitnexus/run.cjs detect-changes --scope all --repo .` (CLI fallback). `partial: true` or `truncated: true` is not a clean check — a zero means unseen, not unaffected; re-run it. For regression review: `detect_changes({scope: "compare", base_ref: "main"})` or `node .gitnexus/run.cjs detect-changes --scope compare --base-ref "main" --repo .`.
- MUST warn on HIGH/CRITICAL `risk` pre-edit; never use `riskSharedAxes` to waive a HIGH/CRITICAL `risk` warning. Compare File/symbol: MCP File omits axes; Graph-RAG expands File.
- **MUST treat `risk: UNKNOWN` as unresolved, not as low.** An empty caller set is not evidence the symbol is unused — it can also mean the callers are not resolvable by the index (plain-object property access, dynamic dispatch, cross-language calls). `impact` pairs `UNKNOWN` with a `riskNote` saying so. Confirm with a text search before treating the symbol as safe to change or delete; do not proceed on the strength of a zero.
- **MUST use `query({search_query: "concept"})` for concepts/flows, `context({name: "symbolName"})` for a named symbol, or `impact` for blast radius, on read-only callers, dependencies, imports, or execution flow.** Graph first; text search only for empty/`UNKNOWN`/literals.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method before MCP/CLI impact analysis.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis, and never read `UNKNOWN` as an all-clear — it means the walk could not answer, which is the one verdict that requires confirming by other means.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit before MCP/CLI graph change analysis.

## Resources

| Resource | Use for |
| --- | --- |
| `gitnexus://repo/cuvis-ai-wur-course-2026/context` | Codebase overview, check index freshness |
| `gitnexus://repo/cuvis-ai-wur-course-2026/clusters` | All functional areas |
| `gitnexus://repo/cuvis-ai-wur-course-2026/processes` | All execution flows |
| `gitnexus://repo/cuvis-ai-wur-course-2026/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
| --- | --- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
