# pinneaple_researcher examples

These examples are designed to show the full workflow:

1) **Discover** relevant papers (arXiv) + repos (GitHub)
2) **Build a Knowledge Base (KB)**: persist artifacts + chunk them into `chunks.jsonl`
3) **Search locally** over the KB chunks (RAG-style)
4) **Extract Problem/Solution pairs** (LLM) + export to a ProblemDesign-like schema
5) **Reproduce** a selected item into a runnable local project (LLM)

## Quick start

```bash
# optional but recommended for deeper GitHub signals
export GITHUB_TOKEN="..."

# optional (only needed for extraction/reproduce)
export GEMINI_API_KEY="..."
# export GEMINI_MODEL="models/gemini-2.0-flash"

python examples/pinneaple_researcher/00_quickstart_end_to_end.py \
  --topic "physics-informed neural networks boundary conditions" \
  --k-papers 6 --k-repos 6 --min-stars 50
```

## Scripts

- `01_discover.py`: discovery only (fast)
- `02_build_kb.py`: discovery + KB build (artifacts + chunks)
- `03_extract.py`: extract problem/solution pairs from a KB run (requires `GEMINI_API_KEY`)
- `04_reproduce.py`: reproduce a selected item into a local runnable project (requires `GEMINI_API_KEY`)
- `05_local_search_over_chunks.py`: local search over `chunks.jsonl` (no embeddings required)
- `06_compare_chunking_settings.py`: compare chunking settings on an existing run
- `07_export_problemdesign_from_extracted.py`: re-export a saved extraction into ProblemDesign JSON

## Output layout (runs)

Runs are stored under:

```
runs/researcher/<slug_topic>/<timestamp>/
  manifest.json
  papers/<slug_id>/{meta.json,content.md,paper.pdf?}
  repos/<slug_id>/{meta.json,content.md}
  kb_index/chunks.jsonl
  extracted_problem_solutions.json             # if extraction ran
  problemdesign_export.json                    # if extraction/export ran
  reproductions/<...>/                         # if reproduce ran
```

If you want: a CLI wrapper (`pinneaple-research`) that calls these pipelines, or an embedding-backed retrieval option (Chroma/FAISS), we can add that next.
