# Model Management — Feature Registry

## Scope
The HuggingFace model discovery and management system must allow users to search, filter, download, and manage MLX models. This harness validates the search API integration, the MLX filter toggle, the UI entry points, and the state management.

## Features

| # | Feature | Status | Test Function | Last Verified |
|---|---------|--------|---------------|---------------|
| 1 | Strict MLX filter queries HF with `library=mlx` param | ✅ PASS | `testStrictMLXFilterEnabled` | 2026-04-07 |
| 2 | Loose MLX filter appends "mlx" to search text | ✅ PASS | `testStrictMLXFilterDisabled` | 2026-04-07 |
| 3 | Empty query with strict MLX returns trending models | 🔲 TODO | — | — |
| 4 | Search debounce prevents rapid-fire API calls | 🔲 TODO | — | — |
| 5 | Load more pagination increments offset correctly | 🔲 TODO | — | — |
| 6 | ModelManagementView shows "Search HuggingFace" button | 🔲 TODO | — | — |
| 7 | ModelManagementView empty state has search entry point | 🔲 TODO | — | — |
| 8 | Error state renders on network failure | 🔲 TODO | — | — |
| 9 | HFModelResult correctly parses param size hints | 🔲 TODO | — | — |
| 10 | HFModelResult correctly detects MoE architecture | 🔲 TODO | — | — |
