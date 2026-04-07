# Model Management — Acceptance Criteria

---

### Feature 1: Strict MLX filter
- **Precondition**: `service.strictMLX = true`
- **Action**: `service.search(query: "mistral", sort: .trending)`
- **Expected**: API called with `library=mlx` query param. Results returned without error.
- **FAIL if**: `errorMessage` is non-nil, or service hangs in `isSearching` state

### Feature 2: Loose MLX filter
- **Precondition**: `service.strictMLX = false`
- **Action**: `service.search(query: "mistral", sort: .trending)`
- **Expected**: API called WITHOUT `library=mlx`. Query text has "mlx" appended → `"mistral mlx"`. Results returned.
- **FAIL if**: `library=mlx` param is still present, or search hangs

### Feature 3: Empty query trending
- **Precondition**: `service.strictMLX = true`
- **Action**: `service.search(query: "", sort: .trending)`
- **Expected**: Returns non-empty results (trending MLX models from HF). `results.count > 0`.
- **FAIL if**: Results are empty despite no error, or errorMessage is set

### Feature 4: Debounce behavior
- **Action**: Call `service.search()` 5 times in rapid succession (<50ms apart)
- **Expected**: Only the last search query executes (debounce cancels previous)
- **FAIL if**: All 5 queries execute independently, or results contain stale data from an earlier query

### Feature 5: Pagination
- **Precondition**: Perform initial search that returns 20 results
- **Action**: Call `service.loadMore()`
- **Expected**: `results.count > 20`. New results are appended, not replacing.
- **FAIL if**: Results count stays at 20, or existing results are cleared

### Feature 6: ModelManagementView search button (populated state)
- **Precondition**: `downloadedModels` is non-empty
- **Expected**: The model list includes a "Search HuggingFace MLX Models" button in a section above the storage card
- **FAIL if**: Button is missing from the view hierarchy

### Feature 7: ModelManagementView empty state search
- **Precondition**: `downloadedModels` is empty
- **Expected**: Empty state shows "Search HuggingFace MLX Models" as the primary action button (not just "Browse Models")
- **FAIL if**: Only a dismiss button is shown with no search entry point

### Feature 8: Error state rendering
- **Precondition**: Network is unavailable or API returns non-200
- **Expected**: `errorMessage` is set to a user-readable string. UI shows error icon + message.
- **FAIL if**: Error swallowed silently, or app crashes

### Feature 9: Param size parsing
- **Input models**: `"Qwen2.5-7B-Instruct-4bit"`, `"gemma-0.5B"`, `"8x7B-MoE"`
- **Expected**: `paramSizeHint` returns `"7B"`, `"0.5B"`, `"8x7B"` respectively
- **FAIL if**: Returns nil for any of these known patterns

### Feature 10: MoE detection
- **Input models**: `"gemma-4-26b-a4b-it-4bit"`, `"Mixtral-8x7B-MoE"`, `"Qwen2.5-3B-Instruct"`
- **Expected**: `.isMoE` returns `true`, `true`, `false` respectively
- **FAIL if**: Incorrectly classifies any model
