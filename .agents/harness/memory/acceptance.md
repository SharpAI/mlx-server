# Memory Handling — Acceptance Criteria

Each feature below defines the exact input→output contract. A test passes **only** if the output matches the expectation precisely.

---

### Feature 1: Hallucinated preamble + markdown fence
- **Input**: `Here is the JSON you requested:\n\`\`\`json\n{"extractions": [{"test": "value"}]}\n\`\`\`\nRemember to eat!`
- **Expected Output**: `{"extractions": [{"test": "value"}]}`
- **FAIL if**: Output includes any text before `{` or after `}`

### Feature 2: Nested internal braces
- **Input**: `{"extractions": [{"key": "{value}"}, {"key2": "value2"}]}`
- **Expected Output**: Identical to input
- **FAIL if**: Output is truncated at the first `}` instead of the last

### Feature 3: Perfect JSON passthrough
- **Input**: `{"extractions": [{"test": "value"}]}`
- **Expected Output**: Identical to input (trimmed whitespace acceptable)
- **FAIL if**: Any character is added or removed

### Feature 4: Missing closing bracket
- **Input**: `Here is data: {"key": "val"`
- **Expected Output**: `{"key": "val"` (best-effort return of partial content — no crash)
- **FAIL if**: Function throws an exception, returns empty string, or crashes
- **NOTE**: The returned string will not be valid JSON, but `cleanJSON` should not crash. The caller (JSONDecoder) handles the parse error gracefully.

### Feature 5: JSON array root
- **Input**: `[{"item": 1}, {"item": 2}]`
- **Expected Output**: `[{"item": 1}, {"item": 2}]`
- **FAIL if**: Returns empty string or only extracts `{"item": 1}`
- **NOTE**: Current implementation only scans for `{` and `}`. Needs to also handle `[` and `]` as valid JSON roots.

### Feature 6: Empty / whitespace input
- **Input**: `   \n\t  `
- **Expected Output**: `` (empty string)
- **FAIL if**: Crashes, throws, or returns whitespace

### Feature 7: Multiple concatenated JSON objects
- **Input**: `{"first": 1}\n{"second": 2}`
- **Expected Output**: `{"first": 1}\n{"second": 2}` (first `{` to last `}`)
- **FAIL if**: Only returns the first object
- **NOTE**: This is a best-effort scenario. The important thing is we don't lose data.

### Feature 8: Unicode and emoji in values
- **Input**: `{"name": "André 🧠", "notes": "日本語テスト"}`
- **Expected Output**: Identical to input
- **FAIL if**: Unicode characters are corrupted or stripped

### Feature 9: ANSI escape codes
- **Input**: `\u001b[32m{"key": "value"}\u001b[0m`
- **Expected Output**: `{"key": "value"}`
- **FAIL if**: ANSI codes remain in output or JSON is corrupted
- **NOTE**: This happens when users paste terminal output containing color codes.
