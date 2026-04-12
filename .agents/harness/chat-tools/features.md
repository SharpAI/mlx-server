# Chat Tool Integration — Feature Registry

## Scope
Enable the LLM inside `ChatViewModel` to autonomously invoke `MemoryPalaceTools` (like `mempalace_search`), execute them natively, and receive the results back in the context window without requiring user assistance.

## Features

| # | Feature | Status | Test Function | Last Verified |
|---|---------|--------|---------------|---------------|
| 1 | ChatMessage supports `.tool` role | ✅ PASS | `testFeature1_ChatMessageToolRole` | 2026-04-09 |
| 2 | System Prompt Tool Schema Injection | ✅ PASS | `testFeature2_ToolSchemaInjection` | 2026-04-09 |
| 3 | LLM Output Tool Parsing (`ExtractionService`) | ✅ PASS | `testFeature3_ToolCallExtraction` | 2026-04-09 |
| 4 | ChatViewModel Autonomous Tool Execution Loop | ✅ PASS | `testFeature4_ToolExecutionLoopAsync` | 2026-04-09 |
