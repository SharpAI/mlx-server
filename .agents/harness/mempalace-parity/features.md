# MemPalace Feature Parity — Feature Registry

## Scope
Verify that our Swift implementation (`MemoryPalaceService`, `ExtractionService`, `MemoryPalaceTools`) achieves feature parity with the open-source Python [MemPalace](https://github.com/milla-jovovich/mempalace) (v3.0.0, 16.2k stars).

## Reference Architecture (upstream Python)
```
Palace Structure:    Wing → Room → Closet → Drawer
Hall Types:          hall_facts, hall_events, hall_discoveries, hall_preferences, hall_advice
Storage:             ChromaDB (vector embeddings, raw verbatim)
Mining Modes:        projects, convos, general
MCP Tools:           19 tools (search, save, traverse, KG, agents, diary)
Knowledge Graph:     Temporal entity-relationship triples (SQLite)
Specialist Agents:   Per-wing agent personas with diary
Compression:         AAAK dialect (experimental)
Auto-Save Hooks:     Stop hook + PreCompact hook
```

## Our Swift Implementation Status
```
Palace Structure:    Wing → Room → MemoryEntry (SwiftData)
Hall Types:          hall_facts, hall_events, hall_preferences, hall_advice (missing: hall_discoveries)
Storage:             SwiftData + Apple NLEmbedding (local, no ChromaDB)
Mining:              ExtractionService (LLM-based extraction only)
Tool Calling:        3 tools (save_fact, search, list_rooms)
Knowledge Graph:     ❌ Not implemented
Specialist Agents:   ❌ Not implemented
Compression:         ❌ Not implemented
Auto-Save Hooks:     ❌ Not implemented
```

## Feature Parity Matrix

### Tier 1: Core Palace Structure
| # | Feature | Upstream | SwiftBuddy | Status | Test |
|---|---------|----------|------------|--------|------|
| 1 | Wings (create, list, delete) | ✅ | Partial (create only) | 🔲 TODO | — |
| 2 | Rooms within wings | ✅ | ✅ | ✅ PASS | — |
| 3 | Hall types (5 categories) | 5 halls | 4 halls (missing `hall_discoveries`) | 🔲 TODO | — |
| 4 | Verbatim raw storage (drawers) | ✅ | ✅ (as MemoryEntry.text) | ✅ PASS | — |
| 5 | Closets (summaries pointing to drawers) | ✅ | ❌ No summary layer | 🔲 TODO | — |
| 6 | Tunnels (cross-wing room linking) | ✅ | ❌ | 🔲 TODO | — |

### Tier 2: Search & Retrieval
| # | Feature | Upstream | SwiftBuddy | Status | Test |
|---|---------|----------|------------|--------|------|
| 7 | Semantic vector search | ChromaDB | Apple NLEmbedding | ✅ PASS | — |
| 8 | Search within wing | ✅ | ✅ | ✅ PASS | — |
| 9 | Search within wing + room filter | ✅ | ❌ (wing-only filter) | 🔲 TODO | — |
| 10 | Search within wing + hall filter | ✅ | ❌ | 🔲 TODO | — |
| 11 | Cross-wing search (all wings) | ✅ | ❌ | 🔲 TODO | — |
| 12 | Duplicate detection before save | ✅ (`check_duplicate`) | ❌ | 🔲 TODO | — |

### Tier 3: Mining & Extraction
| # | Feature | Upstream | SwiftBuddy | Status | Test |
|---|---------|----------|------------|--------|------|
| 13 | Mine project files (code + docs) | ✅ | ❌ | 🔲 TODO | — |
| 14 | Mine conversation exports | ✅ | ❌ | 🔲 TODO | — |
| 15 | General extraction (auto-classify) | ✅ | Partial (LLM-based) | 🔄 WIP | — |
| 16 | Split mega-files into sessions | ✅ | ❌ | 🔲 TODO | — |

### Tier 4: Tool Calling (MCP Parity)
| # | Feature | Upstream MCP Tool | SwiftBuddy Tool | Status | Test |
|---|---------|-------------------|-----------------|--------|------|
| 17 | Save memory | `mempalace_add_drawer` | `mempalace_save_fact` | ✅ PASS | — |
| 18 | Search memory | `mempalace_search` | `mempalace_search` | ✅ PASS | — |
| 19 | List rooms | `mempalace_list_rooms` | `mempalace_list_rooms` | ✅ PASS | — |
| 20 | List wings | `mempalace_list_wings` | ❌ | 🔲 TODO | — |
| 21 | Get taxonomy | `mempalace_get_taxonomy` | ❌ | 🔲 TODO | — |
| 22 | Delete drawer | `mempalace_delete_drawer` | ❌ | 🔲 TODO | — |
| 23 | Traverse (navigate palace graph) | `mempalace_traverse` | ❌ | 🔲 TODO | — |
| 24 | Find tunnels | `mempalace_find_tunnels` | ❌ | 🔲 TODO | — |
| 25 | Palace status | `mempalace_status` | ❌ | 🔲 TODO | — |

### Tier 5: Knowledge Graph
| # | Feature | Upstream | SwiftBuddy | Status | Test |
|---|---------|----------|------------|--------|------|
| 26 | Add triple (entity-relationship) | ✅ (SQLite) | ❌ | 🔲 TODO | — |
| 27 | Query entity | ✅ | ❌ | 🔲 TODO | — |
| 28 | Invalidate fact (temporal windows) | ✅ | ❌ | 🔲 TODO | — |
| 29 | Timeline (chronological story) | ✅ | ❌ | 🔲 TODO | — |
| 30 | Contradiction detection | Partial (`fact_checker.py`) | ❌ | 🔲 TODO | — |

### Tier 6: Advanced
| # | Feature | Upstream | SwiftBuddy | Status | Test |
|---|---------|----------|------------|--------|------|
| 31 | Specialist agents with diary | ✅ | ❌ | 🔲 TODO | — |
| 32 | AAAK compression dialect | Experimental | ❌ | 🔲 TODO | — |
| 33 | Wake-up context (L0 + L1 ~170 tokens) | ✅ | ❌ | 🔲 TODO | — |
| 34 | Auto-save hooks (every N messages) | ✅ | ❌ | 🔲 TODO | — |

## Summary

| Tier | Total | Implemented | Gap |
|------|-------|-------------|-----|
| Core Palace Structure | 6 | 2 | 4 |
| Search & Retrieval | 6 | 2 | 4 |
| Mining & Extraction | 4 | 0.5 | 3.5 |
| Tool Calling (MCP) | 9 | 3 | 6 |
| Knowledge Graph | 5 | 0 | 5 |
| Advanced | 4 | 0 | 4 |
| **Total** | **34** | **7.5** | **26.5** |

Current parity: **~22%**. Highest-priority gaps: search filters (Tier 2), missing tool calls (Tier 4), and the hall_discoveries category (Tier 1).
