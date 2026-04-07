# MemPalace Feature Parity — Acceptance Criteria

These criteria are ordered by implementation priority (Tier 1 → Tier 6).

---

## Tier 1: Core Palace Structure

### Feature 1: Wings CRUD (create, list, delete)
- **Create**: `saveMemory(wingName: "test")` → creates wing if not exists → ✅ already works
- **List**: New method `listWings()` → returns `["wing_a", "wing_b"]`
- **Delete**: New method `deleteWing("wing_a")` → cascades to rooms and memories
- **FAIL if**: listWings returns empty after saveMemory, or deleteWing leaves orphaned rooms

### Feature 3: Add `hall_discoveries` category
- **Input**: `saveMemory(wingName: "test", roomName: "physics", text: "found gravitational waves", type: "hall_discoveries")`
- **Expected**: Memory saved with `hallType = "hall_discoveries"`
- **FAIL if**: The ExtractionService system prompt doesn't include `hall_discoveries` as a valid category

### Feature 5: Closets (summary layer)
- **Expected**: Each Room has an auto-generated summary (closet) that points to its raw drawers
- **Acceptance**: After saving 5 memories to a room, a `getCloset(roomName:wingName:)` method returns a summary string < 200 tokens that captures the gist
- **FAIL if**: No summary exists, or summary is longer than the raw memories combined

### Feature 6: Tunnels (cross-wing room linking)
- **Setup**: Wing "kai" has room "auth-migration". Wing "driftwood" has room "auth-migration".
- **Expected**: `findTunnels(roomName: "auth-migration")` returns both wings
- **FAIL if**: Rooms with the same name across wings are not discovered

---

## Tier 2: Search & Retrieval

### Feature 9: Search with room filter
- **Action**: `searchMemories(query: "auth", wingName: "kai", roomName: "security")`
- **Expected**: Only returns memories from the "security" room, not "billing" or "deploy"
- **FAIL if**: Results include memories from other rooms

### Feature 10: Search with hall filter
- **Action**: `searchMemories(query: "decided", wingName: "kai", hallType: "hall_facts")`
- **Expected**: Only returns memories with `hallType == "hall_facts"`
- **FAIL if**: Results include events or preferences

### Feature 11: Cross-wing search
- **Action**: `searchAllMemories(query: "database choice")`
- **Expected**: Returns results from ALL wings, ranked by relevance
- **FAIL if**: Method doesn't exist or only returns from one wing

### Feature 12: Duplicate detection
- **Action**: Try saving `"User prefers dark mode"` twice to the same wing+room
- **Expected**: Second save returns a warning or is silently deduped (cosine similarity > 0.95)
- **FAIL if**: Both duplicates are stored without warning

---

## Tier 4: Tool Calling (MCP Parity)

### Feature 20: List wings tool
- **Tool name**: `mempalace_list_wings`
- **Expected**: Returns JSON array of all wing names
- **FAIL if**: Tool is not registered in `MemoryPalaceTools.schemas`

### Feature 21: Get taxonomy tool
- **Tool name**: `mempalace_get_taxonomy`
- **Expected**: Returns the full palace structure: wings → rooms → memory counts
- **FAIL if**: Missing from tool schemas or returns flat list instead of hierarchy

### Feature 22: Delete drawer tool
- **Tool name**: `mempalace_delete_drawer`
- **Parameters**: `wing`, `room`, `fact_text` (or `fact_id`)
- **Expected**: Removes the specific memory entry
- **FAIL if**: Deletes the entire room instead of one memory

### Feature 25: Palace status tool
- **Tool name**: `mempalace_status`
- **Expected**: Returns wing count, room count, total memories, disk usage estimate
- **FAIL if**: Any count is missing or incorrect
