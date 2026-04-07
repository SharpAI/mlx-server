---
description: Run the persistent SwiftBuddy TDD harness loop (memory handling + model management)
---

# Run SwiftBuddy Harness

This workflow executes the persistent TDD harness defined in `.agents/harness/`.

## Steps

1. Read the harness README at `.agents/harness/README.md` to understand the system.

2. Read `.agents/harness/harness_config.md` for build commands and the Agent Loop Protocol.

3. **Memory Handling Harness**:
   - Read `.agents/harness/memory/features.md` to find all 🔲 TODO items.
   - For each TODO, read the acceptance criteria in `.agents/harness/memory/acceptance.md`.
   - Load any relevant fixture files from `.agents/harness/memory/fixtures/`.
   - Follow the Agent Loop Protocol: write test → run → implement → verify → update status.

4. **Model Management Harness**:
   - Read `.agents/harness/model-management/features.md` to find all 🔲 TODO items.
   - For each TODO, read the acceptance criteria in `.agents/harness/model-management/acceptance.md`.
   - Load any relevant fixture files from `.agents/harness/model-management/fixtures/`.
   - Follow the Agent Loop Protocol: write test → run → implement → verify → update status.

// turbo-all
5. Run the test suite:
   ```
   swift test --filter SwiftBuddyTests
   ```

6. Write a timestamped run log to the appropriate `runs/` directory.

7. Report completion: list all features with their final status.
