# Harness Configuration

## Build Environment

| Key | Value |
|-----|-------|
| Working Directory | `/Users/simba/workspace/mlx-server` |
| Test Command | `swift test --filter SwiftBuddyTests` |
| Build Command | `swift build` |
| Xcode Build | `xcodebuild -project SwiftBuddy/SwiftBuddy.xcodeproj -scheme SwiftBuddy -destination 'platform=macOS,arch=arm64' ONLY_ACTIVE_ARCH=YES` |
| Architecture | arm64 only (Apple Silicon) |
| Platform | macOS 14.0+ |
| Swift Version | 5.9 |

## Source of Truth

| Component | Location | Notes |
|-----------|----------|-------|
| Package definition | `Package.swift` | Defines the `SwiftBuddyTests` test target |
| Test target | `Tests/SwiftBuddyTests/` | All XCTest files live here |
| Xcode project | `SwiftBuddy/generate_xcodeproj.py` | For GUI builds only, NOT for tests |
| Memory extraction | `SwiftBuddy/SwiftBuddy/ViewModels/ExtractionService.swift` | Core `cleanJSON()` logic |
| HF search service | `Sources/MLXInferenceCore/HFModelSearch.swift` | `HFModelSearchService` singleton |
| Model management UI | `SwiftBuddy/SwiftBuddy/Views/ModelManagementView.swift` | UI layer |
| Model picker UI | `SwiftBuddy/SwiftBuddy/Views/ModelPickerView.swift` | HFSearchTab lives here |

## Agent Loop Protocol

```
┌─────────────────────────────────────────────┐
│  1. Read features.md                        │
│     → Identify first 🔲 TODO item           │
├─────────────────────────────────────────────┤
│  2. Read acceptance.md                      │
│     → Get exact input→output contract       │
├─────────────────────────────────────────────┤
│  3. Write the XCTest case                   │
│     → Test MUST fail initially (TDD red)    │
├─────────────────────────────────────────────┤
│  4. Run: swift test --filter SwiftBuddyTests│
│     → Confirm failure matches expectation   │
├─────────────────────────────────────────────┤
│  5. Implement the feature in source code    │
├─────────────────────────────────────────────┤
│  6. Run: swift test --filter SwiftBuddyTests│
│     → Confirm all tests pass (TDD green)    │
├─────────────────────────────────────────────┤
│  7. Update features.md → ✅ PASS            │
├─────────────────────────────────────────────┤
│  8. Write run log to runs/                  │
├─────────────────────────────────────────────┤
│  9. Loop back to step 1                     │
│     → Until all features are ✅             │
├─────────────────────────────────────────────┤
│ 10. Report completion                       │
└─────────────────────────────────────────────┘
```

### Failure Recovery

If step 6 fails:
1. Read the XCTest failure output carefully
2. Fix the **source code** (not the test)
3. Re-run `swift test`
4. If still failing after 3 attempts, log the failure in `runs/` with details and move to the next feature
5. Mark the feature as ❌ FAIL in `features.md` with a note

### Completion Criteria

A harness is **complete** when:
- Every feature in `features.md` is marked ✅ PASS
- The final `swift test` run exits with code 0
- A run log exists documenting the final green run
