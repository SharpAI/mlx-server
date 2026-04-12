# Gemma 4 Omni: Any-to-Any Acceptance & Test Plan

## Acceptance Criteria
1. **Structural Equivalence**: The MLX Swift models must define the exact architectural layers present in the `mlx-community/gemma-4-e4b-it-4bit` release (Subsample Convolutions, Clipped Linears, Full Conformer Blocks).
2. **Key Resolution**: The `sanitize(weights:)` pass must operate successfully without arbitrary string-manipulation hacks by utilizing matching `@ModuleInfo` binding names natively.
3. **Multimodal Stability**: A graph containing pure `<|audio|>` payloads must not collapse. Audio values must properly shape-match text inputs (`2560` embedding dimension) when dynamically generated during sequence merging.

## Test Plan
This is fully automated within `run_harness.sh` using the following scenarios:

- **Scenario 1: Build & Integrity Check**
    - `swift build -c release`
    - Ensures that Swift 6 compiler passes without `Sendable`, Actor Isolation, or invalid `MLX/MLXFast` module conflicts.
- **Scenario 2: Native Routing Analysis**
    - The `.agents/harness/audio-omni-gemma4/run_harness.sh` injects a simulated integration payload into explicitly triggering `SwiftLMTests.testGemma4Audio`.
    - Captures STDOUT to verify `MLX.zeros(1, 80, SeqLen)` appropriately generates without blowing up the computation graph.
- **Scenario 3: Zero-Shot Any-to-Any Parsing**
    - The `run_harness.sh` generates an Omni JSON payload imitating standard `SwiftBuddy` chat structures where `<|audio|>` tokens are synthetically appended.
    - Validates that `UserInput.Audio` parsing cascades faithfully into `LMInput.ProcessedAudio`, resolving earlier issues where SwiftLM lacked the fundamental `[Audio]` property class.
