# Chat Tool Integration — Acceptance Criteria

## Feature 1: ChatMessage supports tool role
- **Action**: Add `.tool` to `ChatMessage.Role` enum in `MLXInferenceCore/ChatMessage.swift`.
- **Expected**: Instantiating `ChatMessage(role: .tool, content: "result")` works and properly maps to Hugging Face Jinja template roles.
- **Test**: `testFeature1_ChatMessageToolRole` verifies role string conversion.

## Feature 2: System Prompt Tool Schema Injection
- **Action**: Create a method that converts the JSON dictionary schemas from `MemoryPalaceTools.schemas` into a readable YAML/JSON string block.
- **Expected**: `ChatViewModel` dynamically appends this block to the persona's `ChatMessage.system` block at initialization.
- **Test**: `testFeature2_ToolSchemaInjection` verifies that the `system` message contains `"mempalace_search"`.

## Feature 3: LLM Output Tool Parsing 
- **Action**: Add `extractToolCall(from:)` to `ExtractionService`.
- **Expected**: Given an LLM output containing `<tool_call>{"name": "mempalace_search", "parameters": {"wing": "test", "query": "auth"}}</tool_call>`, it returns a structured Swift object containing the name and parameters dictionary.
- **Test**: `testFeature3_ToolCallExtraction` verifies valid and hallucinated JSON edge cases inside `<tool_call>` tags.

## Feature 4: ChatViewModel Autonomous Tool Execution Loop
- **Action**: Modify `ChatViewModel.send()`. If `extractToolCall` detects a tool call midway through generation, the UI hides the `<tool_call>` text.
- **Expected**: `ChatViewModel` cleanly halts user-facing generation, natively executes `MemoryPalaceTools.handleToolCall`, appends the tool response as `ChatMessage(role: .tool, content: result)`, and autonomously triggers `generate()` again to let the LLM see the tool result and answer the user.
- **Test**: `testFeature4_ToolExecutionLoopAsync` mocks an inference stream emitting a tool call and verifies the engine triggers the sequence autonomously.
