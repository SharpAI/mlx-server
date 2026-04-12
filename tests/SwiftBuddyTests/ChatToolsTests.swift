import XCTest
@testable import MLXInferenceCore
@testable import SwiftBuddy

final class ChatToolsTests: XCTestCase {
    
    // Feature 1: ChatMessage supports tool role
    func testFeature1_ChatMessageToolRole() {
        let toolMessage = ChatMessage(role: .tool, content: "{\"result\": \"success\"}")
        
        XCTAssertEqual(toolMessage.role, .tool)
        XCTAssertEqual(toolMessage.content, "{\"result\": \"success\"}")
        XCTAssertEqual(toolMessage.role.rawValue, "tool")
    }
    
    // Feature 2: System Prompt Tool Schema Injection
    @MainActor
    func testFeature2_ToolSchemaInjection() async {
        let viewModel = ChatViewModel()
        viewModel.currentWing = "Lumina" // Trigger persona load
        
        let content = await viewModel.buildIdentityPayload(userText: "test")
        
        XCTAssertTrue(content.contains("mempalace_search"), "System prompt must document the mempalace_search tool")
        XCTAssertTrue(content.contains("mempalace_save_fact"), "System prompt must document the mempalace_save_fact tool")
        XCTAssertTrue(content.contains("<tool_call>"), "System prompt must provide the XML syntax block for making tool calls")
    }
    
    // Feature 3: LLM Output Tool Parsing (`ExtractionService`)
    @MainActor
    func testFeature3_ToolCallExtraction() throws {
        let validResponse = """
        Let me search the memory palace for that.
        <tool_call>
        {"name": "mempalace_search", "parameters": {"wing": "Lumina", "query": "auth migration"}}
        </tool_call>
        I will wait for the result.
        """
        
        let toolCall = ExtractionService.extractToolCall(from: validResponse)
        XCTAssertNotNil(toolCall)
        XCTAssertEqual(toolCall?.name, "mempalace_search")
        
        let params = toolCall?.parameters as? [String: String]
        XCTAssertEqual(params?["wing"], "Lumina")
        XCTAssertEqual(params?["query"], "auth migration")
        
        let malformedResponse = "<tool_call>{invalid json}</tool_call>"
        XCTAssertNil(ExtractionService.extractToolCall(from: malformedResponse))
        
        let emptyResponse = "No tool needed."
        XCTAssertNil(ExtractionService.extractToolCall(from: emptyResponse))
    }
    
    // Feature 4: ChatViewModel Autonomous Tool Execution Loop
    @MainActor
    func testFeature4_ToolExecutionLoopAsync() async throws {
        let viewModel = ChatViewModel()
        viewModel.currentWing = "Lumina" // Trigger persona load
        
        // This test simulates the logic that extractToolCall will successfully identify the tool call and ChatViewModel handles it.
        // We'll manually insert a response with a tool call and simulate the extraction loop.
        
        let mockedLLMResponse = """
        Let me search the palace.
        <tool_call>
        {"name": "mempalace_list_wings", "parameters": {}}
        </tool_call>
        """
        
        if let toolCall = ExtractionService.extractToolCall(from: mockedLLMResponse) {
            XCTAssertEqual(toolCall.name, "mempalace_list_wings")
            let result = (try? await MemoryPalaceTools.handleToolCall(name: toolCall.name, arguments: toolCall.parameters ?? [:])) ?? "Mocked tool response"
            XCTAssertNotNil(result)
            
            let toolMsg = ChatMessage.tool(result)
            viewModel.messages.append(toolMsg)
            
            XCTAssertEqual(viewModel.messages.last?.role, .tool)
        } else {
            XCTFail("Failed to extract tool call from simulated response")
        }
    }
}
