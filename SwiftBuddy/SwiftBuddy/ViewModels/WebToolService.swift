import Foundation
import SwiftSoup
import WebKit

@MainActor
final class WebToolService: NSObject, ObservableObject, WKNavigationDelegate {
    static let shared = WebToolService()
    
    // Hidden webview for JS-rendered SPAs
    private var hiddenWebView: WKWebView!
    private var webViewContinuation: CheckedContinuation<String, Error>?
    
    override init() {
        super.init()
        let config = WKWebViewConfiguration()
        config.preferences.javaScriptEnabled = true
        hiddenWebView = WKWebView(frame: .zero, configuration: config)
        hiddenWebView.navigationDelegate = self
    }
    
    /// Entrypoint: Attempts native HTML fetch first, falls back to WKWebView if empty/error or SPA heuristic.
    func fetchWebPageText(url: URL) async throws -> String {
        do {
            let html = try await fetchHTMLFast(url: url)
            let text = try extractText(from: html)
            
            // Heuristic to detect empty SPA skeleton
            if text.count < 100 && html.contains("<script") {
                print("[WebTool] Fast-fetch yielded little text, trying WebKit SPA fallback...")
                return try await fetchHTMLWithWebKit(url: url)
            }
            return text
            
        } catch {
            print("[WebTool] Fast-fetch failed: \(error), falling back to WebKit...")
            return try await fetchHTMLWithWebKit(url: url)
        }
    }
    
    // MARK: - SwiftSoup Native (Fast)
    
    private func fetchHTMLFast(url: URL) async throws -> String {
        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        request.setValue("Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0", forHTTPHeaderField: "User-Agent")
        
        let (data, _) = try await URLSession.shared.data(for: request)
        guard let htmlString = String(data: data, encoding: .utf8) else {
            throw URLError(.cannotDecodeRawData)
        }
        return htmlString
    }
    
    private func extractText(from html: String) throws -> String {
        let doc = try SwiftSoup.parse(html)
        
        // Remove unwanted elements
        try doc.select("script, style, nav, footer, header, noscript, iframe, .ad, .advertisement").remove()
        
        let body = try doc.body()
        return try body?.text() ?? ""
    }
    
    // MARK: - WKWebView (SPA Fallback)
    
    private func fetchHTMLWithWebKit(url: URL) async throws -> String {
        return try await withCheckedThrowingContinuation { continuation in
            self.webViewContinuation = continuation
            let request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData, timeoutInterval: 15)
            self.hiddenWebView.load(request)
        }
    }
    
    nonisolated func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        Task { @MainActor in
            // Wait slightly for SPA hydration
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            
            let js = "document.body.innerText"
            webView.evaluateJavaScript(js) { result, error in
                if let err = error {
                    self.webViewContinuation?.resume(throwing: err)
                } else if let text = result as? String {
                    self.webViewContinuation?.resume(returning: text)
                } else {
                    self.webViewContinuation?.resume(returning: "")
                }
                self.webViewContinuation = nil
            }
        }
    }
    
    nonisolated func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        Task { @MainActor in
            self.webViewContinuation?.resume(throwing: error)
            self.webViewContinuation = nil
        }
    }
}
