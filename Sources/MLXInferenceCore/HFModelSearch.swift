// HFModelSearch.swift — Live HuggingFace model search for MLX models
//
// API: https://huggingface.co/api/models?library=mlx&pipeline_tag=text-generation
//      &search=<query>&sort=trending&limit=20&full=false
//
// Mirrors the Aegis-AI pattern (LocalLanguageModels.tsx + modelService.ts):
//   library=mlx  →  filters to MLX-format models (mlx-community and others)
//   pipeline_tag →  text-generation or text2text-generation
//   sort         →  trending (default), downloads, likes, lastModified

import Foundation

// MARK: — HF API model result

public struct HFModelResult: Identifiable, Sendable, Decodable {
    public let id: String               // e.g. "mlx-community/Qwen2.5-7B-Instruct-4bit"
    public let likes: Int?
    public let downloads: Int?
    public let pipeline_tag: String?    // "text-generation"
    public let tags: [String]?

    // Computed helpers
    public var repoOwner: String { String(id.split(separator: "/").first ?? "") }
    public var repoName: String  { String(id.split(separator: "/").last  ?? "") }
    public var isMlxCommunity: Bool { repoOwner == "mlx-community" }

    /// Best-effort parameter size extracted from the model ID name.
    public var paramSizeHint: String? {
        let patterns = [
            #"(\d+)[xX](\d+)[Bb]"#, // 8x7B MoE
            #"(\d+\.?\d*)[Bb]"#    // 7B, 0.5B, 3.8B
        ]
        for pattern in patterns {
            if let match = repoName.range(of: pattern, options: .regularExpression) {
                return String(repoName[match])
            }
        }
        return nil
    }

    /// True if the model name suggests MoE architecture.
    public var isMoE: Bool {
        let lower = repoName.lowercased()
        return lower.contains("moe") || lower.contains("-a") || lower.contains("_a")
    }

    public var downloadsDisplay: String {
        guard let d = downloads else { return "" }
        if d >= 1_000_000 { return String(format: "%.1fM↓", Double(d) / 1_000_000) }
        if d >= 1_000     { return String(format: "%.0fk↓", Double(d) / 1_000) }
        return "\(d)↓"
    }

    public var likesDisplay: String {
        guard let l = likes, l > 0 else { return "" }
        if l >= 1_000 { return String(format: "%.0fk♥", Double(l) / 1_000) }
        return "\(l)♥"
    }
}

// MARK: — Sort options (matching Aegis-AI LocalLanguageModels sort selector)

public enum HFSortOption: String, CaseIterable, Sendable {
    case trending    = "trendingScore"
    case downloads   = "downloads"
    case likes       = "likes"
    case lastModified = "lastModified"

    public var label: String {
        switch self {
        case .trending:     return "Trending"
        case .downloads:    return "Downloads"
        case .likes:        return "Likes"
        case .lastModified: return "Newest"
        }
    }
}

// MARK: — HFModelSearchService

@MainActor
public final class HFModelSearchService: ObservableObject {
    public static let shared = HFModelSearchService()

    @Published public var results: [HFModelResult] = []
    @Published public var isSearching = false
    @Published public var errorMessage: String? = nil
    @Published public var hasMore = false
    @Published public var strictMLX: Bool = true

    private let hfBase = "https://huggingface.co/api/models"
    private let pageSize = 20
    private var currentOffset = 0
    private var currentQuery = ""
    private var currentSort = HFSortOption.trending
    private var debounceTask: Task<Void, Never>? = nil

    private init() {}

    // MARK: — Public API

    /// Debounced search — safe to call on every keystroke.
    public func search(query: String, sort: HFSortOption = .trending) {
        debounceTask?.cancel()
        debounceTask = Task {
            // 300ms debounce
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard !Task.isCancelled else { return }
            currentQuery = query
            currentSort  = sort
            currentOffset = 0
            results = []
            await fetchPage()
        }
    }

    /// Load next page of results.
    public func loadMore() {
        guard hasMore, !isSearching else { return }
        Task { await fetchPage() }
    }

    // MARK: — Private

    private func fetchPage() async {
        print("HFSearch: fetchPage started. Query: '\(currentQuery)' Sort: \(currentSort.rawValue)")
        isSearching = true
        errorMessage = nil

        var finalQuery = currentQuery
        if !strictMLX && !finalQuery.lowercased().contains("mlx") && !finalQuery.isEmpty {
            finalQuery = finalQuery + " mlx"
        }

        var components = URLComponents(string: hfBase)!
        var queryItems: [URLQueryItem] = [
            URLQueryItem(name: "pipeline_tag", value: "text-generation"),
            URLQueryItem(name: "sort",         value: currentSort.rawValue),
            URLQueryItem(name: "limit",        value: "\(pageSize)"),
            URLQueryItem(name: "offset",       value: "\(currentOffset)"),
            URLQueryItem(name: "full",         value: "false"),
        ]
        if !finalQuery.isEmpty {
            queryItems.append(URLQueryItem(name: "search", value: finalQuery))
        }
        if strictMLX {
            queryItems.append(URLQueryItem(name: "library", value: "mlx"))
        }
        components.queryItems = queryItems

        guard let url = components.url else {
            print("HFSearch: Failed to build URL")
            isSearching = false
            return
        }

        print("HFSearch: Fetching from url: \(url.absoluteString)")

        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let http = response as? HTTPURLResponse else {
                print("HFSearch: Response was not HTTPURLResponse")
                errorMessage = "HuggingFace search unavailable"
                isSearching = false
                return
            }
            print("HFSearch: Response status code: \(http.statusCode)")
            if http.statusCode != 200 {
                errorMessage = "HuggingFace API returned \(http.statusCode)"
                isSearching = false
                return
            }
            
            do {
                let page = try JSONDecoder().decode([HFModelResult].self, from: data)
                print("HFSearch: Decoded \(page.count) models")
                results.append(contentsOf: page)
                hasMore = page.count == pageSize
                currentOffset += page.count
            } catch {
                print("HFSearch: Decode error: \(error)")
                errorMessage = "Decode error: \(error.localizedDescription)"
            }
        } catch is CancellationError {
            print("HFSearch: Task was cancelled")
        } catch {
            print("HFSearch: URLSession threw error: \(error)")
            errorMessage = "Search failed: \(error.localizedDescription)"
        }

        isSearching = false
        print("HFSearch: fetchPage finished")
    }
}
