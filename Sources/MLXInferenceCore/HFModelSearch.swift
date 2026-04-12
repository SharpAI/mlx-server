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
    
    // Dynamically fetched after initial list
    public var usedStorage: Int64? = nil

    // Computed helpers
    public var repoOwner: String { String(id.split(separator: "/").first ?? "") }
    public var repoName: String  { String(id.split(separator: "/").last  ?? "") }
    public var isMlxCommunity: Bool { repoOwner == "mlx-community" }

    public var formatDisplay: String {
        guard let t = tags else { return "MLX" }
        if t.contains("gguf") { return "GGUF" }
        if t.contains("safetensors") { return "MLX" }
        return "MLX" // Default assumption from mlx-community
    }

    public var storageDisplay: String? {
        guard let s = usedStorage else { return nil }
        if s >= 1_000_000_000 {
            return String(format: "%.1f GB", Double(s) / 1_000_000_000)
        } else {
            return String(format: "%.1f MB", Double(s) / 1_000_000)
        }
    }

    /// Best-effort parameter size extracted from the model ID name.
    public var paramSizeHint: String? {
        let patterns = [
            #"(\d+)[xX](\d+)[Bb]"#, // 8x7B MoE
            #"(\d+\.?\d*)[BbmM]"#   // 7B, 0.5B, 3.8B, 350M, 150m
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

// MARK: — Size Filter

public enum HFSizeFilter: CaseIterable, Sendable, Equatable {
    case under0_5B, under1B, under3B, under7B, under13B, under32B, all

    public var label: String {
        switch self {
        case .under0_5B: return "≤0.5B"
        case .under1B: return "≤1B"
        case .under3B: return "≤3B"
        case .under7B: return "≤7B"
        case .under13B: return "≤13B"
        case .under32B: return "≤32B"
        case .all: return "All"
        }
    }

    public func matches(_ paramSizeText: String?) -> Bool {
        if self == .all { return true }
        guard let txt = paramSizeText?.lowercased() else { return false }
        
        let size: Double
        if txt.hasSuffix("m") {
            let mStr = txt.replacingOccurrences(of: "m", with: "")
            guard let mSize = Double(mStr) else { return false }
            size = mSize / 1000.0 // Convert to Billions
        } else {
            let bStr = txt.replacingOccurrences(of: "b", with: "")
            guard let bSize = Double(bStr) else { return false }
            size = bSize
        }
        
        switch self {
        case .under0_5B: return size <= 0.6
        case .under1B: return size <= 1.5 // Grace margin for 1.3B etc.
        case .under3B: return size <= 3.8 // Grace margin for 3.5B etc.
        case .under7B: return size <= 7.5
        case .under13B: return size <= 14.0
        case .under32B: return size <= 33.0
        case .all: return true
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
    private let maxFetchTries = 3
    private let pageSize = 20
    
    private var nextPageUrlString: String? = nil
    private var currentQuery = ""
    private var currentSort = HFSortOption.trending
    private var currentSizeFilter = HFSizeFilter.all
    private var debounceTask: Task<Void, Never>? = nil

    private init() {}

    // MARK: — Public API

    /// Debounced search — safe to call on every keystroke.
    public func search(query: String, sort: HFSortOption = .trending, sizeFilter: HFSizeFilter = .all) {
        debounceTask?.cancel()
        debounceTask = Task {
            // 300ms debounce
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard !Task.isCancelled else { return }
            currentQuery = query
            currentSort  = sort
            currentSizeFilter = sizeFilter
            nextPageUrlString = nil
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

    /// When the query looks like "owner/repo-name", fetch the model detail directly
    /// from /api/models/{id} to bypass pipeline_tag and library filtering.
    private func isDirectRepoQuery(_ query: String) -> Bool {
        let parts = query.split(separator: "/")
        return parts.count == 2 && !parts[0].isEmpty && !parts[1].isEmpty
    }

    private func fetchDirectRepo(_ modelId: String) async -> HFModelResult? {
        let urlStr = "https://huggingface.co/api/models/\(modelId)"
        guard let url = URL(string: urlStr) else { return nil }
        do {
            let (data, resp) = try await URLSession.shared.data(from: url)
            guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else { return nil }
            struct Detail: Decodable {
                let id: String
                let likes: Int?
                let downloads: Int?
                let pipeline_tag: String?
                let tags: [String]?
                let usedStorage: Int64?
            }
            guard let detail = try? JSONDecoder().decode(Detail.self, from: data) else { return nil }
            var result = HFModelResult(
                id: detail.id,
                likes: detail.likes,
                downloads: detail.downloads,
                pipeline_tag: detail.pipeline_tag,
                tags: detail.tags
            )
            result.usedStorage = detail.usedStorage
            return result
        } catch { return nil }
    }

    private func fetchPage() async {
        print("HFSearch: fetchPage started. Query: '\(currentQuery)' Sort: \(currentSort.rawValue)")
        isSearching = true
        errorMessage = nil

        // ── Direct repo ID fast-path ──────────────────────────────────────
        // If the query looks like "owner/model-name" skip search entirely
        // and hit the model detail endpoint. This is needed because the HF
        // search API filters on pipeline_tag which many MLX uploads don't match.
        if isDirectRepoQuery(currentQuery) {
            if let result = await fetchDirectRepo(currentQuery) {
                results = [result]
            } else {
                // Also try with prefix variants (e.g. user typed partial name without quant suffix)
                results = []
                errorMessage = nil
            }
            isSearching = false
            hasMore = false
            print("HFSearch: fetchPage finished")
            return
        }

        var localResults: [HFModelResult] = []
        var tries = 0

        while localResults.count < 10 && tries < maxFetchTries {
            tries += 1

            var urlToFetch: URL
            if let next = nextPageUrlString, let url = URL(string: next) {
                urlToFetch = url
            } else {
                var finalQuery = currentQuery
                if !strictMLX && !finalQuery.lowercased().contains("mlx") && !finalQuery.isEmpty {
                    finalQuery = finalQuery + " mlx"
                }

                var components = URLComponents(string: hfBase)!
                var queryItems: [URLQueryItem] = [
                    // NOTE: pipeline_tag intentionally omitted — many MLX uploads use
                    // text2text-generation, feature-extraction, etc. Filtering by
                    // pipeline_tag causes legitimate models to disappear from results.
                    URLQueryItem(name: "sort",  value: currentSort.rawValue),
                    URLQueryItem(name: "limit", value: "\(pageSize)"),
                    URLQueryItem(name: "full",  value: "false"),
                ]
                if !finalQuery.isEmpty {
                    queryItems.append(URLQueryItem(name: "search", value: finalQuery))
                }
                if strictMLX {
                    queryItems.append(URLQueryItem(name: "library", value: "mlx"))
                }
                components.queryItems = queryItems
                guard let constructedUrl = components.url else { break }
                urlToFetch = constructedUrl
            }
            
            do {
                let (data, response) = try await URLSession.shared.data(from: urlToFetch)
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    errorMessage = "HuggingFace API unavailable"
                    break
                }
                
                nextPageUrlString = nil
                if let linkHeader = http.value(forHTTPHeaderField: "Link") {
                    let parts = linkHeader.components(separatedBy: ",")
                    for part in parts {
                        if part.contains("rel=\"next\"") {
                            if let start = part.range(of: "<")?.upperBound,
                               let end = part.range(of: ">")?.lowerBound {
                                nextPageUrlString = String(part[start..<end])
                            }
                        }
                    }
                }
                
                var page = try JSONDecoder().decode([HFModelResult].self, from: data)
                let originalPageCount = page.count

                // Strip GGUF models explicitly (SwiftLM engine only locally loads MLX tensors)
                page = page.filter { result in
                    let isGGUF = result.formatDisplay == "GGUF" || result.id.lowercased().contains("gguf")
                    return !isGGUF
                }

                // Local Size Filtering
                if currentSizeFilter != .all {
                    page = page.filter { currentSizeFilter.matches($0.paramSizeHint) }
                }

                if !page.isEmpty {
                    // Fetch usedStorage for each matched model seamlessly without throwing
                    await withTaskGroup(of: (Int, Int64?).self) { group in
                        for i in 0..<page.count {
                            let safeModelId = page[i].id
                            group.addTask {
                                let detailUrl = URL(string: "https://huggingface.co/api/models/\(safeModelId)")!
                                do {
                                    let (detailData, detailResp) = try await URLSession.shared.data(from: detailUrl)
                                    guard let httpD = detailResp as? HTTPURLResponse, httpD.statusCode == 200 else { return (i, nil) }
                                    struct HFFullDetails: Decodable { let usedStorage: Int64? }
                                    let details = try? JSONDecoder().decode(HFFullDetails.self, from: detailData)
                                    return (i, details?.usedStorage)
                                } catch { return (i, nil) }
                            }
                        }
                        for await (index, size) in group {
                            if let size = size { page[index].usedStorage = size }
                        }
                    }
                    localResults.append(contentsOf: page)
                }

                if originalPageCount < pageSize {
                    hasMore = false
                    break // end of HF pagination
                } else {
                    hasMore = true
                }
            } catch is CancellationError {
                break
            } catch {
                errorMessage = "Search failed: \(error.localizedDescription)"
                break
            }
        }

        if !localResults.isEmpty {
            results.append(contentsOf: localResults)
        }
        
        isSearching = false
        print("HFSearch: fetchPage finished")
    }
}
