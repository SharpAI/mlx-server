import Foundation
let hfBase = "https://huggingface.co/api/models"
var components = URLComponents(string: hfBase)!
var queryItems: [URLQueryItem] = [
    URLQueryItem(name: "pipeline_tag", value: "text-generation"),
    URLQueryItem(name: "sort",         value: "trendingScore"),
    URLQueryItem(name: "limit",        value: "20"),
    URLQueryItem(name: "offset",       value: "0"),
    URLQueryItem(name: "full",         value: "false"),
]
queryItems.append(URLQueryItem(name: "library", value: "mlx"))
components.queryItems = queryItems
print(components.url?.absoluteString ?? "NIL URL")
