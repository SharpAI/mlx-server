#!/usr/bin/env python3
import sys
import urllib.request
import json
from urllib.parse import urlencode

def fetch_top_model(query):
    # If query has author/model, split them for better API accuracy
    author = None
    if "/" in query:
        author, search_term = query.split("/", 1)
    else:
        search_term = query
        
    # Construct Hugging Face Hub API Request sorting explicitly by downloads
    params = {
        "search": search_term,
        "sort": "downloads",
        "direction": "-1",
        "limit": 10
    }
    if author:
        params["author"] = author
        
    url = f"https://huggingface.co/api/models?{urlencode(params)}"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'SwiftLM-Benchmark/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            if not data:
                return None
            
            # The API returns a list of dictionaries sorted by downloads
            for model in data:
                model_id = model.get("id")
                if model_id:
                    return model_id
            return None
    except Exception as e:
        print(f"Error fetching from HF Hub: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: hf_discovery.py <query>")
        sys.exit(1)
    
    query = sys.argv[1]
    top_model = fetch_top_model(query)
    
    if top_model:
        # Standard stdout binding to be captured by run_benchmark.sh
        print(top_model)
        sys.exit(0)
    else:
        sys.exit(1)
