---
description: Autonomous Web Design Workflow & Harness for Agentic Product Marketing
---
// turbo-all

# Autonomous Web Design Harness

> **CRITICAL EXECUTION RULE**: Do NOT immediately begin scaffolding UI elements, generating glassmorphic tokens, or assuming dark-mode when tasked with building a web page. You MUST follow these preliminary research and alignment phases strictly.

When tasked with designing a web page or marketing asset for the SwiftLM ecosystem (or any future project), execute the following workflow sequentially.

## Phase 1: Social Listening & User Empathy
Before designing, you must understand what actual users care about.
- **Action**: Use the `search_web` tool to search Reddit, Twitter/X, and relevant forums. For example: `site:reddit.com "local llm" "mlx" "pain points"`
- **Goal**: Identify 2-3 massive user frustrations (e.g., "Ollama is too slow for agents", "VLM context overflow ruins memory").
- **Output**: Mentally synthesize a target user persona and their primary pain point to drive the entire design narrative.

## Phase 2: Establish the Selling Points
Translate the Phase 1 pain points into product strengths.
- **Action**: Draft 3-5 high-impact, heavily technical but readable "Selling Points". 
- **Rule**: Do not use generic marketingspeak (e.g., "Fast and simple"). Use concrete technical assertions (e.g., "1000 tok/s M3 Max prefill", "No GIL overhead", "Zero-copy NVMe streaming").
- **Goal**: These selling points will directly dictate the layout of the site's "Feature Grid" or "Hero Subtext".

## Phase 3: Visual Inspiration & Benchmarking
Do not design in a vacuum.
- **Action**: Reflect on (or search for) industry-leading developer tools in the AI space (e.g., Vercel, Linear, Modal, HuggingFace).
- **Goal**: Establish a baseline for typography (e.g., Inter, Geist), spacing (large padding, sparse layouts), and structural hierarchy. 

## Phase 4: Aesthetic Constraints & Generation
Now you may begin scaffolding the site.
- **Rule 1 (The Light Default)**: Do NOT aggressively default to dark colors or dark mode. Unless the user explicitly requests dark mode, default to a clean, highly accessible, modern light mode aesthetic.
- **Rule 2 (Layout Hierarchy)**:
   1. Dynamic Hero Section (Strong Tagline + Call to Action).
   2. Social Proof / Testimonial Billboard (Actual quotes from Phase 1).
   3. The Feature Grid (The selling points from Phase 2).
   4. Ecosystem Linkages (How it ties into the broader architecture).
- **Action**: Execute code generation using standard TailwindCSS tokens or explicit Vanila CSS constraints.
