---
description: Review a GitHub Issue or PR for SharpAI/SwiftLM — fetch, analyze, implement fixes, address review comments, and push back to the correct branch
---

# Review GitHub Issue / PR

This workflow guides end-to-end handling of a GitHub Issue or Pull Request for the
`SharpAI/SwiftLM` repository: from fetching context, through implementing or
reviewing code changes, to pushing a clean commit back to the correct fork branch.

---

## Prerequisites

- `gh` CLI path on macOS: **`/opt/homebrew/bin/gh`**
  ```bash
  export PATH="/opt/homebrew/bin:$PATH"
  which gh  # → /opt/homebrew/bin/gh
  ```
- `gh` must be authenticated (`gh auth status`)
- Working directory: `/Users/simba/workspace/mlx-server`
- Remote `fork` may need to be added if pushing to a contributor's fork:
  ```bash
  git remote add fork https://github.com/<contributor>/SwiftLM.git
  ```

---

## Steps

### 1. Fetch the Issue or PR

Determine whether the user supplied an **Issue number** or a **PR number**, then
pull the full context using `gh`:

```bash
# For a PR
gh pr view <NUMBER> --repo SharpAI/SwiftLM \
  --json number,title,body,state,baseRefName,headRefName,headRepository,commits,files

# For an Issue
gh issue view <NUMBER> --repo SharpAI/SwiftLM \
  --json number,title,body,state,labels,comments
```

Note the **`headRepository`** field — if it is not `SharpAI/SwiftLM`, the PR comes
from a fork. You must push back to the fork's branch (see Step 6).

---

### 2. Understand the Scope

Read the PR/Issue body and associated comments carefully. Identify:

- **Category** — bug fix, feature, test improvement, CI/CD, documentation.
- **Files touched** — run `gh pr diff <NUMBER> --repo SharpAI/SwiftLM` or read
  the `files` field.
- **CI status** — check the latest run:
  ```bash
  gh run list --repo SharpAI/SwiftLM --branch <headRefName> --limit 3
  ```
- **Review comments** — if Copilot or a human left inline review comments, read
  them all before writing a single line of code:
  ```bash
  gh pr view <NUMBER> --repo SharpAI/SwiftLM --comments
  ```

---

### 3. Check Out the Branch Locally

```bash
# If the PR is from SharpAI directly
git fetch origin
git checkout <headRefName>

# If the PR is from a fork
git remote add fork https://github.com/<forkOwner>/SwiftLM.git   # once only
git fetch fork <headRefName>
git checkout -b <headRefName> fork/<headRefName>
```

Verify you are on the correct branch:
```bash
git status
git log --oneline -5
```

---

### 4. Triage Review Comments (for PRs)

For each Copilot or human review comment:

1. **Classify** the severity:
   - 🔴 **Must fix** — correctness bugs, resource leaks, race conditions, broken CI.
   - 🟡 **Should fix** — test coverage gaps, false-pass logic, missing imports.
   - 🟢 **Optional** — style, wording, architecture refactors beyond the PR scope.

2. **Implement** all 🔴 and 🟡 items. For 🟢 items, document them as follow-up
   work in a code comment or GitHub comment but do not expand the PR scope.

3. **Key patterns learned from SwiftLM history**:
   - Shell scripts use `set -euo pipefail` — every `grep`, `jq`, or pipeline that
     may produce no output **must** be guarded with `|| true` or placed inside an
     `if` condition to prevent silent script abort.
   - Heartbeat / background `Task` objects in Swift **must** be cancelled via
     `defer { task?.cancel() }` so all exit paths (including client disconnect)
     are covered — not just the happy path.
   - CORS-related shell tests must target the dedicated `--cors` server instance,
     not the main server started without the flag.
   - Concurrent-request tests must use `--parallel N` (N ≥ 2) to actually exercise
     parallel code paths.
   - When adding new Swift test files that use `Data` / `JSONSerialization`,
     always add `import Foundation` — XCTest does not re-export it in all SPM environments.

---

### 5. Verify Locally

Build and run the relevant test suite before pushing:

```bash
# Swift unit tests
swift test --filter SwiftLMTests

# Integration tests (server)
./tests/test-server.sh .build/release/SwiftLM 15413

# OpenCode / SDK compatibility test
./tests/test-opencode.sh .build/release/SwiftLM 15414
```

If CI previously failed with a specific test number, reproduce it locally first:
```bash
gh run view <RUN_ID> --repo SharpAI/SwiftLM --log-failed 2>&1 | grep -E "FAIL|error|Test [0-9]+"
```

---

### 6. Commit and Push to the Correct Remote

> [!IMPORTANT]
> Always push to the **fork's branch** when updating a fork-originated PR.
> Pushing to `origin` (SharpAI) creates a new branch and does NOT update the PR.

```bash
git add <files>
git commit -m "<type>(<scope>): <summary>

<body: what changed and why>"

# PR from a fork → push to fork
git push fork <headRefName>:<headRefName>

# PR from SharpAI directly → push to origin
git push origin <headRefName>
```

Verify the PR was updated:
```bash
gh pr view <NUMBER> --repo SharpAI/SwiftLM --json commits --jq '.commits[].messageHeadline'
```

---

### 7. Monitor CI

After pushing, monitor the triggered workflow:

```bash
# List recent runs on the branch
gh run list --repo SharpAI/SwiftLM --branch <headRefName> --limit 5

# Stream logs for the latest run
gh run view <RUN_ID> --repo SharpAI/SwiftLM --log

# Pull only failed steps
gh run view <RUN_ID> --repo SharpAI/SwiftLM --log-failed 2>&1 | grep -E "FAIL|error|exit code"
```

If tests fail, go back to Step 4. Iterate until CI is green.

---

### 8. Respond to Reviewers (Optional)

If a human or Copilot reviewer left inline comments that you have addressed,
leave a reply comment summarising what was changed and why each item was handled
(or deferred):

```bash
gh pr comment <NUMBER> --repo SharpAI/SwiftLM \
  --body "Addressed all 🔴/🟡 review comments in commit <SHA>:
- heartbeat leak: added defer cleanup in both streaming handlers
- import Foundation: added to ServerSSETests.swift
- CORS test: redirected to CORS_PORT server
- parallel test: dedicated --parallel 2 server on PORT+3
- set -e trap: guarded grep/jq pipelines with || true"
```

---

## Quick Reference

| Task | Command |
|------|---------|
| View PR | `gh pr view <N> --repo SharpAI/SwiftLM` |
| View PR diff | `gh pr diff <N> --repo SharpAI/SwiftLM` |
| View PR comments | `gh pr view <N> --repo SharpAI/SwiftLM --comments` |
| View Issue | `gh issue view <N> --repo SharpAI/SwiftLM` |
| List CI runs | `gh run list --repo SharpAI/SwiftLM --branch <branch>` |
| Failed CI logs | `gh run view <ID> --repo SharpAI/SwiftLM --log-failed` |
| Push to fork | `git push fork <branch>:<branch>` |
| Push to SharpAI | `git push origin <branch>` |
| Verify PR commits | `gh pr view <N> --repo SharpAI/SwiftLM --json commits --jq '.commits[].messageHeadline'` |
