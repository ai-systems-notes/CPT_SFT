#!/usr/bin/env python3
"""Build the canonical, source-grounded QA set used for the 12-model ablation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "ai_coding_agent_cpt_data"
DOCS_PATH = DATA_ROOT / "data/scraped_documents.jsonl"
SFT_PATH = DATA_ROOT / "SFT_Dataset/sft_v3_train.jsonl"
SFT_SOURCES_PATH = DATA_ROOT / "SFT_Dataset/sft_v3_domain_sources.jsonl"
OUTPUT_DIR = DATA_ROOT / "QA_Dataset"
CLAUDE_PATH = OUTPUT_DIR / "eval_qa_claude.jsonl"
CODEX_PATH = OUTPUT_DIR / "eval_qa_codex.jsonl"
COMBINED_PATH = OUTPUT_DIR / "eval_qa_combined.jsonl"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def evidence_token(text: str) -> str:
    return normalize(text).strip("`* _.:;()[]{}\"").casefold()


def exact_evidence_excerpt(content: str, keywords: list[str], max_span: int = 2400) -> str | None:
    normalized = normalize(content)
    folded = normalized.casefold()
    positions: list[tuple[int, int]] = []
    for keyword in keywords:
        token = evidence_token(keyword)
        if not token:
            continue
        position = folded.find(token)
        if position < 0:
            return None
        positions.append((position, position + len(token)))
    if not positions:
        return None
    start = min(position[0] for position in positions)
    end = max(position[1] for position in positions)
    if end - start <= max_span:
        return normalized[max(0, start - 180) : min(len(normalized), end + 180)]
    excerpts = []
    for token_start, token_end in positions:
        excerpt = normalized[max(0, token_start - 140) : min(len(normalized), token_end + 180)]
        if excerpt not in excerpts:
            excerpts.append(excerpt)
    return " […] ".join(excerpts)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_title(doc: dict, url: str) -> str:
    """Name the source document without embedding its body.

    Some scraped records carry page body text in `title`, so anything long or
    multi-line is discarded in favour of the document's filename.
    """
    title = doc.get("title") or ""
    if title and len(title) <= 200 and "\n" not in title:
        return title
    return url.rstrip("/").split("/")[-1] or url


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


HELDOUT = [
    # Claude Code: facts not present in the 30 domain-specific SFT v3 rows.
    {
        "category": "claude_code", "topic": "continue_session",
        "question": "What conversation does `claude --continue` load?",
        "answer": "It loads the most recent conversation associated with the current directory, including sessions that added that directory with `/add-dir`.",
        "keywords": ["--continue", "current directory", "/add-dir"],
        "source_url": "https://code.claude.com/docs/en/cli-reference.md",
        "evidence": "`--continue`, `-c` | Load the most recent conversation in the current directory. Includes sessions that added this directory with `/add-dir`",
    },
    {
        "category": "claude_code", "topic": "accept_edits_mode",
        "question": "Which actions does Claude Code's `acceptEdits` permission mode approve automatically?",
        "answer": "It auto-approves reads, file edits, and common filesystem commands such as `mkdir`, `touch`, `mv`, and `cp`.",
        "keywords": ["acceptEdits", "file edits", "mkdir"],
        "source_url": "https://code.claude.com/docs/en/permission-modes.md",
        "evidence": "`acceptEdits` | Reads, file edits, and common filesystem commands (`mkdir`, `touch`, `mv`, `cp`, etc.)",
    },
    {
        "category": "claude_code", "topic": "mcp_project_scope",
        "question": "Where does Claude Code store project-scoped MCP server configuration, and why should it be committed?",
        "answer": "It stores project-scoped servers in `.mcp.json` at the project root; committing it gives the team the same MCP tools and services.",
        "keywords": [".mcp.json", "project root", "version control"],
        "source_url": "https://code.claude.com/docs/en/mcp.md",
        "evidence": "Project-scoped servers enable team collaboration by storing configurations in a `.mcp.json` file at your project's root directory. Check `.mcp.json` into version control so everyone on your team gets the same MCP tools and services.",
    },
    {
        "category": "claude_code", "topic": "worktree_sessions",
        "question": "How do you start an isolated Claude Code session in a named Git worktree?",
        "answer": "Run `claude --worktree <name>` or `claude -w <name>` to create an isolated worktree and start Claude in it.",
        "keywords": ["--worktree", "-w", "isolated worktree"],
        "source_url": "https://code.claude.com/docs/en/worktrees.md",
        "evidence": "Pass `--worktree` or `-w` with a name to create an isolated worktree and start Claude in it.",
    },
    {
        "category": "claude_code", "topic": "headless_output_formats",
        "question": "Which three values can Claude Code use with `--output-format` in non-interactive mode?",
        "answer": "The supported values are `text`, `json`, and `stream-json`.",
        "keywords": ["text", "json", "stream-json"],
        "source_url": "https://code.claude.com/docs/en/headless.md",
        "evidence": "Use `--output-format` to control how responses are returned: * `text` (default): plain text output * `json`: structured JSON with result, session ID, and metadata * `stream-json`: newline-delimited JSON for real-time streaming",
    },
    {
        "category": "claude_code", "topic": "sdk_structured_output",
        "question": "How do the TypeScript and Python Claude Agent SDKs request JSON-Schema-validated output?",
        "answer": "Pass the schema through `outputFormat` in TypeScript or `output_format` in Python; validated data is returned in `structured_output`.",
        "keywords": ["outputFormat", "output_format", "structured_output"],
        "source_url": "https://code.claude.com/docs/en/agent-sdk/structured-outputs.md",
        "evidence": "pass it to `query()` via the `outputFormat` option (TypeScript) or `output_format` option (Python). When the agent finishes, the result message includes a `structured_output` field with validated data matching your schema.",
    },
    {
        "category": "claude_code", "topic": "sdk_file_checkpointing",
        "question": "Which two Agent SDK options enable file checkpointing and expose checkpoint UUIDs in the response stream?",
        "answer": "Enable `enableFileCheckpointing`/`enable_file_checkpointing` and set `replay-user-messages` in `extraArgs`/`extra_args`.",
        "keywords": ["enableFileCheckpointing", "replay-user-messages", "UUIDs"],
        "source_url": "https://code.claude.com/docs/en/agent-sdk/file-checkpointing.md",
        "evidence": "Enable checkpointing | `enable_file_checkpointing=True` | `enableFileCheckpointing: true` | Tracks file changes for rewinding Receive checkpoint UUIDs | `extra_args={\"replay-user-messages\": None}` | `extraArgs: { 'replay-user-messages': null }` | Required to get user message UUIDs in the stream",
    },
    {
        "category": "claude_code", "topic": "github_action_mentions",
        "question": "What can Claude Code GitHub Actions do after an `@claude` mention in an issue or pull-request comment?",
        "answer": "It can analyze code, implement changes, and push commits from a GitHub Actions workflow.",
        "keywords": ["@claude", "implement changes", "push commits"],
        "source_url": "https://code.claude.com/docs/en/github-actions.md",
        "evidence": "Mention `@claude` in a pull request or issue comment to have Claude analyze code, implement changes, and push commits.",
    },
    {
        "category": "claude_code", "topic": "status_line",
        "question": "What does Claude Code's `/statusline` command create and configure?",
        "answer": "It generates a status-line script under `~/.claude/` and updates the settings automatically.",
        "keywords": ["/statusline", "~/.claude/", "settings"],
        "source_url": "https://code.claude.com/docs/en/statusline.md",
        "evidence": "The `/statusline` command accepts natural language instructions describing what you want displayed. Claude Code generates a script file in `~/.claude/` and updates your settings automatically",
    },
    {
        "category": "claude_code", "topic": "chrome_start",
        "question": "Which CLI flag starts Claude Code with Chrome integration?",
        "answer": "Start it with `claude --chrome`.",
        "keywords": ["claude --chrome", "Chrome"],
        "source_url": "https://code.claude.com/docs/en/chrome.md",
        "evidence": "Start Claude Code with the `--chrome` flag: ```bash theme={null} claude --chrome",
    },
    {
        "category": "claude_code", "topic": "remote_control_resume",
        "question": "Which in-session command makes the current Claude Code conversation remotely controllable?",
        "answer": "Run `/remote-control` or `/rc`; it starts Remote Control while carrying over the current conversation history.",
        "keywords": ["/remote-control", "/rc", "conversation history"],
        "source_url": "https://code.claude.com/docs/en/remote-control.md",
        "evidence": "use the `/remote-control` (or `/rc`) command: This starts a Remote Control session that carries over your current conversation history.",
    },
    {
        "category": "claude_code", "topic": "managed_code_review",
        "question": "What kinds of problems does managed Claude Code Review seek, and does it approve or block pull requests?",
        "answer": "Its agents look for logic errors, security vulnerabilities, broken edge cases, and subtle regressions; findings do not approve or block the PR.",
        "keywords": ["logic errors", "security vulnerabilities", "don't approve or block"],
        "source_url": "https://code.claude.com/docs/en/code-review.md",
        "evidence": "A fleet of specialized agents examine the code changes in the context of your full codebase, looking for logic errors, security vulnerabilities, broken edge cases, and subtle regressions. Findings are tagged by severity and don't approve or block your PR",
    },
    {
        "category": "claude_code", "topic": "custom_keybindings",
        "question": "Which command opens Claude Code's keybinding configuration, and where is that file stored?",
        "answer": "Run `/keybindings`; the configuration file is `~/.claude/keybindings.json` and changes apply without a restart.",
        "keywords": ["/keybindings", "~/.claude/keybindings.json", "without restarting"],
        "source_url": "https://code.claude.com/docs/en/keybindings.md",
        "evidence": "Run `/keybindings` to create or open your configuration file at `~/.claude/keybindings.json`. Changes to the keybindings file are automatically detected and applied without restarting Claude Code.",
    },
    {
        "category": "claude_code", "topic": "official_plugins",
        "question": "How do you browse and install the GitHub plugin from Claude Code's official marketplace?",
        "answer": "Browse with `/plugin` in the Discover tab, then run `/plugin install github@claude-plugins-official`.",
        "keywords": ["/plugin", "Discover", "github@claude-plugins-official"],
        "source_url": "https://code.claude.com/docs/en/discover-plugins.md",
        "evidence": "To browse what's available, run `/plugin` and go to the **Discover** tab. To install a plugin from the official marketplace, use `/plugin install @claude-plugins-official`. For example, to install the GitHub integration: `/plugin install github@claude-plugins-official`",
    },
    {
        "category": "claude_code", "topic": "managed_mcp_catalog",
        "question": "Which managed settings create an approved MCP catalog that blocks unlisted servers?",
        "answer": "Configure `allowedMcpServers` together with `allowManagedMcpServersOnly: true`.",
        "keywords": ["allowedMcpServers", "allowManagedMcpServersOnly", "Approved catalog"],
        "source_url": "https://code.claude.com/docs/en/managed-mcp.md",
        "evidence": "Approved catalog | Publish a list of approved servers; users add the ones they want, anything else is blocked | `allowedMcpServers` + `allowManagedMcpServersOnly: true`",
    },
    {
        "category": "claude_code", "topic": "telegram_channel",
        "question": "Which command starts Claude Code with the official Telegram channel plugin?",
        "answer": "Run `claude --channels plugin:telegram@claude-plugins-official`.",
        "keywords": ["--channels", "plugin:telegram@claude-plugins-official"],
        "source_url": "https://code.claude.com/docs/en/channels.md",
        "evidence": "claude --channels plugin:telegram@claude-plugins-official",
    },
    {
        "category": "claude_code", "topic": "artifact_visibility",
        "question": "Who can see a newly created Claude Code artifact before it is shared?",
        "answer": "A new artifact is visible only to its author until it is shared through the artifact page's Share control.",
        "keywords": ["visible only to you", "Share"],
        "source_url": "https://code.claude.com/docs/en/artifacts.md",
        "evidence": "A new artifact is visible only to you. To share it, open the artifact in your browser and use the **Share** control in the page header.",
    },
    {
        "category": "claude_code", "topic": "voice_requirements",
        "question": "What authentication and environment limitations apply to Claude Code voice dictation?",
        "answer": "`/voice` requires a Claude.ai account and local microphone; it is unavailable with API-key authentication or in remote environments such as SSH.",
        "keywords": ["/voice", "Claude.ai account", "local microphone", "API key"],
        "source_url": "https://code.claude.com/docs/en/voice-dictation.md",
        "evidence": "A Claude.ai account: the speech-to-text service is only available when you authenticate with one, and is not available when Claude Code is configured to use an Anthropic API key directly. A local microphone: voice dictation does not work in remote environments such as Claude Code on the web or SSH sessions.",
    },
    {
        "category": "claude_code", "topic": "sdk_cost_estimates",
        "question": "Are Claude Agent SDK `total_cost_usd` and `costUSD` values authoritative billing figures?",
        "answer": "No. They are client-side estimates; authoritative billing should come from the Usage and Cost API or the Claude Console Usage page.",
        "keywords": ["client-side estimates", "Usage and Cost API", "Claude Console"],
        "source_url": "https://code.claude.com/docs/en/agent-sdk/cost-tracking.md",
        "evidence": "The `total_cost_usd` and `costUSD` fields are client-side estimates, not authoritative billing data. For authoritative billing, use the Usage and Cost API or the Usage page in the Claude Console.",
    },
    {
        "category": "claude_code", "topic": "mcp_precedence",
        "question": "What is Claude Code's precedence order when the same MCP server name exists in local, project, and user scopes?",
        "answer": "Local scope wins over project scope, which wins over user scope; Claude uses the whole entry from the highest-precedence source without merging fields.",
        "keywords": ["Local scope", "Project scope", "User scope", "not merged"],
        "source_url": "https://code.claude.com/docs/en/mcp.md",
        "evidence": "The entire server entry from that source is used; fields are not merged across scopes. 1. Local scope 2. Project scope 3. User scope",
    },

    # OpenAI Codex / ChatGPT: facts not present in the 15 Codex SFT v3 rows.
    {
        "category": "openai_codex", "topic": "agents_md_discovery",
        "question": "How does Codex choose global and project `AGENTS.md` instruction files?",
        "answer": "Globally it prefers `AGENTS.override.md` over `AGENTS.md`; in a project it walks from the project root to the working directory and selects at most one instruction file per directory.",
        "keywords": ["AGENTS.override.md", "AGENTS.md", "project root"],
        "source_url": "https://learn.chatgpt.com/docs/agent-configuration/agents-md.md",
        "evidence": "Codex reads `AGENTS.override.md` if it exists. Otherwise, Codex reads `AGENTS.md`. Starting at the project root (typically the Git root), Codex walks down to your current working directory. Codex includes at most one file per directory",
    },
    {
        "category": "openai_codex", "topic": "command_rules",
        "question": "Where is a typical user-level Codex command rule stored, and which function defines a command-prefix policy?",
        "answer": "Store it under `~/.codex/rules/`, commonly `~/.codex/rules/default.rules`, and define the policy with `prefix_rule()`.",
        "keywords": ["~/.codex/rules/default.rules", "prefix_rule"],
        "source_url": "https://learn.chatgpt.com/docs/agent-configuration/rules.md",
        "evidence": "Create a `.rules` file under a `rules/` folder next to an active config layer (for example, `~/.codex/rules/default.rules`). Add a rule. `prefix_rule(`",
    },
    {
        "category": "openai_codex", "topic": "subagent_limit",
        "question": "Which Codex setting controls concurrent subagent threads per session, and what is its legacy alias?",
        "answer": "Use `agents.max_concurrent_threads_per_session`; existing configurations may use the legacy alias `agents.max_threads`.",
        "keywords": ["agents.max_concurrent_threads_per_session", "agents.max_threads"],
        "source_url": "https://learn.chatgpt.com/docs/agent-configuration/subagents.md",
        "evidence": "When you leave `agents.max_concurrent_threads_per_session` unset, Codex chooses the default. Existing configurations can keep using `agents.max_threads` as a legacy alias.",
    },
    {
        "category": "openai_codex", "topic": "app_server_transport",
        "question": "What protocol and default transport does `codex app-server` use?",
        "answer": "It uses bidirectional JSON-RPC 2.0 semantics; the default `stdio://` transport sends newline-delimited JSON.",
        "keywords": ["JSON-RPC 2.0", "stdio://", "JSONL"],
        "source_url": "https://learn.chatgpt.com/docs/app-server.md",
        "evidence": "`codex app-server` supports bidirectional communication using JSON-RPC 2.0 messages. `stdio` (`--listen stdio://`, default): newline-delimited JSON (JSONL).",
    },
    {
        "category": "openai_codex", "topic": "mcp_codex_sandbox",
        "question": "Which sandbox modes can a session started through the Codex MCP server request?",
        "answer": "The `sandbox` field accepts `read-only`, `workspace-write`, or `danger-full-access`.",
        "keywords": ["read-only", "workspace-write", "danger-full-access"],
        "source_url": "https://learn.chatgpt.com/docs/mcp-server.md",
        "evidence": "`sandbox` | `string` | Sandbox mode: `read-only`, `workspace-write`, or `danger-full-access`.",
    },
    {
        "category": "openai_codex", "topic": "skill_structure",
        "question": "What must a Codex skill directory contain, and which front-matter fields are required?",
        "answer": "It must contain `SKILL.md`; that file must declare `name` and `description`, while scripts and references are optional.",
        "keywords": ["SKILL.md", "name", "description"],
        "source_url": "https://learn.chatgpt.com/docs/build-skills.md",
        "evidence": "A skill is a directory with a `SKILL.md` file plus optional scripts and references. The `SKILL.md` file must include `name` and `description`.",
    },
    {
        "category": "openai_codex", "topic": "plugin_manifest",
        "question": "Where is a Codex plugin's required manifest located?",
        "answer": "The required manifest is `.codex-plugin/plugin.json` inside the plugin directory.",
        "keywords": [".codex-plugin/plugin.json", "manifest"],
        "source_url": "https://learn.chatgpt.com/docs/build-plugins.md",
        "evidence": "The skill creates the required `.codex-plugin/plugin.json` manifest",
    },
    {
        "category": "openai_codex", "topic": "sandbox_modes",
        "question": "How do the `read-only`, `workspace-write`, and `danger-full-access` Codex sandbox modes differ?",
        "answer": "`read-only` only inspects; `workspace-write` permits edits and routine commands within the workspace; `danger-full-access` removes filesystem and network sandbox boundaries.",
        "keywords": ["read-only", "workspace-write", "danger-full-access"],
        "source_url": "https://learn.chatgpt.com/docs/sandboxing.md",
        "evidence": "`read-only`: The agent can inspect files, but it can't edit files or run commands without approval. `workspace-write`: The agent can read files, edit within the workspace, and run routine local commands inside that boundary. `danger-full-access`: The agent runs without sandbox restrictions. This removes the filesystem and network boundaries",
    },
    {
        "category": "openai_codex", "topic": "exec_streams",
        "question": "In non-interactive `codex exec`, where are progress updates and the final agent message written?",
        "answer": "Progress streams to `stderr`, while only the final agent message is printed to `stdout`.",
        "keywords": ["stderr", "stdout", "final agent message"],
        "source_url": "https://learn.chatgpt.com/docs/non-interactive-mode.md",
        "evidence": "While `codex exec` runs, Codex streams progress to `stderr` and prints only the final agent message to `stdout`.",
    },
    {
        "category": "openai_codex", "topic": "exec_jsonl",
        "question": "What does `codex exec --json` emit on standard output?",
        "answer": "It emits a JSON Lines event stream containing lifecycle and item events such as `thread.started`, `turn.completed`, `item.*`, and `error`.",
        "keywords": ["--json", "JSON Lines", "thread.started"],
        "source_url": "https://learn.chatgpt.com/docs/non-interactive-mode.md",
        "evidence": "When you enable `--json`, `stdout` becomes a JSON Lines (JSONL) stream so you can capture every event Codex emits while it's running. Event types include `thread.started`, `turn.started`, `turn.completed`, `turn.failed`, `item.*`, and `error`.",
    },
    {
        "category": "openai_codex", "topic": "config_locations",
        "question": "Where do Codex user defaults and project-specific configuration live?",
        "answer": "User defaults live in `~/.codex/config.toml`; project overrides use `.codex/config.toml` and load only for trusted projects.",
        "keywords": ["~/.codex/config.toml", ".codex/config.toml", "trusted"],
        "source_url": "https://learn.chatgpt.com/docs/config-file/config-basic.md",
        "evidence": "Your personal defaults live in `~/.codex/config.toml`, and you can add project overrides with `.codex/config.toml` files. For security, Codex loads project `.codex/` layers only when you trust the project.",
    },
    {
        "category": "openai_codex", "topic": "config_profiles",
        "question": "How does a Codex CLI profile overlay the base user configuration?",
        "answer": "`--profile <name>` loads `~/.codex/config.toml` and then overlays `~/.codex/<name>.config.toml`.",
        "keywords": ["--profile", "~/.codex/config.toml", ".config.toml"],
        "source_url": "https://learn.chatgpt.com/docs/config-file/config-advanced.md",
        "evidence": "When you pass `--profile profile-name`, Codex loads `~/.codex/config.toml`, then overlays `~/.codex/profile-name.config.toml`.",
    },
    {
        "category": "openai_codex", "topic": "local_review",
        "question": "What does Codex `/review` do to the selected diff and working tree?",
        "answer": "It launches a dedicated reviewer that reads the selected diff and reports prioritized, actionable findings without changing the working tree.",
        "keywords": ["/review", "dedicated reviewer", "without changing"],
        "source_url": "https://learn.chatgpt.com/docs/code-review.md",
        "evidence": "Codex starts a dedicated reviewer that reads the selected diff and reports prioritized, actionable findings without changing your working tree.",
    },
    {
        "category": "openai_codex", "topic": "remote_capabilities",
        "question": "Which four activities does Codex Remote support from a phone?",
        "answer": "It lets you start tasks, steer running work, approve requested actions, and review responses, diffs, changed files, and test results.",
        "keywords": ["Start tasks", "Guide work", "Approve requested actions", "Review the result"],
        "source_url": "https://learn.chatgpt.com/docs/remote.md",
        "evidence": "Start tasks from your phone. Guide work as it happens. Approve requested actions. Review the result: Inspect responses, changed files, diffs, and test results",
    },
    {
        "category": "openai_codex", "topic": "scheduled_tasks",
        "question": "Where are Codex scheduled tasks created and managed, and can Codex CLI provide that management interface?",
        "answer": "Create them from Chat or ChatGPT Work and manage runs in Scheduled; Codex CLI does not provide the Scheduled management interface.",
        "keywords": ["ChatGPT Work", "Scheduled", "Codex CLI"],
        "source_url": "https://learn.chatgpt.com/docs/automations.md",
        "evidence": "create them from Chat or ChatGPT Work on the web and manage their runs from **Scheduled**. Codex CLI doesn't provide the Scheduled management interface.",
    },
    {
        "category": "openai_codex", "topic": "browser_availability",
        "question": "Is the built-in browser available in Codex CLI or the Codex IDE extension?",
        "answer": "No. The built-in browser is used through ChatGPT on the web or in the ChatGPT desktop app, not Codex CLI or the Codex IDE extension.",
        "keywords": ["isn't available", "Codex CLI", "desktop app"],
        "source_url": "https://learn.chatgpt.com/docs/browser.md",
        "evidence": "Browser isn't available in Codex CLI or the Codex IDE extension. Open the ChatGPT desktop app to use the built-in browser. Browser is available in ChatGPT on the web and in the ChatGPT desktop app.",
    },
    {
        "category": "openai_codex", "topic": "appshots_capture",
        "question": "What does a ChatGPT appshot capture, and on which desktop platform is it available?",
        "answer": "On macOS, an appshot captures the frontmost window, including its visible image and available text.",
        "keywords": ["macOS", "frontmost window", "available text"],
        "source_url": "https://learn.chatgpt.com/docs/appshots.md",
        "evidence": "Appshots are available in the ChatGPT desktop app on macOS. An appshot captures the frontmost window only. It can include: An image of the visible window. Available text from that window",
    },
    {
        "category": "openai_codex", "topic": "goal_mode",
        "question": "What does entering `/goal` establish for a long-running Codex task?",
        "answer": "It starts Goal mode and uses the goal text as both the first prompt and the task's completion criteria.",
        "keywords": ["/goal", "first prompt", "completion criteria"],
        "source_url": "https://learn.chatgpt.com/docs/long-running-work.md",
        "evidence": "Type `/goal` in the ChatGPT desktop app, Codex CLI, or the IDE extension. The goal text becomes both the first prompt and the completion criteria for the task.",
    },
    {
        "category": "openai_codex", "topic": "config_precedence",
        "question": "Which Codex configuration layers take precedence over the user `~/.codex/config.toml` file?",
        "answer": "CLI overrides, trusted project `.codex/config.toml` files, and the selected profile layer all take precedence over user configuration.",
        "keywords": ["--config", ".codex/config.toml", "--profile", "~/.codex/config.toml"],
        "source_url": "https://learn.chatgpt.com/docs/config-file/config-basic.md",
        "evidence": "Configuration precedence Codex resolves values in this order: 1. CLI flags and `-c` / `--config` overrides 2. Project config files: `.codex/config.toml` 3. Profile files selected with `--profile profile-name` 4. User config: `~/.codex/config.toml`",
    },
    {
        "category": "openai_codex", "topic": "app_server_health",
        "question": "Which HTTP health endpoints does Codex app-server expose with a WebSocket listener?",
        "answer": "It exposes `GET /readyz` for readiness and `GET /healthz` for health checks without an Origin header.",
        "keywords": ["/readyz", "/healthz", "Origin"],
        "source_url": "https://learn.chatgpt.com/docs/app-server.md",
        "evidence": "`GET /readyz` returns `200 OK` once the listener accepts new connections. `GET /healthz` returns `200 OK` when the request doesn't include an `Origin` header. Requests with an `Origin` header are rejected with `403 Forbidden`.",
    },
]


def main() -> int:
    docs = read_jsonl(DOCS_PATH)
    docs_by_url: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        docs_by_url.setdefault(doc.get("url", ""), []).append(doc)

    sft_rows = read_jsonl(SFT_PATH)
    source_rows = read_jsonl(SFT_SOURCES_PATH)
    seen: list[dict[str, Any]] = []
    counters = Counter()
    for source in source_rows:
        row_index = source["row_index"]
        sft = sft_rows[row_index]
        category = "claude_code" if source["category"] == "claude_code" else "openai_codex"
        counters[category] += 1
        seen.append({
            "id": f"seen_{'claude' if category == 'claude_code' else 'codex'}_{counters[category]:03d}",
            "split": "sft_seen",
            "sft_relation": "exact_sft_v3_training_item",
            "category": category,
            "topic": source["topic"],
            "question": sft["instruction"],
            "answer": sft["output"],
            "keywords": list(dict.fromkeys(source.get("required_exact_tokens", []) + source.get("answer_exact_tokens", []))),
            "source_url": source["source_url"],
            "evidence": source["evidence_text"],
            "expected_source_document_sha256": source["source_sha256"],
            "sft_v3_row_index": row_index,
        })

    heldout: list[dict[str, Any]] = []
    counters = Counter()
    for definition in HELDOUT:
        category = definition["category"]
        counters[category] += 1
        heldout.append({
            "id": f"heldout_{'claude' if category == 'claude_code' else 'codex'}_{counters[category]:03d}",
            "split": "heldout",
            "sft_relation": "fact_and_question_excluded_from_sft_v3_domain_rows",
            **definition,
        })

    rows = seen + heldout
    failures: list[str] = []
    for row in rows:
        candidates = docs_by_url.get(row["source_url"], [])
        if not candidates:
            failures.append(f"{row['id']}: source URL is absent from scraped corpus: {row['source_url']}")
            continue
        evidence_summary = row.pop("evidence")
        expected_sha = row.pop("expected_source_document_sha256", None)
        if expected_sha:
            candidates = [doc for doc in candidates if sha256_text(doc.get("content", "")) == expected_sha]
        matching = []
        for candidate in candidates:
            excerpt = exact_evidence_excerpt(candidate.get("content", ""), row["keywords"])
            if excerpt is not None:
                matching.append((candidate, excerpt))
        if not matching:
            failures.append(f"{row['id']}: one or more required evidence tokens are missing from the claimed official document")
            continue
        doc, evidence_excerpt = matching[0]
        row["evidence_summary"] = evidence_summary
        row["evidence"] = evidence_excerpt
        row["evidence_validation"] = "all_required_keywords_in_one_official_document"
        row["source_title"] = source_title(doc, row["source_url"])
        row["source_document_sha256"] = sha256_text(doc.get("content", ""))
        row["question_sha256"] = sha256_text(normalize(row["question"]))
        row["answer_sha256"] = sha256_text(normalize(row["answer"]))

    normalized_questions = [normalize(row["question"]).casefold() for row in rows]
    duplicate_questions = [question for question, count in Counter(normalized_questions).items() if count > 1]
    if duplicate_questions:
        failures.append(f"duplicate normalized questions: {duplicate_questions}")
    normalized_answers = [normalize(row["answer"]).casefold() for row in rows]
    duplicate_answers = [answer for answer, count in Counter(normalized_answers).items() if count > 1]
    if duplicate_answers:
        failures.append(f"duplicate normalized answers: {duplicate_answers}")
    if len(seen) != 30 or len(heldout) != 40:
        failures.append(f"unexpected row counts: seen={len(seen)} heldout={len(heldout)}")
    if failures:
        raise SystemExit("QA generation failed before overwrite:\n- " + "\n- ".join(failures))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    claude_rows = [row for row in rows if row["category"] == "claude_code"]
    codex_rows = [row for row in rows if row["category"] == "openai_codex"]
    write_jsonl_atomic(CLAUDE_PATH, claude_rows)
    write_jsonl_atomic(CODEX_PATH, codex_rows)
    write_jsonl_atomic(COMBINED_PATH, rows)
    metadata = {
        "status": "generated_and_source_grounded_requires_manual_semantic_review",
        "canonical_dataset": str(COMBINED_PATH.relative_to(REPO_ROOT)),
        "items": len(rows),
        "breakdown": {
            "sft_seen": len(seen),
            "heldout": len(heldout),
            "claude_code": sum(row["category"] == "claude_code" for row in rows),
            "openai_codex": sum(row["category"] == "openai_codex" for row in rows),
        },
        "design": {
            "sft_seen": "Exact questions and answers from the 30 domain-specific SFT v3 rows; measures direct SFT memorization.",
            "heldout": "Distinct facts and questions grounded in other official collected pages; measures CPT knowledge retrieval after SFT.",
        },
        "hashes": {
            "combined_sha256": sha256_file(COMBINED_PATH),
            "claude_sha256": sha256_file(CLAUDE_PATH),
            "codex_sha256": sha256_file(CODEX_PATH),
            "scraped_documents_sha256": sha256_file(DOCS_PATH),
            "sft_v3_train_sha256": sha256_file(SFT_PATH),
        },
    }
    audit = {
        "status": "pass",
        "checks": {
            "all_sources_in_collected_official_corpus": True,
            "all_evidence_excerpts_derived_from_required_keywords_in_one_source": True,
            "unique_normalized_questions": len(set(normalized_questions)) == len(rows),
            "unique_normalized_answers": len(set(normalized_answers)) == len(rows),
            "fixed_split_counts": {"sft_seen": 30, "heldout": 40},
        },
        "manual_semantic_review_required": True,
    }
    print(json.dumps({"metadata": metadata, "audit": audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
