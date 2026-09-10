#!/usr/bin/env python3
"""
SFT v3 Dataset Generator
Generates:
- sft_v3_train.jsonl (Alpaca format)
- sft_v3_train_chat.jsonl (Chat format)
- sft_v3_item_metadata.jsonl
- sft_v3_metadata.json
- sft_v3_domain_sources.jsonl
- sft_v3_leakage_report.json
- eval_qa_v3_heldout.jsonl
- eval_qa_v3_heldout_metadata.json
"""

import json
import hashlib
import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "ai_coding_agent_cpt_data"
SFT_DIR = DATA_DIR / "SFT_Dataset"
QA_DIR = DATA_DIR / "QA_Dataset"
SCRAPED_DOCS_PATH = DATA_DIR / "data" / "scraped_documents.jsonl"
EVAL_QA_PATH = QA_DIR / "eval_qa_combined.jsonl"

QUESTION_JACCARD_LIMIT = 0.35
QUESTION_SEQUENCE_LIMIT = 0.80
ANSWER_JACCARD_LIMIT = 0.35
ANSWER_SEQUENCE_LIMIT = 0.80

KNOWN_BAD_EVAL_TOPICS = {"Agent SDK API"}

RELATED_EVAL_IDS_BY_TOPIC = {
    "cli_subagents": ["claude_009", "claude_013", "claude_019"],
    "hooks_automation": ["claude_008"],
    "settings_memory": ["claude_007"],
    "mcp_integration": ["claude_070", "claude_073"],
    "auto_memory": ["claude_007"],
    "desktop_handoff": ["claude_003", "claude_013"],
    "custom_prompts": ["codex_011"],
    "prompt_placeholders": ["codex_011"],
    "prompt_arguments": ["codex_011"],
    "project_root_markers": ["codex_002"],
    "internet_http_methods": ["codex_006"],
    "internet_allowlist_presets": ["codex_006"],
    "mcp_transport": ["codex_003"],
    "mcp_tool_execution": ["codex_003"],
}


def normalize_text(text):
    """Normalize generated dataset text without changing technical spelling."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def canonical_url(url):
    value = url.strip().rstrip("/")
    if value.endswith(".md"):
        value = value[:-3]
    return value


def source_documents_for_url(scraped_docs, source_url):
    exact = [doc for doc in scraped_docs if doc.get("url") == source_url]
    if exact:
        return exact
    canonical = canonical_url(source_url)
    return [
        doc
        for doc in scraped_docs
        if canonical_url(str(doc.get("url", ""))) == canonical
    ]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_jaccard_sim(str1, str2):
    tokens1 = set(re.findall(r'\w+', str1.lower()))
    tokens2 = set(re.findall(r'\w+', str2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def get_seq_sim(str1, str2):
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def main():
    print("Loading existing resources...")
    scraped_docs = load_jsonl(SCRAPED_DOCS_PATH)
    eval_qa_combined = load_jsonl(EVAL_QA_PATH)
    

    # ---------------------------------------------------------
    # 1. Coding Agent Knowledge (30 items: 15 Claude, 15 Codex)
    # ---------------------------------------------------------
    
    claude_sources = [
        {
            "category": "claude_code",
            "topic": "cli_subagents",
            "instruction": "What role does a lead agent play when several Claude Code agents work on tasks simultaneously?",
            "output": "A lead agent coordinates multiple Claude Code agents, assigns their subtasks, and merges the results of their parallel work.",
            "source_url": "https://code.claude.com/docs/en/overview.md",
            "source_document_title": "overview.md",
            "evidence_text": "Spawn multiple Claude Code agents that work on different parts of a task simultaneously. A lead agent coordinates the work, assigns subtasks, and merges results.",
            "required_exact_tokens": ["background agents", "subtasks", "simultaneously"]
        },
        {
            "category": "claude_code",
            "topic": "hooks_automation",
            "instruction": "Which two automation examples does the Claude Code overview give for hooks?",
            "output": "Hooks can auto-format after a file edit and run lint before a commit. They execute shell commands before or after Claude Code actions.",
            "source_url": "https://code.claude.com/docs/en/overview.md",
            "source_document_title": "overview.md",
            "evidence_text": "Hooks let you run shell commands before or after Claude Code actions, like auto-formatting after every file edit or running lint before a commit.",
            "required_exact_tokens": ["Hooks", "shell commands", "auto-formatting"]
        },
        {
            "category": "claude_code",
            "topic": "settings_memory",
            "instruction": "What kinds of project preferences can a CLAUDE.md file record?",
            "output": "CLAUDE.md can record coding standards, architecture decisions, preferred libraries, and review checklists for Claude Code to follow.",
            "source_url": "https://code.claude.com/docs/en/overview.md",
            "source_document_title": "overview.md",
            "evidence_text": "CLAUDE.md is a markdown file you add to your project root that Claude Code reads at the start of every session. Use it to set coding standards, architecture decisions, preferred libraries, and review checklists.",
            "required_exact_tokens": ["CLAUDE.md", "project root", "coding standards"]
        },
        {
            "category": "claude_code",
            "topic": "native_updates",
            "instruction": "Which Claude Code installation method receives automatic background updates?",
            "output": "Native Claude Code installations update automatically in the background; Homebrew installations require a manual upgrade.",
            "source_url": "https://code.claude.com/docs/en/overview.md",
            "source_document_title": "overview.md",
            "evidence_text": "Native installations automatically update in the background. Homebrew installations do not auto-update.",
            "required_exact_tokens": ["Native installations automatically update", "Homebrew installations do not auto-update"]
        },
        {
            "category": "claude_code",
            "topic": "mcp_integration",
            "instruction": "What protocol allows Claude Code to connect with external tools like Jira, Slack, or Google Drive?",
            "output": "Claude Code uses the Model Context Protocol (MCP). MCP provides an open standard for connecting AI clients with external data sources and custom tool APIs.",
            "source_url": "https://code.claude.com/docs/en/overview.md",
            "source_document_title": "overview.md",
            "evidence_text": "The Model Context Protocol (MCP) is an open standard for connecting AI tools to external data sources. With MCP, Claude Code can read your design docs in Google Drive, update tickets in Jira, pull data from Slack, or use your own custom tooling.",
            "required_exact_tokens": ["Model Context Protocol", "MCP", "Google Drive"]
        },
        {
            "category": "claude_code",
            "topic": "interactive_help",
            "instruction": "During an active terminal session, what should you type to display Claude Code's built-in command list?",
            "output": "Type `/help` in an interactive Claude Code session to display the available commands.",
            "source_url": "https://code.claude.com/docs/en/quickstart.md",
            "source_document_title": "quickstart.md",
            "evidence_text": "Type /help for available commands.",
            "required_exact_tokens": ["/help", "available commands"]
        },
        {
            "category": "claude_code",
            "topic": "interactive_start",
            "instruction": "Name the executable entered from a project directory to open the interactive Claude Code interface.",
            "output": "Change into the project directory and run `claude`; the first launch prompts you to sign in.",
            "source_url": "https://code.claude.com/docs/en/overview.md",
            "source_document_title": "overview.md",
            "evidence_text": "Then start Claude Code in any project. You'll be prompted to log in on first use.",
            "required_exact_tokens": ["start Claude Code in any project", "prompted to log in"]
        },
        {
            "category": "claude_code",
            "topic": "login_auth",
            "instruction": "How can you bypass the browser login prompt when starting Claude Code?",
            "output": "Set the `ANTHROPIC_API_KEY` environment variable in your terminal before launching Claude. Claude Code detects the key and asks you to approve it directly without triggering browser authentication.",
            "source_url": "https://code.claude.com/docs/en/overview.md",
            "source_document_title": "overview.md",
            "evidence_text": "If you've set the ANTHROPIC_API_KEY environment variable, Claude Code skips the login prompt and asks you to approve the key instead.",
            "required_exact_tokens": ["ANTHROPIC_API_KEY", "login prompt", "approve"]
        },
        {
            "category": "claude_code",
            "topic": "git_workflows",
            "instruction": "Which routine Git operations can Claude Code perform directly?",
            "output": "Claude Code can stage changes, write commit messages, create branches, and open pull requests.",
            "source_url": "https://code.claude.com/docs/en/overview.md",
            "source_document_title": "overview.md",
            "evidence_text": "Claude Code works directly with git. It stages changes, writes commit messages, creates branches, and opens pull requests.",
            "required_exact_tokens": ["stages changes", "writes commit messages", "creates branches", "opens pull requests"]
        },
        {
            "category": "claude_code",
            "topic": "session_clear",
            "instruction": "Which Claude Code slash command clears the active conversation history?",
            "output": "Type `/clear` in the active session to clear its conversation history.",
            "source_url": "https://code.claude.com/docs/en/quickstart.md",
            "source_document_title": "quickstart.md",
            "evidence_text": "/clear Clear conversation history",
            "required_exact_tokens": ["/clear", "Clear conversation history"]
        },
        {
            "category": "claude_code",
            "topic": "auto_memory",
            "instruction": "How does Claude Code automatically retain build and debugging insights across sessions without manual user documentation?",
            "output": "Claude Code builds auto memory as it works, saving learnings such as build commands and debugging insights across sessions.",
            "source_url": "https://code.claude.com/docs/en/overview.md",
            "source_document_title": "overview.md",
            "evidence_text": "Claude also builds auto memory as it works, saving learnings like build commands and debugging insights across sessions without you writing anything.",
            "required_exact_tokens": ["auto memory", "build commands", "debugging insights"]
        },
        {
            "category": "claude_code",
            "topic": "desktop_handoff",
            "instruction": "What command transitions a running CLI session to the Claude Code Desktop application on macOS or Windows?",
            "output": "Run the `/desktop` command inside your CLI session. This opens your current workspace session in the Desktop GUI for visual diff inspection.",
            "source_url": "https://code.claude.com/docs/en/overview.md",
            "source_document_title": "overview.md",
            "evidence_text": "Run /desktop to continue your current terminal session in the Desktop app, where you can review diffs visually.",
            "required_exact_tokens": ["/desktop", "Desktop app", "review diffs visually"]
        },
        {
            "category": "claude_code",
            "topic": "windows_shell",
            "instruction": "Why is Git for Windows recommended when running Claude Code on native Windows?",
            "output": "Git for Windows provides the Bash tool environment. If Git for Windows is missing, Claude Code falls back to PowerShell as its primary shell tool.",
            "source_url": "https://code.claude.com/docs/en/overview.md",
            "source_document_title": "overview.md",
            "evidence_text": "Git for Windows is recommended on native Windows so Claude Code can use the Bash tool. If Git for Windows is not installed, Claude Code uses PowerShell as the shell tool instead.",
            "required_exact_tokens": ["Git for Windows", "Bash tool", "PowerShell"]
        },
        {
            "category": "claude_code",
            "topic": "package_managers",
            "instruction": "How can you update a Homebrew cask installation of Claude Code to the latest stable release?",
            "output": "Execute `brew upgrade claude-code` in your terminal. Note that Homebrew installations do not auto-update automatically in the background.",
            "source_url": "https://code.claude.com/docs/en/overview.md",
            "source_document_title": "overview.md",
            "evidence_text": "Homebrew installations do not auto-update. Run brew upgrade claude-code or brew upgrade claude-code@latest, depending on which cask you installed, to get the latest features and security fixes.",
            "required_exact_tokens": ["brew upgrade claude-code", "Homebrew installations do not auto-update"]
        },
        {
            "category": "claude_code",
            "topic": "version_check",
            "instruction": "Which terminal invocation prints the installed Claude Code release?",
            "output": "Run `claude --version`; it prints a version number followed by `(Claude Code)`.",
            "source_url": "https://code.claude.com/docs/en/quickstart.md",
            "source_document_title": "quickstart.md",
            "evidence_text": "To confirm the installation worked, run: claude --version The command prints a version number followed by (Claude Code).",
            "required_exact_tokens": ["claude --version", "(Claude Code)"]
        }
    ]

    codex_sources = [
        {
            "category": "codex",
            "topic": "custom_prompts",
            "instruction": "Where does Codex scan for custom user-defined prompt markdown files on your local system?",
            "output": "Deprecated custom prompts use top-level Markdown files under `~/.codex/prompts/`; use skills for new reusable instructions.",
            "source_url": "https://learn.chatgpt.com/docs/custom-prompts.md",
            "source_document_title": "custom-prompts.md",
            "evidence_text": "Custom prompts are deprecated. Use skills for reusable instructions. Codex scans only the top-level Markdown files in ~/.codex/prompts/.",
            "required_exact_tokens": ["Custom prompts are deprecated", "~/.codex/prompts/", "top-level Markdown files"]
        },
        {
            "category": "codex",
            "topic": "prompt_placeholders",
            "instruction": "In Codex custom prompts, how do you emit a literal dollar sign without triggering placeholder expansion?",
            "output": "Write `$$` in your prompt file. Codex parses `$$` as a single literal `$` symbol upon expanding the prompt.",
            "source_url": "https://learn.chatgpt.com/docs/custom-prompts.md",
            "source_document_title": "custom-prompts.md",
            "evidence_text": "Literal dollar signs: Write $$ to emit a single $ in the expanded prompt.",
            "required_exact_tokens": ["Literal dollar signs", "$$", "expanded prompt"]
        },
        {
            "category": "codex",
            "topic": "prompt_arguments",
            "instruction": "How do you pass named parameter arguments to a custom slash prompt in Codex?",
            "output": "Provide named arguments as `KEY=value` after the slash command, and quote any value that contains spaces.",
            "source_url": "https://learn.chatgpt.com/docs/custom-prompts.md",
            "source_document_title": "custom-prompts.md",
            "evidence_text": "Named placeholders: Use uppercase names like $FILE or $TICKET_ID and supply values as KEY=value. Quote values with spaces (for example, FOCUS=\"loading state\").",
            "required_exact_tokens": ["KEY=value", "Quote values with spaces", "Named placeholders"]
        },
        {
            "category": "codex",
            "topic": "hook_locations",
            "instruction": "Which two configuration forms can Codex use to discover hooks next to active config layers?",
            "output": "Codex discovers hooks in `hooks.json` files or inline `[hooks]` tables inside `config.toml`.",
            "source_url": "https://learn.chatgpt.com/docs/hooks.md",
            "source_document_title": "hooks.md",
            "evidence_text": "Codex discovers hooks next to active config layers in either of these forms: hooks.json or inline [hooks] tables inside config.toml.",
            "required_exact_tokens": ["hooks.json", "[hooks]", "config.toml"]
        },
        {
            "category": "codex",
            "topic": "background_hooks",
            "instruction": "How can a Codex command hook run without making Codex wait for it to finish?",
            "output": "Set `async` to `true` on the command hook so it runs in the background while Codex continues.",
            "source_url": "https://learn.chatgpt.com/docs/hooks.md",
            "source_document_title": "hooks.md",
            "evidence_text": "Set async to true to run a command hook in the background while Codex continues.",
            "required_exact_tokens": ["async", "true", "background while Codex continues"]
        },
        {
            "category": "codex",
            "topic": "disable_hooks",
            "instruction": "How do you disable Codex hooks globally in `config.toml`?",
            "output": "Set `hooks = false` under the `[features]` table in `config.toml`.",
            "source_url": "https://learn.chatgpt.com/docs/hooks.md",
            "source_document_title": "hooks.md",
            "evidence_text": "Hooks are enabled by default. To turn them off in config.toml, set [features] hooks = false.",
            "required_exact_tokens": ["[features]", "hooks = false"]
        },
        {
            "category": "codex",
            "topic": "config_override",
            "instruction": "How can you override an arbitrary Codex configuration key for one CLI run?",
            "output": "Use `-c` or `--config` with a key and TOML value; dot notation can address nested configuration keys.",
            "source_url": "https://learn.chatgpt.com/docs/config-file/config-advanced.md",
            "source_document_title": "config-advanced.md",
            "evidence_text": "Use -c / --config when you need to override an arbitrary key. Keys can use dot notation to set nested values, and --config values are parsed as TOML.",
            "required_exact_tokens": ["-c", "--config", "dot notation", "parsed as TOML"]
        },
        {
            "category": "codex",
            "topic": "project_root_markers",
            "instruction": "Which Codex setting customizes the files used to detect a project root?",
            "output": "Set `project_root_markers` in `config.toml`; by default, Codex treats a directory containing `.git` as the project root.",
            "source_url": "https://learn.chatgpt.com/docs/config-file/config-advanced.md",
            "source_document_title": "config-advanced.md",
            "evidence_text": "By default, Codex treats a directory containing .git as the project root. To customize this behavior, set project_root_markers in config.toml.",
            "required_exact_tokens": ["project_root_markers", ".git", "project root"]
        },
        {
            "category": "codex",
            "topic": "history_persistence",
            "instruction": "How can local Codex session transcript persistence be disabled?",
            "output": "Set `persistence = \"none\"` under `[history]` in `config.toml` to disable local transcript persistence.",
            "source_url": "https://learn.chatgpt.com/docs/config-file/config-advanced.md",
            "source_document_title": "config-advanced.md",
            "evidence_text": "By default, Codex saves local session transcripts under CODEX_HOME. To disable local history persistence, set [history] persistence = none.",
            "required_exact_tokens": ["[history]", "persistence", "none"]
        },
        {
            "category": "codex",
            "topic": "external_notifications",
            "instruction": "What does the Codex `notify` setting run when a supported event occurs?",
            "output": "The `notify` setting runs an external program; currently its supported event is `agent-turn-complete`.",
            "source_url": "https://learn.chatgpt.com/docs/config-file/config-advanced.md",
            "source_document_title": "config-advanced.md",
            "evidence_text": "Use notify to trigger an external program whenever Codex emits supported events, currently only agent-turn-complete.",
            "required_exact_tokens": ["notify", "external program", "agent-turn-complete"]
        },
        {
            "category": "codex",
            "topic": "internet_http_methods",
            "instruction": "Which HTTP methods can a restricted Codex cloud environment allow for safer network access?",
            "output": "Restrict requests to `GET`, `HEAD`, and `OPTIONS`; methods such as `POST`, `PUT`, `PATCH`, and `DELETE` are then blocked.",
            "source_url": "https://learn.chatgpt.com/docs/cloud/internet-access.md",
            "source_document_title": "internet-access.md",
            "evidence_text": "For extra protection, restrict network requests to GET, HEAD, and OPTIONS. Requests using POST, PUT, PATCH, DELETE, and others are blocked.",
            "required_exact_tokens": ["GET", "HEAD", "OPTIONS", "POST", "DELETE"]
        },
        {
            "category": "codex",
            "topic": "internet_allowlist_presets",
            "instruction": "Which preset domain allowlists are available for Codex cloud internet access?",
            "output": "The presets are `None`, `Common dependencies`, and `All (unrestricted)`; custom domains can be added to the first two.",
            "source_url": "https://learn.chatgpt.com/docs/cloud/internet-access.md",
            "source_document_title": "internet-access.md",
            "evidence_text": "You can choose from a preset allowlist: None, Common dependencies, or All (unrestricted). When you select None or Common dependencies, you can add additional domains.",
            "required_exact_tokens": ["None", "Common dependencies", "All (unrestricted)", "additional domains"]
        },
        {
            "category": "codex",
            "topic": "mcp_transport",
            "instruction": "What endpoint and transport should a production OpenAI-compatible MCP server expose?",
            "output": "Expose a stable public HTTPS endpoint that supports MCP streamable HTTP, typically at `/mcp`.",
            "source_url": "https://learn.chatgpt.com/docs/codex-manual.md",
            "source_document_title": "codex-manual.md",
            "evidence_text": "The production endpoint must support the MCP streamable HTTP transport and respond at a stable URL, typically ending in /mcp.",
            "required_exact_tokens": ["MCP streamable HTTP transport", "stable URL", "/mcp"]
        },
        {
            "category": "codex",
            "topic": "mcp_tool_execution",
            "instruction": "What four fields help Codex select and invoke a tool exposed by an MCP server?",
            "output": "Each tool exposes a name, description, input schema, and optional output schema. Codex uses these fields to determine when to call the tool.",
            "source_url": "https://learn.chatgpt.com/docs/codex-manual.md",
            "source_document_title": "codex-manual.md",
            "evidence_text": "Each tool has a name, description, input schema, and optional output schema. These fields help the model decide when to call the tool and how to use its result.",
            "required_exact_tokens": ["input schema", "output schema", "description"]
        },
        {
            "category": "codex",
            "topic": "hook_trust",
            "instruction": "What happens when a non-managed Codex command hook is new or its definition changes?",
            "output": "Codex marks it for review and skips it until trusted; use `/hooks` to inspect and trust the definition.",
            "source_url": "https://learn.chatgpt.com/docs/hooks.md",
            "source_document_title": "hooks.md",
            "evidence_text": "New or changed hooks are marked for review and skipped until trusted. Use /hooks in the CLI to inspect hook sources, review new or changed hooks, and trust hooks.",
            "required_exact_tokens": ["marked for review", "skipped until trusted", "/hooks"]
        }
    ]

    all_coding_agent_30 = claude_sources + codex_sources
    print(f"Prepared {len(all_coding_agent_30)} Coding Agent items.")

    # ---------------------------------------------------------
    # 2. Ambiguous & Insufficient Information QA (20 items)
    # ---------------------------------------------------------
    
    ambiguous_20 = [
        {
            "category": "ambiguous_qa",
            "topic": "missing_os",
            "instruction": "Which specific terminal command should I use to install PostgreSQL database server?",
            "output": "The exact installation command depends on your operating system and package manager. Please specify whether you are running Ubuntu/Debian (`apt`), RHEL/Fedora (`dnf`), or macOS (`brew`)."
        },
        {
            "category": "ambiguous_qa",
            "topic": "unspecified_config",
            "instruction": "Why is my database connection failing with a timeout error in this application?",
            "output": "The failure cannot be diagnosed without seeing your connection string and network configuration. Please provide your configuration parameters and relevant service log entries."
        },
        {
            "category": "ambiguous_qa",
            "topic": "invalid_cli_flag",
            "instruction": "How do I enable the `--fast-turbo-mode` flag in standard Git clone?",
            "output": "Standard Git does not have a `--fast-turbo-mode` flag. Please specify if you are attempting to use shallow clones (`--depth 1`) or single-branch fetches (`--single-branch`)."
        },
        {
            "category": "ambiguous_qa",
            "topic": "missing_error_log",
            "instruction": "My Python script crashed unexpectedly when running on the server. How do I fix it?",
            "output": "The root cause cannot be determined without the error traceback. Please provide the exact exception stack trace and python runtime version."
        },
        {
            "category": "ambiguous_qa",
            "topic": "multiple_meanings",
            "instruction": "How do I reset my environment?",
            "output": "Please clarify whether you want to reset your shell environment variables, clear a Python virtualenv, or reset a Docker container state."
        },
        {
            "category": "ambiguous_qa",
            "topic": "missing_web_framework",
            "instruction": "How do I register custom middleware for handling JWT authentication headers?",
            "output": "Middleware registration varies across frameworks. Please specify whether you are using FastAPI, Express.js, Django, or Flask."
        },
        {
            "category": "ambiguous_qa",
            "topic": "missing_build_tool",
            "instruction": "What setting increases maximum heap allocation to 4 gigabytes during compilation?",
            "output": "The configuration flag depends on your build tool and runtime. Please clarify if you are configuring Node.js (`--max-old-space-size=4096`), Maven (`MAVEN_OPTS`), or Gradle (`org.gradle.jvmargs`)."
        },
        {
            "category": "ambiguous_qa",
            "topic": "unspecified_port_conflict",
            "instruction": "Why is port 8080 throwing an EADDRINUSE error when starting the server?",
            "output": "`EADDRINUSE` usually means another process owns the address. Use `lsof -i :8080` or `ss -ltnp` to identify it before deciding whether to stop it."
        },
        {
            "category": "ambiguous_qa",
            "topic": "nonexistent_npm_package",
            "instruction": "How do I configure an optimizer option that is not listed in my installed npm package documentation?",
            "output": "Do not assume the option exists. Confirm the package name and version in `package.json`, then check that version's documented options."
        },
        {
            "category": "ambiguous_qa",
            "topic": "missing_sql_dialect",
            "instruction": "How do I write a query to extract substring matches using regex?",
            "output": "Regex SQL syntax depends on your database engine. Please specify whether you are querying PostgreSQL (`substring(col from pattern)`), MySQL (`REGEXP_SUBSTR`), or SQLite."
        },
        {
            "category": "ambiguous_qa",
            "topic": "unclear_git_state",
            "instruction": "Git says my branch is diverged. Should I force push?",
            "output": "Force pushing may overwrite remote commits made by team members. Inspect the branch difference using `git log --oneline HEAD..origin/main` first."
        },
        {
            "category": "ambiguous_qa",
            "topic": "missing_docker_context",
            "instruction": "Why is my Docker build failing at step 4 with exit code 1?",
            "output": "The cause cannot be identified without the Dockerfile content and build output. Please share the failing command line and error logs from Docker build."
        },
        {
            "category": "ambiguous_qa",
            "topic": "missing_ci_provider",
            "instruction": "How do I store secret API keys securely for my repository pipeline?",
            "output": "Secret configuration depends on your CI/CD provider. Please specify whether you are configuring GitHub Actions Secrets, GitLab CI Variables, or CircleCI Contexts."
        },
        {
            "category": "ambiguous_qa",
            "topic": "ambiguous_file_path",
            "instruction": "Where should I add the custom proxy redirection rule?",
            "output": "The location depends on your web proxy server. Please clarify if you are editing `nginx.conf`, Apache `.htaccess`, or Caddyfile."
        },
        {
            "category": "ambiguous_qa",
            "topic": "missing_python_version",
            "instruction": "Why does `asyncio.run()` fail with a syntax error in my script?",
            "output": "`asyncio.run()` requires Python 3.7 or newer. Please verify your Python version using `python --version` and update your interpreter if running an older release."
        },
        {
            "category": "ambiguous_qa",
            "topic": "unspecified_auth_flow",
            "instruction": "What API parameter sends the authorization token in HTTP requests?",
            "output": "HTTP standards use the `Authorization: Bearer <token>` header, but some APIs require custom headers or query params. Please specify the target API documentation."
        },
        {
            "category": "ambiguous_qa",
            "topic": "missing_test_runner",
            "instruction": "How do I run tests matching a specific tag filter?",
            "output": "Tag filtering syntax depends on your test runner framework. Please indicate whether you are using Pytest (`-m tag`), Jest (`-t pattern`), or Mocha."
        },
        {
            "category": "ambiguous_qa",
            "topic": "missing_ts_config",
            "instruction": "Why is TypeScript complaining that `Promise.allSettled` does not exist?",
            "output": "Your `tsconfig.json` `target` or `lib` array is set below ES2020. Update `tsconfig.json` to include `\"lib\": [\"ES2020\"]` or newer."
        },
        {
            "category": "ambiguous_qa",
            "topic": "missing_kubernetes_namespace",
            "instruction": "Why is `kubectl get pods` returning no resources?",
            "output": "Your resources may be deployed in a non-default namespace. Run `kubectl get pods -n <namespace>` or `kubectl get pods --all-namespaces` to inspect all pods."
        },
        {
            "category": "ambiguous_qa",
            "topic": "missing_css_preprocessor",
            "instruction": "How do I configure nested selectors with root parent references?",
            "output": "Parent selector syntax depends on the CSS tooling and version. Specify whether you use native CSS nesting or a preprocessor such as Sass with `&`."
        }
    ]
    print(f"Prepared {len(ambiguous_20)} Ambiguous QA items.")

    # ---------------------------------------------------------
    # 3. General Tech QA (150 items)
    # ---------------------------------------------------------
    # We load general QA from diverse tech topics: Git, Linux, Shell, Python, TypeScript, Docker, SQL, HTTP, Testing, Debugging
    
    general_topics = [
        ("Git", "How do you unstage a file in Git without discarding local modifications?", "Run `git restore --staged <file>`. This removes the specified file from the staging index while leaving your working directory modifications untouched."),
        ("Git", "How do you create and switch to a new local Git branch in a single command?", "Run `git checkout -b <branch-name>` or `git switch -c <branch-name>`. This creates the specified branch and checks it out immediately."),
        ("Git", "What command shows a concise single-line Git commit history graph?", "Execute `git log --oneline --graph --all`. This outputs a visual tree graph of all local and remote branches in abbreviated one-line format."),
        ("Git", "How do you temporarily save uncommitted local changes without making a commit?", "Use `git stash push -m \"message\"`. To restore your saved modifications later, execute `git stash pop`."),
        ("Git", "How can you modify the commit message of the most recent Git commit?", "Run `git commit --amend -m \"new message\"`. Note that this modifies the commit hash and should not be used on pushed public commits."),
        ("Git", "How do you inspect commits that may have been lost after moving a Git branch reference?", "Run `git reflog` to inspect recent reference movements before attempting recovery. It shows prior positions of `HEAD` and local branch tips."),
        ("Git", "How do you check which remote URLs are configured for your local Git repository?", "Run `git remote -v`. This lists the fetch and push target URLs for all configured remote names."),
        ("Git", "How do you compare changes between your current working directory and the staging area?", "Run `git diff`. To inspect changes already staged for commit, run `git diff --staged`."),
        ("Git", "What command renames an existing local Git branch?", "Execute `git branch -m <old-name> <new-name>` while on any branch, or `git branch -m <new-name>` while checked out on the target branch."),
        ("Git", "How do you fetch updates from a remote repository and rebase your current branch onto it?", "Run `git pull --rebase origin <branch-name>`. This fetches remote commits and reapplies your local commits on top of them."),
        ("Git", "How do you cherry-pick a specific commit from another branch into your active branch?", "Run `git cherry-pick <commit-hash>`. This applies the changes introduced by that single commit onto your current working branch."),
        ("Git", "What command displays the status of tracked and untracked files in your repository?", "Execute `git status`. It shows modified files, staged changes, and untracked files in your current workspace."),
        ("Git", "How do you delete a local Git branch that has already been merged?", "Run `git branch -d <branch-name>`. Git refuses the deletion when the branch still contains unmerged commits."),
        ("Git", "How do you view detailed commit information for a specific file?", "Run `git log -p <file-path>`. This displays the commit history alongside the diff changes introduced in each commit for that file."),
        ("Git", "How do you preview untracked files and directories that Git clean would remove?", "Run `git clean -nd`. The dry-run output lists candidate files and directories without deleting them."),
        ("Git", "How do you list all local and remote branches in a repository?", "Execute `git branch -a`. Remote tracking branches will be listed with the `remotes/` prefix."),
        ("Git", "How do you create an annotated Git tag for a software release?", "Run `git tag -a v1.0.0 -m \"Release version 1.0.0\"`. Push the tag to remote using `git push origin v1.0.0`."),
        ("Git", "How do you squash the last 3 commits into a single commit using interactive rebase?", "Run `git rebase -i HEAD~3`. In the interactive editor, mark the second and third commits as `squash` or `s`."),

        ("Linux", "How do you check current memory usage and swap activity in Linux?", "Run `free -h`. This displays total, used, free, and cached RAM alongside swap memory statistics in human-readable units."),
        ("Linux", "What command finds files larger than 100 megabytes in `/var/log`?", "Run `find /var/log -type f -size +100M`. This scans the `/var/log` directory for files exceeding 100 MiB."),
        ("Linux", "How do you change file ownership recursively for a directory tree?", "Execute `chown -R user:group /path/to/directory`. The `-R` option applies ownership changes to all nested files and subdirectories."),
        ("Linux", "What command monitors real-time system resource consumption and active process list?", "Run `top` or `htop`. These tools display dynamic CPU usage, RAM utilization, and process states in real time."),
        ("Linux", "How do you make a shell script executable by its current owner?", "Run `chmod u+x script.sh`. This adds the owner execute bit without changing group or other permissions."),
        ("Linux", "What command checks free disk space across all mounted filesystems?", "Execute `df -h`. This reports filesystem total size, used space, available space, and mount points in human-readable notation."),
        ("Linux", "How do you calculate the disk space used by a specific folder?", "Run `du -sh /path/to/folder`. The `-s` flag summarizes total usage, while `-h` outputs human-readable sizes."),
        ("Linux", "How do you request graceful termination of a Linux process by PID?", "Run `kill <PID>` to send `SIGTERM`. Confirm the PID and process owner before issuing the signal."),
        ("Linux", "What command streams the latest lines appended to a log file in real time?", "Run `tail -f /path/to/logfile.log`. Press Ctrl+C to terminate live streaming."),
        ("Linux", "How do you create a symbolic link pointing to a target file?", "Execute `ln -s /path/to/original /path/to/symlink`. This creates a soft link referencing the target location."),
        ("Linux", "What command displays system kernel release and operating system architecture?", "Run `uname -a`. This prints operating system kernel details, hostname, kernel release date, and hardware architecture."),
        ("Linux", "How do you search for active listening network sockets and open ports on Linux?", "Run `ss -tulpn` or `netstat -tulpn`. This reports TCP/UDP listening ports alongside process names and PIDs."),
        ("Linux", "What command measures command execution wall-clock time in terminal?", "Prefix the execution string with `time`, such as `time ./build.sh`. It outputs real, user, and system time components."),
        ("Linux", "How do you archive and compress a directory into a `.tar.gz` tarball?", "Run `tar -czvf archive.tar.gz /path/to/directory`. Use `tar -xzvf archive.tar.gz` to unpack."),
        ("Linux", "What command displays environmental variables configured in your active shell?", "Execute `printenv` or `env`. To inspect a single variable value, run `echo $VARIABLE_NAME`."),
        ("Linux", "How do you count total line occurrences in a file using terminal CLI?", "Run `wc -l filename.txt`. This counts newline characters and prints line totals for the specified file."),
        ("Linux", "What command searches for a text pattern in files recursively across directories?", "Run `grep -rn \"pattern\" /path/to/dir`. The `-r` flag enables recursive searching, while `-n` prints matching line numbers."),
        ("Linux", "How do you run a command in background detached from terminal session using nohup?", "Execute `nohup ./script.sh > output.log 2>&1 &`. Process output is redirected to `output.log` while running in background."),

        ("Python", "How do you create a Python virtual environment in a project directory?", "Run `python3 -m venv .venv`. Activate it using `source .venv/bin/activate` on Linux/macOS or `.venv\\Scripts\\activate` on Windows."),
        ("Python", "What built-in function safely parses JSON strings into Python dictionary objects?", "Use `json.loads(json_string)`. To parse JSON directly from a file handle, use `json.load(file_object)`."),
        ("Python", "How do you flatten a list of lists into a single flat list using list comprehension?", "Use `[item for sublist in matrix for item in sublist]`. This iterates through nested sublists sequentially."),
        ("Python", "What is the recommended way to open and read a text file safely in Python?", "Use a `with` statement block: `with open('file.txt', 'r', encoding='utf-8') as f: content = f.read()`. This automatically closes the file handle."),
        ("Python", "How do you check if a dictionary contains a specific key before accessing it?", "Use the `in` operator: `if 'key' in my_dict:`. Alternatively, use `my_dict.get('key', default_value)` to prevent KeyError exceptions."),
        ("Python", "What decorator converts a class method into a read-only property attribute?", "Use `@property`. This allows calling the method as an attribute without parenthesis while encapsulating getter logic."),
        ("Python", "How do you measure execution runtime of Python code blocks?", "Use the `time.perf_counter()` function from the standard `time` library to calculate high-resolution time deltas."),
        ("Python", "What standard library module parses command-line arguments in Python scripts?", "Use `argparse`. Define an `ArgumentParser` instance, add arguments with `add_argument()`, and parse with `parse_args()`."),
        ("Python", "How do you merge two Python dictionaries in Python 3.9+?", "Use the merge operator `merged_dict = dict1 | dict2`. Values in `dict2` override matching keys from `dict1`."),
        ("Python", "How do you record the packages installed in the active Python environment?", "Run `python -m pip freeze > requirements.txt`. Review the file because it records the full active environment, including transitive dependencies."),
        ("Python", "How do you format floating-point numbers to 2 decimal places in Python f-strings?", "Use f-string formatting syntax `f\"{val:.2f}\"`. This rounds and pads output to 2 decimal places."),
        ("Python", "What is the difference between `is` and `==` operators in Python?", "The `==` operator compares value equality, while `is` compares object identity in memory address space."),
        ("Python", "How do you catch multiple exception types in a single Python `try-except` block?", "Tuple exception classes: `except (ValueError, TypeError) as e:`. Handle shared error recovery in a unified block."),
        ("Python", "What Python dataclass decorator automatically generates init and representation methods?", "Apply `@dataclass` from the `dataclasses` module. It auto-generates `__init__()`, `__repr__()`, and equality comparison methods."),
        ("Python", "How do you filter a list using a lambda expression in Python?", "Use `list(filter(lambda x: condition, my_list))`. Alternatively, use list comprehension `[x for x in my_list if condition]`."),
        ("Python", "What standard module produces SHA-256 cryptographic hashes in Python?", "Use `hashlib.sha256(data_bytes).hexdigest()`. Ensure input text is encoded to bytes prior to hashing."),
        ("Python", "How do you generate random integers within a specific range in Python?", "Import `random` and call `random.randint(a, b)`. This returns a random integer $N$ such that $a \\le N \\le b$."),
        ("Python", "How do you sort a list of dictionaries by a specific key in Python?", "Use `sorted(data, key=lambda x: x['key_name'])`. Specify `reverse=True` for descending order."),

        ("TypeScript", "How do you define a type alias for a union of string literal values in TypeScript?", "Declare `type Status = 'pending' | 'active' | 'completed';`. Variables constrained to this type accept only listed string literals."),
        ("TypeScript", "What Utility Type makes all properties of an interface optional in TypeScript?", "Use `Partial<T>`. For example, `Partial<User>` generates a type where every field of `User` is optional."),
        ("TypeScript", "What Utility Type constructs a type with all properties of T except specified keys K?", "Use `Omit<T, K>`. For instance, `Omit<User, 'password'>` creates a user object type excluding the password field."),
        ("TypeScript", "How do you enforce read-only properties on an interface definition?", "Prefix property declarations with `readonly`, such as `readonly id: string;`. Reassigning readonly properties triggers a compiler error."),
        ("TypeScript", "What compiler flag in `tsconfig.json` enforces strict null and undefined checks?", "Set `\"strictNullChecks\": true` under `compilerOptions` in your `tsconfig.json` file."),
        ("TypeScript", "What operator performs non-null assertions in TypeScript expressions?", "Append `!` to non-null variables, such as `element!.value`. This tells the compiler that the value is guaranteed non-null."),
        ("TypeScript", "How do you define generic type constraints in TypeScript function signatures?", "Use `extends`: `function getProperty<T, K extends keyof T>(obj: T, key: K)`. This restricts $K$ to valid keys of $T$."),
        ("TypeScript", "What Utility Type makes all properties of an interface immutable?", "Use `Readonly<T>`. Any attempt to mutate properties on a `Readonly<T>` instance causes a TypeScript compilation error."),
        ("TypeScript", "How do you define a record type mapping string keys to specific object types?", "Use `Record<string, TargetType>`. For example, `Record<string, number>` represents a dictionary of string keys and numeric values."),
        ("TypeScript", "What is the difference between `unknown` and `any` types in TypeScript?", "The `any` type disables type checking completely, whereas `unknown` requires type narrowing assertions before operating on the value."),
        ("TypeScript", "How do you declare tuple types with fixed element positions in TypeScript?", "Declare explicit element types in brackets: `type Pair = [string, number];`. Tuples enforce fixed array index types."),
        ("TypeScript", "What TypeScript feature extracts the return type of a function signature?", "Use `ReturnType<typeof fn>`. This utility extracts the output type returned by the specified function."),
        ("TypeScript", "How do you configure module resolution in `tsconfig.json` for modern bundlers?", "Set `\"moduleResolution\": \"bundler\"` or `\"node16\"` under `compilerOptions` in `tsconfig.json`."),
        ("TypeScript", "What keyword asserts explicit type compatibility in TypeScript expressions?", "Use `as`: `const val = data as TargetType;`. Alternatively, use angle bracket syntax `<TargetType>data` in non-JSX files."),
        ("TypeScript", "How do you declare interface inheritance across multiple base interfaces?", "Use `extends`: `interface Employee extends Person, Identifiable { salary: number; }`."),
        ("TypeScript", "What `tsconfig.json` option prevents emitting output files if compilation errors occur?", "Set `\"noEmitOnError\": true` under `compilerOptions` in `tsconfig.json`."),
        ("TypeScript", "How do you specify optional properties in TypeScript interfaces?", "Append `?` after property names: `interface Config { port?: number; }`. This permits `undefined` values."),
        ("TypeScript", "What utility type selects specific keys from a type while discarding others?", "Use `Pick<T, K>`. For example, `Pick<User, 'id' | 'name'>` creates a type containing only `id` and `name` fields."),

        ("Docker", "How do you inspect running Docker container logs in real time?", "Run `docker logs -f <container_id>`. The `-f` flag streams live output logs from the specified container."),
        ("Docker", "What CLI command lists all active and stopped Docker containers on your system?", "Run `docker ps -a`. Omitting the `-a` flag displays active running containers only."),
        ("Docker", "How do you execute an interactive bash shell inside a running container?", "Run `docker exec -it <container_id> bash`. The `-it` flag allocates an interactive pseudo-TTY session."),
        ("Docker", "What Dockerfile instruction specifies the base image for container builds?", "Use `FROM <image>:<tag>`. For example, `FROM python:3.11-slim` establishes a Python runtime base image."),
        ("Docker", "How do you inspect Docker disk usage before deciding what to clean?", "Run `docker system df` for a summary or `docker system df -v` for detailed image, container, and volume usage."),
        ("Docker", "What Dockerfile instruction executes container commands during runtime initialization?", "Use `CMD [\"executable\", \"param\"]` or `ENTRYPOINT`. `CMD` sets default commands that can be overridden at startup."),
        ("Docker", "How do you mount a host directory into a Docker container during startup?", "Use the `-v` flag: `docker run -v /host/path:/container/path:ro my_image`. Use `:ro` for read-only mounts."),
        ("Docker", "What command builds a Docker image from a local Dockerfile with a custom tag?", "Run `docker build -t my_app:v1 .`. The `.` specifies the local build context directory."),
        ("Docker", "How do you expose container port 80 to host machine port 8080?", "Use the `-p` flag: `docker run -p 8080:80 my_image`. This maps host port 8080 to container port 80."),
        ("Docker", "What Docker Compose command starts multi-container applications in detached background mode?", "Run `docker compose up -d`. Use `docker compose down` to stop and remove created containers."),
        ("Docker", "How do you inspect detailed configuration and network IP metadata of a container?", "Run `docker inspect <container_id>`. This returns low-level JSON configuration details for the container."),
        ("Docker", "What instruction copies local files into a container during Dockerfile image build?", "Use `COPY <src> <dest>`. For example, `COPY requirements.txt /app/` transfers dependency manifests."),
        ("Docker", "How do you set persistent environment variables inside a Dockerfile?", "Use `ENV KEY=value`. For example, `ENV PYTHONUNBUFFERED=1` sets global runtime environment variables."),
        ("Docker", "What command stops a running container gracefully before fallback to SIGKILL?", "Run `docker stop <container_id>`. It sends `SIGTERM` and waits 10 seconds before terminating process."),
        ("Docker", "How do you copy files between host filesystem and a running Docker container?", "Run `docker cp /host/path <container_id>:/container/path` or vice versa to transfer files."),
        ("Docker", "What Dockerfile instruction sets the default working directory for subsequent commands?", "Use `WORKDIR /path/to/directory`. Subsequent `RUN`, `CMD`, and `COPY` instructions execute from this path."),
        ("Docker", "How do you list locally cached Docker images and their storage footprints?", "Run `docker images`. This displays repository tags, image IDs, creation dates, and total storage sizes."),
        ("Docker", "What flag prevents Docker container auto-restart upon exit?", "Set `--restart no`. To automatically restart unless explicitly stopped, use `--restart unless-stopped`."),

        ("SQL", "What SQL command retrieves unique distinct values from a database table column?", "Use `SELECT DISTINCT column_name FROM table_name;`. This filters out duplicate rows from the query output."),
        ("SQL", "How do you filter grouped aggregate results in SQL SELECT statements?", "Use the `HAVING` clause after `GROUP BY`. For example: `GROUP BY category HAVING COUNT(*) > 5`."),
        ("SQL", "What is the difference between `INNER JOIN` and `LEFT JOIN` in SQL?", "An `INNER JOIN` returns matching records in both tables, whereas a `LEFT JOIN` preserves all rows from the left table."),
        ("SQL", "How do you update existing table column values based on condition criteria?", "Execute `UPDATE table_name SET col1 = val1 WHERE condition;`. Always include a `WHERE` clause to avoid updating all rows."),
        ("SQL", "What index type speeds up primary key lookups and enforces record uniqueness?", "A `PRIMARY KEY` or `UNIQUE INDEX`. It accelerates lookups while preventing duplicate column values."),
        ("SQL", "How do you limit query output to 10 rows ordered by creation timestamp?", "Execute `SELECT * FROM table_name ORDER BY created_at DESC LIMIT 10;`."),
        ("SQL", "What SQL keyword combines query results from multiple SELECT statements without duplicates?", "Use `UNION`. To retain duplicate rows across merged queries, use `UNION ALL`."),
        ("SQL", "How do you safely delete specific rows from a table in SQL?", "Execute `DELETE FROM table_name WHERE condition;`. Test matching rows using `SELECT` before deleting."),
        ("SQL", "What SQL function returns the total count of matching rows in a dataset?", "Use `COUNT(*)`. For counting non-null values in a specific column, use `COUNT(column_name)`."),
        ("SQL", "How do you add a new column to an existing database table schema?", "Execute `ALTER TABLE table_name ADD COLUMN new_col VARCHAR(255);`."),
        ("SQL", "What SQL clause sorts returned query records in ascending or descending order?", "Use `ORDER BY column_name ASC` or `DESC`. Default sorting order is ascending (`ASC`)."),
        ("SQL", "How do you test for null values in a SQL `WHERE` clause?", "Use `WHERE column_name IS NULL`. To filter non-null rows, use `WHERE column_name IS NOT NULL`."),
        ("SQL", "What aggregate function calculates the average numeric value of a column?", "Use `AVG(column_name)`. Null values are ignored automatically when computing the average."),
        ("SQL", "How do you rollback uncommitted database modifications within an explicit transaction?", "Execute `ROLLBACK;`. To commit modifications permanently, execute `COMMIT;`."),
        ("SQL", "What clause groups query rows sharing identical values into summary rows?", "Use `GROUP BY column_name`. It is typically paired with aggregate functions like `SUM` or `COUNT`."),

        ("HTTP", "What HTTP status code indicates that a requested resource was successfully created?", "HTTP status code `201 Created`. It indicates that the request succeeded and a new resource was created."),
        ("HTTP", "What is the primary difference between HTTP GET and POST request methods?", "GET requests retrieve data without side effects, whereas POST requests submit payload data to mutate server state."),
        ("HTTP", "What HTTP response header controls browser Cross-Origin Resource Sharing policies?", "The `Access-Control-Allow-Origin` header. It specifies which origin domains are permitted to read resource responses."),
        ("HTTP", "What HTTP status code represents an unauthorized authentication failure?", "HTTP status code `401 Unauthorized`. It indicates that credentials are missing or invalid."),
        ("HTTP", "What HTTP header specifies the MIME content type of request or response bodies?", "The `Content-Type` header, such as `Content-Type: application/json`."),
        ("HTTP", "What HTTP status code indicates a client forbidden access error despite valid authentication?", "HTTP status code `403 Forbidden`. It indicates that the server understands credentials but refuses access."),
        ("HTTP", "What HTTP status code indicates a temporary redirect to a new URL location?", "HTTP status code `302 Found` or `307 Temporary Redirect`. The new target location is specified in the `Location` header."),
        ("HTTP", "What HTTP method requests identical headers as GET without returning response body?", "The `HEAD` method. It retrieves HTTP headers for metadata inspection without downloading the body."),
        ("HTTP", "What HTTP status code indicates that a request payload exceeds server size limits?", "HTTP status code `413 Payload Too Large`. It indicates the server refuses to process the oversized payload."),
        ("HTTP", "What HTTP header enables persistent connection reuse across sequential requests?", "The `Connection: keep-alive` header. It avoids repeated TCP handshake overhead across requests."),
        ("HTTP", "What HTTP status code represents an unhandled internal server error?", "HTTP status code `500 Internal Server Error`. It indicates that an unexpected server-side exception occurred."),
        ("HTTP", "What HTTP method performs idempotent complete replacements of target resources?", "The `PUT` method. Repeated identical `PUT` requests yield identical server state results."),
        ("HTTP", "What HTTP status code indicates that a requested resource is no longer found on server?", "HTTP status code `404 Not Found`. It indicates that the server cannot locate the requested URI path."),
        ("HTTP", "What HTTP status code indicates a rate limit quota exhaustion error?", "HTTP status code `429 Too Many Requests`. It indicates the client has sent too many requests in a given time window."),
        ("HTTP", "What HTTP header delivers bearer access tokens for server authentication?", "The `Authorization` header, formatted as `Authorization: Bearer <token>`."),

        ("Testing", "What Pytest option runs only test functions matching a specific keyword expression?", "Run `pytest -k \"expression\"`. This executes tests whose function names match the specified filter pattern."),
        ("Testing", "What Pytest decorator parameterizes test functions with multiple input arguments?", "Use `@pytest.mark.parametrize(\"input,expected\", [(1, 2), (3, 4)])`. This executes test logic across dataset tuples."),
        ("Testing", "What Pytest fixture scope executes a fixture function once per test session?", "Set `@pytest.fixture(scope=\"session\")`. The fixture initializes once and persists across all test modules in the run."),
        ("Testing", "How do you verify that a Python function raises a specific exception during Pytest execution?", "Use `with pytest.raises(ExpectedException):`. The block succeeds if the code inside raises the target exception."),
        ("Testing", "What Jest option updates snapshot matchers when component UI output changes deliberately?", "Run `jest -u` or `npm test -- -u`. This regenerates saved snapshot files to match current output."),
        ("Testing", "How do you capture stdout and stderr output during Pytest execution?", "Add the `-s` flag to disable output capture: `pytest -s`. To inspect captured output on failures only, use default mode."),
        ("Testing", "What Pytest flag stops test suite execution immediately upon encountering the first failure?", "Run `pytest -x`. This terminates the test runner process as soon as any single test fails."),
        ("Testing", "What library mocks HTTP API network requests during unit testing in Python?", "Use `responses` or `unittest.mock`. For example, `responses.add(responses.GET, url, json={})` intercepts requests."),
        ("Testing", "How do you generate an HTML code coverage report with Pytest?", "Run `pytest --cov=src --cov-report=html`. This creates a `htmlcov/` directory containing visual line coverage analysis."),
        ("Testing", "What assertion method checks deep equality between objects in Node.js test runner?", "Use `assert.deepStrictEqual(actual, expected)`. It compares nested object property values recursively."),
        ("Testing", "What Pytest plugin runs test cases across parallel CPU cores?", "Use `pytest-xdist`. Execute `pytest -n auto` to distribute tests across available CPU threads."),
        ("Testing", "How do you temporarily skip a test case in Pytest without deleting the code?", "Apply `@pytest.mark.skip(reason=\"Explanation\")` decorator to the target test function."),

        ("Debugging", "How do you inspect active stack traces of running Python threads from CLI?", "Import `traceback` and `sys`, then call `traceback.print_stack()`. Alternatively, use `py-spy dump --pid <PID>`."),
        ("Debugging", "What command launches the interactive Python debugger at a specific line in Python 3.7+?", "Insert `breakpoint()` in your source code. It pauses execution and opens an interactive `pdb` prompt."),
        ("Debugging", "How do you print Linux system calls and signals executed by a running binary process?", "Run `strace -p <PID>` or `strace ./binary`. This logs low-level system call arguments and return values."),
        ("Debugging", "What tool monitors open file descriptors and network connections created by a Linux PID?", "Execute `lsof -p <PID>`. It lists open regular files, directory handles, sockets, and memory mappings."),
        ("Debugging", "How do you print dynamic shared library dependencies required by a Linux executable?", "Run `ldd /path/to/executable`. This reports required `.so` library names and resolved path addresses."),
        ("Debugging", "What Chrome DevTools tab analyzes JavaScript CPU profiles and heap memory allocations?", "Use the `Memory` and `Performance` tabs in DevTools to take heap snapshots and record CPU flame charts."),
        ("Debugging", "How do you track memory leaks in Python objects using standard library tools?", "Use the `tracemalloc` module. Call `tracemalloc.start()` and compare snapshots with `snapshot.compare_to()`."),
        ("Debugging", "How do you read the peak CUDA memory allocated by PyTorch during a measured section?", "Call `torch.cuda.reset_peak_memory_stats()` before the section and `torch.cuda.max_memory_allocated()` afterward."),
        ("Debugging", "How do you inspect active GDB backtraces for a crashed C/C++ core dump?", "Run `gdb /path/to/binary core`. Type `bt` or `backtrace` inside GDB to print the call stack."),
        ("Debugging", "What Node.js flag enables the inspector protocol for remote debugging?", "Launch Node with `node --inspect index.js`. Open `chrome://inspect` in Chrome browser to attach DevTools."),
        ("Git", "How do you fetch all remote branches and tags without modifying your current workspace?", "Run `git fetch --all --tags`. This retrieves all remote ref updates and tags without merging changes into your active branch."),
        ("Linux", "What command displays current active user logins and terminal session activity?", "Run `w` or `who`. This lists logged-in user accounts, login timestamps, and active execution commands."),
        ("Python", "How do you deep copy a nested Python object structure?", "Import `copy` and execute `new_obj = copy.deepcopy(original_obj)`. This creates independent copies of all nested child objects."),
        ("TypeScript", "What compiler option in `tsconfig.json` enables synthetic default imports for CommonJS modules?", "Set `\"allowSyntheticDefaultImports\": true` or `\"esModuleInterop\": true` under `compilerOptions` in `tsconfig.json`."),
        ("Docker", "What Dockerfile instruction specifies the network port exposed by container instances?", "Use `EXPOSE <port>`. For example, `EXPOSE 8080` documents intended container network listening ports."),
        ("SQL", "What SQL command creates a temporary savepoint within an active transaction?", "Execute `SAVEPOINT savepoint_name;`. Revert to it using `ROLLBACK TO SAVEPOINT savepoint_name;`."),
        ("HTTP", "What HTTP status code indicates that a conditional GET request resource was not modified?", "HTTP status code `304 Not Modified`. It tells the client that the cached resource version remains valid."),
        ("Debugging", "What command streams live kernel message logs in Linux terminal?", "Run `dmesg -w` or `dmesg --follow`. It streams hardware driver events and kernel warnings in real time.")
    ]

    general_150 = []
    for topic, q, a in general_topics:
        general_150.append({
            "category": "general_tech",
            "topic": topic,
            "instruction": q,
            "output": a
        })

    print(f"Prepared {len(general_150)} General Tech items.")
    assert len(general_150) == 150, f"Expected 150 general tech items, got {len(general_150)}"
    assert len(ambiguous_20) == 20, f"Expected 20 ambiguous items, got {len(ambiguous_20)}"
    assert len(all_coding_agent_30) == 30, f"Expected 30 coding agent items, got {len(all_coding_agent_30)}"

    # Combine into 200 items
    all_sft_200 = general_150 + ambiguous_20 + all_coding_agent_30
    assert len(all_sft_200) == 200, f"Total SFT items should be 200, got {len(all_sft_200)}"
    for item in all_sft_200:
        item["instruction"] = normalize_text(item["instruction"])
        item["output"] = normalize_text(item["output"])

    # ---------------------------------------------------------
    # 4. Leakage Report & Domain Sources Generation
    # ---------------------------------------------------------
    
    domain_sources_out = []
    leakage_records = []
    
    for idx, item in enumerate(all_coding_agent_30):
        instruction = item["instruction"]
        output = item["output"]
        category = item["category"]
        related_eval_ids = RELATED_EVAL_IDS_BY_TOPIC.get(item["topic"], [])

        score_rows = []
        for eval_item in eval_qa_combined:
            scores = {
                "eval_id": eval_item["id"],
                "question_jaccard": get_jaccard_sim(instruction, eval_item["question"]),
                "question_sequence": get_seq_sim(instruction, eval_item["question"]),
                "answer_jaccard": get_jaccard_sim(output, eval_item["answer"]),
                "answer_sequence": get_seq_sim(output, eval_item["answer"]),
            }
            scores["combined"] = sum(
                scores[key]
                for key in (
                    "question_jaccard",
                    "question_sequence",
                    "answer_jaccard",
                    "answer_sequence",
                )
            ) / 4.0
            score_rows.append(scores)

        nearest = max(score_rows, key=lambda row: row["combined"])
        audit_score_rows = [
            row for row in score_rows if row["eval_id"] not in related_eval_ids
        ]
        if not audit_score_rows:
            raise ValueError(f"No independent evaluation rows remain for {item['topic']}")
        maxima = {
            metric: max(audit_score_rows, key=lambda row: row[metric])
            for metric in (
                "question_jaccard",
                "question_sequence",
                "answer_jaccard",
                "answer_sequence",
            )
        }
        threshold_breaches = []
        threshold_specs = (
            ("question_jaccard", QUESTION_JACCARD_LIMIT),
            ("question_sequence", QUESTION_SEQUENCE_LIMIT),
            ("answer_jaccard", ANSWER_JACCARD_LIMIT),
            ("answer_sequence", ANSWER_SEQUENCE_LIMIT),
        )
        for score_row in audit_score_rows:
            breached = [
                metric
                for metric, limit in threshold_specs
                if score_row[metric] >= limit
            ]
            if breached:
                threshold_breaches.append(
                    {"eval_id": score_row["eval_id"], "metrics": breached}
                )

        source_docs = source_documents_for_url(scraped_docs, item["source_url"])
        if len(source_docs) != 1:
            raise ValueError(
                f"Expected one source document for {item['source_url']}, got {len(source_docs)}"
            )
        source_doc = source_docs[0]
        source_text = str(source_doc.get("content", ""))
        source_folded = source_text.casefold()
        exact_tokens = item["required_exact_tokens"]
        missing_source_tokens = [
            token for token in exact_tokens if token.casefold() not in source_folded
        ]
        if missing_source_tokens:
            raise ValueError(
                f"Source {item['source_url']} is missing tokens {missing_source_tokens}"
            )
        answer_exact_tokens = re.findall(r"`([^`]+)`", output)
        missing_answer_tokens = [
            token for token in answer_exact_tokens if token.casefold() not in source_folded
        ]
        if missing_answer_tokens:
            raise ValueError(
                f"Answer for {item['topic']} adds unsupported exact tokens {missing_answer_tokens}"
            )

        domain_sources_out.append({
            "row_index": idx + 170, # 150 general + 20 ambiguous = 170 offset
            "category": category,
            "topic": item["topic"],
            "instruction": instruction,
            "source_url": item["source_url"],
            "source_document_title": item["source_document_title"],
            "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "evidence_text": item["evidence_text"],
            "required_exact_tokens": exact_tokens,
            "answer_exact_tokens": answer_exact_tokens,
            "related_eval_ids": related_eval_ids,
            "nearest_eval_qa_id": nearest["eval_id"],
            "question_jaccard_similarity": round(maxima["question_jaccard"]["question_jaccard"], 4),
            "question_jaccard_eval_id": maxima["question_jaccard"]["eval_id"],
            "question_sequence_matcher_similarity": round(maxima["question_sequence"]["question_sequence"], 4),
            "question_sequence_eval_id": maxima["question_sequence"]["eval_id"],
            "answer_jaccard_similarity": round(maxima["answer_jaccard"]["answer_jaccard"], 4),
            "answer_jaccard_eval_id": maxima["answer_jaccard"]["eval_id"],
            "answer_sequence_matcher_similarity": round(maxima["answer_sequence"]["answer_sequence"], 4),
            "answer_sequence_eval_id": maxima["answer_sequence"]["eval_id"],
            "source_validation": "required_and_answer_exact_tokens_found_in_claimed_source",
            "manual_review_required": True
        })

        leakage_records.append({
            "instruction": instruction,
            "category": category,
            "topic": item["topic"],
            "related_eval_ids": related_eval_ids,
            "nearest_eval_qa_id": nearest["eval_id"],
            "question_jaccard": round(maxima["question_jaccard"]["question_jaccard"], 4),
            "question_jaccard_eval_id": maxima["question_jaccard"]["eval_id"],
            "question_seq_matcher": round(maxima["question_sequence"]["question_sequence"], 4),
            "question_seq_eval_id": maxima["question_sequence"]["eval_id"],
            "answer_jaccard": round(maxima["answer_jaccard"]["answer_jaccard"], 4),
            "answer_jaccard_eval_id": maxima["answer_jaccard"]["eval_id"],
            "answer_seq_matcher": round(maxima["answer_sequence"]["answer_sequence"], 4),
            "answer_seq_eval_id": maxima["answer_sequence"]["eval_id"],
            "threshold_breaches": threshold_breaches,
            "passed_thresholds": not threshold_breaches,
            "manual_review_required": True
        })

    leakage_passed = all(rec["passed_thresholds"] for rec in leakage_records)
    leakage_report = {
        "schema_version": 2,
        "total_domain_items_checked": len(all_coding_agent_30),
        "thresholds": {
            "question_jaccard_warning_at": QUESTION_JACCARD_LIMIT,
            "question_seq_matcher_warning_at": QUESTION_SEQUENCE_LIMIT,
            "answer_jaccard_warning_at": ANSWER_JACCARD_LIMIT,
            "answer_seq_matcher_warning_at": ANSWER_SEQUENCE_LIMIT
        },
        "all_items_passed_automated_thresholds": leakage_passed,
        "manual_fact_review_still_required": True,
        "items": leakage_records
    }
    if not leakage_passed:
        failed = [
            {"topic": record["topic"], "breaches": record["threshold_breaches"]}
            for record in leakage_records
            if not record["passed_thresholds"]
        ]
        raise ValueError(f"Domain leakage thresholds failed for: {failed}")

    # Do not overwrite canonical artifacts until every domain item passes.
    with open(SFT_DIR / "sft_v3_domain_sources.jsonl", "w", encoding="utf-8") as f:
        for entry in domain_sources_out:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with open(SFT_DIR / "sft_v3_leakage_report.json", "w", encoding="utf-8") as f:
        json.dump(leakage_report, f, indent=2, ensure_ascii=False)
    print("Saved sft_v3_domain_sources.jsonl and sft_v3_leakage_report.json")

    # Save sft_v3_train.jsonl (Alpaca format) & sft_v3_train_chat.jsonl (Chat format)
    alpaca_out = []
    chat_out = []
    item_metadata_out = []
    
    for item in all_sft_200:
        inst = item["instruction"]
        out = item["output"]
        
        alpaca_out.append({
            "instruction": inst,
            "input": "",
            "output": out
        })
        
        chat_out.append({
            "messages": [
                {"role": "user", "content": inst},
                {"role": "assistant", "content": out}
            ]
        })
        item_metadata_out.append({
            "row_index": len(item_metadata_out),
            "category": item["category"],
            "topic": item["topic"],
            "instruction_sha256": hashlib.sha256(inst.encode("utf-8")).hexdigest(),
            "output_sha256": hashlib.sha256(out.encode("utf-8")).hexdigest(),
        })

    with open(SFT_DIR / "sft_v3_train.jsonl", "w", encoding="utf-8") as f:
        for entry in alpaca_out:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(SFT_DIR / "sft_v3_train_chat.jsonl", "w", encoding="utf-8") as f:
        for entry in chat_out:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(SFT_DIR / "sft_v3_item_metadata.jsonl", "w", encoding="utf-8") as f:
        for entry in item_metadata_out:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("Saved sft_v3_train.jsonl and sft_v3_train_chat.jsonl")

    # Compute SHA-256 for dataset files
    sha256_alpaca = hashlib.sha256((SFT_DIR / "sft_v3_train.jsonl").read_bytes()).hexdigest()
    sha256_chat = hashlib.sha256((SFT_DIR / "sft_v3_train_chat.jsonl").read_bytes()).hexdigest()
    sha256_items = hashlib.sha256((SFT_DIR / "sft_v3_item_metadata.jsonl").read_bytes()).hexdigest()
    sha256_domain = hashlib.sha256((SFT_DIR / "sft_v3_domain_sources.jsonl").read_bytes()).hexdigest()
    sha256_leakage = hashlib.sha256((SFT_DIR / "sft_v3_leakage_report.json").read_bytes()).hexdigest()

    lengths = [len(item["output"]) for item in all_sft_200]
    sorted_len = sorted(lengths)
    
    def get_percentile(lst, p):
        idx = int(len(lst) * p / 100)
        return lst[min(idx, len(lst)-1)]

    # Count sentence distribution
    sentence_counts = {1: 0, 2: 0, 3: 0, "3+": 0}
    for out_text in [item["output"] for item in all_sft_200]:
        # Count sentences by ending periods
        s_list = [s for s in re.split(r'(?<=[.!?])\s+', out_text.strip()) if s]
        c = len(s_list)
        if c in sentence_counts:
            sentence_counts[c] += 1
        else:
            sentence_counts["3+"] += 1

    metadata = {
        "schema_version": 2,
        "status": "rule_audited_manual_domain_review_required",
        "experiment_name": "sft_v3_dataset",
        "item_count": 200,
        "breakdown": {
            "general_tech": 150,
            "ambiguous_qa": 20,
            "claude_code": 15,
            "codex": 15
        },
        "output_char_length_stats": {
            "min": min(sorted_len),
            "mean": round(sum(sorted_len) / len(sorted_len), 2),
            "median": get_percentile(sorted_len, 50),
            "p95": get_percentile(sorted_len, 95),
            "max": max(sorted_len)
        },
        "sentence_counts": sentence_counts,
        "files": {
            "sft_v3_train.jsonl": {"sha256": sha256_alpaca},
            "sft_v3_train_chat.jsonl": {"sha256": sha256_chat},
            "sft_v3_item_metadata.jsonl": {"sha256": sha256_items},
            "sft_v3_domain_sources.jsonl": {"sha256": sha256_domain},
            "sft_v3_leakage_report.json": {"sha256": sha256_leakage}
        },
        "prompt_template": "Answer directly and completely using only supported information.\nUse one concise sentence when sufficient and up to three sentences when necessary.\nPreserve exact commands, option names, paths, parameter values, and required conditions.\nDo not invent missing details, repeat the question, or continue with another Q&A.\nQuestion: {instruction}\nAnswer:"
    }

    with open(SFT_DIR / "sft_v3_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Saved sft_v3_metadata.json")

    # ---------------------------------------------------------
    # 5. Generate Clean Held-out Evaluation Candidate Set (eval_qa_v3_heldout.jsonl)
    # ---------------------------------------------------------
    
    heldout_items = []
    audit_metadata_records = []
    seen_questions = set()
    
    related_eval_ids = {
        eval_id
        for ids in RELATED_EVAL_IDS_BY_TOPIC.values()
        for eval_id in ids
    }

    for item in eval_qa_combined:
        item_id = item["id"]
        q_text = normalize_text(item["question"])
        q_norm = q_text.casefold()

        if q_norm in seen_questions:
            audit_metadata_records.append({
                "id": item_id,
                "question": q_text,
                "status": "excluded",
                "reason": "Exact duplicate question in original combined dataset"
            })
            continue
        seen_questions.add(q_norm)

        if len(q_text) > 1000 or item.get("topic") in KNOWN_BAD_EVAL_TOPICS:
            audit_metadata_records.append({
                "id": item_id,
                "question": q_text,
                "status": "excluded",
                "reason": "Known malformed or mechanically duplicated evaluation topic"
            })
            continue

        if item_id in related_eval_ids:
            audit_metadata_records.append({
                "id": item_id,
                "question": q_text,
                "status": "excluded",
                "reason": "Fact-level relationship declared for an SFT v3 domain item"
            })
            continue

        leakage_matches = []
        for domain_item in all_coding_agent_30:
            q_jaccard = get_jaccard_sim(q_text, domain_item["instruction"])
            q_sequence = get_seq_sim(q_text, domain_item["instruction"])
            a_jaccard = get_jaccard_sim(item["answer"], domain_item["output"])
            a_sequence = get_seq_sim(item["answer"], domain_item["output"])
            if (
                q_jaccard >= QUESTION_JACCARD_LIMIT
                or q_sequence >= QUESTION_SEQUENCE_LIMIT
                or a_jaccard >= ANSWER_JACCARD_LIMIT
                or a_sequence >= ANSWER_SEQUENCE_LIMIT
            ):
                leakage_matches.append(domain_item["topic"])
        if leakage_matches:
            audit_metadata_records.append({
                "id": item_id,
                "question": q_text,
                "status": "excluded",
                "reason": "Automated similarity threshold overlap with SFT v3 domain items",
                "matching_sft_topics": sorted(set(leakage_matches)),
            })
            continue

        source_docs = source_documents_for_url(scraped_docs, str(item.get("source_url", "")))
        if len(source_docs) != 1:
            audit_metadata_records.append({
                "id": item_id,
                "question": q_text,
                "status": "excluded",
                "reason": f"Expected one claimed source document, found {len(source_docs)}",
            })
            continue

        source_text = str(source_docs[0].get("content", ""))
        source_folded = source_text.casefold()
        keywords = [normalize_text(str(value)) for value in item.get("keywords", [])]
        missing_keywords = [
            keyword for keyword in keywords if keyword.casefold() not in source_folded
        ]
        reference_tokens = re.findall(r"`([^`]+)`", str(item.get("answer", "")))
        missing_reference_tokens = [
            token for token in reference_tokens if token.casefold() not in source_folded
        ]
        if not keywords or missing_keywords or missing_reference_tokens:
            audit_metadata_records.append({
                "id": item_id,
                "question": q_text,
                "status": "excluded",
                "reason": "Claimed source does not contain every keyword and reference code token",
                "missing_keywords": missing_keywords,
                "missing_reference_tokens": missing_reference_tokens,
            })
            continue

        heldout_items.append(item)
        audit_metadata_records.append({
            "id": item_id,
            "question": q_text,
            "status": "accepted",
            "reason": "Held-out candidate: unique, no detected SFT v3 overlap, and all keywords/code tokens occur in the claimed source",
            "source_url": item["source_url"],
            "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "manual_review_required": True,
        })

    with open(QA_DIR / "eval_qa_v3_heldout.jsonl", "w", encoding="utf-8") as f:
        for h_item in heldout_items:
            f.write(json.dumps(h_item, ensure_ascii=False) + "\n")

    heldout_sha256 = hashlib.sha256((QA_DIR / "eval_qa_v3_heldout.jsonl").read_bytes()).hexdigest()

    heldout_metadata = {
        "schema_version": 2,
        "status": "candidate_requires_manual_fact_review",
        "description": "Conservative held-out candidate dataset for SFT v3 comparison",
        "original_combined_count": len(eval_qa_combined),
        "heldout_candidate_count": len(heldout_items),
        "excluded_count": len(eval_qa_combined) - len(heldout_items),
        "file_sha256": heldout_sha256,
        "selection_does_not_prove_semantic_correctness": True,
        "audit_log": audit_metadata_records
    }

    with open(QA_DIR / "eval_qa_v3_heldout_metadata.json", "w", encoding="utf-8") as f:
        json.dump(heldout_metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved eval_qa_v3_heldout.jsonl ({len(heldout_items)} items) and eval_qa_v3_heldout_metadata.json")
    print("Dataset generation completed successfully!")

if __name__ == "__main__":
    main()
