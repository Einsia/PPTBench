from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import shutil
import subprocess
import sys
import sysconfig
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from typing import Any

from .clip_score import DEFAULT_CLIP_MODEL, TransformersClipEncoder, score_pairs
from .config import Config
from .pptx import (
    PptxError,
    evaluate_pptx,
    render_pptx,
    validate_reconstruction_pptx,
)
from .samples import load_samples
from .tools.check_pptx_bounds import check_deck
from .usage import aggregate_usage_ledger, ledger_from_invocation_paths


UTC = timezone.utc
IMAGE_ONLY_SINGLE_RUN_PROTOCOL = "image-only-single-run"
CODEX_FILESYSTEM_ISOLATION = "bubblewrap-sparse"
CODEX_IDENTITY_ISOLATION = "identity-blind-role-separated"
CODEX_PROCESS_ISOLATION = "private-proc-with-opaque-bind-sources"
CODEX_ENVIRONMENT_ISOLATION = "minimal-main-and-tool-env"
CLAUDE_CODE_FILESYSTEM_ISOLATION = "bubblewrap-sparse"
CLAUDE_CODE_IDENTITY_ISOLATION = "identity-blind-bare-role-separated"
CLAUDE_CODE_PROCESS_ISOLATION = "private-proc-with-opaque-bind-sources"
CLAUDE_CODE_ENVIRONMENT_ISOLATION = "minimal-claude-api-and-tool-env"
OPENCODE_FILESYSTEM_ISOLATION = "bubblewrap-sparse"
OPENCODE_IDENTITY_ISOLATION = "identity-blind-bare-role-separated"
OPENCODE_PROCESS_ISOLATION = "private-proc-with-opaque-bind-sources"
OPENCODE_ENVIRONMENT_ISOLATION = "minimal-opencode-api-and-tool-env"
AGENT_WORKSPACE_ALIAS = PurePosixPath("/work")
AGENT_OUTPUT_ALIAS = PurePosixPath("/turn-output")
AGENT_NODE_RUNTIME_ALIAS = PurePosixPath("/runtime/node")
AGENT_PYTHON_ENV_ALIAS = PurePosixPath("/runtime/venv")
AGENT_PYTHON_BASE_ALIAS = PurePosixPath("/runtime/python-base")

class HarnessError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AgentSpec:
    kind: str
    executable: str
    model: str = ""
    reasoning_effort: str = ""
    max_turns: int = 40

    def __post_init__(self) -> None:
        """Validate the selected agent runtime."""
        if self.kind not in {"codex", "claude", "claude-code", "opencode"}:
            raise ValueError(f"unsupported agent kind: {self.kind}")
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")


def _write_json(path: Path, value: Any) -> None:
    """Write a JSON value, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object, returning an empty object when it is unusable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _valid_iteration_checkpoint(path: Path, expected_index: int) -> dict[str, Any]:
    """Load a complete iteration checkpoint whose referenced artifacts still exist."""

    value = _read_json_object(path)
    if value.get("iteration") != expected_index or value.get("target") != "pptx":
        return {}
    for key in ("pptx_path", "screenshot_path", "components_path"):
        artifact = Path(str(value.get(key, "")))
        if not artifact.is_file():
            return {}
    return value


def _completed_invocation_checkpoint(path: Path) -> dict[str, Any]:
    """Load a successfully completed physical invocation checkpoint."""

    value = _read_json_object(path)
    if (
        value.get("returncode") != 0
        or value.get("timed_out")
        or value.get("error_events")
        or value.get("json_parse_errors")
        or not value.get("thread_id")
    ):
        return {}
    usage = value.get("usage")
    if not isinstance(usage, dict) or not all(
        isinstance(usage.get(field), int)
        and not isinstance(usage.get(field), bool)
        and usage[field] >= 0
        for field in ("input_tokens", "output_tokens")
    ):
        return {}
    return value


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    """Convert a value to a filesystem-safe slug."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    if not slug:
        raise ValueError("run identifier cannot be empty")
    return slug


def _opaque_run_id(value: str) -> str:
    """Return a deterministic run directory name that carries no task identity."""

    slug = _slug(value)
    if re.fullmatch(r"(?:rollout-pptx|run)-[0-9a-f]{32}", slug):
        return slug
    digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:32]
    return f"run-{digest}"


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).isoformat()


def default_run_id() -> str:
    """Return a timestamp-based run identifier."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _ensure_pptx_reconstruction(
    workspace: Path,
    approved_resource_paths: list[Path] | None = None,
    reference: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Locate the reconstruction PPTX and return its artifact diagnostics."""
    pptx_path = workspace / "reconstruction.pptx"
    if not pptx_path.is_file():
        raise HarnessError("agent did not create reconstruction.pptx")
    try:
        report = validate_reconstruction_pptx(
            pptx_path,
            reference_path=reference,
            approved_resource_paths=approved_resource_paths,
            enforce_resource_integrity=False,
        )
    except PptxError as exc:
        raise HarnessError(str(exc)) from exc
    return pptx_path, report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON objects from a JSONL file."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _jsonl_diagnostics(path: Path) -> dict[str, Any]:
    """Summarize parse errors and row counts in a JSONL file."""
    parse_errors: list[dict[str, Any]] = []
    error_events: list[dict[str, Any]] = []
    warning_events: list[dict[str, Any]] = []
    pending_errors: list[dict[str, Any]] = []
    event_count = 0
    if not path.exists():
        return {
            "event_count": 0,
            "json_parse_errors": [{"line": 0, "error": "events file is missing"}],
            "error_events": [],
            "warning_events": [],
        }
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        event_count += 1
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            parse_errors.append({"line": line_number, "error": str(exc)})
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        record = {"line": line_number, "event": event}
        if event_type == "error":
            pending_errors.append(record)
        elif event_type == "turn.completed":
            warning_events.extend(pending_errors)
            pending_errors.clear()
        elif event_type == "turn.failed":
            error_events.extend(pending_errors)
            pending_errors.clear()
            error_events.append(record)
        elif event_type == "result":
            # Claude Code reports gateway/model failures as a final result
            # object.  Some gateways still exit the CLI with status zero, so
            # process status alone is not sufficient evidence of success.
            if event.get("is_error") or event.get("subtype") != "success":
                error_events.extend(pending_errors)
                pending_errors.clear()
                error_events.append(record)
            else:
                warning_events.extend(pending_errors)
                pending_errors.clear()
    error_events.extend(pending_errors)
    return {
        "event_count": event_count,
        "json_parse_errors": parse_errors,
        "error_events": error_events,
        "warning_events": warning_events,
    }


def _only_claude_max_turns_errors(error_events: list[dict[str, Any]]) -> bool:
    """Return whether every fatal event is Claude's max-turns terminator."""

    if not error_events:
        return False
    for record in error_events:
        event = record.get("event") if isinstance(record, dict) else None
        if not isinstance(event, dict) or event.get("type") != "result":
            return False
        if not (
            event.get("terminal_reason") == "max_turns"
            or event.get("subtype") == "error_max_turns"
        ):
            return False
    return True


def _codex_event_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract usage and session metadata from Codex events."""
    thread_id = ""
    usage: dict[str, Any] = {}
    final_message = ""
    for row in rows:
        if row.get("type") == "thread.started":
            thread_id = str(row.get("thread_id", ""))
        elif row.get("type") == "turn.completed" and isinstance(row.get("usage"), dict):
            usage = dict(row["usage"])
        elif row.get("type") == "item.completed":
            item = row.get("item", {})
            if isinstance(item, dict) and item.get("type") == "agent_message":
                final_message = str(item.get("text", ""))
    return {"thread_id": thread_id, "usage": usage, "final_message": final_message}


def _claude_event_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract usage and session metadata from Claude events."""
    result: dict[str, Any] = {}
    init: dict[str, Any] = {}
    for row in rows:
        if row.get("type") == "system" and row.get("subtype") == "init":
            init = row
        if row.get("type") == "result":
            result = row
    provider_usage = result.get("usage")
    if not isinstance(provider_usage, dict):
        provider_usage = {}
    has_reported_usage = any(
        field in provider_usage
        for field in (
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "output_tokens",
        )
    )

    def counter(name: str) -> int:
        value = provider_usage.get(name, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

    cache_creation = counter("cache_creation_input_tokens")
    cache_read = counter("cache_read_input_tokens")
    direct_input = counter("input_tokens")
    normalized_usage = (
        {
            # The common ledger treats cached_input_tokens as a subset of total
            # input. Claude reports direct, cache-write, and cache-read tokens as
            # disjoint counters, so construct the inclusive total explicitly.
            "input_tokens": direct_input + cache_creation + cache_read,
            "cached_input_tokens": cache_read,
            "output_tokens": counter("output_tokens"),
            "reasoning_output_tokens": 0,
        }
        if has_reported_usage
        else {}
    )
    return {
        "thread_id": str(result.get("session_id") or init.get("session_id") or ""),
        "usage": normalized_usage,
        "provider_usage": provider_usage,
        "final_message": str(result.get("result", "")),
        "total_cost_usd": result.get("total_cost_usd"),
        "num_turns": result.get("num_turns"),
        "effective_model": str(init.get("model", "")),
        "model_usage": result.get("modelUsage") if isinstance(result.get("modelUsage"), dict) else {},
        "terminal_reason": result.get("terminal_reason"),
    }


def _opencode_event_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize OpenCode's per-step JSON events into the common usage ledger."""

    thread_id = ""
    final_message = ""
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    reasoning_output_tokens = 0
    total_cost_usd = 0.0
    num_turns = 0
    terminal_reason: str | None = None
    provider_steps: list[dict[str, Any]] = []
    for row in rows:
        if row.get("sessionID"):
            thread_id = str(row["sessionID"])
        part = row.get("part")
        if not isinstance(part, dict):
            continue
        if row.get("type") == "text" and part.get("text"):
            final_message = str(part["text"])
        if row.get("type") != "step_finish":
            continue
        tokens = part.get("tokens")
        if not isinstance(tokens, dict):
            tokens = {}

        def counter(value: Any) -> int:
            return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0

        cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
        input_tokens += counter(tokens.get("input"))
        cached_input_tokens += counter(cache.get("read"))
        output_tokens += counter(tokens.get("output"))
        reasoning_output_tokens += counter(tokens.get("reasoning"))
        cost = part.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
            total_cost_usd += float(cost)
        num_turns += 1
        terminal_reason = str(part.get("reason")) if part.get("reason") is not None else None
        provider_steps.append(
            {
                "tokens": tokens,
                "cost": cost,
                "reason": part.get("reason"),
            }
        )
    usage = (
        {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "reasoning_output_tokens": reasoning_output_tokens,
        }
        if num_turns
        else {}
    )
    return {
        "thread_id": thread_id,
        "usage": usage,
        "provider_usage": {"steps": provider_steps} if provider_steps else {},
        "final_message": final_message,
        "total_cost_usd": total_cost_usd if provider_steps else None,
        "num_turns": num_turns or None,
        "terminal_reason": terminal_reason,
    }


def build_agent_command(
    spec: AgentSpec,
    *,
    workspace: Path,
    prompt: str,
    images: list[Path],
    final_message: Path,
    writable: bool,
    output_schema: Path | None = None,
    resume_session_id: str | None = None,
    persistent: bool = False,
) -> list[str]:
    """Build the command used to invoke an agent in its isolated workspace."""
    if spec.kind == "codex":
        if resume_session_id:
            command = [
                spec.executable,
                "exec",
                "resume",
                "--json",
                "--skip-git-repo-check",
                "--ignore-rules",
            ]
        else:
            command = [spec.executable, "exec"]
            if not persistent:
                command.append("--ephemeral")
            command.extend(
                [
                    "--json",
                    "--color",
                    "never",
                    "--sandbox",
                    "workspace-write" if writable else "read-only",
                    "--skip-git-repo-check",
                    "--ignore-rules",
                    "--cd",
                    str(workspace),
                ]
            )
        if spec.model:
            command.extend(["--model", spec.model])
        if spec.reasoning_effort:
            command.extend(["--config", f'model_reasoning_effort="{spec.reasoning_effort}"'])
        if resume_session_id:
            sandbox_mode = "workspace-write" if writable else "read-only"
            command.extend(
                [
                    "--config",
                    f'sandbox_mode="{sandbox_mode}"',
                    "--config",
                    'approval_policy="never"',
                ]
            )
        for image in images:
            command.extend(["--image", str(image)])
        if output_schema is not None:
            command.extend(["--output-schema", str(output_schema)])
        command.extend(["--output-last-message", str(final_message)])
        if resume_session_id:
            command.append(resume_session_id)
        command.append(prompt)
        return command

    if spec.kind == "opencode":
        # Keep the positional message before --file. OpenCode's repeatable file
        # option otherwise consumes a trailing prompt as another attachment.
        command = [spec.executable, "run", prompt, "--pure", "--format", "json"]
        if resume_session_id:
            command.extend(["--session", resume_session_id])
        if spec.model:
            command.extend(["--model", spec.model])
        if spec.reasoning_effort:
            command.extend(["--variant", spec.reasoning_effort])
        for image in images:
            command.extend(["--file", str(image)])
        return command

    # Keep the prompt immediately after --print.  Claude's --allowedTools is a
    # variadic option and would otherwise consume a trailing positional prompt.
    command = [spec.executable, "--bare", "--print", prompt]
    if resume_session_id:
        command.extend(["--resume", resume_session_id])
    command.extend(
        [
            "--output-format",
            "stream-json",
            "--verbose",
            "--max-turns",
            str(spec.max_turns),
            "--dangerously-skip-permissions",
        ]
    )
    if not persistent:
        command.append("--no-session-persistence")
    if spec.model:
        command.extend(["--model", spec.model])
    if spec.reasoning_effort:
        command.extend(["--effort", spec.reasoning_effort])
    if writable:
        command.extend(["--allowedTools", "Read,Write,Edit,Glob,Grep,Bash"])
    else:
        command.extend(
            [
                "--allowedTools",
                "Read,Glob,Grep",
                "--disallowedTools",
                "Write,Edit,Bash",
            ]
        )
    return command


def _resolve_executable(value: str) -> str:
    """Resolve an executable name against the configured runtime."""
    path = shutil.which(value)
    if path:
        return str(Path(path).resolve())
    candidate = Path(value)
    if candidate.is_file():
        return str(candidate.resolve())
    raise HarnessError(f"agent executable is not available: {value}")


def _codex_runtime_root(executable: Path) -> Path:
    """Return the smallest installed Node runtime tree containing Codex."""

    executable = executable.resolve()
    for parent in executable.parents:
        if parent != Path("/") and (parent / "bin" / "node").is_file():
            return parent
    if executable.is_file():
        return executable.parent
    raise HarnessError(f"cannot locate the Codex runtime for isolation: {executable}")


def _prepare_isolated_python_environment(
    agent_run: Path,
    session_namespace: str,
) -> tuple[Path, Path, PurePosixPath]:
    """Create a tiny identity-blind venv facade for the agent sandbox.

    uv and other relocatable venvs often use absolute interpreter symlinks.
    Binding such a venv at ``/runtime/venv`` leaves those links broken and also
    reveals the host home path. The facade keeps only a sanitized interpreter
    link and pyvenv metadata while the real site-packages directory is mounted
    read-only at the same relative location.
    """

    python_environment = Path(sys.executable).absolute().parent.parent
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    try:
        purelib_relative = purelib.relative_to(python_environment.resolve())
    except ValueError as exc:
        raise HarnessError("Python site-packages is outside the active environment") from exc
    facade = agent_run / f".python-env-image-only-{session_namespace}"
    facade_purelib = facade / purelib_relative
    facade_purelib.mkdir(parents=True, exist_ok=True)
    bin_dir = facade / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    base_executable = Path(sys._base_executable).name
    # System Python remains visible under /usr in the sparse filesystem;
    # relocatable runtimes are mounted at AGENT_PYTHON_BASE_ALIAS instead.
    base_prefix = Path(sys.base_prefix).resolve()
    visible_base = (
        PurePosixPath(str(base_prefix))
        if base_prefix.is_relative_to("/usr")
        else AGENT_PYTHON_BASE_ALIAS
    )
    target = str(visible_base / "bin" / base_executable)
    for name in {"python", "python3", f"python{sys.version_info.major}.{sys.version_info.minor}"}:
        link = bin_dir / name
        link.unlink(missing_ok=True)
        link.symlink_to(target)
    (facade / "pyvenv.cfg").write_text(
        "\n".join(
            [
                f"home = {visible_base}/bin",
                "include-system-site-packages = false",
                f"version = {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                f"executable = {target}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    purelib_alias = PurePosixPath(str(AGENT_PYTHON_ENV_ALIAS)) / purelib_relative.as_posix()
    return facade, purelib, purelib_alias


def _toml_key(value: str) -> str:
    """Encode a TOML key used in the isolated Codex provider configuration."""

    return value if re.fullmatch(r"[A-Za-z0-9_-]+", value) else json.dumps(value)


def _toml_value(value: Any) -> str:
    """Encode the small TOML value subset accepted in provider definitions."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        entries = ", ".join(
            f"{_toml_key(str(key))} = {_toml_value(item)}"
            for key, item in value.items()
            if item is not None
        )
        return "{ " + entries + " }"
    raise HarnessError(f"unsupported Codex provider config value: {type(value).__name__}")


def _write_isolated_codex_config(
    host_codex_home: Path,
    destination: Path,
    *,
    agent_home_alias: PurePosixPath = PurePosixPath("/home/agent"),
) -> None:
    """Copy only connection-provider settings, excluding unrelated local project state."""

    lines: list[str] = []
    host_config = host_codex_home / "config.toml"
    if host_config.is_file():
        try:
            parsed = tomllib.loads(host_config.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise HarnessError(f"cannot parse Codex provider configuration: {exc}") from exc
        provider_name = parsed.get("model_provider")
        providers = parsed.get("model_providers")
        if isinstance(provider_name, str) and isinstance(providers, dict):
            provider = providers.get(provider_name)
            if not isinstance(provider, dict):
                raise HarnessError(f"Codex model provider is not configured: {provider_name}")
            service_tier = parsed.get("service_tier")
            if isinstance(service_tier, str):
                lines.append(f"service_tier = {_toml_value(service_tier)}")
            lines.extend(
                [
                    f"model_provider = {_toml_value(provider_name)}",
                    "",
                    f"[model_providers.{_toml_key(provider_name)}]",
                ]
            )
            lines.extend(
                f"{_toml_key(str(key))} = {_toml_value(value)}"
                for key, value in provider.items()
                if value is not None
            )
    tool_path = os.pathsep.join(
        [
            str(AGENT_PYTHON_ENV_ALIAS / "bin"),
            str(AGENT_NODE_RUNTIME_ALIAS / "bin"),
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        ]
    )
    lines.extend(
        [
            "",
            "[shell_environment_policy]",
            'inherit = "none"',
            "ignore_default_excludes = false",
            "set = "
            + _toml_value(
                {
                    "HOME": str(agent_home_alias),
                    "LANG": "C.UTF-8",
                    "NO_COLOR": "1",
                    "PATH": tool_path,
                    "PWD": str(AGENT_WORKSPACE_ALIAS),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "TMPDIR": "/tmp",
                    "VIRTUAL_ENV": str(AGENT_PYTHON_ENV_ALIAS),
                }
            ),
        ]
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _codex_isolation_command(
    command: list[str],
    *,
    workspace: Path,
    output_dir: Path,
) -> tuple[list[str], list[str], dict[str, Any], dict[str, str]]:
    """Wrap Codex in a sparse filesystem that contains no evaluator source files."""

    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise HarnessError("bubblewrap is required for the image-only Codex protocol")

    workspace_argument_root = workspace.absolute()
    output_argument_root = output_dir.absolute()
    workspace = workspace.resolve()
    output_dir = output_dir.resolve()
    if not workspace.is_dir() or not output_dir.is_dir():
        raise HarnessError("Codex isolation requires existing workspace and output directories")
    physical_case = workspace.parent
    if (
        not re.fullmatch(r"case-\d{4}", physical_case.name)
        or not re.fullmatch(r"(?:rollout-pptx|run)-[0-9a-f]{32}", physical_case.parent.name)
        or not output_dir.is_relative_to(physical_case)
    ):
        raise HarnessError("Codex bind sources must use opaque run and case directories")

    executable = Path(command[0]).resolve()
    runtime_root = _codex_runtime_root(executable)
    # A venv interpreter can be a symlink into a relocatable runtime such as
    # ``/install/bin/python``. Mount its actual base prefix, not an assumed
    # three-level ancestor, so the standard library remains visible in bwrap.
    python_runtime_collection = Path(sys.base_prefix).resolve()
    agent_run = output_dir.parent.parent if output_dir.parent.name == "turns" else output_dir.parent
    session_namespace = "generator"
    if not output_dir.name.startswith("generator-"):
        raise HarnessError(f"unexpected single-run output directory: {output_dir.name}")
    python_environment, python_purelib, python_purelib_alias = (
        _prepare_isolated_python_environment(agent_run, session_namespace)
    )
    agent_codex_home_alias = PurePosixPath(f"/codex-home-{session_namespace}")
    agent_home_alias = PurePosixPath(f"/home/agent-{session_namespace}")
    codex_home = agent_run / f".codex-image-only-{session_namespace}"
    sandbox_home = agent_run / f".agent-home-{session_namespace}"
    codex_home.mkdir(parents=True, exist_ok=True)
    sandbox_home.mkdir(parents=True, exist_ok=True)

    host_codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
    _write_isolated_codex_config(
        host_codex_home,
        codex_home / "config.toml",
        agent_home_alias=agent_home_alias,
    )
    host_auth = host_codex_home / "auth.json"
    auth_destination = codex_home / "auth.json"
    if not host_auth.is_file():
        raise HarnessError(
            "Codex image-only isolation requires CODEX_HOME/auth.json authentication"
        )
    auth_destination.touch(exist_ok=True)

    wrapped = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--cap-drop",
        "ALL",
        "--tmpfs",
        "/",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--symlink",
        "usr/sbin",
        "/sbin",
        "--ro-bind",
        "/etc",
        "/etc",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/runtime",
        "--dir",
        str(AGENT_NODE_RUNTIME_ALIAS),
        "--dir",
        str(AGENT_PYTHON_ENV_ALIAS),
        "--dir",
        str(AGENT_PYTHON_BASE_ALIAS),
        "--dir",
        str(AGENT_WORKSPACE_ALIAS),
        "--dir",
        str(AGENT_OUTPUT_ALIAS),
        "--dir",
        str(agent_codex_home_alias),
        "--dir",
        str(agent_home_alias),
    ]
    resolver = Path("/run/systemd/resolve/stub-resolv.conf")
    if resolver.is_file():
        wrapped.extend(["--ro-bind", str(resolver), str(resolver)])
    # /etc is mounted read-only, but WSL points resolv.conf outside /etc.
    # Include its resolved target so DNS works in the sparse filesystem.
    resolver_target = Path("/etc/resolv.conf").resolve()
    if resolver_target.is_file() and not resolver_target.is_relative_to("/etc"):
        wrapped.extend(["--ro-bind", str(resolver_target), str(resolver_target)])
    wrapped.extend(
        [
            "--ro-bind",
            str(runtime_root),
            str(AGENT_NODE_RUNTIME_ALIAS),
            "--ro-bind",
            str(python_environment),
            str(AGENT_PYTHON_ENV_ALIAS),
            "--ro-bind",
            str(python_purelib),
            str(python_purelib_alias),
        ]
    )
    if (
        python_runtime_collection != Path("/")
        and python_runtime_collection.is_dir()
        and not python_runtime_collection.is_relative_to("/usr")
    ):
        # Virtual-environment interpreters use an absolute symlink into this
        # installation collection. It contains runtimes only, never task data.
        wrapped.extend(
            [
                "--ro-bind",
                str(python_runtime_collection),
                str(AGENT_PYTHON_BASE_ALIAS),
            ]
        )
    wrapped.extend(
        [
            "--bind",
            str(workspace),
            str(AGENT_WORKSPACE_ALIAS),
            "--bind",
            str(output_dir),
            str(AGENT_OUTPUT_ALIAS),
            "--bind",
            str(codex_home),
            str(agent_codex_home_alias),
            "--bind",
            str(sandbox_home),
            str(agent_home_alias),
        ]
    )
    wrapped.extend(
        [
            "--ro-bind",
            str(host_auth.resolve()),
            str(agent_codex_home_alias / auth_destination.name),
        ]
    )
    path = os.pathsep.join(
        [
            str(AGENT_PYTHON_ENV_ALIAS / "bin"),
            str(AGENT_NODE_RUNTIME_ALIAS / "bin"),
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        ]
    )
    path_aliases = (
        (workspace_argument_root, AGENT_WORKSPACE_ALIAS),
        (workspace, AGENT_WORKSPACE_ALIAS),
        (output_argument_root, AGENT_OUTPUT_ALIAS),
        (output_dir, AGENT_OUTPUT_ALIAS),
        (runtime_root, AGENT_NODE_RUNTIME_ALIAS),
    )

    def isolated_argument(argument: str) -> str:
        for source, alias in path_aliases:
            source_text = str(source)
            if argument == source_text:
                return str(alias)
            if argument.startswith(source_text + os.sep):
                return str(alias / Path(argument).relative_to(source))
        return argument

    agent_command = [isolated_argument(argument) for argument in command]
    executable_alias = AGENT_NODE_RUNTIME_ALIAS / executable.relative_to(runtime_root)
    agent_command[0] = str(executable_alias)
    wrapped.extend(
        [
            "--chdir",
            str(AGENT_WORKSPACE_ALIAS),
            "--setenv",
            "HOME",
            str(agent_home_alias),
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "CODEX_HOME",
            str(agent_codex_home_alias),
            "--setenv",
            "PATH",
            path,
            "--setenv",
            "VIRTUAL_ENV",
            str(AGENT_PYTHON_ENV_ALIAS),
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "PWD",
            str(AGENT_WORKSPACE_ALIAS),
            "--",
            *agent_command,
        ]
    )
    isolation = {
        "filesystem": CODEX_FILESYSTEM_ISOLATION,
        "identity_contract": CODEX_IDENTITY_ISOLATION,
        "process_view": CODEX_PROCESS_ISOLATION,
        "environment_policy": CODEX_ENVIRONMENT_ISOLATION,
        "network": "shared for Codex API; model tools remain governed by Codex sandbox",
        "session_namespace": session_namespace,
        "identity_blind": True,
        "workspace_alias": str(AGENT_WORKSPACE_ALIAS),
        "workspace_access": "read-write",
        "output_alias": str(AGENT_OUTPUT_ALIAS),
        "codex_home_alias": str(agent_codex_home_alias),
        "sandbox_home_alias": str(agent_home_alias),
        "evaluator_sources_mounted": False,
        "readonly_runtime_mounts": [
            str(AGENT_NODE_RUNTIME_ALIAS),
            str(AGENT_PYTHON_ENV_ALIAS),
        ],
        "writable_mounts": [
            str(AGENT_OUTPUT_ALIAS),
            str(agent_codex_home_alias),
            str(agent_home_alias),
        ] + [str(AGENT_WORKSPACE_ALIAS)],
        "readonly_case_mounts": [],
    }
    environment = {
        "NO_COLOR": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": path,
        "VIRTUAL_ENV": str(AGENT_PYTHON_ENV_ALIAS),
        "PYTHONHOME": (
            str(AGENT_PYTHON_BASE_ALIAS)
            if not python_runtime_collection.is_relative_to("/usr")
            else str(python_runtime_collection)
        ),
        "HOME": str(agent_home_alias),
        "CODEX_HOME": str(agent_codex_home_alias),
        "TMPDIR": "/tmp",
        "PWD": str(AGENT_WORKSPACE_ALIAS),
        "LANG": "C.UTF-8",
    }
    return wrapped, agent_command, isolation, environment


def _claude_code_isolation_command(
    command: list[str],
    *,
    workspace: Path,
    output_dir: Path,
    agent_kind: str = "claude-code",
) -> tuple[list[str], list[str], dict[str, Any], dict[str, str]]:
    """Run Claude Code or OpenCode against only the opaque image-only workspace.

    The CLI needs only its explicitly allowlisted gateway variables in the main
    process and must not inherit the host filesystem, user configuration,
    project checkout, evaluator PDF, or unrelated credentials.
    """

    if agent_kind not in {"claude", "claude-code", "opencode"}:
        raise HarnessError(f"unsupported isolated CLI kind: {agent_kind}")
    display_name = "OpenCode" if agent_kind == "opencode" else "Claude Code"

    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise HarnessError(f"bubblewrap is required for the image-only {display_name} protocol")

    workspace_argument_root = workspace.absolute()
    output_argument_root = output_dir.absolute()
    workspace = workspace.resolve()
    output_dir = output_dir.resolve()
    if not workspace.is_dir() or not output_dir.is_dir():
        raise HarnessError(
            f"{display_name} isolation requires existing workspace and output directories"
        )
    physical_case = workspace.parent
    if (
        not re.fullmatch(r"case-\d{4}", physical_case.name)
        or not re.fullmatch(r"(?:rollout-pptx|run)-[0-9a-f]{32}", physical_case.parent.name)
        or not output_dir.is_relative_to(physical_case)
    ):
        raise HarnessError(f"{display_name} bind sources must use opaque run and case directories")

    executable = Path(command[0]).resolve()
    if agent_kind == "opencode":
        runtime_root = executable.parent.parent
        if not (runtime_root / "bin" / executable.name).is_file():
            raise HarnessError(f"cannot locate the OpenCode runtime for isolation: {executable}")
    else:
        runtime_root = _codex_runtime_root(executable)
    # Keep the base interpreter and standard library visible when the venv's
    # executable is a symlink into a relocatable runtime (for example /install).
    python_runtime_collection = Path(sys.base_prefix).resolve()
    agent_run = output_dir.parent.parent if output_dir.parent.name == "turns" else output_dir.parent
    session_namespace = "generator"
    if not output_dir.name.startswith("generator-"):
        raise HarnessError(f"unexpected single-run output directory: {output_dir.name}")
    python_environment, python_purelib, python_purelib_alias = (
        _prepare_isolated_python_environment(agent_run, session_namespace)
    )
    home_prefix = "opencode" if agent_kind == "opencode" else "claude"
    sandbox_prefix = "opencode" if agent_kind == "opencode" else "claude-code"
    agent_home_alias = PurePosixPath(f"/home/{home_prefix}-{session_namespace}")
    sandbox_home = agent_run / f".{sandbox_prefix}-image-only-{session_namespace}"
    sandbox_home.mkdir(parents=True, exist_ok=True)

    wrapped = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--cap-drop",
        "ALL",
        "--tmpfs",
        "/",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--symlink",
        "usr/sbin",
        "/sbin",
        "--ro-bind",
        "/etc",
        "/etc",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/runtime",
        "--dir",
        str(AGENT_NODE_RUNTIME_ALIAS),
        "--dir",
        str(AGENT_PYTHON_ENV_ALIAS),
        "--dir",
        str(AGENT_PYTHON_BASE_ALIAS),
        "--dir",
        str(AGENT_WORKSPACE_ALIAS),
        "--dir",
        str(AGENT_OUTPUT_ALIAS),
        "--dir",
        str(agent_home_alias),
    ]
    resolver = Path("/run/systemd/resolve/stub-resolv.conf")
    if resolver.is_file():
        wrapped.extend(["--ro-bind", str(resolver), str(resolver)])
    wrapped.extend(
        [
            "--ro-bind",
            str(runtime_root),
            str(AGENT_NODE_RUNTIME_ALIAS),
            "--ro-bind",
            str(python_environment),
            str(AGENT_PYTHON_ENV_ALIAS),
            "--ro-bind",
            str(python_purelib),
            str(python_purelib_alias),
        ]
    )
    if (
        python_runtime_collection != Path("/")
        and python_runtime_collection.is_dir()
        and not python_runtime_collection.is_relative_to("/usr")
    ):
        wrapped.extend(
            [
                "--ro-bind",
                str(python_runtime_collection),
                str(AGENT_PYTHON_BASE_ALIAS),
            ]
        )
    wrapped.extend(
        [
            "--bind",
            str(workspace),
            str(AGENT_WORKSPACE_ALIAS),
            "--bind",
            str(output_dir),
            str(AGENT_OUTPUT_ALIAS),
            "--bind",
            str(sandbox_home),
            str(agent_home_alias),
        ]
    )

    path = os.pathsep.join(
        [
            str(AGENT_PYTHON_ENV_ALIAS / "bin"),
            str(AGENT_NODE_RUNTIME_ALIAS / "bin"),
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        ]
    )
    path_aliases = (
        (workspace_argument_root, AGENT_WORKSPACE_ALIAS),
        (workspace, AGENT_WORKSPACE_ALIAS),
        (output_argument_root, AGENT_OUTPUT_ALIAS),
        (output_dir, AGENT_OUTPUT_ALIAS),
        (runtime_root, AGENT_NODE_RUNTIME_ALIAS),
    )

    def isolated_argument(argument: str) -> str:
        for source, alias in path_aliases:
            source_text = str(source)
            if argument == source_text:
                return str(alias)
            if argument.startswith(source_text + os.sep):
                return str(alias / Path(argument).relative_to(source))
        return argument

    agent_command = [isolated_argument(argument) for argument in command]
    executable_alias = AGENT_NODE_RUNTIME_ALIAS / executable.relative_to(runtime_root)
    agent_command[0] = str(executable_alias)
    wrapped.extend(["--chdir", str(AGENT_WORKSPACE_ALIAS), "--", *agent_command])

    environment = {
        "HOME": str(agent_home_alias),
        "LANG": "C.UTF-8",
        "NO_COLOR": "1",
        "PATH": path,
        "PWD": str(AGENT_WORKSPACE_ALIAS),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": "/tmp",
        "VIRTUAL_ENV": str(AGENT_PYTHON_ENV_ALIAS),
        "PYTHONHOME": (
            str(AGENT_PYTHON_BASE_ALIAS)
            if not python_runtime_collection.is_relative_to("/usr")
            else str(python_runtime_collection)
        ),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
        "DISABLE_TELEMETRY": "1",
    }
    claude_gateway_environment_names = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_CUSTOM_HEADERS",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "API_TIMEOUT_MS",
        "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
        "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
        "CLAUDE_STREAM_IDLE_TIMEOUT_MS",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
    )
    opencode_gateway_environment_names = (
        "OPENCODE_CONFIG_CONTENT",
        "OPENCODE_MODEL",
        "OPENAI_API_KEY",
        "KIMI_API_KEY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
    )
    gateway_environment_names = (
        opencode_gateway_environment_names
        if agent_kind == "opencode"
        else claude_gateway_environment_names
    )
    for name in gateway_environment_names:
        value = os.environ.get(name)
        if value:
            environment[name] = value
    if agent_kind == "opencode" and not environment.get("OPENCODE_CONFIG_CONTENT"):
        raise HarnessError(
            "OpenCode image-only isolation requires OPENCODE_CONFIG_CONTENT"
        )
    if (
        agent_kind != "opencode"
        and not environment.get("ANTHROPIC_API_KEY")
        and not environment.get("ANTHROPIC_AUTH_TOKEN")
    ):
        raise HarnessError(
            "Claude Code image-only isolation requires ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN"
        )

    filesystem_policy = (
        OPENCODE_FILESYSTEM_ISOLATION
        if agent_kind == "opencode"
        else CLAUDE_CODE_FILESYSTEM_ISOLATION
    )
    identity_policy = (
        OPENCODE_IDENTITY_ISOLATION
        if agent_kind == "opencode"
        else CLAUDE_CODE_IDENTITY_ISOLATION
    )
    process_policy = (
        OPENCODE_PROCESS_ISOLATION
        if agent_kind == "opencode"
        else CLAUDE_CODE_PROCESS_ISOLATION
    )
    environment_policy = (
        OPENCODE_ENVIRONMENT_ISOLATION
        if agent_kind == "opencode"
        else CLAUDE_CODE_ENVIRONMENT_ISOLATION
    )

    isolation = {
        "filesystem": filesystem_policy,
        "identity_contract": identity_policy,
        "process_view": process_policy,
        "environment_policy": environment_policy,
        "network": f"shared for the configured API and {display_name} tool process",
        "session_namespace": session_namespace,
        "identity_blind": True,
        "workspace_alias": str(AGENT_WORKSPACE_ALIAS),
        "workspace_access": "read-write",
        "output_alias": str(AGENT_OUTPUT_ALIAS),
        "sandbox_home_alias": str(agent_home_alias),
        "evaluator_sources_mounted": False,
        "forwarded_environment_names": sorted(
            name for name in gateway_environment_names if name in environment
        ),
        "readonly_runtime_mounts": [
            str(AGENT_NODE_RUNTIME_ALIAS),
            str(AGENT_PYTHON_ENV_ALIAS),
        ],
        "writable_mounts": [str(AGENT_OUTPUT_ALIAS), str(agent_home_alias), str(AGENT_WORKSPACE_ALIAS)],
        "readonly_case_mounts": [],
    }
    return wrapped, agent_command, isolation, environment


def _agent_version(spec: AgentSpec) -> str:
    """Return the installed agent version when it is available."""
    executable = _resolve_executable(spec.executable)
    result = subprocess.run(
        [executable, "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return (result.stdout or result.stderr).strip()


def invoke_agent(
    spec: AgentSpec,
    *,
    workspace: Path,
    prompt: str,
    images: list[Path],
    output_dir: Path,
    writable: bool,
    timeout_seconds: int,
    output_schema: dict[str, Any] | None = None,
    resume_session_id: str | None = None,
    persistent: bool = False,
) -> dict[str, Any]:
    """Run an agent command and collect its events and usage metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "prompt.md"
    events_path = output_dir / "events.jsonl"
    stderr_path = output_dir / "stderr.log"
    final_message = output_dir / "final-message.md"
    result_path = output_dir / "invocation.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    schema_path: Path | None = None
    if output_schema is not None and spec.kind == "codex":
        schema_path = output_dir / "output-schema.json"
        _write_json(schema_path, output_schema)

    effective_spec = AgentSpec(
        kind=spec.kind,
        executable=_resolve_executable(spec.executable),
        model=spec.model,
        reasoning_effort=spec.reasoning_effort,
        max_turns=spec.max_turns,
    )
    command = build_agent_command(
        effective_spec,
        workspace=workspace,
        prompt=prompt,
        images=images,
        final_message=final_message,
        writable=writable,
        output_schema=schema_path,
        resume_session_id=resume_session_id,
        persistent=persistent,
    )
    started = time.monotonic()
    timed_out = False
    timeout_reason: str | None = None
    isolation: dict[str, Any] = {"filesystem": "none"}
    execution_command = command
    if spec.kind == "codex":
        execution_command, agent_command, isolation, agent_environment = _codex_isolation_command(
            command,
            workspace=workspace,
            output_dir=output_dir,
        )
        logged_command = [
            "<prompt from prompt.md>" if part == prompt else part for part in agent_command
        ]
    else:
        execution_command, agent_command, isolation, agent_environment = (
            _claude_code_isolation_command(
                command,
                workspace=workspace,
                output_dir=output_dir,
                agent_kind=spec.kind,
            )
        )
        logged_command = ["<prompt from prompt.md>" if part == prompt else part for part in command]
    _write_json(output_dir / "command.json", logged_command)
    _write_json(output_dir / "isolation.json", isolation)
    with events_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            execution_command,
            cwd=Path("/"),
            env=agent_environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        first_event_timeout = float(os.environ.get("AGENT_FIRST_EVENT_TIMEOUT_SECONDS", "0") or 0)
        stream_idle_timeout = float(os.environ.get("AGENT_STREAM_IDLE_TIMEOUT_SECONDS", "0") or 0)
        last_activity = started
        last_size = 0
        while process.poll() is None:
            now = time.monotonic()
            current_size = events_path.stat().st_size + stderr_path.stat().st_size
            if current_size != last_size:
                last_size = current_size
                last_activity = now
            if timeout_seconds > 0 and now - started >= timeout_seconds:
                timeout_reason = "wall_clock"
                break
            if first_event_timeout > 0 and last_size == 0 and now - started >= first_event_timeout:
                timeout_reason = "first_event_idle"
                break
            if stream_idle_timeout > 0 and last_size > 0 and now - last_activity >= stream_idle_timeout:
                timeout_reason = "stream_idle"
                break
            time.sleep(1)
        if timeout_reason is None:
            returncode = process.wait()
        else:
            timed_out = True
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover - Windows execution path
                process.terminate()
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:  # pragma: no cover - Windows execution path
                    process.kill()
                returncode = process.wait()

    rows = _read_jsonl(events_path)
    diagnostics = _jsonl_diagnostics(events_path)
    if spec.kind == "codex":
        metadata = _codex_event_metadata(rows)
    elif spec.kind == "opencode":
        metadata = _opencode_event_metadata(rows)
    else:
        metadata = _claude_event_metadata(rows)
    if not final_message.exists() and metadata.get("final_message"):
        final_message.write_text(str(metadata["final_message"]), encoding="utf-8")
    result = {
        "agent": spec.kind,
        "model": spec.model,
        "reasoning_effort": spec.reasoning_effort,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_reason": timeout_reason,
        "duration_seconds": round(time.monotonic() - started, 3),
        "event_count": len(rows),
        "thread_id": metadata.get("thread_id", ""),
        "session_mode": "resume" if resume_session_id else "start",
        "resume_session_id": resume_session_id,
        "usage_semantics": "cumulative" if spec.kind == "codex" else "incremental",
        "usage": metadata.get("usage", {}),
        "provider_usage": metadata.get("provider_usage", {}),
        "total_cost_usd": metadata.get("total_cost_usd"),
        "num_turns": metadata.get("num_turns"),
        "effective_model": metadata.get("effective_model", spec.model),
        "model_usage": metadata.get("model_usage", {}),
        "terminal_reason": metadata.get("terminal_reason"),
        "prompt_path": str(prompt_path.resolve()),
        "events_path": str(events_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
        "final_message_path": str(final_message.resolve()),
        "image_paths": [str(path.resolve()) for path in images],
        "isolation": isolation,
        **diagnostics,
    }
    salvage_reason: str | None = None
    if (
        spec.kind in {"claude-code", "opencode"}
        and not diagnostics["json_parse_errors"]
        and (workspace / "reconstruction.pptx").is_file()
    ):
        if (
            spec.kind == "claude-code"
            and
            not timed_out
            # Claude Code exits 1 for its structured error_max_turns terminator.
            # That process status does not invalidate a deck already written to
            # the workspace; the artifact still has to pass the normal validators.
            and returncode in {0, 1}
            and metadata.get("terminal_reason") == "max_turns"
            and _only_claude_max_turns_errors(diagnostics["error_events"])
        ):
            salvage_reason = "max_turns"
        elif (
            spec.kind == "claude-code"
            and
            not timed_out
            and returncode != 0
            and metadata.get("terminal_reason") == "completed"
            and not diagnostics["error_events"]
        ):
            # Claude Code can occasionally emit an unambiguous successful
            # terminal result and a finished deck, then propagate exit code 1
            # from its outer process. Do not discard that provider-confirmed
            # artifact solely because the wrapper status disagrees; the deck
            # still has to pass all normal package, render, resource, and bounds
            # validators before the case can become complete.
            salvage_reason = "completed_nonzero_exit"
        elif timed_out and timeout_reason == "wall_clock" and not diagnostics["error_events"]:
            # The wall-clock deadline defines the end of an Agent attempt. If
            # the agent already wrote a deck, keep that exact deadline artifact
            # and let the normal package/render/bounds validators decide it.
            # Provider usage/cost may legitimately be absent because the CLI
            # never emitted its final result event.
            salvage_reason = "timeout"
    terminal_artifact_salvaged = salvage_reason is not None
    result["terminal_artifact_salvaged"] = terminal_artifact_salvaged
    result["terminal_artifact_salvage_reason"] = salvage_reason
    _write_json(result_path, result)
    if timed_out and not terminal_artifact_salvaged:
        raise HarnessError(
            f"{spec.kind} invocation timed out ({timeout_reason}) after "
            f"{round(time.monotonic() - started, 1)}s"
        )
    if diagnostics["json_parse_errors"]:
        raise HarnessError(f"{spec.kind} emitted malformed JSONL events")
    if diagnostics["error_events"] and not terminal_artifact_salvaged:
        event_record = diagnostics["error_events"][-1]
        event = event_record.get("event", {}) if isinstance(event_record, dict) else {}
        if not isinstance(event, dict):
            event = {}
        event_summary = {
            key: event.get(key)
            for key in (
                "type",
                "subtype",
                "is_error",
                "api_error_status",
                "terminal_reason",
                "result",
            )
            if event.get(key) is not None
        }
        serialized = json.dumps(event_summary, ensure_ascii=False)[-2000:]
        raise HarnessError(f"{spec.kind} emitted a fatal result event: {serialized}")
    if returncode and not terminal_artifact_salvaged:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise HarnessError(f"{spec.kind} invocation failed with code {returncode}: {tail}")
    if not result["thread_id"]:
        raise HarnessError(f"{spec.kind} did not report a session/thread ID")
    if resume_session_id and result["thread_id"] != resume_session_id:
        raise HarnessError(
            f"{spec.kind} resumed {resume_session_id} but reported {result['thread_id']}"
        )
    return result


def _append_bytes(source: Path, destination: Path) -> tuple[int, int]:
    """Append a source file to a destination file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    start = destination.stat().st_size if destination.exists() else 0
    payload = source.read_bytes()
    with destination.open("ab") as handle:
        handle.write(payload)
    return start, start + len(payload)


def _record_turn(
    agent_run: Path,
    *,
    role: str,
    turn_index: int,
    invocation: dict[str, Any],
) -> dict[str, Any]:
    """Persist one generator turn and its invocation evidence."""

    role_dir = agent_run / role
    aggregate_events = role_dir / "events.jsonl"
    aggregate_stderr = role_dir / "stderr.log"
    existing_events = _jsonl_diagnostics(aggregate_events)["event_count"]
    turn_events = Path(invocation["events_path"])
    turn_stderr = Path(invocation["stderr_path"])
    event_count = int(invocation["event_count"])
    events_byte_start, events_byte_end = _append_bytes(turn_events, aggregate_events)
    stderr_byte_start, stderr_byte_end = _append_bytes(turn_stderr, aggregate_stderr)
    prompt_path = Path(invocation["prompt_path"])
    final_message_path = Path(invocation["final_message_path"])
    return {
        "role": role,
        "turn": turn_index,
        "stage": f"G{turn_index}",
        "session_mode": invocation["session_mode"],
        "thread_id": invocation["thread_id"],
        "resume_session_id": invocation["resume_session_id"],
        "event_start_seq": existing_events,
        "event_end_seq": existing_events + event_count - 1,
        "events_byte_start": events_byte_start,
        "events_byte_end": events_byte_end,
        "stderr_byte_start": stderr_byte_start,
        "stderr_byte_end": stderr_byte_end,
        "prompt_sha256": _sha256(prompt_path),
        "final_message_sha256": (
            _sha256(final_message_path) if final_message_path.exists() else None
        ),
        "duration_seconds": invocation["duration_seconds"],
        "usage": invocation["usage"],
        "usage_semantics": invocation.get("usage_semantics", "cumulative"),
        "provider_usage": invocation.get("provider_usage", {}),
        "total_cost_usd": invocation.get("total_cost_usd"),
        "effective_model": invocation.get("effective_model", ""),
        "model_usage": invocation.get("model_usage", {}),
        "image_paths": list(invocation.get("image_paths") or []),
        "isolation": dict(invocation.get("isolation") or {}),
        "terminal_artifact_salvaged": bool(invocation.get("terminal_artifact_salvaged")),
        "terminal_artifact_salvage_reason": invocation.get(
            "terminal_artifact_salvage_reason"
        ),
        "invocation_path": str((prompt_path.parent / "invocation.json").resolve()),
        "turn_artifacts": str(prompt_path.parent.resolve()),
    }


def _workspace_input_state(workspace: Path) -> dict[str, Any]:
    """Describe the files mounted into an agent workspace."""
    state: dict[str, Any] = {}
    for name in ("reference.png",):
        path = workspace / name
        state[name] = {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": _sha256(path) if path.is_file() else None,
        }
    state["components.json_present"] = (workspace / "components.json").exists()
    state["resources"] = [
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted((workspace / "resources").glob("resource_*.png"))
        if path.is_file()
    ]
    return state


def _write_case_usage(
    agent_run: Path,
    invocation_paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Persist raw and incremental token accounting for every physical call."""

    ledger = ledger_from_invocation_paths(invocation_paths)
    summary = aggregate_usage_ledger(ledger)
    _write_json(agent_run / "token-ledger.json", ledger)
    _write_json(agent_run / "usage-summary.json", summary)
    return ledger, summary


def _generator_prompt(
    sample: dict[str, Any],
    include_caption: bool,
    agent_kind: str = "",
) -> str:
    """Build the reconstruction prompt for one task."""
    context = ""
    if include_caption and sample.get("description"):
        context = f"\nPaper caption (context only): {sample['description']}\n"
    resource_names = [Path(value).name for value in sample.get("resource_paths") or []]
    resource_list = "\n".join(f"  - resources/{name}" for name in resource_names)
    resource_context = (
        "\n- resources/: manually approved irreducible raster subfigures. Use every file "
        "exactly from this folder; do not redraw, trace, or replace these assets.\n"
        f"{resource_list}\n"
        if resource_names
        else "\n- No approved raster subfigures are provided for this task.\n"
    )
    pptx_resource_constraint = (
        "- Embed every approved resource as an exact, unmodified picture and use it visibly in its "
        "intended figure location. Do not copy reference.png (or any crop/near-copy) into the deck; "
        "the validator compares every other raster picture with the reference and rejects a high "
        "similarity match. Do not place compliance copies off-slide, at tiny size, hidden, "
        "transparent, or behind opaque shapes.\n"
        if resource_names
        else "- This task has no approved raster resources. Keep diagram content as native objects "
        "and do not embed reference.png, a crop, or a raster substitute for the diagram.\n"
    )
    return f"""You are the Generator in a scientific-figure-to-PowerPoint reconstruction benchmark.

Inputs in the current working directory:
- reference.png: the only visual target. Inspect this file directly with your image-capable file
  inspection tool before authoring the slide.
- /runtime/venv/bin/python: the local authoring runtime with python-pptx and Pillow.
{resource_context}
{context}
Create the target DIRECTLY as a one-slide PowerPoint deck and WRITE it to reconstruction.pptx.

Requirements:
- Use editable native PowerPoint objects for text, boxes, arrows, connectors, and other diagram parts.
- Reproduce every visible label, shape, connector, color, border, grouping, and alignment.
- You may choose any locally available authoring method. python-pptx is available, but it is not
  required; use whichever method gives the best editable PPTX result.
- Never crop, export, render, screenshot, or otherwise derive raster content from reference.png,
  in whole or in part. Do not use generated raster substitutes for diagram content.
- Do not embed external fonts, attachments, or other binary payloads. Use locally installed fonts
  and native PowerPoint content only.
{pptx_resource_constraint}- Keep all other content as editable native PowerPoint objects.
- Use exactly one slide and make it match the reference aspect ratio and composition.
- Keep every shape, picture, text box, and connector bounding box fully inside the slide canvas;
  no object may use negative coordinates or extend past any slide edge.
- Work autonomously. Inspect reference.png carefully, then create the deck.
- Do not merely describe the solution. Before finishing, open/parse the deck and verify that
  reconstruction.pptx exists and is valid.

Your final message should be a concise summary; reconstruction.pptx is the authoritative output."""


def _evaluator_source_path(sample: dict[str, Any]) -> Path:
    """Resolve the evaluator-only source document without exposing it to an agent."""

    raw_path = sample.get("source_pdf_path")
    if not raw_path:
        raise HarnessError(f"sample has no evaluator source: {sample['sample_id']}")
    path = Path(str(raw_path)).resolve()
    try:
        with path.open("rb") as handle:
            signature = handle.read(5)
    except OSError:
        signature = b""
    if signature != b"%PDF-":
        raise HarnessError(f"sample has an invalid evaluator source: {sample['sample_id']}")
    return path


def _prepare_workspace(sample: dict[str, Any], workspace: Path) -> Path:
    """Prepare the isolated workspace for one task."""
    workspace.mkdir(parents=True, exist_ok=True)
    existing_documents = sorted(workspace.rglob("*.pdf"))
    if existing_documents:
        raise HarnessError("agent workspace contains a forbidden evaluator document")
    reference = workspace / "reference.png"
    shutil.copy2(Path(sample["reference_path"]), reference)
    resource_paths = [Path(value) for value in sample.get("resource_paths") or []]
    if resource_paths:
        resources = workspace / "resources"
        resources.mkdir(parents=True, exist_ok=True)
        for source in resource_paths:
            if not source.is_file() or not re.fullmatch(r"resource_\d{4}\.png", source.name):
                raise HarnessError(f"invalid task resource: {source}")
            shutil.copy2(source, resources / source.name)
    return reference


def _write_agent_input_manifest(
    agent_run: Path,
    *,
    sample: dict[str, Any],
    workspace: Path,
    protocol: str,
) -> Path:
    """Persist an auditable list of the only task inputs mounted for agents."""

    visible_files = [workspace / "reference.png"]
    visible_files.extend(sorted((workspace / "resources").glob("resource_*.png")))
    manifest_path = agent_run / "agent-input-manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": "pptbench-agent-inputs-v2",
            "protocol": protocol,
            "sample_id": sample["sample_id"],
            "workspace": str(workspace.resolve()),
            "visible_task_inputs": [
                {
                    "path": str(path.relative_to(workspace)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in visible_files
            ],
            "evaluator_source_mounted": False,
        },
    )
    return manifest_path


def _agent_visible_component_report(
    report: dict[str, Any],
    *,
    workspace: Path,
) -> dict[str, Any]:
    """Return generated-deck diagnostics without evaluator host paths."""

    workspace = workspace.resolve()

    def sanitize(value: Any, *, field: str | None = None) -> Any:
        if isinstance(value, dict):
            return {str(key): sanitize(item, field=str(key)) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item, field=field) for item in value]
        if field == "text":
            return value
        if not isinstance(value, str):
            return value
        path = Path(value)
        if not path.is_absolute():
            return value
        try:
            relative = path.resolve().relative_to(workspace)
        except ValueError as exc:
            raise HarnessError(
                "agent-visible component report contains a host absolute path"
            ) from exc
        return str(PurePosixPath(str(AGENT_WORKSPACE_ALIAS)) / relative.as_posix())

    sanitized = sanitize(report)
    if not isinstance(sanitized, dict):  # pragma: no cover - type guard
        raise HarnessError("component report is not an object")
    sanitized["source_path"] = str(
        PurePosixPath(str(AGENT_WORKSPACE_ALIAS)) / "reconstruction.pptx"
    )
    return sanitized


def _record_iteration(
    *,
    workspace: Path,
    reference: Path,
    iteration_dir: Path,
    index: int,
    soffice_command: str | Path | None = None,
    pdf_component_manifest: Path | None = None,
    source_pdf_path: Path | None = None,
    approved_resource_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Persist one iteration's render and diagnostics.

    Args:
        source_pdf_path: Source PDF used when no component manifest is supplied.
    """
    pptx_source = workspace / "reconstruction.pptx"
    pptx_snapshot = iteration_dir / "reconstruction.pptx"
    screenshot = iteration_dir / "screenshot.png"
    components_path = iteration_dir / "components.json"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    report = validate_reconstruction_pptx(
        pptx_source,
        reference_path=reference,
        approved_resource_paths=approved_resource_paths,
        enforce_resource_integrity=False,
    )
    shutil.copy2(pptx_source, pptx_snapshot)
    render_pptx(pptx_snapshot, screenshot, soffice_command=soffice_command)
    _write_json(components_path, report)
    metrics = evaluate_pptx(
        reference,
        screenshot,
        report,
        pdf_component_manifest,
        source_pdf_path,
    )
    _write_json(iteration_dir / "metrics.json", metrics)
    # Only path-sanitized generated-deck diagnostics are visible to the next agent call.
    agent_visible_report = _agent_visible_component_report(report, workspace=workspace)
    _write_json(workspace / "current-pptx-components.json", agent_visible_report)
    return {
        "iteration": index,
        "target": "pptx",
        "pptx_path": str(pptx_snapshot.resolve()),
        "screenshot_path": str(screenshot.resolve()),
        "components_path": str(components_path.resolve()),
        "pptx_sha256": _sha256(pptx_snapshot),
        "screenshot_sha256": _sha256(screenshot),
        "components_sha256": _sha256(components_path),
        "metrics": metrics,
        "resource_integrity": report.get("resource_integrity", {}),
    }


def _finalize_case(
    case_dir: Path,
    sample: dict[str, Any],
    iterations: list[dict[str, Any]],
    *,
    agent_run: Path,
    turns: list[dict[str, Any]],
    input_state_before: dict[str, Any],
    generator: AgentSpec,
    started: float,
    stop_reason: str,
) -> dict[str, Any]:
    """Finalize one single-run case and persist its audit records."""

    if not iterations:
        raise HarnessError("case completed without an iteration")
    final_iteration = iterations[-1]
    single_iteration = iterations[0]
    best_iteration = max(iterations, key=lambda row: float(row["metrics"]["composite"]))
    single_dir = case_dir / "single-run"
    final_dir = case_dir / "final"
    single_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    for key, name in (
        ("pptx_path", "reconstruction.pptx"),
        ("screenshot_path", "screenshot.png"),
        ("components_path", "components.json"),
    ):
        shutil.copy2(Path(single_iteration[key]), single_dir / name)
        shutil.copy2(Path(final_iteration[key]), final_dir / name)

    input_state_after = _workspace_input_state(case_dir / "workspace")
    violations: list[str] = []
    if input_state_before != input_state_after:
        violations.append("immutable reference image or approved resources changed during the run")
    workspace_documents = sorted((case_dir / "workspace").rglob("*.pdf"))
    if workspace_documents:
        violations.append("agent workspace contains a forbidden document")
    if input_state_after["components.json_present"]:
        violations.append("components.json leaked into the agent workspace")

    if [(turn.get("role"), turn.get("turn")) for turn in turns] != [("generator", 0)]:
        violations.append("single-run must contain exactly one generator turn")
    if turns:
        turn = turns[0]
        if turn.get("session_mode") != "start":
            violations.append("single-run turn must start a new session")
        if not turn.get("thread_id"):
            violations.append("single-run turn has no session/thread id")

    stream_diagnostics: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    events = agent_run / "generator" / "events.jsonl"
    stderr = agent_run / "generator" / "stderr.log"
    if events.exists():
        diagnostics = _jsonl_diagnostics(events)
        stream_diagnostics["generator"] = diagnostics
        salvaged_max_turns = bool(
            any(turn.get("terminal_artifact_salvaged") for turn in turns)
            and _only_claude_max_turns_errors(diagnostics["error_events"])
        )
        stream_diagnostics["generator"]["salvaged_max_turns"] = salvaged_max_turns
        if diagnostics["json_parse_errors"]:
            violations.append("generator event stream contains malformed JSONL")
        if diagnostics["error_events"] and not salvaged_max_turns:
            violations.append("generator event stream contains fatal errors")
        artifacts["generator_events_sha256"] = _sha256(events)
        if stderr.exists():
            artifacts["generator_stderr_sha256"] = _sha256(stderr)

    single_run_resource_usage = dict(single_iteration.get("resource_integrity") or {})
    final_resource_usage = dict(final_iteration.get("resource_integrity") or {})
    violations.extend(
        f"single-run: {issue}" for issue in single_run_resource_usage.get("violations", [])
    )
    violations.extend(str(issue) for issue in final_resource_usage.get("violations", []))
    violations.extend(
        f"single-run: {issue}" for issue in check_deck(Path(single_iteration["pptx_path"]))
    )
    violations.extend(check_deck(Path(final_iteration["pptx_path"])))

    invocation_paths = [Path(str(turn["invocation_path"])) for turn in turns]
    token_ledger, usage_summary = _write_case_usage(agent_run, invocation_paths)
    if len(token_ledger) != len(turns):
        violations.append("token ledger does not contain one row per physical call")
    if not usage_summary["usage_complete"]:
        violations.append("one or more physical calls has incomplete token usage")
    for turn, usage_row in zip(turns, token_ledger, strict=True):
        turn["reported_usage"] = usage_row["reported_usage"]
        turn["incremental_usage"] = usage_row["incremental_usage"]
        turn["usage_missing"] = usage_row["usage_missing"]
        turn["incremental_usage_missing"] = usage_row["incremental_usage_missing"]

    integrity = {
        "input_state_before": input_state_before,
        "input_state_after": input_state_after,
        "immutable_inputs_unchanged": input_state_before == input_state_after,
        "components_json_visible_to_agents": input_state_after["components.json_present"],
        "workspace_pdf_count": len(workspace_documents),
        "stream_diagnostics": stream_diagnostics,
        "violations": violations,
        "usage_incomplete_after_terminal_artifact_salvage": False,
        "pptx_resource_usage": final_resource_usage,
        "single_run_pptx_resource_usage": single_run_resource_usage,
    }
    result = {
        "schema_version": 2,
        "protocol": IMAGE_ONLY_SINGLE_RUN_PROTOCOL,
        "status": "complete" if not violations else "invalid",
        "harness": "single-run",
        "target": "pptx",
        "sample_id": sample["sample_id"],
        "candidate_id": sample.get("candidate_id", ""),
        "generator": {
            "agent": generator.kind,
            "model": generator.model,
            "reasoning_effort": generator.reasoning_effort,
        },
        "started_at": datetime.fromtimestamp(
            time.time() - (time.monotonic() - started), UTC
        ).isoformat(),
        "finished_at": _utc_now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "stop_reason": stop_reason,
        "final_iteration": final_iteration["iteration"],
        "best_iteration": best_iteration["iteration"],
        "final_metrics": final_iteration["metrics"],
        "single_run_metrics": single_iteration["metrics"],
        "best_metrics": best_iteration["metrics"],
        "thread_ids": {"generator": turns[0].get("thread_id") if turns else None},
        "usage": usage_summary,
        "token_ledger_path": str((agent_run / "token-ledger.json").resolve()),
        "usage_summary_path": str((agent_run / "usage-summary.json").resolve()),
        "agent_input_manifest_path": str((agent_run / "agent-input-manifest.json").resolve()),
        "turns": turns,
        "versions": iterations,
        "single_run_pptx_path": str((single_dir / "reconstruction.pptx").resolve()),
        "single_run_screenshot_path": str((single_dir / "screenshot.png").resolve()),
        "single_run_components_path": str((single_dir / "components.json").resolve()),
        "final_screenshot_path": str((final_dir / "screenshot.png").resolve()),
        "final_pptx_path": str((final_dir / "reconstruction.pptx").resolve()),
        "final_components_path": str((final_dir / "components.json").resolve()),
        "integrity": integrity,
        "artifacts": artifacts,
    }
    result["run_path"] = str((agent_run / "run.json").resolve())
    _write_json(agent_run / "run.json", result)
    return result


def run_single_case(
    sample: dict[str, Any],
    *,
    case_dir: Path,
    agent: AgentSpec,
    timeout_seconds: int,
    include_caption: bool,
    soffice_command: str | Path | None = None,
) -> dict[str, Any]:
    """Run one reconstruction task and persist its result."""
    started = time.monotonic()
    workspace = case_dir / "workspace"
    evaluator_source = _evaluator_source_path(sample)
    reference = _prepare_workspace(sample, workspace)
    input_state_before = _workspace_input_state(workspace)
    approved_resource_paths = [
        Path(value).resolve() for value in sample.get("resource_paths") or []
    ]
    agent_run = case_dir / "agent_run"
    _write_agent_input_manifest(
        agent_run,
        sample=sample,
        workspace=workspace,
        protocol=IMAGE_ONLY_SINGLE_RUN_PROTOCOL,
    )
    round_dir = agent_run / "versions" / "00"
    turn_dir = agent_run / "turns" / "generator-00"
    invocation = invoke_agent(
        agent,
        workspace=workspace,
        prompt=_generator_prompt(sample, include_caption, agent.kind),
        images=[reference],
        output_dir=turn_dir,
        writable=True,
        timeout_seconds=timeout_seconds,
    )
    _ensure_pptx_reconstruction(workspace, approved_resource_paths, reference)
    iteration = _record_iteration(
        workspace=workspace,
        reference=reference,
        iteration_dir=round_dir,
        index=0,
        soffice_command=soffice_command,
        pdf_component_manifest=(
            Path(sample["component_manifest_path"])
            if sample.get("component_manifest_path")
            else None
        ),
        source_pdf_path=evaluator_source,
        approved_resource_paths=approved_resource_paths,
    )
    iteration["generator_turn"] = 0
    iteration["generator_invocation_path"] = str((turn_dir / "invocation.json").resolve())
    _write_json(round_dir / "iteration.json", iteration)
    turns = [
        _record_turn(
            agent_run,
            role="generator",
            turn_index=0,
            invocation=invocation,
        )
    ]
    return _finalize_case(
        case_dir,
        sample,
        [iteration],
        agent_run=agent_run,
        turns=turns,
        input_state_before=input_state_before,
        generator=agent,
        started=started,
        stop_reason="single-run-complete",
    )


def _select_samples(
    config: Config,
    selectors: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Select task samples from the configured manifest."""
    minimum_count = None
    if selectors:
        # Frozen task IDs are not numerically contiguous (curation removes
        # candidates), so a numeric suffix is not a manifest position.  Load
        # the complete manifest before filtering selectors; otherwise a request
        # such as task_0977 could silently stop before the selected row.
        try:
            minimum_count = sum(
                bool(line.strip())
                for line in config.benchmark.tasks_manifest.read_text(
                    encoding="utf-8-sig"
                ).splitlines()
            )
        except OSError as exc:
            raise HarnessError(f"cannot read the frozen task manifest: {exc}") from exc
    samples = load_samples(config.benchmark, minimum_count=minimum_count, selectors=selectors)
    if selectors:
        samples = [
            sample
            for sample in samples
            if any(
                selector == sample["sample_id"] or selector in sample["sample_id"]
                for selector in selectors
            )
        ]
        if not samples:
            raise HarnessError("no samples matched --sample selectors")
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise HarnessError("the finalized tasks manifest contains no benchmark tasks")
    return samples


def _score_clip(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    device: str,
    batch_size: int,
) -> None:
    """Optionally compute CLIP diagnostics for completed cases."""
    complete = [row for row in rows if row.get("status") == "complete"]
    pairs = [
        {
            "sample_id": row["sample_id"],
            "reference_path": str(
                Path(row["final_screenshot_path"]).parents[1] / "workspace" / "reference.png"
            ),
            "candidate_path": row["final_screenshot_path"],
        }
        for row in complete
    ]
    if not pairs:
        return
    encoder = TransformersClipEncoder(model_name, device=device)
    scores = score_pairs(pairs, encoder=encoder, batch_size=batch_size)
    by_sample = {score["sample_id"]: score for score in scores}
    for row in complete:
        score = by_sample[row["sample_id"]]
        row["clip"] = {
            "model": score["clip_model"],
            "cosine": score["clip_cosine"],
            "score": score["clip_score"],
        }
        _write_json(Path(row["run_path"]), row)


def run_harness(
    config: Config,
    *,
    harness: str,
    agent: AgentSpec,
    selectors: list[str] | None = None,
    limit: int | None = None,
    run_id: str | None = None,
    timeout_seconds: int = 900,
    include_caption: bool = False,
    with_clip: bool = False,
    clip_model: str = DEFAULT_CLIP_MODEL,
    clip_device: str = "auto",
    clip_batch_size: int = 16,
    soffice_command: str | Path | None = None,
) -> dict[str, Any]:
    """Run the single-run PPT reconstruction harness for selected tasks."""

    if harness != "single-run":
        raise ValueError(f"unsupported harness: {harness}")
    if include_caption:
        raise ValueError("image-only PPT runs do not allow caption input")
    if not agent.model or not agent.reasoning_effort:
        raise ValueError("agent requires explicit model and reasoning effort")

    samples = _select_samples(config, selectors, limit)
    pairing_root = config.benchmark.output_dir / "harnesses" / harness / agent.kind
    effective_run_id = _opaque_run_id(run_id or default_run_id())
    run_root = pairing_root / effective_run_id
    existing_run_config = _read_json_object(run_root / "run-config.json")
    if run_root.exists() and any(run_root.iterdir()) and not existing_run_config:
        raise HarnessError(f"run directory has no valid checkpoint config: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        "harness": harness,
        "target": "pptx",
        "protocol": IMAGE_ONLY_SINGLE_RUN_PROTOCOL,
        "run_id": effective_run_id,
        "created_at": _utc_now(),
        "samples": [sample["sample_id"] for sample in samples],
        "generator": {
            "kind": agent.kind,
            "executable": agent.executable,
            "model": agent.model,
            "reasoning_effort": agent.reasoning_effort,
            "max_turns": agent.max_turns,
            "version": _agent_version(agent),
        },
        "include_caption": include_caption,
        "with_clip": with_clip,
    }
    if existing_run_config:
        identity_fields = (
            "harness",
            "target",
            "protocol",
            "run_id",
            "samples",
            "include_caption",
        )
        if any(
            existing_run_config.get(field) != run_config.get(field) for field in identity_fields
        ):
            raise HarnessError(f"run checkpoint identity does not match: {run_root}")
        observed = existing_run_config.get("generator")
        expected = run_config["generator"]
        if not isinstance(observed, dict) or any(
            observed.get(field) != expected.get(field)
            for field in ("kind", "model", "reasoning_effort")
        ):
            raise HarnessError("run checkpoint generator config does not match")
        run_config["created_at"] = existing_run_config.get("created_at", run_config["created_at"])
        run_config["resumed_at"] = _utc_now()
    _write_json(run_root / "run-config.json", run_config)

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        print(
            f"[harness:{harness}] sample={index}/{len(samples)} {sample['sample_id']}",
            flush=True,
        )
        case_dir = run_root / sample["sample_id"]
        physical_case_dir = run_root / f"case-{index - 1:04d}"
        if case_dir.is_symlink():
            if case_dir.resolve() != physical_case_dir.resolve():
                raise HarnessError(f"case alias points outside its opaque storage: {case_dir}")
        elif case_dir.exists():
            raise HarnessError(f"image-only case directory is not an opaque alias: {case_dir}")
        else:
            physical_case_dir.mkdir(parents=True, exist_ok=True)
            case_dir.symlink_to(physical_case_dir.name, target_is_directory=True)
        try:
            row = run_single_case(
                sample,
                case_dir=case_dir,
                agent=agent,
                timeout_seconds=timeout_seconds,
                include_caption=include_caption,
                soffice_command=soffice_command,
            )
        except Exception as exc:  # keep a multi-case benchmark moving
            agent_run = case_dir / "agent_run"
            invocation_paths = [
                path
                for path in (agent_run / "turns" / "generator-00" / "invocation.json",)
                if path.is_file()
            ]
            usage_summary: dict[str, Any] | None = None
            if invocation_paths:
                _, usage_summary = _write_case_usage(agent_run, invocation_paths)
            row = {
                "status": "failed",
                "harness": harness,
                "target": "pptx",
                "protocol": run_config["protocol"],
                "sample_id": sample["sample_id"],
                "candidate_id": sample.get("candidate_id", ""),
                "error": repr(exc),
                "finished_at": _utc_now(),
                "usage": usage_summary,
                "token_ledger_path": (
                    str((agent_run / "token-ledger.json").resolve())
                    if usage_summary is not None
                    else None
                ),
                "usage_summary_path": (
                    str((agent_run / "usage-summary.json").resolve())
                    if usage_summary is not None
                    else None
                ),
            }
            _write_json(agent_run / "run.json", row)
            print(f"[harness:failed] {sample['sample_id']}: {exc}", flush=True)
        rows.append(row)

    if with_clip:
        _score_clip(
            rows,
            model_name=clip_model,
            device=clip_device,
            batch_size=clip_batch_size,
        )
    complete = [row for row in rows if row["status"] == "complete"]
    summary = {
        **run_config,
        "finished_at": _utc_now(),
        "run_root": str(run_root.resolve()),
        "requested_cases": len(samples),
        "complete_cases": len(complete),
        "failed_cases": len(rows) - len(complete),
        "mean_final_composite": (
            round(
                sum(float(row["final_metrics"]["composite"]) for row in complete) / len(complete),
                6,
            )
            if complete
            else None
        ),
        "mean_clip_score": (
            round(sum(float(row["clip"]["score"]) for row in complete) / len(complete), 4)
            if complete and all("clip" in row for row in complete)
            else None
        ),
        "cases": rows,
    }
    _write_json(run_root / "run.json", summary)
    _write_json(
        pairing_root / "latest-run.json",
        {"run_id": effective_run_id, "run_root": str(run_root.resolve())},
    )
    return summary
