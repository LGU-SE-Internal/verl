"""ARL-backed tools for R2E-Gym scaffold.

These tools execute inside K8s sandbox pods via arl-env ManagedSession.
They implement the verl BaseTool interface with OpenAI function calling.
"""

import asyncio
import base64
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("ARL_TOOL_LOG_LEVEL", "ERROR"))

TOOLS_DIR = os.path.join(os.path.dirname(__file__), "r2egym_tools")

# Matches R2E-Gym's DOCKER_PATH
DOCKER_PATH = (
    "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin"
    ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)

CMD_TIMEOUT = 120  # seconds per action
_TRUNCATION_LINES = 40

BLOCKED_COMMANDS = frozenset(["git", "ipython", "jupyter", "nohup"])

TOOL_FILES = [
    os.path.join(TOOLS_DIR, "file_editor.py"),
    os.path.join(TOOLS_DIR, "search.py"),
]


def _mirror_image(docker_image: str) -> str:
    """Rewrite docker image to mirror registry."""
    registry = os.environ.get(
        "ARL_MIRROR_REGISTRY", "pair-diag-cn-guangzhou.cr.volces.com"
    )
    if not registry:
        return docker_image
    namespace = os.environ.get("ARL_MIRROR_NAMESPACE", "code")
    parts = docker_image.split("/", 1)
    image_path = parts[1] if len(parts) == 2 else docker_image
    return f"{registry}/{namespace}/{image_path}"


def _format_observation(output: str, error_code: str, action_name: str) -> str:
    """Format command output with truncation for bash commands."""
    if action_name in ("execute_bash", "bash"):
        lines = output.splitlines() if output else []
        if len(lines) > 2 * _TRUNCATION_LINES:
            top = "\n".join(lines[:_TRUNCATION_LINES])
            bottom = "\n".join(lines[-_TRUNCATION_LINES:])
            divider = "-" * 50
            output = (
                f"{top}\n"
                f"{divider}\n"
                f"<Observation truncated in middle for saving context>\n"
                f"{divider}\n"
                f"{bottom}"
            )
        return f"Exit code: {error_code}\nExecution output of [{action_name}]:\n{output}"
    return f"Execution output of [{action_name}]:\n{output}"


@dataclass
class SessionInfo:
    """Holds an ARL session and its metadata."""
    session: Any  # ManagedSession
    docker_image: str
    swebench_verified: bool
    repo_path: str
    alt_path: str
    extra_info: dict


class ArlSessionManager:
    """Thread-safe registry for ARL sessions, shared across tools."""

    def __init__(self):
        self._sessions: Dict[str, SessionInfo] = {}
        self._lock = asyncio.Lock()

    async def register(self, instance_id: str, info: SessionInfo):
        async with self._lock:
            self._sessions[instance_id] = info

    async def get(self, instance_id: str) -> Optional[SessionInfo]:
        async with self._lock:
            return self._sessions.get(instance_id)

    async def unregister(self, instance_id: str) -> Optional[SessionInfo]:
        async with self._lock:
            return self._sessions.pop(instance_id, None)


class _ArlToolBase(BaseTool):
    """Base class for ARL tools — provides shared command execution."""

    session_manager: ArlSessionManager = None  # set by ToolAgentLoop init

    def _execute_in_session(
        self, session_info: SessionInfo, cmd: str,
        timeout: int = CMD_TIMEOUT, workdir: str = None,
    ) -> Tuple[str, str, int]:
        """Execute a command in the ARL sandbox. Returns (stdout, stderr, exit_code)."""
        session = session_info.session
        workdir = workdir or session_info.repo_path

        if session_info.swebench_verified:
            shell_cmd = (
                f"source /opt/miniconda3/bin/activate && "
                f"conda activate testbed && "
                f"{cmd}"
            )
            step = {
                "name": "tool_cmd",
                "command": ["bash", "-c", shell_cmd],
                "workDir": workdir,
                "timeout": timeout,
            }
        else:
            step = {
                "name": "tool_cmd",
                "command": ["bash", "-c", cmd],
                "env": {"PATH": DOCKER_PATH},
                "workDir": workdir,
                "timeout": timeout,
            }

        response = session.execute(steps=[step])
        result = response.results[0]
        stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", result.output.stdout or "")
        stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", result.output.stderr or "")
        return stdout, stderr, result.output.exit_code

    def _run(
        self, session_info: SessionInfo, cmd: str,
        timeout: int = CMD_TIMEOUT, workdir: str = None,
    ) -> Tuple[str, str]:
        """Execute command, return (output, error_code_str)."""
        stdout, stderr, exit_code = self._execute_in_session(
            session_info, cmd, timeout, workdir
        )
        output = stdout
        if stderr:
            output = output + "\n" + stderr if output else stderr
        if exit_code == 124:
            return f"The command took too long to execute (>{timeout}s)", "-1"
        if exit_code != 0:
            return output, f"Error: Exit code {exit_code}"
        return output, str(exit_code)

    def _copy_to_sandbox(self, session_info: SessionInfo, src_path: str, dest_path: str):
        """Copy a local file into the sandbox via base64 encoding."""
        with open(src_path, "rb") as f:
            content = f.read()
        b64 = base64.b64encode(content).decode()
        dir_path = os.path.dirname(dest_path)
        chunk_size = 65536
        if len(b64) <= chunk_size:
            self._run(
                session_info,
                f"mkdir -p {dir_path} && printf '%s' '{b64}' | base64 -d > {dest_path}",
            )
        else:
            self._run(session_info, f"mkdir -p {dir_path} && : > {dest_path}")
            for i in range(0, len(b64), chunk_size):
                chunk = b64[i : i + chunk_size]
                self._run(
                    session_info,
                    f"printf '%s' '{chunk}' | base64 -d >> {dest_path}",
                )

    def _setup_env(self, session_info: SessionInfo):
        """Initialize the sandbox environment."""
        if session_info.swebench_verified:
            self._run(session_info, "chmod +x /run_tests.sh")
            self._run(session_info, "ln -sf /opt/miniconda3/envs/testbed /root/.venv")
            self._run(session_info, "/root/.venv/bin/python -m pip install chardet")
        else:
            repo_path = session_info.repo_path
            alt_path = session_info.alt_path
            self._run(session_info, f"ln -sf {repo_path}/.venv {alt_path}/.venv")
            self._run(
                session_info,
                f"ln -sf {repo_path}/.venv/bin/python {alt_path}/.local/bin/python",
            )
            self._run(
                session_info,
                f"ln -sf {repo_path}/.venv/bin/python {alt_path}/.local/bin/python3",
            )
            self._run(
                session_info,
                f"find {repo_path}/.venv/bin -type f -executable "
                f"-exec ln -sf {{}} {alt_path}/.local/bin/ \\;",
            )
            self._run(session_info, "uv pip install chardet")
            self._run(session_info, "find . -name '*.pyc' -delete")
            self._run(session_info, "find . -name '__pycache__' -exec rm -rf {} +")
            self._run(session_info, "find /r2e_tests -name '*.pyc' -delete")
            self._run(session_info, "find /r2e_tests -name '__pycache__' -exec rm -rf {} +")
            # Move run_tests.sh out of testbed
            self._run(
                session_info,
                f"mv {repo_path}/run_tests.sh {alt_path}/run_tests.sh",
            )
            self._run(
                session_info,
                f"mv /r2e_tests {alt_path}/r2e_tests",
            )
            self._run(
                session_info,
                f"ln -s {alt_path}/r2e_tests {repo_path}/r2e_tests",
            )

    def _provision_tools(self, session_info: SessionInfo):
        """Copy tool scripts into sandbox and make them executable."""
        for tool_file in TOOL_FILES:
            cmd_name = os.path.basename(tool_file)
            if cmd_name.endswith(".py"):
                container_cmd_name = cmd_name[:-3]
            else:
                container_cmd_name = cmd_name
            container_path = f"/usr/local/bin/{container_cmd_name}"
            self._copy_to_sandbox(session_info, tool_file, container_path)
            self._run(session_info, f"chmod +x {container_path}")


class ArlFileEditor(_ArlToolBase):
    """R2E-Gym file_editor tool backed by ARL sandbox.

    This is the primary tool — it owns the ARL session lifecycle.
    On create(), it provisions the pod and sets up the environment.
    On release(), it tears down the pod.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.gateway_url = config.get("gateway_url") or os.environ.get(
            "ARL_GATEWAY_URL", "http://localhost:8080"
        )
        self.namespace = config.get("namespace", "default")
        self.step_timeout = config.get("step_timeout", 90)
        self.reward_timeout = config.get("reward_timeout", 300)
        self.experiment_id = config.get("experiment_id") or os.environ.get(
            "ARL_EXPERIMENT_ID", "default"
        )

    async def create(
        self, instance_id: str, docker_image: str = "", **kwargs
    ) -> Tuple[str, ToolResponse]:
        from arl import ManagedSession

        if not docker_image:
            logger.error(f"[ArlFileEditor.create] docker_image is empty for {instance_id}")
            return instance_id, ToolResponse(
                text="Error: docker_image is required for ARL file_editor"
            )

        image = _mirror_image(docker_image)
        swebench_verified = "swebench" in docker_image
        repo_path = "/testbed"
        alt_path = "/" if swebench_verified else "/root"

        session = ManagedSession(
            image=image,
            experiment_id=self.experiment_id,
            namespace=self.namespace,
            gateway_url=self.gateway_url,
            timeout=max(self.reward_timeout, self.step_timeout) + 60,
        )

        # Create pod in thread pool (blocking I/O)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, session.create_sandbox)

        info = SessionInfo(
            session=session,
            docker_image=docker_image,
            swebench_verified=swebench_verified,
            repo_path=repo_path,
            alt_path=alt_path,
            extra_info=kwargs,
        )

        # Setup env and provision tools (blocking I/O)
        await loop.run_in_executor(None, self._setup_env, info)
        await loop.run_in_executor(None, self._provision_tools, info)

        await self.session_manager.register(instance_id, info)

        return instance_id, ToolResponse(text="Environment ready.")

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[ToolResponse, float, dict]:
        info = await self.session_manager.get(instance_id)
        if info is None:
            return ToolResponse(text="Error: no ARL session found"), 0.0, {}

        # Convert OpenAI function call params to bash command
        # file_editor <command> --path <path> [--old_str <old_str>] [--new_str <new_str>] ...
        command = parameters.get("command", "view")
        args = [f"file_editor {command}"]
        for key in ["path", "old_str", "new_str", "file_text", "insert_line", "view_range",
                     "search_term", "enable_linting", "concise"]:
            val = parameters.get(key)
            if val is not None:
                if isinstance(val, bool):
                    args.append(f"--{key} {'true' if val else 'false'}")
                elif isinstance(val, list):
                    args.append(f"--{key} '{val}'")
                else:
                    # Escape single quotes in value
                    val_str = str(val).replace("'", "'\"'\"'")
                    args.append(f"--{key} '{val_str}'")

        bash_cmd = " ".join(args)
        loop = asyncio.get_event_loop()
        output, error_code = await loop.run_in_executor(
            None, self._run, info, bash_cmd, self.step_timeout
        )
        obs = _format_observation(output, error_code, "file_editor")
        return ToolResponse(text=obs), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        info = await self.session_manager.unregister(instance_id)
        if info and info.session:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, info.session.delete_sandbox)
            except Exception as e:
                logger.warning(f"delete_sandbox failed: {e}")
            try:
                await loop.run_in_executor(None, info.session.close)
            except Exception as e:
                logger.warning(f"session.close failed: {e}")


class ArlSearch(_ArlToolBase):
    """R2E-Gym search tool backed by ARL sandbox.

    Shares session with ArlFileEditor via ArlSessionManager.
    """

    async def create(self, instance_id: str, **kwargs) -> Tuple[str, ToolResponse]:
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[ToolResponse, float, dict]:
        info = await self.session_manager.get(instance_id)
        if info is None:
            return ToolResponse(text="Error: no ARL session found"), 0.0, {}

        search_term = parameters.get("search_term", "")
        path = parameters.get("path", ".")
        # Escape single quotes
        search_term_esc = search_term.replace("'", "'\"'\"'")
        path_esc = path.replace("'", "'\"'\"'")
        bash_cmd = f"search --search_term '{search_term_esc}' --path '{path_esc}'"

        loop = asyncio.get_event_loop()
        output, error_code = await loop.run_in_executor(
            None, self._run, info, bash_cmd, CMD_TIMEOUT
        )
        obs = _format_observation(output, error_code, "search")
        return ToolResponse(text=obs), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        pass  # Session owned by ArlFileEditor


class ArlBashExec(_ArlToolBase):
    """Execute bash commands in ARL sandbox.

    Shares session with ArlFileEditor via ArlSessionManager.
    """

    async def create(self, instance_id: str, **kwargs) -> Tuple[str, ToolResponse]:
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[ToolResponse, float, dict]:
        info = await self.session_manager.get(instance_id)
        if info is None:
            return ToolResponse(text="Error: no ARL session found"), 0.0, {}

        cmd = parameters.get("command", "")
        first_token = cmd.strip().split()[0] if cmd.strip() else ""

        if first_token in BLOCKED_COMMANDS:
            output = (
                f"Bash command '{first_token}' is not allowed. "
                "Please use a different command or tool."
            )
            error_code = "Error: Exit code 1"
        else:
            loop = asyncio.get_event_loop()
            stdout, stderr, exit_code = await loop.run_in_executor(
                None, self._execute_in_session, info, cmd, CMD_TIMEOUT
            )
            if exit_code == 124:
                output = f"The command took too long to execute (>{CMD_TIMEOUT}s)"
                error_code = "-1"
            elif exit_code != 0:
                output = (
                    f"Error executing command:\n\n"
                    f"[STDOUT]\n\n{stdout.strip()}\n\n"
                    f"[STDERR]\n\n{stderr.strip()}"
                )
                error_code = f"Error: Exit code {exit_code}"
            else:
                output = f"[STDOUT]\n\n{stdout.strip()}\n\n[STDERR]\n\n{stderr.strip()}"
                error_code = str(exit_code)

        obs = _format_observation(output, error_code, "execute_bash")
        return ToolResponse(text=obs), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        pass  # Session owned by ArlFileEditor


class ArlFinish(_ArlToolBase):
    """Finish tool — generates patch diff from ARL sandbox and signals termination.

    The patch is wrapped in [PATCH]...[/PATCH] tags so that both:
    - RM-based reward (extract_patch) can extract it from the solution string
    - ARL testing reward can apply it in a fresh pod
    """

    async def create(self, instance_id: str, **kwargs) -> Tuple[str, ToolResponse]:
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[ToolResponse, float, dict]:
        info = await self.session_manager.get(instance_id)
        if info is None:
            return ToolResponse(text="[PATCH]\n\n[/PATCH]"), 0.0, {}

        # Generate diff from the sandbox
        loop = asyncio.get_event_loop()
        # Use git diff if git is available, otherwise diff against original
        diff_output, _ = await loop.run_in_executor(
            None,
            self._run,
            info,
            "cd /testbed && git diff",
            30,
        )

        patch = diff_output if diff_output else ""
        response_text = f"[PATCH]\n{patch}\n[/PATCH]"
        return ToolResponse(text=response_text), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        pass  # Session owned by ArlFileEditor
