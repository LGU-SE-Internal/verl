import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl_utils.data.envs.WS import WorkSpace

SNIPPET_LINES: int = 4
MAX_RESPONSE_LEN: int = 16000


class EditTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.root_dir = self.config["root_dir"]
        self.temp_dir = self.config["temp_dir"]

        self.workspace_manager = None
        
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)


    async def create(self, instance_id: str, id: str, sha: str, **kwargs) -> str:
        if await self.workspace_manager.get(instance_id):
            await self.release(instance_id)

        workspace = WorkSpace(self.root_dir, self.temp_dir, id)

        await self.workspace_manager.register(instance_id, workspace)

        await asyncio.to_thread(workspace.create_ws, sha)

        return await super().create(instance_id, **kwargs)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        workspace = await self.workspace_manager.get(instance_id)
        if workspace is None:
            error_msg = f"No workspace found for instance_id '{instance_id}'. `create` must be called first."
            print(error_msg)
            return error_msg, 0.0, {}

        required_params = ["path", "start_line", "end_line", "new_str"]
        
        missing_params = [
            param for param in required_params if parameters.get(param) is None
        ]

        if missing_params:
            error_msg = f"No edit was performed. The following required arguments were not provided: {', '.join(missing_params)}. Please provide all required arguments and try again."
            return error_msg, 0.0, {}

        path = parameters.get("path")
        start_line = parameters.get("start_line")
        end_line = parameters.get("end_line")
        new_str = parameters.get("new_str")

        response = await asyncio.to_thread(
            self._blocking_execute,
            workspace.path,
            path,
            start_line,
            end_line,
            new_str
        )

        return response, 0.0, {}
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        workspace = await self.workspace_manager.unregister(instance_id)
        if workspace:
            await asyncio.to_thread(workspace.del_ws)
        await super().release(instance_id, **kwargs)

    def _blocking_execute(self, workspace_path: str, path: str, start_line: str, end_line: str, new_str: str) -> str:
        # get absolute path from relative path
        short_path = path
        full_path_obj = Path(workspace_path) / path
        path = str(full_path_obj)

        if not full_path_obj.exists():
            if '/' not in short_path:
                return f"No edit was performed. The file '{short_path}' does not exist. Please use relative path (e.g., a/b/c.py) rather than simple file name (e.g., c.py)."
            return f"No edit was performed. The file '{short_path}' does not exist."
        if not full_path_obj.is_file():
            return f"No edit was performed. The path '{short_path}' points to a directory, not a file."

        file_content = self._read_file(path).expandtabs()
        
        new_str = (new_str or "").expandtabs()

        if not path:
            return "No edit was performed. The required argument `path` is empty."

        # Validate line numbers
        try:
            start_line = int(start_line)
            end_line = int(end_line)
        except (ValueError, TypeError):
            return "No edit was performed. `start_line` and `end_line` must be integers."

        if start_line <= 0 or end_line <= 0:
            return "No edit was performed. Line numbers must be positive integers."
        if start_line > end_line:
            return "No edit was performed. `start_line` must be less than or equal to `end_line`."

        # Split file into lines
        file_lines = file_content.split("\n")
        total_lines = len(file_lines)

        # Check if line numbers are within file bounds
        if start_line > total_lines or end_line > total_lines:
            return f"No edit was performed. File only has {total_lines} lines."

        # Perform the replacement
        before_lines = file_lines[:start_line-1]
        after_lines = file_lines[end_line:]
        new_lines = new_str.split("\n")

        modified_lines = before_lines + new_lines + after_lines
        new_file_content = "\n".join(modified_lines)

        # Create a snippet of the edited section
        snippet_start = max(0, start_line - 1 - SNIPPET_LINES)
        snippet_end = start_line - 1 + len(new_lines) + SNIPPET_LINES
        snippet_lines = modified_lines[snippet_start:snippet_end]
        
        numbered_snippet = []
        if snippet_lines:
            max_line_num = snippet_start + len(snippet_lines)
            max_line_num_length = len(str(max_line_num))

            for i, line in enumerate(snippet_lines, start=snippet_start + 1):
                numbered_snippet.append(f"{i:<{max_line_num_length}}\t{line}")
        
        snippet = "\n".join(numbered_snippet)

        # Check for linter errors before writing
        if path.endswith('.py'):
            original_file_name = os.path.basename(path)
            with tempfile.NamedTemporaryFile(mode='w', delete=True, prefix=original_file_name, suffix='.py', dir=workspace_path) as temp:
                temp.write(new_file_content)
                temp.flush()
                
                command_options = f"pylint {temp.name} --disable=all --enable=E0001"
                result = subprocess.run(command_options, shell=True, capture_output=True, text=True)
                
                if 'E0001' in result.stdout or 'E0001' in result.stderr:
                    error_msg = f"No edit was performed. This is because your edit would introduce syntax errors in {short_path}.\n"
                    error_msg += f"Specifically, this is a snapshot of your intended edit (lines {start_line}-{end_line} edited):\n{snippet}\n"
                    error_msg += f"However, Pylint finds the following syntax errors:\n{result.stdout + result.stderr}\n"
                    error_msg += "Thus, your edit has been aborted. Please fix these errors and retry."
                    return error_msg

        self._write_file(path, new_file_content)
        
        if len(snippet) > MAX_RESPONSE_LEN:
            snippet = snippet[:MAX_RESPONSE_LEN] + "\n<response clipped>"

        success_msg = f"The file {short_path} has been edited (lines {start_line}-{end_line} replaced). "
        success_msg += f"Here's a snapshot of your edited snippet:\n{snippet}\n"
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return success_msg

    def _read_file(self, path: str) -> str:
        """Reads the content of a file."""
        try:
            return Path(path).read_text()
        except Exception as e:
            print(f"Error reading file {path}: {str(e)}")
            return ""

    def _write_file(self, path: str, content: str) -> None:
        """Writes content to a file."""
        try:
            Path(path).write_text(content)
        except Exception as e:
            print(f"Error writing to file {path}: {str(e)}")