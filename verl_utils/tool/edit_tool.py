from verl_utils.data.envs.WS import WorkSpace
from pathlib import Path
import tempfile
import subprocess
import os

SNIPPET_LINES: int = 4
MAX_RESPONSE_LEN: int = 16000

class EditTool():
    """Tool to replace content between lines in a file."""

    def __init__(self, root_dir, temp_dir, instance_id):
        self.instance_id = instance_id
        self.workspace = WorkSpace(root_dir, temp_dir, instance_id)
        self.uuid = self.workspace.uuid
        self.workspace_path = self.workspace.path

    def execute(self, path, start_line, end_line, new_str):
        # get absolute path from relative path
        short_path = path
        full_path_obj = Path(self.workspace_path) / path
        path = str(full_path_obj)

        if not full_path_obj.exists():
            if '/' not in short_path:
                return f"No edit was performed. The file '{short_path}' does not exist. Please use relative path (e.g., a/b/c.py) rather than simple file name (e.g., c.py)."
            return f"No edit was performed. The file '{short_path}' does not exist."
        if not full_path_obj.is_file():
            return f"No edit was performed. The path '{short_path}' points to a directory, not a file."

        file_content = self._read_file(path).expandtabs()
        new_str = new_str.expandtabs()

        # Check availability
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
        before_lines = file_lines[:start_line-1]  # lines before the replacement
        after_lines = file_lines[end_line:]       # lines after the replacement
        new_lines = new_str.split("\n")           # new content to insert

        # Combine all parts
        modified_lines = before_lines + new_lines + after_lines
        new_file_content = "\n".join(modified_lines)

        # Create a snippet of the edited section
        snippet_start = max(0, start_line - 1 - SNIPPET_LINES)
        snippet_end = start_line - 1 + len(new_lines) + SNIPPET_LINES
        snippet_lines = modified_lines[snippet_start:snippet_end]
        
        # Add line numbers and format the snippet
        numbered_snippet = []
        if snippet_lines:
            max_line_num = snippet_start + len(snippet_lines)
            max_line_num_length = len(str(max_line_num))

            for i, line in enumerate(snippet_lines, start=snippet_start + 1):
                numbered_snippet.append(f"{i:<{max_line_num_length}}\t{line}")
        
        snippet = "\n".join(numbered_snippet)

        # Check for linter errors before writing
        if path.endswith('.py'):
            # Create a temporary file with the new content
            original_file_name = os.path.basename(path)
            with tempfile.NamedTemporaryFile(mode='w', delete=True, prefix=original_file_name, suffix='.py') as temp:
                temp.write(new_file_content)
                temp.flush()
                
                # Run pylint to check for syntax errors
                command_options = f"pylint {temp.name} --disable=all --enable=E0001"
                result = subprocess.run(command_options, shell=True, capture_output=True, text=True)
                
                # Check if there are syntax errors
                if 'E0001' in result.stdout or 'E0001' in result.stderr:
                    error_msg = f"No edit was performed. This is because your edit would introduce syntax errors in {short_path}.\n"
                    error_msg += f"Specifically, this is a snapshot of your intended edit (lines {start_line}-{end_line} edited):\n{snippet}\n"
                    error_msg += f"However, Pylint finds the following syntax errors:\n{result.stdout + result.stderr}\n"
                    error_msg += "Thus, your edit has been aborted. Please fix these errors and retry."
                    return error_msg

        # Write the new content to the file
        self.write_file(path, new_file_content)
        
        if len(snippet) > MAX_RESPONSE_LEN:
            snippet = snippet[:MAX_RESPONSE_LEN] + "\n<response clipped>"

        # Prepare the success message
        success_msg = f"The file {short_path} has been edited (lines {start_line}-{end_line} replaced). "
        success_msg += f"Here's a snapshot of your edited snippet:\n{snippet}\n"
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return success_msg

    def _read_file(self, path):
        """Read the content of a file from a given path; raise a ToolError if an error occurs."""
        path = Path(path)
        try:
            return path.read_text()
        except Exception as e:
            print(str(e))
            return ""

    def write_file(self, path, file: str):
        """Write the content of a file to a given path; raise a ToolError if an error occurs."""
        path = Path(path)
        try:
            _ = path.write_text(file)
        except Exception as e:
            print(str(e))
            return