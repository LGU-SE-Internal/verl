import asyncio
import sqlite3
from typing import Any, Dict, Optional, Tuple

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema

MAX_RESPONSE_LEN: int = 16000


class SearchTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.root_dir = self.config["root_dir"]
        
        self.db_paths: Dict[str, str] = {}

        self.workspace_manager = None

    async def create(self, instance_id: str, id: str, sha: str, **kwargs) -> str:
        db_path = f'{self.root_dir}/codegraph/{id}.db'
        self.db_paths[instance_id] = db_path
        return await super().create(instance_id, **kwargs)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        db_path = self.db_paths.get(instance_id)
        if not db_path:
            error_msg = f"No database path found for instance_id '{instance_id}'. `create` must be called first."
            print(error_msg)
            return error_msg, 0.0, {}

        required_params = ["construct", "entity"]
        
        missing_params = [
            param for param in required_params if parameters.get(param) is None
        ]

        if missing_params:
            error_msg = f"No search was performed. The following required arguments were not provided: {', '.join(missing_params)}. Please provide all required arguments and try again."
            return error_msg, 0.0, {}

        construct = parameters.get("construct")
        entity = parameters.get("entity")

        response = await asyncio.to_thread(
            self._blocking_execute,
            db_path,
            construct,
            entity
        )
        
        return response, 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self.db_paths:
            del self.db_paths[instance_id]
        await super().release(instance_id, **kwargs)

    def _blocking_execute(self, db_path: str, construct: str, entity: str) -> str:
        # Check availability
        if not construct:
            return "`construct` argument is empty. Please ensure the function calling format is valid."
        elif not entity:
            return "`entity` argument is empty. Please ensure the function calling format is valid."
        
        try:
            with sqlite3.connect(db_path) as db_connection:
                match construct:
                    case "directory":
                        return self._search_directory(db_connection, entity)
                    case "file":
                        return self._search_file(db_connection, entity)
                    case "function":
                        return self._search_function(db_connection, entity)
                    case "class":
                        return self._search_class(db_connection, entity)
                    case "class_method":
                        return self._search_class_method(db_connection, entity)
                    case _:
                        return f"Invalid program construct: {construct}"
        except sqlite3.OperationalError as e:
            return f"Database error: {e}. The database file at '{db_path}' might be missing or corrupted."
        except Exception as e:
            return f"An unexpected error occurred during search: {e}"

    def _add_line_numbers(self, code: str, start_line: int) -> str:
        if not code:
            return ""
        lines = code.split('\n')
        numbered_lines = []
        max_line_num_length = len(str(start_line + len(lines) - 1))
        for i, line in enumerate(lines):
            line_num = start_line + i
            numbered_line = f"{line_num:<{max_line_num_length}}\t{line}"
            numbered_lines.append(numbered_line)
        return '\n'.join(numbered_lines)

    def _search_function(self, db_connection: sqlite3.Connection, entity: str) -> str:
        entries = db_connection.execute(
            "SELECT file_path, start_line, end_line, body FROM functions WHERE name = ?", (entity,)
        ).fetchall()
        if not entries:
            msg = f"No function named `{entity}` found."
            if '.' in entity:
                return f"{msg} Please use **simple name** rather than full qualified name."
            return f"{msg} Maybe you are looking for a class method or a class?"
        output = ""
        for entry in entries:
            file_path, start_line, end_line, body = entry
            numbered_body = self._add_line_numbers(body, start_line)
            output += f"{file_path}:{start_line}-{end_line}\n{numbered_body}\n\n"
            if len(output) > MAX_RESPONSE_LEN:
                output = output[:MAX_RESPONSE_LEN] + "\n<response clipped>"
                break
        return output

    def _search_class(self, db_connection: sqlite3.Connection, entity: str) -> str:
        entries = db_connection.execute(
            "SELECT file_path, start_line, end_line, fields, methods, body FROM classes WHERE name = ?", (entity,)
        ).fetchall()
        if not entries:
            msg = f"No class named `{entity}` found."
            if '.' in entity:
                return f"{msg} Please use **simple name** rather than full qualified name."
            return f"{msg} Maybe you are looking for a function or a class method?"
        output = ""
        for entry in entries:
            file_path, start_line, end_line, fields, methods, body = entry
            numbered_body = self._add_line_numbers(body, start_line)
            output += f"{file_path}:{start_line}-{end_line}\n"
            output += f"Fields:\n{fields}\n" if fields else "Fields: None\n"
            output += f"Methods:\n{methods}\n" if methods else "Methods: None\n"
            output += f"Class body:\n{numbered_body}\n\n"
            if len(output) > MAX_RESPONSE_LEN:
                output = output[:MAX_RESPONSE_LEN] + "\n<response clipped>"
                break
        return output

    def _search_class_method(self, db_connection: sqlite3.Connection, entity: str) -> str:
        entries = db_connection.execute(
            "SELECT file_path, start_line, end_line, body, class_name FROM class_methods WHERE name = ?", (entity,)
        ).fetchall()
        if not entries:
            msg = f"No class method named `{entity}` found."
            if '.' in entity:
                return f"{msg} Please use **simple name** rather than full qualified name."
            return f"{msg} Maybe you are looking for a function or a class?"
        output = ""
        for entry in entries:
            file_path, start_line, end_line, body, class_name = entry
            numbered_body = self._add_line_numbers(body, start_line)
            output += f"{file_path}:{start_line}-{end_line} Within class {class_name}\n{numbered_body}\n\n"
            if len(output) > MAX_RESPONSE_LEN:
                output = output[:MAX_RESPONSE_LEN] + "\n<response clipped>"
                break
        return output

    def _search_directory(self, db_connection: sqlite3.Connection, entity: str) -> str:
        if entity != '.':
            directory_path = entity.rstrip('/') + '/'
        else:
            directory_path = '.'
        entries = db_connection.execute(
            "SELECT path, files FROM directories WHERE path = ?", (directory_path,)
        ).fetchall()
        if not entries:
            msg = f"No directory named `{directory_path}` found."
            if entity.startswith('/'):
                return f"{msg} Please use relative path of a directory."
            return f"{msg} Maybe you should first check the directory's relative path to ensure the file exists?"
        entry = entries[0]
        output = f"{entry[0]} contains following files:\n{entry[1]}"
        if len(output) > MAX_RESPONSE_LEN:
            output = output[:MAX_RESPONSE_LEN] + "\n<response clipped>"
        return output

    def _search_file(self, db_connection: sqlite3.Connection, entity: str) -> str:
        entries = db_connection.execute(
            "SELECT path, entities FROM files WHERE path = ?", (entity,)
        ).fetchall()
        if not entries:
            msg = f"No file named `{entity}` found."
            if entity.startswith('/'):
                return f"{msg} Please use relative path (e.g., a/b/c.py) rather than absolute path (e.g., /root/a/b/c.py)."
            elif '/' not in entity:
                return f"{msg} Please use relative path (e.g., a/b/c.py) rather than a simple file name (e.g., c.py)."
            return f"{msg} Maybe you should first check from the root directory `.` (top-down) to ensure the directory exists?"
        entry = entries[0]
        full_name_list = entry[1].split('\n')
        functions, classes, methods = set(), set(), set()
        for name in full_name_list:
            if '.' in name:
                parts = name.split('.')
                methods.add(parts[-1])
                classes.add(parts[0])
            else:
                functions.add(name)
        functions = functions - classes
        functions_str = '\n'.join(sorted(list(functions))) if functions else '(No functions)'
        classes_str = '\n'.join(sorted(list(classes))) if classes else '(No classes)'
        methods_str = '\n'.join(sorted(list(methods))) if methods else '(No class methods)'
        names = f"# Function:\n{functions_str}\n\n# Class:\n{classes_str}\n\n# Class Method:\n{methods_str}"
        output = f"{entry[0]} contains following entries:\n{names}"
        if len(output) > MAX_RESPONSE_LEN:
            output = output[:MAX_RESPONSE_LEN] + "\n<response clipped>"
        output += "\n\nNote: The `file` construct shows only the skeleton structure of a file entity. To view the source code, please search for an entity in `function`, `class`, or `class_method` construct."
        return output