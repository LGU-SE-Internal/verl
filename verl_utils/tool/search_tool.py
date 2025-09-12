import sqlite3

MAX_RESPONSE_LEN: int = 16000


class SearchTool():
    """Tool to query the code knowledge graph of a codebase."""

    def __init__(self, root_dir, instance_id):
        self.instance_id = instance_id
        self.root_dir = root_dir
        self.db_path = f'{root_dir}/codegraph/{instance_id}.db'

    def execute(self, construct, entity):
        # Check availability
        if not construct:
            return "`construct` argument is empty. Please ensure the function calling format is valid."
        elif not entity:
            return "`entity` argument is empty. Please ensure the function calling format is valid."

        match construct:
            case "directory":
                return self._search_directory(entity)
            case "file":
                return self._search_file(entity)
            case "function":
                return self._search_function(entity)
            case "class":
                return self._search_class(entity)
            case "class_method":
                return self._search_class_method(entity)
            case _:
                return f"Invalid program construct: {construct}"

    def _add_line_numbers(self, code: str, start_line: int) -> str:
        """Add line numbers to code while preserving indentation."""
        if not code:
            return ""
        
        lines = code.split('\n')
        numbered_lines = []
        max_line_num_length = len(str(start_line + len(lines) - 1))
        
        for i, line in enumerate(lines):
            line_num = start_line + i
            # Format: [line number][space][code with original indentation]
            numbered_line = f"{line_num:<{max_line_num_length}}\t{line}"
            numbered_lines.append(numbered_line)
        
        return '\n'.join(numbered_lines)

    def _search_function(self, entity) -> str:
        """Search for a function in the ckg database."""
        db_connection = sqlite3.connect(self.db_path)
        entries = db_connection.execute(
            """
            SELECT file_path, start_line, end_line, body FROM functions WHERE name = ?
            """,
            (entity,),
        ).fetchall()

        if len(entries) == 0:
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

    def _search_class(self, entity) -> str:
        """Search for a class in the ckg database."""
        db_connection = sqlite3.connect(self.db_path)
        entries = db_connection.execute(
            """
            SELECT file_path, start_line, end_line, fields, methods, body FROM classes WHERE name = ?
            """,
            (entity,),
        ).fetchall()

        if len(entries) == 0:
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

    def _search_class_method(self, entity) -> str:
        """Search for a class method in the ckg database."""
        db_connection = sqlite3.connect(self.db_path)
        entries = db_connection.execute(
            """
            SELECT file_path, start_line, end_line, body, class_name FROM class_methods WHERE name = ?
            """,
            (entity,),
        ).fetchall()

        if len(entries) == 0:
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

    def _search_directory(self, entity) -> str:
        """Search for a directory in the ckg database."""
        if entity != '.':
            directory_path = entity.rstrip('/') + '/'
        else:
            directory_path = '.'
        
        db_connection = sqlite3.connect(self.db_path)
        entries = db_connection.execute(
            """
            SELECT path, files FROM directories WHERE path = ?
            """,
            (directory_path,),
        ).fetchall()

        if len(entries) == 0:
            msg = f"No directory named `{directory_path}` found."
            if entity.startswith('/'):
                return f"{msg} Please use relative path of a directory."
            return f"{msg} Maybe you should first check the directory's relative path to ensure the file exists?"
        
        entry = entries[0]
        output = f"{entry[0]} contains following files:\n{entry[1]}"

        if len(output) > MAX_RESPONSE_LEN:
            output = output[:MAX_RESPONSE_LEN] + "\n<response clipped>"

        return output

    def _search_file(self, entity) -> str:
        """Search for a file in the ckg database."""
        db_connection = sqlite3.connect(self.db_path)
        entries = db_connection.execute(
            """
            SELECT path, entities FROM files WHERE path = ?
            """,
            (entity,),
        ).fetchall()

        if len(entries) == 0:
            msg = f"No file named `{entity}` found."
            if entity.startswith('/'): # absolute path error
                return f"{msg} Please use relative path (e.g., a/b/c.py) rather than absolute path (e.g., /root/a/b/c.py)."
            elif '/' not in entity: # simple name error
                return f"{msg} Please use relative path (e.g., a/b/c.py) rather than a simple file name (e.g., c.py)."
            return f"{msg} Maybe you should first check from the root directory `.` (top-down) to ensure the directory exists?"

        entry = entries[0]
        full_name_list = entry[1].split('\n')
        functions = set()
        classes = set()
        methods = set()
        for name in full_name_list:
            if '.' in name:
                methods.add(name.split('.')[-1])
                classes.add(name.split('.')[0])
            else:
                functions.add(name)
        functions = functions - classes
        
        functions = '\n'.join(sorted(list(functions))) if len(functions) > 0 else '(No functions)'
        classes = '\n'.join(sorted(list(classes))) if len(classes) > 0 else '(No classes)'
        methods = '\n'.join(sorted(list(methods))) if len(methods) > 0 else '(No class methods)'
        names = f"# Function:\n{functions}\n\n# Class:\n{classes}\n\n# Class Method:\n{methods}"
        output = f"{entry[0]} contains following entries:\n{names}"

        if len(output) > MAX_RESPONSE_LEN:
            output = output[:MAX_RESPONSE_LEN] + "\n<response clipped>"

        output += "\n\nNote: The `file` construct shows only the skeleton structure of a file entity. To view the source code, please search for an entity in `function`, `class`, or `class_method` construct."

        return output