## WSL Path Handling
When working with files in WSL paths (e.g., \\wsl.localhost\Ubuntu\... ), you MUST use the wsl command prefix to execute operations in the WSL terminal.

### Correct Approach
Use wsl command prefix for file operations:

- wsl touch /path/to/file - Create empty file
- wsl ls -la /path/to/dir - List directory contents
- wsl cat /path/to/file - Read file contents
- wsl rm /path/to/file - Delete file
- wsl mkdir -p /path/to/dir - Create directory
### Incorrect Approach
Do NOT use the Write tool directly with WSL paths:

- The Write tool with \\wsl.localhost\Ubuntu\... paths may report success but fail to actually create files
### Reason
The Write tool does not reliably work with WSL filesystem paths. Using wsl command prefix ensures operations are executed within the WSL environment and work correctly.

## Sandbox Limitations
The trae-sandbox tool has parsing restrictions for command-line arguments. Complex commands may fail with errors like `unexpected argument`.

### Known Limitations
- Parentheses `()` in content cause parsing errors
- Nested quotes are problematic
- Newline characters `\n` cannot be passed correctly
- Here-doc syntax `<<EOF` is blocked by the parser
- Spaces in content may also cause parsing errors

### Workarounds
1. For simple content: Use `wsl sh -c 'printf %s "content" > /path/to/file'`
2. For complex content: Use `wsl touch` to create file, then edit in IDE
3. For reading files: `wsl cat /path/to/file` works fine

## File Content Writing Workflow
When writing complex content to files (especially code files like .py, .js, etc.), follow this workflow:

### Creating New Files
1. Create empty file using `wsl touch /path/to/file`
2. Output the code content directly in the chat window
3. Wait for user to manually paste the content into the file
4. User confirms completion by replying
5. AI reads the file to verify the content is correct
6. Only proceed to next step after verification

### Modifying Existing Files
1. AI reads the existing file content first using `wsl cat /path/to/file`
2. AI outputs the complete modified content in the chat window
3. Wait for user to manually replace the file content
4. User confirms completion by replying
5. AI reads the file to verify the content is correct
6. Only proceed to next step after verification

### When to Use This Workflow
- Writing code files with complex syntax
- Content that may trigger sandbox parsing errors
- Any file that requires precise formatting

### Reason
This approach avoids sandbox parsing issues and ensures the user has full control over the final file content. The verification step catches any copy-paste errors.