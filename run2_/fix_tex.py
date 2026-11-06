import re
from pathlib import Path

tex_path = Path('paper/main.tex')
text = tex_path.read_text(encoding='utf-8')

# Find all inline code formatted with backticks
def replace_backtick(match):
    content = match.group(1)
    escaped = content.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    return r'\texttt{' + escaped + '}'

new_text = re.sub(r'`([^`\n]+)`', replace_backtick, text)

# Also check for any standalone filenames or underscores
tex_path.write_text(new_text, encoding='utf-8')
print("Successfully replaced backticks with \\texttt{...} and escaped underscores.")
