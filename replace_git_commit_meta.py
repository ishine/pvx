import glob
import re

replacement = """def git_commit_meta() -> tuple[str, str]:
    return "[COMMIT_HASH]", "[COMMIT_DATE]"
"""

for filepath in glob.glob("scripts/scripts_generate_*.py"):
    with open(filepath, "r") as f:
        content = f.read()

    # We want to match the whole git_commit_meta function and replace it
    content = re.sub(
        r'def git_commit_meta\(\) -> tuple\[str, str\]:[\s\S]*?return commit, commit_date\n',
        replacement,
        content
    )
    with open(filepath, "w") as f:
        f.write(content)
print("success")
