import os
import subprocess
from functools import lru_cache

from modules.utils import constants, hf

git = os.environ.get("GIT", "git")


@lru_cache()
def commit_hash():
    try:
        if hf.is_spaces_env:
            return "<hf>"
        return subprocess.check_output(
            [git, "-C", constants.ROOT_DIR, "rev-parse", "HEAD"],
            shell=False,
            encoding="utf8",
        ).strip()
    except Exception:
        return "<none>"


@lru_cache()
def git_tag():
    try:
        if hf.is_spaces_env:
            return "<hf>"
        return subprocess.check_output(
            [git, "-C", constants.ROOT_DIR, "describe", "--tags"],
            shell=False,
            encoding="utf8",
        ).strip()
    except Exception:
        try:

            changelog_md = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "CHANGELOG.md"
            )
            with open(changelog_md, "r", encoding="utf-8") as file:
                line = next((line.strip() for line in file if line.strip()), "<none>")
                line = line.replace("## ", "")
                return line
        except Exception:
            return "<none>"


@lru_cache()
def branch_name():
    try:
        if hf.is_spaces_env:
            return "<hf>"
        return subprocess.check_output(
            [git, "-C", constants.ROOT_DIR, "rev-parse", "--abbrev-ref", "HEAD"],
            shell=False,
            encoding="utf8",
        ).strip()
    except Exception:
        return "<none>"


if __name__ == "__main__":
    print(commit_hash())
    print(git_tag())
    print(branch_name())
