import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# 드라이브 문자 경로, 홈 폴더 경로, file:// 절대 링크
ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|(?<![\w.])/Users/[^/\s]+/|(?<![\w.])/home/[^/\s]+/|file:///")


class NoAbsolutePathTests(unittest.TestCase):
    def test_tracked_text_files_use_relative_paths(self):
        files = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout.decode("utf-8").split("\0")
        hits = []
        for name in filter(None, files):
            path = ROOT / name
            if path == Path(__file__).resolve():
                continue
            if path.suffix.lower() not in {".py", ".md", ".kt", ".kts", ".js", ".html", ".yaml", ".yml", ".json", ".txt", ".css", ".example"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, FileNotFoundError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if ABSOLUTE_PATH.search(line):
                    hits.append(f"{name}:{lineno}")
        self.assertEqual(hits, [], "절대 경로가 남아 있습니다: " + ", ".join(hits[:20]))


if __name__ == "__main__":
    unittest.main()
