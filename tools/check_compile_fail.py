"""Build every program in `tests_compile_fail/` and require each to fail
with the message it names.

A mistake numax promises to stop at compile time -- an operator between
two dtypes, two compile-time shapes that do not broadcast, an oversized
`Array`-tier matrix -- is a claim `std.testing` cannot check, since a test
that does not compile does not run. Each file here is one such mistake,
headed by one or more `# expect: <text>` lines; the check passes when
`mojo build -I .` exits non-zero and every expected text appears in its
output, and fails when a program builds (the guard is gone) or fails
with a different message (the guard no longer speaks).

    python3 tools/check_compile_fail.py
"""

import glob
import os
import re
import subprocess
import sys
import tempfile


def main():
    paths = sorted(glob.glob("tests_compile_fail/*.mojo"))
    if not paths:
        print("no programs under tests_compile_fail/")
        sys.exit(1)
    failures = 0
    with tempfile.TemporaryDirectory() as scratch:
        for path in paths:
            expected = re.findall(r"(?m)^# expect: (.+)$", open(path).read())
            if not expected:
                print(f"FAIL {path}: no `# expect:` line")
                failures += 1
                continue
            out = os.path.join(scratch, os.path.basename(path)[:-5])
            result = subprocess.run(
                ["mojo", "build", "-I", ".", path, "-o", out],
                capture_output=True,
                text=True,
            )
            log = result.stdout + result.stderr
            missing = [e for e in expected if e not in log]
            if result.returncode == 0:
                print(f"FAIL {path}: built, and was meant not to")
                failures += 1
            elif missing:
                print(f"FAIL {path}: failed without the expected message")
                for e in missing:
                    print(f"    missing: {e}")
                for line in [l for l in log.split("\n") if "error" in l or "constraint" in l][:5]:
                    print(f"    {line}")
                failures += 1
            else:
                print(f"ok   {path}")
    print(f"{len(paths) - failures} of {len(paths)} programs failed as expected")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
