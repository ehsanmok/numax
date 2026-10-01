"""Compile every fenced `mojo` block in the given Markdown files.

A block with a `def main` is compiled as written. A block without one is
wrapped: its import lines are hoisted and the rest becomes the body of a
`def main() raises`. Each block is compiled with `mojo build -I .` into a
scratch directory, and the script exits non-zero if any block fails,
printing the file, the block's first line number and the compiler error.

A block that asks for the device (`gpu=True`, or a `DeviceContext()`) is compiled with
`--target-accelerator sm_80`, as `examples-gpu-build` compiles the GPU
examples: nothing here runs a block, and without a named target a
GPU-less machine -- CI's Linux runner -- cannot instantiate the device
path at all.

    python3 tools/check_snippets.py llms.txt docs/quickstart.md
"""

import os
import re
import subprocess
import sys
import tempfile


def blocks(path):
    text = open(path).read()
    for m in re.finditer(r"(?ms)^```mojo\n(.*?)^```", text):
        yield text[: m.start()].count("\n") + 2, m.group(1)


def program(code):
    if re.search(r"(?m)^def main\(", code):
        return code
    imports = [l for l in code.split("\n") if re.match(r"(from|import) ", l)]
    body = [l for l in code.split("\n") if not re.match(r"(from|import) ", l)]
    lines = ["    " + l if l.strip() else "" for l in body]
    return "\n".join(imports) + "\n\n\ndef main() raises:\n" + "\n".join(lines) + "\n    pass\n"


def main():
    failures = 0
    total = 0
    with tempfile.TemporaryDirectory() as scratch:
        for path in sys.argv[1:]:
            for line, code in blocks(path):
                total += 1
                src = os.path.join(scratch, f"snippet_{total}.mojo")
                open(src, "w").write(program(code))
                command = ["mojo", "build", "-I", ".", src, "-o", src[:-5]]
                if "gpu=True" in code or "DeviceContext()" in code:
                    command[2:2] = ["--target-accelerator", "sm_80"]
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    failures += 1
                    errors = [l for l in (result.stdout + result.stderr).split("\n") if "error" in l]
                    print(f"FAIL {path}:{line}")
                    for l in errors[:5]:
                        print("    " + l.replace(src, f"{path}:{line}"))
                else:
                    print(f"ok   {path}:{line}")
    print(f"{total - failures} of {total} blocks compiled")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
