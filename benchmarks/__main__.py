"""python -m benchmarks [suite] [args...] — with no suite, list them."""
import ast
import importlib
import sys
from pathlib import Path

SUITES = Path(__file__).parent / 'suites'


def suites() -> dict[str, str]:
    """name -> first docstring line, read without importing (suites may need optional deps)."""
    out = {}
    for path in sorted(SUITES.glob('*.py')):
        if path.stem.startswith('_'):
            continue
        doc = ast.get_docstring(ast.parse(path.read_text())) or ''
        out[path.stem] = doc.splitlines()[0] if doc else ''
    return out


def main(argv: list[str]) -> None:
    available = suites()
    if not argv or argv[0] in ('-h', '--help', 'list'):
        print(__doc__)
        width = max(map(len, available))
        for name, desc in available.items():
            print(f'  {name:{width}s}  {desc}')
        return
    name, args = argv[0], argv[1:]
    if name not in available:
        sys.exit(f'unknown suite {name!r}; run `python -m benchmarks` to list them')
    importlib.import_module(f'benchmarks.suites.{name}').main(args)


if __name__ == '__main__':
    main(sys.argv[1:])
