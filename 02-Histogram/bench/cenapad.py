from pathlib import Path
from typing import Final, Iterable, Iterator, TextIO
import sys
import re
from itertools import cycle

import pandas as pd


def path(*parts: str | Path, suffix: str | None = None, strict: bool = True) -> Path:
    """Builds an absolute Path from a sequence of segments."""
    result = Path(*parts)

    if suffix is not None:
        assert result.suffix == ""
        result = result.with_suffix(suffix)

    return result.resolve(strict=strict)

# Source directories
SCRIPTPATH: Final = path(sys.argv[0])
SCRIPTDIR: Final = path(SCRIPTPATH.parent)
SRCDIR: Final = path(SCRIPTDIR.parent)

# Results CSV file
SOURCE: Final = path(SCRIPTDIR, 'cenapad.txt')
OUTPUT: Final = path(SCRIPTDIR, 'cenapad.csv', strict=False)


TESTS = cycle(str(n + 1) for n in range(5))

SPEEDUP_RE: Final = re.compile(
    "\s*Serial\s+runtime:\s+(?P<serial>\d*.\d+)s"
    "\s*Parallel\s+runtime:\s+(?P<parallel>\d*.\d+)s"
    "\s*Speedup:\s+(?P<speedup>\d*.\d+)x",
    re.ASCII | re.MULTILINE
)

def parse(file: Path, /, *, tests: Iterable[str] = TESTS) -> Iterator[tuple[str, str, str, str]]:
    timings = file.read_text()
    for match in SPEEDUP_RE.finditer(timings):
        test = next(tests)
        serial = str(match.group('serial'))
        parallel = str(match.group('parallel'))
        speedup = str(match.group('speedup'))
        yield test, serial, parallel, speedup

def results(file: Path, /) -> Iterator[str]:
    yield 'test,nthreads,serial,parallel,speedup,total_serial,total_parallel,total_speedup'
    for test, serial, parallel, speedup in sorted(parse(file), key=lambda t: (t[0], float(t[3]))):
        yield f'{test},0,{serial},{parallel},{speedup},0,0,0'


if __name__ == "__main__":
    with open(OUTPUT, 'wt') as csv:
        for line in results(SOURCE):
            print(line, file=csv, flush=True)
