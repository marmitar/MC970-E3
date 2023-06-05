from pathlib import Path
from typing import Final
import sys

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
OUTPUT: Final = path(SCRIPTDIR, 'results.csv')


def tests(df: pd.DataFrame, /) -> pd.DataFrame:
    df = df[['test', 'speedup','total_speedup']]
    return df.groupby('test').describe()

if __name__ == "__main__":
    df = pd.read_csv(OUTPUT, usecols=['test','nthreads','speedup','total_speedup'])

    for nthreads, group in df.groupby('nthreads'):
        print('THREADS_PER_BLOCK', '=', nthreads, '>')
        print(tests(group))

    print('TOTAL', '>')
    print(tests(df))
