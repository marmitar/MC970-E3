from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, unique
from functools import cached_property
from itertools import chain
from pathlib import Path
from time import time
from typing import ClassVar, Final, Iterable, Iterator, TextIO, final

from tqdm import tqdm, trange

# # # # # # # # # #
# PATH  UTILITIES #

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
OUTPUT: Final = path(SCRIPTDIR, 'results.csv', strict=False)

@contextmanager
def chdir(target: str | Path, /, *parts: str | Path) -> Iterator[Path]:
    """Changes current directory on `__enter__` and restores on `__exit__`."""
    prevdir = path(os.curdir)
    targetdir = path(target, *parts)
    try:
        os.chdir(targetdir)
        yield targetdir
    finally:
        os.chdir(prevdir)

@contextmanager
def tempdir(*, parent: Path = path('/tmp'), clear_on_err: bool = False) -> Iterator[Path]:
    """Creates a temporary directory and `chdir` to it, removing everything on `__exit__`."""
    dirpath = tempfile.mkdtemp(dir=parent)
    try:
        fullpath = path(dirpath)
        with chdir(fullpath):
            yield fullpath
        shutil.rmtree(dirpath, ignore_errors=True)
    finally:
        if clear_on_err:
            shutil.rmtree(dirpath, ignore_errors=True)

@contextmanager
def tempbuilddir(*content: str | Path, source_dir: Path = SRCDIR) -> Iterator[Path]:
    """Creates a `tempdir` with `contents` copied from `source_dir`."""
    with tempdir() as build:
        for item in content:
            input = path(source_dir, item)
            output = path(build, item, strict=False)

            if input.is_dir():
                shutil.copytree(input, output)
            else:
                shutil.copy2(input, output)

        yield build

# # # # # # # # # # #
# COMMAND UTILITIES #

@final
@dataclass(frozen=True)
class Execution:
    command: str | Path
    args: tuple[str, ...]
    stdout: bytes
    stderr: bytes
    time: float

def encode(value: str | Path) -> bytes:
    return str(value).encode()

def exec(command: str | Path, *args: str) -> Execution:
    """Executes `command` and measures execution time."""
    cmd = (encode(command), *map(encode, args))

    start = time()
    proc = subprocess.run(cmd, capture_output=True, check=True, text=False, input=b'')
    end = time()

    elapsed = end - start
    return Execution(command, args, proc.stdout, proc.stderr, elapsed)

# # # # # # # # # #
# BUILD UTILITIES #

@final
@dataclass(frozen=True)
class CMakeVar:
    name: str
    value: str | None = None

    VARIABLE_RE: ClassVar[re.Pattern[str]] = re.compile('^(?P<name>\w+(_\w+)*):\w+=(?P<value>.*)$', re.MULTILINE)

    def __post_init__(self):
        cache_var = f'{self.name}:VARIABLE={self.value or ""}'
        assert self.VARIABLE_RE.fullmatch(cache_var), f"invalid CMake var {self}"

    def __str__(self) -> str:
        if self.value is None:
            return self.name
        else:
            return f'{self.name}={self.value}'

    @classmethod
    def parse(cls, text: str, /) -> Iterator[CMakeVar]:
        for match in cls.VARIABLE_RE.finditer(text):
            name = match.group('name')
            assert isinstance(name, str), "CMake var missing name"
            value = match.group('value')
            if isinstance(value, str) and value != "":
                yield CMakeVar(name, value)
            else:
                yield CMakeVar(name)

    @staticmethod
    def of(target: str | CMakeVar) -> CMakeVar:
        if isinstance(target, str):
            return CMakeVar(target)
        else:
            return CMakeVar(target.name)

@final
@dataclass(frozen=True)
class CMake:
    @classmethod
    @property
    def srcdir(cls) -> Path:
        return path(os.curdir)

    @classmethod
    @property
    def builddir(cls) -> Path:
        return path(cls.srcdir, 'build', strict=False)

    @classmethod
    @property
    def testsdir(cls) -> Path:
        return path(cls.srcdir, 'tests')

    @classmethod
    def clear(cls):
        shutil.rmtree(cls.builddir, ignore_errors=True)

    @classmethod
    def prepare(cls, **kwargs: str):
        cls.clear()

        variables = (CMakeVar(name, value) for name, value in kwargs.items())
        defines = chain.from_iterable(('-D', str(var)) for var in variables)
        exec('cmake', '-B', str(cls.builddir), *defines, str(cls.srcdir))

    @classmethod
    def build(cls):
        exec('make', '-C', str(cls.builddir))

    @classmethod
    @property
    def cache(cls) -> Path:
        return path(cls.builddir, 'CMakeCache.txt')

    @classmethod
    @property
    def cache_vars(cls) -> Iterable[CMakeVar]:
        return CMakeVar.parse(cls.cache.read_text())

    @classmethod
    def var(cls, name: str | CMakeVar, /) -> CMakeVar:
        target = CMakeVar.of(name)

        for variable in cls.cache_vars:
            if variable.name == target.name:
                return variable

        raise KeyError(target)

# # # # # # # #
# TEST  CASES #

@unique
class Binary(str, Enum):
    SERIAL = 'serial'
    PARALELL = 'parallel'

    @property
    def path(self) -> Path:
        return path(CMake.builddir, self.value)

@final
@dataclass(frozen=True)
class Timing:
    proc: float
    total: float

@final
@dataclass(frozen=True)
class TestResult:
    test: TestCase
    nthreads: int
    serial: float
    parallel: float
    total_serial: float
    total_parallel: float

    @property
    def speedup(self):
        return self.serial / self.parallel

    @property
    def total_speedup(self):
        return self.total_serial / self.total_parallel

    KEYS: ClassVar[tuple[str, ...]] = \
        ('test', 'nthreads', 'serial', 'parallel', 'speedup', \
         'total_serial', 'total_parallel', 'total_speedup')

    @classmethod
    def csv_header(cls) -> str:
        return ','.join(cls.KEYS)

    def to_csv(self) -> str:
        return ','.join(str(getattr(self, key)) for key in self.KEYS)

    @staticmethod
    def parse(csv: str, /) -> TestResult:
        keys = (value.strip() for value in csv.split(','))
        test = TestCase(next(keys))
        nthreads = int(next(keys))
        serial = float(next(keys))
        parallel = float(next(keys))
        float(next(keys))
        total_serial = float(next(keys))
        total_parallel = float(next(keys))
        float(next(keys))
        return TestResult(test, nthreads, serial, parallel, total_serial, total_parallel)

@final
@dataclass(frozen=True)
class TestCase:
    name: str

    @property
    def input_file(self) -> Path:
        return path(CMake.testsdir, self.name, suffix='.in')

    @property
    def output_file(self) -> Path:
        return path(CMake.testsdir, self.name, suffix='.out.ppm')

    def __str__(self) -> str:
        return self.name

    @cached_property
    def expected_output(self) -> bytes:
        assert self.output_file.is_file()
        return self.output_file.read_bytes()

    def compare_output(self, output: bytes, /):
        if self.expected_output == output:
            return

        expected = len(self.expected_output)
        size = len(output)

        matched = sum(a == b for a, b in zip(self.expected_output, output))
        share = 100 * matched / expected
        raise ValueError(f"output is {size} bytes, with {matched} bytes ({share:.1f}%) equal to the expected {expected} bytes")

    def exec(self, binary: Binary) -> Timing:
        result = exec(binary.path, str(self.input_file))
        self.compare_output(result.stdout)
        return Timing(float(result.stderr), result.time)

    def run(self) -> TestResult:
        serial = self.exec(Binary.SERIAL)
        parallel = self.exec(Binary.PARALELL)
        threads = CMake.var('THREADS_PER_BLOCK')
        return TestResult(self, int(threads.value), serial.proc, parallel.proc, serial.total, parallel.total)

# # # # # # # # #
# BENCHMARKING  #

@final
@dataclass(frozen=True)
class ResultsCSV:
    csv: TextIO | None = None
    results: list[TestResult] = field(default_factory=list)

    def write_line(self, line: str, /):
        if self.csv is not None:
            print(line, file=self.csv, flush=True)

    def insert(self, result: TestResult, /):
        self.write_line(result.to_csv())
        self.results.append(result)

    @classmethod
    @property
    def header(cls) -> str:
        return TestResult.csv_header()

    @staticmethod
    def read(path: Path) -> ResultsCSV:
        results = ResultsCSV()

        with path.open('rt') as csv:
            assert csv.readline().strip() == ResultsCSV.header
            for line in csv:
                line = line.strip()
                if line != "":
                    results.insert(TestResult.parse(line))

        return results

    @contextmanager
    @staticmethod
    def open(path: Path, /, *, ignore_previous: bool = False) -> Iterator[ResultsCSV]:
        try:
            assert not ignore_previous
            previous = ResultsCSV.read(path)
        except:
            previous = None

        with open(path, 'wt') as file:
            csv = ResultsCSV(file)
            csv.write_line(TestResult.csv_header())
            if previous is not None:
                for result in previous.results:
                    csv.insert(result)

            yield csv

@tempbuilddir('src', 'tests', 'CMakeLists.txt')
def run(*tests: TestCase, repeats: int = 50, **cmake_args: str):
    CMake.prepare(CMAKE_BUILD_TYPE='Release', CMAKE_CUDA_ARCHITECTURES='native', **cmake_args)
    CMake.build()

    with ResultsCSV.open(OUTPUT) as results:
        for _ in trange(repeats, desc='Repetition'):
            for test in tqdm(tests, desc='Test Case'):
                results.insert(test.run())

        results = sorted(results.results, key=lambda r: (r.nthreads, r.test.name, r.speedup, r.total_speedup))

    # then rewrite the sorted results
    with ResultsCSV.open(OUTPUT, ignore_previous=True) as csv:
        for result in results:
            csv.insert(result)

if __name__ == "__main__":
    TESTS = TestCase('1'), TestCase('2'), TestCase('3'), TestCase('4'), TestCase('5')

    try:
        THREADS = int(sys.argv[1])
    except:
        THREADS = None

    if THREADS is not None:
        run(*TESTS, THREADS_PER_BLOCK = str(THREADS))
    else:
        run(*TESTS)
