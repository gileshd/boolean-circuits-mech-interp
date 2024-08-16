import itertools
from typing import Iterable, Generator, Tuple, Any


def all_combinations(
    iterable: Iterable[Any], min_length: int = 0, max_length: int | None = None
) -> Generator[Tuple[Any, ...], None, None]:
    """Generate all combinations of all lengths for the given iterable."""
    max_length = max_length if max_length is not None else len(tuple(iterable))
    for r in range(min_length, max_length + 1):
        for combination in itertools.combinations(iterable, r):
            yield combination
