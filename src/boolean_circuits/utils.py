import itertools


def all_combinations(iterable, min_length=0, max_length=None):
    """Generate all combinations of all lengths for the given iterable."""
    max_length = max_length if max_length is not None else len(iterable)
    for r in range(min_length, max_length + 1):
        for combination in itertools.combinations(iterable, r):
            yield combination
