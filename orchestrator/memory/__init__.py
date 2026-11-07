
from functools import lru_cache


@lru_cache(maxsize=1)
def get_ltm():
    from .long_term import LongTermMemory
    return LongTermMemory()


@lru_cache(maxsize=1)
def get_wm():
    from .working import WorkingMemory
    return WorkingMemory()