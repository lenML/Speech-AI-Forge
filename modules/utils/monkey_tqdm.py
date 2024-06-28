import contextlib


@contextlib.contextmanager
def disable_tqdm(enabled=True):
    if not enabled:
        yield
        return

    try:
        _tqdm = __import__("tqdm")
    except ModuleNotFoundError:
        yield
        return

    # Backup original methods
    original_init = _tqdm.tqdm.__init__
    original_update = _tqdm.tqdm.update
    original_close = _tqdm.tqdm.close
    original_exit = _tqdm.tqdm.__exit__
    original_iter = _tqdm.tqdm.__iter__

    def init_tqdm(
        self, iterable=None, desc=None, total=None, disable=True, *args, **kwargs
    ):
        kwargs.setdefault("disable", disable)
        self.__init__orig__(iterable, desc, total, *args, **kwargs)

    def iter_tqdm(self):
        return self.__iter__orig__()

    def update_tqdm(self, n=1):
        return self.__update__orig__(n)

    def close_tqdm(self):
        return self.__close__orig__()

    def exit_tqdm(self, exc_type, exc_value, traceback):
        return self.__exit__orig__(exc_type, exc_value, traceback)

    # Patch methods
    _tqdm.tqdm.__init__ = init_tqdm
    _tqdm.tqdm.update = update_tqdm
    _tqdm.tqdm.close = close_tqdm
    _tqdm.tqdm.__exit__ = exit_tqdm
    _tqdm.tqdm.__iter__ = iter_tqdm

    # Ensure original methods are called
    if not hasattr(_tqdm.tqdm, "__init__orig__"):
        _tqdm.tqdm.__init__orig__ = original_init
    if not hasattr(_tqdm.tqdm, "__update__orig__"):
        _tqdm.tqdm.__update__orig__ = original_update
    if not hasattr(_tqdm.tqdm, "__close__orig__"):
        _tqdm.tqdm.__close__orig__ = original_close
    if not hasattr(_tqdm.tqdm, "__exit__orig__"):
        _tqdm.tqdm.__exit__orig__ = original_exit
    if not hasattr(_tqdm.tqdm, "__iter__orig__"):
        _tqdm.tqdm.__iter__orig__ = original_iter

    try:
        yield  # Allow code within the with block to execute
    finally:
        # Restore original methods
        _tqdm.tqdm.__init__ = original_init
        _tqdm.tqdm.update = original_update
        _tqdm.tqdm.close = original_close
        _tqdm.tqdm.__exit__ = original_exit
        _tqdm.tqdm.__iter__ = original_iter

        if hasattr(_tqdm, "auto") and hasattr(_tqdm.auto, "tqdm"):
            _tqdm.auto.tqdm = _tqdm.tqdm


if __name__ == "__main__":
    import time

    from tqdm import tqdm

    with disable_tqdm():
        for i in tqdm(range(10), desc="Processing"):
            time.sleep(0.1)

    print("Progress bar should be back:")
    for i in tqdm(range(10), desc="Processing"):
        time.sleep(0.1)
