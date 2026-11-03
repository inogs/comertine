from collections.abc import Sequence
from typing import Any

import dask
from dask.callbacks import Callback
from dask.delayed import Delayed
from tqdm import tqdm


def _get_distributed_client():
    """Return the active `distributed` client, or None if there is none."""
    try:
        from distributed import get_client
    except ImportError:
        return None

    try:
        return get_client()
    except ValueError:
        return None


class _TqdmCallback(Callback):
    """Update a progress bar every time one of the given keys is computed."""

    def __init__(self, keys, progress_bar: tqdm):
        super().__init__()
        self._keys = set(keys)
        self._progress_bar = progress_bar

    def _posttask(self, key, result, dsk, state, id):
        if key in self._keys:
            self._progress_bar.update(1)


def run_tasks(
    tasks: Sequence[Delayed],
    *,
    show_progress: bool = True,
    description: str | None = None,
) -> list[Any]:
    """
    Compute some delayed tasks with the scheduler that is currently active.

    The caller does not choose the scheduler: if a `distributed` client is
    active (e.g. created by `main` with a `LocalCluster`), the tasks are
    submitted to it; otherwise they are computed with `dask.compute`, i.e.
    with the scheduler set in the dask configuration (for example with
    `dask.config.set(scheduler="processes")`).

    Args:
        tasks: The tasks to compute.
        show_progress: Whether to show a progress bar that advances every
            time one of `tasks` is completed.
        description: The description of the progress bar.

    Returns:
        The results of the tasks, in the same order of `tasks`.
    """
    progress_bar = tqdm(
        total=len(tasks), desc=description, disable=not show_progress
    )

    with progress_bar:
        client = _get_distributed_client()
        if client is not None:
            from distributed import as_completed

            futures = client.compute(list(tasks))
            for future in as_completed(futures):
                # Raises immediately if the task failed
                future.result()
                progress_bar.update(1)
            return client.gather(futures)

        keys = [task.key for task in tasks]
        with _TqdmCallback(keys, progress_bar):
            return list(dask.compute(*tasks))
