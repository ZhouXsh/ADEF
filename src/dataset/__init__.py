import copy

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.seed import GLOBAL_SEED, make_generator, seed_worker


def _restartable_next(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _clone_loader(data_loader, dataset, seed):
    """Clone the ordinary train DataLoader with deterministic RNG handling."""
    if isinstance(data_loader.sampler, RandomSampler):
        shuffle = True
    elif isinstance(data_loader.sampler, SequentialSampler):
        shuffle = False
    else:
        # Custom samplers may encode project-specific semantics; keep them intact.
        return data_loader

    kwargs = dict(
        dataset=dataset,
        batch_size=data_loader.batch_size,
        shuffle=shuffle,
        num_workers=data_loader.num_workers,
        collate_fn=data_loader.collate_fn,
        pin_memory=data_loader.pin_memory,
        drop_last=data_loader.drop_last,
        timeout=data_loader.timeout,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
    if data_loader.num_workers > 0:
        kwargs["persistent_workers"] = data_loader.persistent_workers
        if data_loader.prefetch_factor is not None:
            kwargs["prefetch_factor"] = data_loader.prefetch_factor
    return DataLoader(**kwargs)


def _make_begin_loader(data_loader):
    dataset = copy.copy(data_loader.dataset)
    dataset.crop_strategy = "begin64"
    return _clone_loader(data_loader, dataset, GLOBAL_SEED + 1)


def infinite_data_loader(data_loader):
    """Convert a DataLoader into an infinite deterministic iterator.

    For the 0803 emotion dataset, alternate exactly with the training loop's
    parity convention: even iterations receive a random 80-frame continuation
    crop; odd iterations receive a true video-start 64-frame crop.
    Other ordinary loaders keep their previous infinite-loop behavior but use
    the same project-wide seed for shuffling and worker-side NumPy/Python RNGs.
    """
    seeded_random_loader = _clone_loader(
        data_loader, data_loader.dataset, GLOBAL_SEED
    )

    if not getattr(data_loader.dataset, "alternate_start_random", False):
        while True:
            for batch in seeded_random_loader:
                yield batch
        return

    if data_loader.batch_size is None:
        raise ValueError(
            "Alternating start/random loading requires a regular batch_size DataLoader."
        )

    begin_loader = _make_begin_loader(data_loader)
    random_iter = iter(seeded_random_loader)
    begin_iter = iter(begin_loader)
    use_begin = False  # it=0 continuation/random; it=1 starting/true begin64.

    while True:
        if use_begin:
            batch, begin_iter = _restartable_next(begin_loader, begin_iter)
        else:
            batch, random_iter = _restartable_next(
                seeded_random_loader, random_iter
            )
        yield batch
        use_begin = not use_begin
