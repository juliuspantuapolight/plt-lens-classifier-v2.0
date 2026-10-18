from scripts.prepare_slot_dataset import balanced_session_splits


def test_balanced_session_splits_keep_sessions_separate_and_representative():
    counts = {
        "dense-a": (2400, 1580),
        "dense-b": (2400, 1240),
        "medium-a": (1200, 400),
        "medium-b": (1200, 380),
        "sparse-a": (1200, 20),
        "sparse-b": (1200, 10),
        "mixed-a": (800, 250),
        "mixed-b": (800, 240),
    }

    splits = balanced_session_splits(counts)

    assert set.union(*splits.values()) == set(counts)
    assert sum(len(values) for values in splits.values()) == len(counts)
    assert all(splits[name] for name in ("train", "val", "test"))
