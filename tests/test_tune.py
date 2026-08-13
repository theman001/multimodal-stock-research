from src.models.tune import sample_candidates
from src.models.train import BASELINE_PARAMS


def test_sample_candidates_returns_requested_count():
    candidates = sample_candidates(10)
    assert len(candidates) == 10


def test_first_candidate_is_the_current_baseline():
    candidates = sample_candidates(5)
    assert candidates[0] == dict(BASELINE_PARAMS)


def test_sampling_is_reproducible_with_same_seed():
    a = sample_candidates(8, seed=7)
    b = sample_candidates(8, seed=7)
    assert a == b


def test_different_seeds_can_produce_different_candidates():
    a = sample_candidates(8, seed=1)
    b = sample_candidates(8, seed=2)
    assert a != b
