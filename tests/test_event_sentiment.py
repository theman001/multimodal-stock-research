from src.data_collection.event_sentiment import sliding_windows, split_sentences


def test_split_sentences_splits_on_period_exclamation_question():
    text = "First sentence here. Second sentence! Third sentence?"
    result = split_sentences(text)
    assert result == ["First sentence here.", "Second sentence!", "Third sentence?"]


def test_split_sentences_drops_short_fragments():
    """표 데이터 파편(짧은 숫자 조각 등)은 문장으로 안 친다."""
    text = "1. A real sentence with enough content here."
    result = split_sentences(text, min_length=10)
    assert "1." not in result


def test_sliding_windows_matches_user_specified_stride_1_example():
    """사용자가 직접 지정한 예시: 문장 6개, window=3, stride=1 -> 4묶음
    (1,2,3)/(2,3,4)/(3,4,5)/(4,5,6)."""
    sentences = [f"s{i}" for i in range(1, 7)]
    result = sliding_windows(sentences, window_size=3, stride=1)
    assert result == ["s1 s2 s3", "s2 s3 s4", "s3 s4 s5", "s4 s5 s6"]


def test_sliding_windows_handles_fewer_sentences_than_window_size():
    """문장이 window_size 이하인 짧은 공시는 전체를 윈도우 1개로 취급한다."""
    sentences = ["s1", "s2"]
    result = sliding_windows(sentences, window_size=3, stride=1)
    assert result == ["s1 s2"]


def test_sliding_windows_handles_empty_input():
    assert sliding_windows([], window_size=3, stride=1) == []


def test_sliding_windows_exact_window_size_produces_one_window():
    sentences = ["s1", "s2", "s3"]
    result = sliding_windows(sentences, window_size=3, stride=1)
    assert result == ["s1 s2 s3"]
