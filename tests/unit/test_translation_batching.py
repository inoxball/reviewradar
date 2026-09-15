from reviewradar.translation.translator import length_sorted_batches


def test_groups_texts_of_similar_length_longest_first() -> None:
    texts = ["a", "a long review text", "mid text", "ab", "a much longer review text"]

    assert length_sorted_batches(texts, 2) == [[4, 1], [2, 3], [0]]


def test_covers_every_text_exactly_once() -> None:
    texts = ["x" * (index % 7) for index in range(50)]

    batches = length_sorted_batches(texts, 16)

    assert sorted(index for batch in batches for index in batch) == list(range(50))
    assert [len(batch) for batch in batches] == [16, 16, 16, 2]


def test_empty_input_has_no_batches() -> None:
    assert length_sorted_batches([], 4) == []
