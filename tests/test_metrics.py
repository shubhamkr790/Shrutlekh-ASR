import pytest

from shrutlekh.metrics import align, cer, normalize, tokenize, wer, word_errors, wrong_hypothesis_tokens


def test_normalize_drops_punctuation_and_case():
    assert normalize("Hello, World!  It's  FINE.") == "hello world its fine"


def test_normalize_keeps_hindi_vowel_signs():
    # a naive [^\w\s] regex would strip the matras and split these words
    assert tokenize("नमस्ते, दुनिया!") == ["नमस्ते", "दुनिया"]


def test_perfect_match_is_zero():
    assert wer("the cat sat", "The cat sat.") == 0.0


def test_textbook_example_one_substitution():
    # said "turn on the kitchen light", heard "turn on a kitchen light"
    e = word_errors("turn on the kitchen light", "turn on a kitchen light")
    assert (e.substitutions, e.deletions, e.insertions, e.reference_length) == (1, 0, 0, 5)
    assert e.rate == pytest.approx(0.2)


def test_deletion_and_insertion_counted():
    e = word_errors("a b c d", "a c d e")
    assert e.deletions == 1 and e.insertions == 1 and e.substitutions == 0
    assert e.rate == pytest.approx(0.5)


def test_wer_can_exceed_one():
    # three extra words against a one-word reference = 300%
    assert wer("hi", "oh hi there friend") == pytest.approx(3.0)


def test_empty_reference():
    assert wer("", "") == 0.0
    assert wer("", "something") == 1.0


def test_cer_ignores_spaces():
    assert cer("sea shells", "seashells") == 0.0
    assert cer("abcd", "abed") == pytest.approx(0.25)


def test_align_reports_operations_in_order():
    ops = [o.op for o in align(["a", "b", "c"], ["a", "x", "c", "d"])]
    assert ops == ["equal", "sub", "equal", "ins"]


def test_wrong_hypothesis_tokens_points_at_the_bad_words():
    hyp = ["the", "doctor", "prescribed", "a", "moxicillin"]
    assert wrong_hypothesis_tokens("the doctor prescribed amoxicillin", hyp) == {3, 4} or \
        wrong_hypothesis_tokens("the doctor prescribed amoxicillin", hyp) == {4, 3}
