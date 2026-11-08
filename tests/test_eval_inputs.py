"""Evaluator input regressions from the contest-guide audit."""
import pytest

from poker_eval import assess, draw_routes, preflop_score, sample_indices, visible_score


@pytest.mark.parametrize("function", [assess, visible_score, draw_routes, sample_indices])
@pytest.mark.parametrize("invalid", ["overlap", "negative", "boolean", "short_board"])
def test_public_evaluator_rejects_invalid_physical_cards(function, invalid):
    hole, board = (0, 5, 10, 15, 20), (24, 29, 34)
    if invalid == "overlap":
        board = (0, 29, 34)
    elif invalid == "negative":
        hole = (-1,) + hole[1:]
    elif invalid == "boolean":
        hole = (False,) + hole[1:]
    else:
        board = board[:2]
    args = (hole, board, 0) if function is draw_routes else (hole, board)
    with pytest.raises(ValueError):
        function(*args)


def test_evaluator_accepts_mixed_sequence_types():
    hole, board = (0, 5, 10, 15, 20), (24, 29, 34)
    assert assess(list(hole), board) == assess(hole, list(board))
    assert draw_routes(list(hole), board, 0) == draw_routes(hole, list(board), 0)


@pytest.mark.parametrize("hole", [(0, 1, 2, 3), (0, 0, 2, 3, 4), (-1, 1, 2, 3, 4)])
def test_preflop_score_rejects_invalid_hands(hole):
    with pytest.raises(ValueError):
        preflop_score(hole)


@pytest.mark.parametrize("salt", [-1, 2**32, True])
def test_sampler_salt_has_a_defined_unsigned_32_bit_domain(salt):
    with pytest.raises(ValueError):
        sample_indices((0, 5, 10, 15, 20), (24, 29, 34), salt)
