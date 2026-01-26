from sparket.shared.probability import (
    eu_to_implied_prob,
    normalize_vector,
    normalize_probs,
    implied_from_eu_vector,
    implied_from_eu_odds,
)


def test_eu_to_implied_prob():
    assert abs(eu_to_implied_prob(2.0) - 0.5) < 1e-9


def test_normalize_vector_and_overround():
    raw = [0.6, 0.6]
    norm, over = normalize_vector(raw)
    assert abs(sum(norm) - 1.0) < 1e-9
    assert abs(over - (sum(raw) - 1.0)) < 1e-9


def test_normalize_probs_map():
    norm, over = normalize_probs({"home": 0.6, "away": 0.6})
    assert abs(sum(norm.values()) - 1.0) < 1e-9
    assert abs(over - 0.2) < 1e-9


def test_implied_from_eu_helpers():
    raw, norm, over = implied_from_eu_vector([2.0, 2.0])
    assert raw == [0.5, 0.5]
    assert abs(sum(norm) - 1.0) < 1e-9
    rmap, nmap, over2 = implied_from_eu_odds({"home": 2.0, "away": 2.0})
    assert rmap["home"] == 0.5 and rmap["away"] == 0.5
    assert abs(sum(nmap.values()) - 1.0) < 1e-9
    assert abs(over - over2) < 1e-9
