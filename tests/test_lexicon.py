from crossword_agent.lexicon import PatternLexicon


def test_pattern_lookup_and_frequency_ranking() -> None:
    lexicon = PatternLexicon(
        ["cat", "car", "can", "dog"],
        frequency_scores={"CAT": 0.9, "CAR": 0.8, "CAN": 0.7},
    )

    assert lexicon.contains("Cat")
    assert lexicon.find("CA?") == ["CAT", "CAR", "CAN"]
    assert lexicon.find("?O?", limit=1) == ["DOG"]
    assert lexicon.frequency_score("unknown") == 0.0

