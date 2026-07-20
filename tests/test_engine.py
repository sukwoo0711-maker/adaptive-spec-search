from adaptive_spec_search import Document, FeedbackEvent, HybridSearchEngine


DOCS = [
    Document("power", "예약 타이머 만료 후 자동 전원 종료 scheduled power off"),
    Document("booking", "회의실 예약 reservation booking 시간 제한"),
    Document("network", "네트워크 연결 시간 초과 timeout retry"),
]


def tiny_embed(texts):
    vocabulary = ["전원", "종료", "예약", "회의실", "네트워크", "timeout"]
    return [[text.casefold().count(term) for term in vocabulary] for text in texts]


def test_hybrid_search_finds_semantic_and_exact_terms():
    engine = HybridSearchEngine(
        DOCS,
        embed=tiny_embed,
        synonyms={"꺼지는": ["전원 종료", "power off"], "예약": ["scheduled"]},
    )
    assert engine.search("예약 후 전원이 꺼지는 현상")[0].document.id == "power"


def test_metadata_filter_is_applied_after_candidate_generation():
    docs = [
        Document("old", "power shutdown timer", {"version": "1"}),
        Document("new", "power shutdown timer", {"version": "2"}),
    ]
    engine = HybridSearchEngine(docs)
    result = engine.search("shutdown", metadata_filter={"version": "2"})
    assert [row.document.id for row in result] == ["new"]


def test_explicit_feedback_can_correct_a_similar_future_query():
    engine = HybridSearchEngine(DOCS)
    before = engine.search("예약 시간", expand=False)
    engine.add_feedback(FeedbackEvent("예약 시간", "power", True, "correct function"))
    engine.add_feedback(FeedbackEvent("예약 시간", "booking", False, "wrong domain"))
    after = engine.search("예약 시간", expand=False)
    assert before
    assert after[0].document.id == "power"


def test_unknown_feedback_document_is_rejected():
    engine = HybridSearchEngine(DOCS)
    try:
        engine.add_feedback(FeedbackEvent("query", "missing", True))
    except KeyError:
        pass
    else:
        raise AssertionError("unknown document feedback must fail")
