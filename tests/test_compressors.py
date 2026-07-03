from sclab.compressors import Document, get_compressor


def test_all_requested_compressors_return_results():
    doc = Document(
        text=(
            "Contract. Monthly rent is GBP 1,100 per month.\n\n"
            "The warranty expires on 2027-04-15.\n\n"
            "Unrelated office note about chairs."
        ),
        metadata={"source_span": "Monthly rent is GBP 1,100 per month"},
    )
    for name in [
        "raw",
        "gzip_b64_control",
        "extractive_relevance",
        "semantic_brief",
        "fact_table",
        "hybrid_brief_excerpts",
        "oracle",
    ]:
        result = get_compressor(name).compress(doc, "What is the monthly rent?")
        assert result.method == name
        assert result.compressed_text
        assert result.original_tokens is not None
        assert result.compressed_tokens is not None


def test_extractive_preserves_exact_source_text():
    doc = Document(text="Noise paragraph.\n\nPayment terms are net 30 days with GBP 900 due.\n\nOther noise.")
    result = get_compressor("extractive_relevance").compress(doc, "What are the payment terms?")
    assert "Payment terms are net 30 days" in result.compressed_text


def test_extractive_recognizes_obligation_intent_without_exact_term_overlap():
    doc = Document(
        text=(
            "Office note about plants and badges.\n\n"
            "Service agreement. Vendor must deliver weekly status reports every Friday. "
            "Vendor must maintain 99.5% uptime each calendar month. "
            "Customer must pay invoices within 30 days of receipt.\n\n"
            "Archive note about furniture."
        )
    )
    result = get_compressor("extractive_relevance", budget=0.2).compress(
        doc,
        "Summarise the three main obligations and include deadlines or thresholds.",
    )
    assert "every Friday" in result.compressed_text
    assert "99.5% uptime" in result.compressed_text
    assert "30 days" in result.compressed_text


def test_extractive_intent_terms_do_not_match_inside_distractor_words():
    doc = Document(
        text=(
            "Facilities note: plants and badges were handled.\n\n"
            "Policy memo section A. Refund requests must be filed within 30 days of purchase. "
            "Policy memo section B. Refund requests must be filed within 14 days of purchase for the same product line.\n\n"
            "Archive note: furniture was discussed."
        )
    )
    result = get_compressor("extractive_relevance", budget=0.2).compress(
        doc,
        "Is there a contradiction about the refund request window?",
    )
    assert "30 days" in result.compressed_text
    assert "14 days" in result.compressed_text


def test_semantic_brief_budget_changes_output_size_but_keeps_relevant_precision():
    noise = "\n\n".join(
        f"Operations note {idx}: the office schedule mentioned room {idx}, chairs, coffee, and badge printing."
        for idx in range(40)
    )
    doc = Document(
        text=(
            f"{noise}\n\n"
            "Escalation playbook. The remediation amount is GBP 2,400 and the escalation deadline is 2026-11-20. "
            "The backup contact is Mira Patel.\n\n"
            "Audit appendix. The archival threshold is GBP 9,900 and the old review date was 2025-01-15."
        )
    )
    question = "What remediation amount and escalation deadline are listed?"

    tight = get_compressor("semantic_brief", budget=0.05).compress(doc, question)
    roomy = get_compressor("semantic_brief", budget=0.80).compress(doc, question)

    assert tight.compressed_tokens < roomy.compressed_tokens
    assert "GBP 2,400" in tight.compressed_text
    assert "2026-11-20" in tight.compressed_text


def test_budgeted_semantic_brief_prioritizes_exact_relevant_excerpt():
    doc = Document(
        text=(
            "Distractor after 1. Procurement note: envelopes and monitor arms were reordered.\n\n"
            "Quote comparison. Option A costs GBP 80 per month with a GBP 400 setup fee. "
            "Option B costs GBP 95 per month with no setup fee. For a 12 month term, taxes are excluded.\n\n"
            "Distractor after 2. People note: lighting and commute preferences were discussed."
        )
    )
    result = get_compressor("semantic_brief", budget=0.2).compress(
        doc,
        "Which option is cheaper over 12 months after setup fees?",
    )
    assert "Option B costs GBP 95 per month with no setup fee" in result.compressed_text
    assert "12 month term" in result.compressed_text
    assert "Distractor after" not in result.compressed_text
