from frank_eq.rate_compute.decision import _compiled_gate_status


def test_compiled_gate_status_keeps_prior_and_direct_diagnoses_separate() -> None:
    aggregate = {
        "compiled_over_prior_passed": True,
        "compiled_over_direct_passed": True,
    }
    groups = [
        {
            "compiled_over_prior_passed": True,
            "compiled_over_direct_passed": False,
        },
        {
            "compiled_over_prior_passed": True,
            "compiled_over_direct_passed": True,
        },
    ]

    prior_passed, direct_passed = _compiled_gate_status(groups, aggregate)

    assert prior_passed is True
    assert direct_passed is False
