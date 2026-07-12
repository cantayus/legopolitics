from legopolitics.validation import classification_metrics, cohens_kappa, percent_agreement


def test_validation_metrics():
    true = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", "b"]
    metrics = classification_metrics(true, pred, "a")
    assert metrics["accuracy"] == 0.75
    assert percent_agreement(true, pred) == 0.75
    assert -1 <= cohens_kappa(true, pred) <= 1
