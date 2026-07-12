from legopolitics.prompting.schemas import ObservationOutput
from legopolitics.vision.ensemble import vote
from legopolitics.vision.parsing import parse_model


def test_parse_fenced_json():
    parsed = parse_model(
        '```json\n{"visible_evidence":["flag"],"uncertain_observations":[],"cannot_determine":[]}\n```',
        ObservationOutput,
    )
    assert parsed.visible_evidence == ["flag"]


def test_voting_modes():
    predictions = [("hero", 0.8, 1.0), ("hero", 0.6, 0.8), ("victim", 0.95, 0.5)]
    assert vote(predictions, "majority")["label"] == "hero"
    assert vote(predictions, "maximum_confidence")["candidate_label"] == "victim"
