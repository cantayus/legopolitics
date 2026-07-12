from legopolitics.analysis.relations import infer_proximity_relations
from legopolitics.analysis.salience import compute_salience
from legopolitics.schemas import BoundingBox, DetectionRecord, TrackObservation


def test_salience_components():
    detection = DetectionRecord(
        detection_id="d",
        video_id="v",
        frame_id="f",
        detector_id="x",
        class_name="figure",
        confidence=0.9,
        box=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        normalized_box=BoundingBox(x1=0.4, y1=0.4, x2=0.6, y2=0.6),
        frame_area_fraction=0.04,
    )
    result = compute_salience(detection)
    assert 0 <= result.composite <= 1


def test_relations_from_cooccurrence():
    obs = [
        TrackObservation(
            track_id="a", frame_id="f", detection_id="d1", timestamp_seconds=1, confidence=0.9
        ),
        TrackObservation(
            track_id="b", frame_id="f", detection_id="d2", timestamp_seconds=1, confidence=0.8
        ),
    ]
    relations = infer_proximity_relations(obs, "v")
    assert len(relations) == 1
    assert relations[0].predicate == "co_occurs_with"
