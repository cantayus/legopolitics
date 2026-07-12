import numpy as np

from legopolitics.video.hashing import difference_hash, hamming_distance
from legopolitics.video.quality import analyze_frame_quality


def test_quality_and_hash():
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    white = np.full((64, 64, 3), 255, dtype=np.uint8)
    q = analyze_frame_quality(black)
    assert q.black_frame_score > 0.9
    assert hamming_distance(difference_hash(black), difference_hash(black)) == 0
    assert isinstance(hamming_distance(difference_hash(black), difference_hash(white)), int)
