from legopolitics.detection.registry import DETECTORS

class MyDetector:
    def load(self):
        return None
    def predict(self, images, frame_ids, video_id):
        return []
    def metadata(self):
        return {"adapter": "my_detector"}
    def unload(self):
        return None

DETECTORS.register("my_detector", MyDetector)
