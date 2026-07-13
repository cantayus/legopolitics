from legopolitics.tracking.association import IoUTracker


class ByteTrackAdapter(IoUTracker):
    """Portable approximation for table-level detections; use YoloTrackAdapter for native ByteTrack."""
