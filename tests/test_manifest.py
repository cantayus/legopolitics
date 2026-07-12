from pathlib import Path

from legopolitics.storage import RunManifest


def test_manifest_state_and_lock(tmp_path: Path):
    with RunManifest(tmp_path / "manifest.sqlite") as manifest:
        run = manifest.begin_run("hash")
        manifest.record_video("v", "video.mp4", "fp")
        started = manifest.begin_stage(run, "v", "sample", "key")
        manifest.finish_stage(run, "v", "sample", started)
        assert manifest.stage_status(run, "v", "sample") == "complete"
        with manifest.lock("v:sample"):
            pass
        manifest.finish_run(run)
        assert manifest.inspect()["runs"][0]["status"] == "complete"
