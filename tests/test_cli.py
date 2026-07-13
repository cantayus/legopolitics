from click.utils import strip_ansi
from typer.testing import CliRunner

from legopolitics.cli import app

runner = CliRunner()


def test_about_and_version():
    about = runner.invoke(app, ["about"])
    assert about.exit_code == 0
    assert "not affiliated" in about.stdout
    version = runner.invoke(app, ["version"])
    assert version.exit_code == 0


def test_schema_and_config_template(tmp_path):
    schema = runner.invoke(app, ["schema"])
    assert schema.exit_code == 0
    target = tmp_path / "analysis.yaml"
    result = runner.invoke(app, ["config-template", "--output", str(target)])
    assert result.exit_code == 0 and target.exists()
    yolo_target = tmp_path / "lego-yolo.yaml"
    profile = runner.invoke(
        app,
        ["config-template", "--profile", "lego-yolo", "--output", str(yolo_target)],
    )
    assert profile.exit_code == 0 and "lego_figure_best.pt" in yolo_target.read_text()


def test_analyze_video_uses_named_output_option() -> None:
    result = runner.invoke(
        app,
        ["analyze-video", "--help"],
        color=False,
    )

    assert result.exit_code == 0, result.output

    help_text = strip_ansi(result.output)

    assert "--video" in help_text
    assert "--output" in help_text
