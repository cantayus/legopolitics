from pathlib import Path
from legopolitics import AnalysisConfig, LegoPoliticsAnalyzer

config = AnalysisConfig.from_yaml(Path("configs/minimal.yaml"))
result = LegoPoliticsAnalyzer(config).analyze_video(
    Path("videos/example.mp4"), Path("outputs/example")
)
print(result.output_paths)
