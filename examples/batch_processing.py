from pathlib import Path
from legopolitics import AnalysisConfig, LegoPoliticsAnalyzer

analyzer = LegoPoliticsAnalyzer(AnalysisConfig.from_yaml("configs/minimal.yaml"))
analyzer.analyze_directory(Path("videos"), Path("outputs"), num_batches=4, batch_id=0)
