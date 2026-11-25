"""Paso 5: compuertas, métricas selectivas y pruebas estadísticas."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from vgate import config, evaluate  # noqa: E402

runs = pd.read_parquet(config.RESULTS / "runs.parquet")
report = evaluate.run(runs)
print(json.dumps(report, indent=2, ensure_ascii=False))
