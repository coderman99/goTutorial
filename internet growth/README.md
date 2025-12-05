# Internet Growth Toolkit

This folder contains simple Python utilities for synthesizing internet usage
records, projecting growth trajectories, and saving the results for downstream
analysis. Add the folder to your `PYTHONPATH` (e.g., `export PYTHONPATH="$(pwd)/internet growth:$PYTHONPATH"`)
so the modules can be imported as `data_fetcher`, `model_builder`, and
`data_storage`.

## Modules
- **data_fetcher.py**: Builds or retrieves internet usage observations and
  organizes them by region.
- **model_builder.py**: Computes a compounded growth projection that tapers near
  a configurable saturation point.
- **data_storage.py**: Persists raw records to CSV and model summaries to JSON,
  creating parent directories as needed.

## Quickstart
```
python - <<'PY'
import sys
from pathlib import Path

module_root = Path('internet growth').resolve()
sys.path.append(str(module_root))

import data_fetcher
import model_builder
import data_storage

regions = ['Europe', 'Latin America']
records = data_fetcher.fetch_usage_data(regions, 2002, 2008, seed=21)
summary = model_builder.build_growth_model(records, years_ahead=5, saturation=0.95)

data_storage.save_usage_records_csv(records, module_root / 'examples' / 'usage.csv')
data_storage.save_model_summary(summary, module_root / 'examples' / 'summary.json')
PY
```
