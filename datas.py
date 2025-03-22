from pathlib import Path

import pandas as pd

app_dir = Path(__file__).parent
data = pd.read_csv(app_dir / "stockdata.csv")

