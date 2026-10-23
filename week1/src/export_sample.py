"""Export a small real-data example without merchant/URL/source-ID columns."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
columns = ["price", "log_price", "brand", "model", "variant", "city", "registration_year", "vehicle_age_current", "age_at_listing_months", "mileage_km", "annual_mileage_km", "engine_liters", "gearbox", "emission_standard", "wheelbase_mm", "fuelType", "power"]
data = pd.read_csv(ROOT / "data/processed_capital_cars.csv", usecols=columns)
sample = data.sample(n=min(100, len(data)), random_state=2026)[columns].reset_index(drop=True)
sample.insert(0, "sample_id", range(1, len(sample) + 1))
sample.to_csv(ROOT / "data/sample_capital_cars.csv", index=False, encoding="utf-8-sig")
print(f"Exported {len(sample)} real rows.")
