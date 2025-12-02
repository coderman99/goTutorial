from db_loader import load_indicator_data, load_sp500_data
from labelling import label_business_cycle
from model import create_future_targets

print("\n=== Loading Indicators ===")
df = load_indicator_data()
print(df.head())
print("Indicator rows:", len(df))

print("\n=== Loading SP500 ===")
spx = load_sp500_data()
print(spx.head())
print("SPX rows:", len(spx))

print("\n=== Label Business Cycle ===")
labeled = label_business_cycle(df.set_index("timestamp"), spx)
print(labeled.head())
print("Labeled rows:", len(labeled))

print("\n=== Create Future Targets ===")
future = create_future_targets(labeled)
print(future.head())
print("Rows after target creation:", len(future))
