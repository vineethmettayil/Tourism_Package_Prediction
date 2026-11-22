import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"  # raw dataset inside the data folder (paths are relative to the repo root)

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Flag any column that is not in the data dictionary (the raw file carries a saved row index)
extra = [c for c in df.columns if c not in expected_columns]
if extra:
    print(f"Note: columns not in the data dictionary (dropped during data prep): {extra}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Missing values in total:", int(df.isna().sum().sum()))
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())
