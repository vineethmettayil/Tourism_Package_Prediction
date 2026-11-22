import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")   # registered dataset inside the data folder

# Drop columns that carry no predictive meaning:
#   CustomerID  - unique customer identifier
#   Unnamed: 0  - a saved row index left over from an earlier export (it equals CustomerID - 200000)
df.drop(columns=["CustomerID", "Unnamed: 0"], inplace=True, errors="ignore")

# Data cleaning: 155 rows record Gender as "Fe Male", a typing error for "Female".
# Left as is, the model would treat it as a third gender.
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

target = "ProdTaken"  # 1 if the customer purchased the package, else 0
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Train rows: {len(Xtrain)}, Test rows: {len(Xtest)}")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
print("ProdTaken distribution in test:")
print(ytest.value_counts())
