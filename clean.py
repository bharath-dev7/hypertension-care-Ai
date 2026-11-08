import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Load the data and keep the raw copy untouched
raw = pd.read_csv("datasets/Hypertension-risk-model-main.csv")
df = raw.copy()

# 2. Inspect: size, types, missing values, duplicates, class balance
print(df.shape)
df.info()
print(df.isnull().sum())
print("duplicates:", df.duplicated().sum())
print(df["Risk"].value_counts(normalize=True))

# 3. Remove duplicates (none expected here)
df = df.drop_duplicates()

# 4. Check for impossible values
print("sys <= dia rows:", (df.sysBP <= df.diaBP).sum())   # should be 0
print(df.describe())                                       # check min and max of each column

# 5. Separate features and target, then split BEFORE imputing or scaling
X = df.drop(columns="Risk")
y = df["Risk"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)      # stratify keeps the 31/69 ratio in both sets

# 6. Impute missing values: fit on train only, then apply to test
imp = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(imp.transform(X_test), columns=X.columns)

# 7. Build the early-risk feature set (drops the BP-related columns)
bp_cols = ["sysBP", "diaBP", "BPMeds"]
X_train_early = X_train.drop(columns=bp_cols)
X_test_early = X_test.drop(columns=bp_cols)

# 8. Scale (needed only for logistic regression; fit on train only)
sc = StandardScaler().fit(X_train)
X_train_scaled = sc.transform(X_train)
X_test_scaled = sc.transform(X_test)

# 9. Save the cleaned data
X_train.assign(Risk=y_train.values).to_csv("train_clean.csv", index=False)
X_test.assign(Risk=y_test.values).to_csv("test_clean.csv", index=False)


# Sanity check: no missing values left, shapes and class ratio are correct
print(X_train.shape, X_test.shape, X_train_early.shape)
print("nulls left:", X_train.isnull().sum().sum(), X_test.isnull().sum().sum())
print(y_train.mean().round(3), y_test.mean().round(3))   # both should be about 0.311