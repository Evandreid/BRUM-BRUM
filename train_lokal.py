import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

print("⏳ Sedang melatih model di laptop...")

# 1. Baca data
df = pd.read_csv("ds_cars.csv")
df.replace("?", np.nan, inplace=True)
df.dropna(subset=["price"], inplace=True)
df["price"] = df["price"].astype(float)

# 2. Paksa konversi kolom angka dari teks ke float
num_cols = [
    "normalized-losses",
    "bore",
    "stroke",
    "horsepower",
    "peak-rpm",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-size",
    "compression-ratio",
    "city-mpg",
    "highway-mpg",
]
for col in num_cols:
  if col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3. Categorical encoding hanya untuk kolom teks murni
nominal_cols = [
    "make",
    "fuel-type",
    "aspiration",
    "body-style",
    "drive-wheels",
    "engine-location",
    "engine-type",
    "fuel-system",
    "num-of-doors",
    "num-of-cylinders",
]
df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# 4. Isi data kosong (NaN) dengan median
X = df_encoded.drop(columns=["price"], errors="ignore")
X = X.fillna(X.median(numeric_only=True))
y = df_encoded["price"]

# 5. Scaling & Training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 6. Simpan model, scaler, dan daftar nama kolom fitur
joblib.dump(model, "model_mobil.pkl")
joblib.dump(scaler, "scaler_mobil.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print(
    "🎉 Sukses! File model_mobil.pkl, scaler_mobil.pkl, & feature_columns.pkl"
    " berhasil dibuat!"
)