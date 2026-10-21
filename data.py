import pandas as pd
import numpy as np

df = pd.read_csv('nist_corrosion_clean.csv')

input_cols = ['pH', 'Cl_mol_kg', 'SO4_mol_kg',
              'resistivity_ohm_m', 'moisture_equiv_pct', 'exp_time_yrs']
X = df[input_cols].values
y = df['mean_penetration_mm'].values
sites = df['location'].values            # <- your location column name

# choose 4 whole sites to hold out as the test set
rng = np.random.default_rng(3)
test_sites = rng.choice(np.unique(sites), size=4, replace=False)
is_test = np.isin(sites, test_sites)

X_tr, X_te = X[~is_test], X[is_test]
y_tr, y_te = y[~is_test], y[is_test]

# normalize using TRAIN statistics only
X_mean, X_std = X_tr.mean(axis=0), X_tr.std(axis=0)
y_mean, y_std = y_tr.mean(), y_tr.std()

np.savez('data.npz',
         X_train=(X_tr - X_mean) / X_std, y_train=(y_tr - y_mean) / y_std,
         X_test=(X_te - X_mean) / X_std,  y_test=(y_te - y_mean) / y_std,
         X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)

print("test sites:", test_sites)
print("train rows:", len(y_tr), " test rows:", len(y_te))