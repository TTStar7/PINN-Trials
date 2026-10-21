"""
Collocation points: extra (unlabeled) input combinations, sampled across the
realistic range of each variable. We don't know the TRUE pit depth at these
points -- we only require that the network's OWN behavior obeys the physics
equation there. This is what makes it "physics-informed" rather than just
"fitted to 139 data points."
"""
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

df = pd.read_csv('/home/claude/corrosion_data/nist_corrosion_clean.csv')
FEATURES = ['pH', 'Cl_mol_kg', 'SO4_mol_kg', 'resistivity_ohm_m', 'moisture_equiv_pct', 'exp_time_yrs']

bounds = {f: (float(df[f].min()), float(df[f].max())) for f in FEATURES}

def sample_collocation_points(key, n_points, bounds, X_mean, X_std):
    """Returns n_points x 6 array, normalized the same way as training data."""
    raw = []
    keys = jax.random.split(key, len(FEATURES))
    for k, f in zip(keys, FEATURES):
        lo, hi = bounds[f]
        raw.append(jax.random.uniform(k, (n_points,), minval=lo, maxval=hi))
    raw = jnp.stack(raw, axis=1)
    normed = (raw - X_mean) / X_std
    return normed

if __name__ == "__main__":
    data = np.load('/home/claude/corrosion_nn/data.npz')
    X_mean, X_std = jnp.array(data['X_mean']), jnp.array(data['X_std'])

    print("Sampling bounds (real units):")
    for f, (lo, hi) in bounds.items():
        print(f"  {f}: [{lo:.3f}, {hi:.3f}]")

    key = jax.random.PRNGKey(0)
    coll_points = sample_collocation_points(key, 500, bounds, X_mean, X_std)
    print(f"\nCollocation points shape: {coll_points.shape}")
    print("First 3 (normalized):\n", coll_points[:3])
