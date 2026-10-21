"""
The loss function. Two parts added together:
  1. data_loss     -- standard MSE against our 139 real measurements
  2. physics_loss  -- how badly the network's OWN output violates
                       dh/dt = physical_rate(pH, Cl, SO4, resistivity, moisture)
                       at the collocation points, with NO known answer needed.

Physical constants (A, B, beta, k_s) are LEARNABLE too -- the network and the
physics coefficients train together, which is the "inverse PINN" idea from
our original design discussion.
"""
import jax
import jax.numpy as jnp
from model import model, model_batch

def physical_rate(phys_params, x_row, y_mean, y_std, X_mean, X_std):
    """
    x_row: normalized (pH, Cl, SO4, resistivity, moisture, time)
    Un-normalizes to real units, computes a Butler-Volmer/Faraday-style rate
    law for dh/dt (mm/yr), then re-expresses it in NORMALIZED units so it's
    comparable to the network's own (normalized) dh/dt.
    """
    x_real = x_row * X_std + X_mean
    pH, Cl, SO4, resistivity, moisture = x_real[0], x_real[1], x_real[2], x_real[3], x_real[4]

    A, B, beta, k_s, m = phys_params["A"], phys_params["B"], phys_params["beta"], phys_params["k_s"], phys_params["m"]

    # Butler-Volmer-inspired exchange-current proxy -> Faraday-converted rate
    rate_mm_per_yr = (
        A
        * jnp.exp(-B * (pH - 7.0))              # lower pH -> faster corrosion
        * (jnp.clip(Cl, 1e-6, None) ** beta)     # more chloride -> faster (sub-linear)
        * (1.0 + k_s * SO4)                      # sulfate -> mild accelerant
        * (moisture ** m) / (resistivity + 1.0)  # wetter, less resistive soil -> faster
    )
    # convert real mm/yr rate into the NORMALIZED dh/dt scale the network uses
    return rate_mm_per_yr * X_std[-1] / y_std[0]   # X_std[-1] = time's std (chain rule for normalization)

def pinn_loss(params, phys_params, X_data, y_data, X_coll, y_mean, y_std, X_mean, X_std, lambda_phys):
    # ---- 1. data loss ----
    preds = model_batch(params, X_data)
    data_loss = jnp.mean((preds - y_data[:, 0]) ** 2)

    # ---- 2. physics loss, one collocation point at a time ----
    def residual_at_point(x_row):
        # dh/dt from the NETWORK: differentiate its output w.r.t. the time input (index 5)
        dmodel_dx = jax.grad(model, argnums=1)(params, x_row)   # vector of d(output)/d(each input)
        dh_dt_network = dmodel_dx[5] / X_std[-1] * 1.0            # chain rule: input was normalized

        dh_dt_physics = physical_rate(phys_params, x_row, y_mean, y_std, X_mean, X_std)
        return (dh_dt_network - dh_dt_physics) ** 2

    residuals = jax.vmap(residual_at_point)(X_coll)
    physics_loss = jnp.mean(residuals)

    total = data_loss + lambda_phys * physics_loss
    return total, (data_loss, physics_loss)

if __name__ == "__main__":
    import numpy as np
    from model import init_params
    from collocation import sample_collocation_points, bounds

    data = np.load('/home/claude/corrosion_nn/data.npz')
    X_train = jnp.array(data['X_train']); y_train = jnp.array(data['y_train'])
    X_mean = jnp.array(data['X_mean']); X_std = jnp.array(data['X_std'])
    y_mean = jnp.array(data['y_mean']); y_std = jnp.array(data['y_std'])

    key = jax.random.PRNGKey(42)
    params = init_params(key)
    phys_params = {"A": jnp.array(0.1), "B": jnp.array(0.1), "beta": jnp.array(0.5),
                   "k_s": jnp.array(0.5), "m": jnp.array(0.5)}

    X_coll = sample_collocation_points(jax.random.PRNGKey(0), 500, bounds, X_mean, X_std)

    total, (dl, pl) = pinn_loss(params, phys_params, X_train, y_train, X_coll,
                                  y_mean, y_std, X_mean, X_std, lambda_phys=0.1)
    print(f"data_loss={dl:.4f}  physics_loss={pl:.4f}  total={total:.4f}")
