"""
The optimizer: Adam, via optax (JAX's standard optimizer library).

Why Adam instead of our plain "W -= lr * dW" from the NumPy version?
PINN loss landscapes (data loss + physics loss, often at very different
scales) are much bumpier than plain MSE. Adam adapts its step size per
parameter using running averages of past gradients, which handles that
bumpiness far better than fixed-learning-rate gradient descent.
"""
import jax
import optax

def make_optimizer(learning_rate=1e-3):
    return optax.adam(learning_rate)

def train_step(params, phys_params, opt_state, optimizer, loss_fn, *loss_args):
    all_params = {"net": params, "phys": phys_params}

    def wrapped_loss(all_params):
        total, aux = loss_fn(all_params["net"], all_params["phys"], *loss_args)
        return total, aux

    (total, aux), grads = jax.value_and_grad(wrapped_loss, has_aux=True)(all_params)
    updates, opt_state = optimizer.update(grads, opt_state)
    all_params = optax.apply_updates(all_params, updates)
    return all_params["net"], all_params["phys"], opt_state, total, aux

if __name__ == "__main__":
    import numpy as np, jax.numpy as jnp
    from model import init_params
    from loss import pinn_loss
    from collocation import sample_collocation_points, bounds

    data = np.load('/home/claude/corrosion_nn/data.npz')
    X_train = jnp.array(data['X_train']); y_train = jnp.array(data['y_train'])
    X_mean = jnp.array(data['X_mean']); X_std = jnp.array(data['X_std'])
    y_mean = jnp.array(data['y_mean']); y_std = jnp.array(data['y_std'])

    key = jax.random.PRNGKey(42)
    params = init_params(key)
    phys_params = {"A": jnp.array(0.1), "B": jnp.array(0.1), "beta": jnp.array(0.5),
                   "k_s": jnp.array(0.5), "m": jnp.array(0.5)}

    optimizer = make_optimizer(1e-3)
    opt_state = optimizer.init({"net": params, "phys": phys_params})

    X_coll = sample_collocation_points(jax.random.PRNGKey(0), 500, bounds, X_mean, X_std)
    loss_args = (X_train, y_train, X_coll, y_mean, y_std, X_mean, X_std, 0.1)

    print("Testing 5 training steps:")
    for step in range(5):
        params, phys_params, opt_state, total, (dl, pl) = train_step(
            params, phys_params, opt_state, optimizer, pinn_loss, *loss_args)
        print(f"  step {step}: total={total:.4f}  data={dl:.4f}  physics={pl:.4f}  A={phys_params['A']:.4f}")
