"""
The model: same 6-in -> 8-hidden -> 1-out network as before,
but written in JAX so we can differentiate its OUTPUT w.r.t. its INPUT
(not just w.r.t. the weights, which is all plain backprop gives you).
"""
import jax
import jax.numpy as jnp

def init_params(key, n_features=6, n_hidden=8, n_output=1):
    k1, k2 = jax.random.split(key)
    W1 = jax.random.normal(k1, (n_features, n_hidden)) * jnp.sqrt(1.0 / n_features)
    b1 = jnp.zeros((n_hidden,))
    W2 = jax.random.normal(k2, (n_hidden, n_output)) * jnp.sqrt(1.0 / n_hidden)
    b2 = jnp.zeros((n_output,))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def model(params, x):
    """x: shape (6,) -- ONE site's inputs. Returns scalar pit-depth guess."""
    z1 = x @ params["W1"] + params["b1"]
    a1 = jnp.tanh(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    return z2[0]   # scalar

# batched version (applies model() to many rows at once)
model_batch = jax.vmap(model, in_axes=(None, 0))

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    params = init_params(key)
    x_test = jnp.array([0.356, -0.265, -0.416, -0.588, 0.69, -1.134])
    print("Single prediction:", model(params, x_test))

    X_batch = jnp.stack([x_test, x_test])
    print("Batched prediction:", model_batch(params, X_batch))
