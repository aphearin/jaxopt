"""
"""
import argparse
from jax import value_and_grad
from jax import random as jax_random
from jax.experimental import optimizers as jax_opt

from elsewhere import read_init_data
from elsewhere import get_init_params
from elsewhere import get_target_data
from elsewhere import get_mse_loss_data
from elsewhere import calculate_mse_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_iter", type=int, help="Number of GD steps")
    parser.add_argument("-seed", type=int, default=43, help="Random number seed")
    parser.add_argument(
        "-resample_rate",
        type=float,
        default=0.0,
        help="Frequency of target data resampling",
    )
    args = parser.parse_args()
    n_iter = args.n_iter
    resample_rate = args.resample_rate
    ran_key = jax_random.PRNGKey(args.seed)

    data_init = read_init_data()
    params_init = get_init_params()

    data_target, ran_key, metadata = get_target_data(data_init, ran_key)

    data_mse_loss, ran_key = get_mse_loss_data(data_target, metadata, ran_key)

    loss_init = calculate_mse_loss(params_init, data_mse_loss)

    opt_init, opt_update, get_params = jax_opt.adam(1e-3)
    opt_state = opt_init(params_init)

    for istep in range(n_iter):
        loss, grads = value_and_grad(calculate_mse_loss, argnums=0)(
            get_params(opt_state), data_mse_loss
        )
        opt_state = opt_update(istep, grads, opt_state)
