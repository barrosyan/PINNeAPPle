from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from pinneaple_arena.bundle.loader import BundleData
from pinneaple_arena.registry import register_backend


@register_backend
@dataclass
class JAXPINNBackend:
    """
    Full PINN backend in JAX for FlowObstacle2D (steady 2D Navier–Stokes).

    Training:
      - PDE residual on collocation points (momentum + continuity)
      - BC loss on boundary points
      - Optional supervised loss on sensors (if w_data>0 and sensors exist)

    Output:
      - predict_fn: numpy callable (N,2)->(N,3)
      - metrics: includes train_* and eval keys:
          test_pde_rms, test_div_rms, bc_mse, test_l2_uv
    """

    name: str = "jax_pinn"

    def train(self, bundle: BundleData, run_cfg: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import jax
            import jax.numpy as jnp
            import optax
        except Exception as e:
            raise RuntimeError(
                "JAXPINNBackend requires jax + optax.\n"
                "Install (CPU): pip install 'jax[cpu]' optax\n"
                "GPU install: follow https://jax.readthedocs.io/en/latest/installation.html"
            ) from e

        train_cfg = dict(run_cfg.get("train", {}))
        model_cfg = dict(run_cfg.get("model", {}))
        arena_cfg = dict(run_cfg.get("arena", {}))

        steps = int(train_cfg.get("steps", train_cfg.get("epochs", 20000)))
        lr = float(train_cfg.get("lr", 1e-3))
        seed = int(train_cfg.get("seed", 0))
        log_every = int(train_cfg.get("log_every", 500))

        weights = dict(train_cfg.get("weights", {}))
        w_pde = float(weights.get("pde", 1.0))
        w_bc = float(weights.get("bc", 10.0))
        w_data = float(weights.get("data", 0.0))

        n_collocation = int(arena_cfg.get("n_collocation", 4096))
        n_boundary = int(arena_cfg.get("n_boundary", 2048))
        n_data = int(arena_cfg.get("n_data", 0))

        width = int(model_cfg.get("width", 256))
        depth = int(model_cfg.get("depth", 4))
        act = str(model_cfg.get("activation", "tanh")).lower()

        nu = float(bundle.manifest["nu"])

        rng = np.random.default_rng(seed)

        def _sample_df(df, n: int):
            n = min(int(n), len(df))
            if n <= 0:
                return df.iloc[:0]
            return df.sample(n=n, replace=(len(df) < n), random_state=rng.integers(0, 2**31 - 1))

        col_df = bundle.points_collocation
        bnd_df = bundle.points_boundary

        sensors_df = None
        if bundle.sensors is not None and {"x", "y", "u", "v"}.issubset(bundle.sensors.columns):
            sensors_df = bundle.sensors
            if "split" in sensors_df.columns:
                sensors_df = sensors_df[sensors_df["split"].astype(str) == "train"]
            if len(sensors_df) == 0:
                sensors_df = None

        key = jax.random.PRNGKey(seed)

        def activation(x):
            if act == "tanh":
                return jnp.tanh(x)
            if act == "relu":
                return jnp.maximum(x, 0.0)
            if act == "gelu":
                return jax.nn.gelu(x)
            if act in ("silu", "swish"):
                return jax.nn.silu(x)
            return jnp.tanh(x)

        def init_params(key) -> list[Tuple[jnp.ndarray, jnp.ndarray]]:
            keys = jax.random.split(key, depth + 1)
            params = []
            d0 = 2
            for i in range(depth):
                k = keys[i]
                w = jax.random.normal(k, (d0, width), dtype=jnp.float32) * jnp.sqrt(2.0 / d0)
                b = jnp.zeros((width,), dtype=jnp.float32)
                params.append((w, b))
                d0 = width
            k = keys[-1]
            w = jax.random.normal(k, (d0, 3), dtype=jnp.float32) * jnp.sqrt(2.0 / d0)
            b = jnp.zeros((3,), dtype=jnp.float32)
            params.append((w, b))
            return params

        def mlp_apply(params, xy: jnp.ndarray) -> jnp.ndarray:
            h = xy
            for (w, b) in params[:-1]:
                h = activation(jnp.dot(h, w) + b)
            w, b = params[-1]
            return jnp.dot(h, w) + b

        params = init_params(key)

        def uvp_single(params, xy_single: jnp.ndarray) -> jnp.ndarray:
            return mlp_apply(params, xy_single[None, :])[0]

        jac_uvp = jax.jacrev(uvp_single, argnums=1)

        def u_single(params, xy_single):
            return uvp_single(params, xy_single)[0]

        def v_single(params, xy_single):
            return uvp_single(params, xy_single)[1]

        hess_u = jax.hessian(u_single, argnums=1)
        hess_v = jax.hessian(v_single, argnums=1)

        @jax.jit
        def residuals_batch(params, xy: jnp.ndarray):
            uvp = jax.vmap(lambda z: uvp_single(params, z))(xy)  # (N,3)
            J = jax.vmap(lambda z: jac_uvp(params, z))(xy)       # (N,3,2)

            u = uvp[:, 0:1]
            v = uvp[:, 1:2]
            p = uvp[:, 2:3]

            u_x = J[:, 0:1, 0:1]
            u_y = J[:, 0:1, 1:2]
            v_x = J[:, 1:2, 0:1]
            v_y = J[:, 1:2, 1:2]
            p_x = J[:, 2:3, 0:1]
            p_y = J[:, 2:3, 1:2]

            Hu = jax.vmap(lambda z: hess_u(params, z))(xy)  # (N,2,2)
            Hv = jax.vmap(lambda z: hess_v(params, z))(xy)

            u_xx = Hu[:, 0:1, 0:1]
            u_yy = Hu[:, 1:2, 1:2]
            v_xx = Hv[:, 0:1, 0:1]
            v_yy = Hv[:, 1:2, 1:2]

            mom_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
            mom_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
            cont = u_x + v_y
            return mom_u, mom_v, cont

        def encode_regions(regions_np: np.ndarray) -> np.ndarray:
            out = np.full_like(regions_np, 1, dtype=np.int32)  # default walls
            out[regions_np == "inlet"] = 0
            out[regions_np == "walls"] = 1
            out[regions_np == "obstacle"] = 2
            out[regions_np == "outlet"] = 3
            return out

        def bc_loss(params, xy_b: jnp.ndarray, regions: jnp.ndarray) -> jnp.ndarray:
            uvp = mlp_apply(params, xy_b)
            u = uvp[:, 0]
            v = uvp[:, 1]
            p = uvp[:, 2]

            m_in = regions == 0
            m_w = regions == 1
            m_o = regions == 2
            m_out = regions == 3

            terms = []
            if jnp.any(m_in):
                terms.append(jnp.mean((u[m_in] - 1.0) ** 2))
                terms.append(jnp.mean((v[m_in] - 0.0) ** 2))
            if jnp.any(m_w):
                terms.append(jnp.mean((u[m_w] - 0.0) ** 2))
                terms.append(jnp.mean((v[m_w] - 0.0) ** 2))
            if jnp.any(m_o):
                terms.append(jnp.mean((u[m_o] - 0.0) ** 2))
                terms.append(jnp.mean((v[m_o] - 0.0) ** 2))
            if jnp.any(m_out):
                terms.append(jnp.mean((p[m_out] - 0.0) ** 2))

            return jnp.mean(jnp.stack(terms)) if len(terms) else jnp.array(0.0, dtype=jnp.float32)

        def data_loss(params, xy_s: jnp.ndarray, y_s: jnp.ndarray) -> jnp.ndarray:
            uvp = mlp_apply(params, xy_s)
            return jnp.mean((uvp - y_s) ** 2)

        @jax.jit
        def total_loss(params, batch: Dict[str, jnp.ndarray]):
            mom_u, mom_v, cont = residuals_batch(params, batch["xy_col"])
            pde = jnp.mean(mom_u**2) + jnp.mean(mom_v**2) + jnp.mean(cont**2)

            bc = bc_loss(params, batch["xy_bnd"], batch["reg_bnd"])

            data = jnp.array(0.0, dtype=jnp.float32)
            if "xy_data" in batch:
                data = data_loss(params, batch["xy_data"], batch["uvp_data"])

            total = w_pde * pde + w_bc * bc + w_data * data
            return total, {"pde": pde, "bc": bc, "data": data}

        opt = optax.adam(lr)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state, batch):
            (loss, parts), grads = jax.value_and_grad(total_loss, has_aux=True)(params, batch)
            updates, opt_state2 = opt.update(grads, opt_state, params)
            params2 = optax.apply_updates(params, updates)
            return params2, opt_state2, loss, parts

        last = {"total": np.nan, "pde": np.nan, "bc": np.nan, "data": np.nan}

        for s in range(steps):
            col = _sample_df(col_df, n_collocation)
            bnd = _sample_df(bnd_df, n_boundary)

            xy_col = col[["x", "y"]].to_numpy(dtype=np.float32)
            xy_bnd = bnd[["x", "y"]].to_numpy(dtype=np.float32)
            reg_bnd = encode_regions(bnd["region"].astype(str).to_numpy())

            batch = {
                "xy_col": jnp.asarray(xy_col),
                "xy_bnd": jnp.asarray(xy_bnd),
                "reg_bnd": jnp.asarray(reg_bnd),
            }

            if w_data > 0.0 and sensors_df is not None and n_data > 0:
                sen = _sample_df(sensors_df, n_data)
                xy_s = sen[["x", "y"]].to_numpy(dtype=np.float32)
                uvp_s = np.zeros((xy_s.shape[0], 3), dtype=np.float32)
                uvp_s[:, 0] = sen["u"].to_numpy(dtype=np.float32)
                uvp_s[:, 1] = sen["v"].to_numpy(dtype=np.float32)
                if "p" in sen.columns:
                    uvp_s[:, 2] = sen["p"].to_numpy(dtype=np.float32)
                batch["xy_data"] = jnp.asarray(xy_s)
                batch["uvp_data"] = jnp.asarray(uvp_s)

            params, opt_state, loss, parts = step(params, opt_state, batch)

            if (s % log_every) == 0 or s == steps - 1:
                last = {
                    "total": float(loss),
                    "pde": float(parts["pde"]),
                    "bc": float(parts["bc"]),
                    "data": float(parts["data"]),
                }
                print(
                    f"[jax_pinn] step={s:06d} total={last['total']:.3e} "
                    f"pde={last['pde']:.3e} bc={last['bc']:.3e} data={last['data']:.3e}"
                )

        # ---------- evaluation ----------
        def eval_metrics(params) -> Dict[str, float]:
            col = col_df.sample(n=min(8192, len(col_df)), replace=False, random_state=0)
            bnd = bnd_df.sample(n=min(4096, len(bnd_df)), replace=False, random_state=0)

            xy_col = jnp.asarray(col[["x", "y"]].to_numpy(dtype=np.float32))
            mom_u, mom_v, cont = residuals_batch(params, xy_col)
            test_pde_rms = jnp.sqrt((jnp.mean(mom_u**2) + jnp.mean(mom_v**2)) / 2.0)
            test_div_rms = jnp.sqrt(jnp.mean(cont**2))

            xy_bnd = jnp.asarray(bnd[["x", "y"]].to_numpy(dtype=np.float32))
            reg_bnd = jnp.asarray(encode_regions(bnd["region"].astype(str).to_numpy()))
            bc = bc_loss(params, xy_bnd, reg_bnd)

            l2_uv = jnp.array(jnp.nan, dtype=jnp.float32)
            if bundle.sensors is not None and {"x", "y", "u", "v"}.issubset(bundle.sensors.columns):
                sen = bundle.sensors
                sen = sen[sen["split"].astype(str) == "test"] if "split" in sen.columns else sen
                if len(sen) > 0:
                    sen = sen.sample(n=min(4096, len(sen)), replace=False, random_state=0)
                    xy_s = jnp.asarray(sen[["x", "y"]].to_numpy(dtype=np.float32))
                    uvp_s = mlp_apply(params, xy_s)
                    u_hat = np.array(uvp_s[:, 0])
                    v_hat = np.array(uvp_s[:, 1])
                    u_true = sen["u"].to_numpy(dtype=np.float32)
                    v_true = sen["v"].to_numpy(dtype=np.float32)
                    l2_uv = jnp.asarray(
                        np.sqrt(np.mean((u_hat - u_true) ** 2 + (v_hat - v_true) ** 2)),
                        dtype=jnp.float32,
                    )

            return {
                "test_pde_rms": float(test_pde_rms),
                "test_div_rms": float(test_div_rms),
                "bc_mse": float(bc),
                "test_l2_uv": float(l2_uv),
            }

        metrics_eval = eval_metrics(params)

        @jax.jit
        def predict_jit(params, x):
            return mlp_apply(params, x)

        def predict_fn(xy: np.ndarray) -> np.ndarray:
            xy = np.asarray(xy, dtype=np.float32)
            y = np.array(predict_jit(params, jnp.asarray(xy)), dtype=np.float32)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if y.shape[1] != 3:
                out = np.zeros((y.shape[0], 3), dtype=np.float32)
                out[:, : min(3, y.shape[1])] = y[:, : min(3, y.shape[1])]
                return out
            return y

        return {
            "predict_fn": predict_fn,
            "metrics": {
                "backend": "jax_pinn",
                "train_total": float(last["total"]),
                "train_pde": float(last["pde"]),
                "train_bc": float(last["bc"]),
                "train_data": float(last["data"]),
                **metrics_eval,
            },
        }