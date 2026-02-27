from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from pinneaple_arena.bundle.loader import BundleData
from pinneaple_arena.registry import register_backend


@register_backend
class DeepXDEBackend:
    """DeepXDE backend (baseline supervisionado em sensores).

    Requer: bundle.sensors com colunas x,y,u,v,(p opcional).

    Output:
      - predict_fn: numpy callable (N,2)->(N,3)
      - metrics: metadados de treino
    """

    name: str = "deepxde"

    def train(self, bundle: BundleData, run_cfg: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import deepxde as dde  # type: ignore
        except Exception as e:
            raise RuntimeError("DeepXDE backend requested but deepxde is not installed. Install: pip install deepxde") from e

        if bundle.sensors is None:
            raise RuntimeError("DeepXDEBackend requires bundle.sensors with columns x,y,u,v,(p optional).")

        cfg = dict(run_cfg.get("train", {}))
        model_cfg = dict(run_cfg.get("model", {}))

        backend = str(cfg.get("deepxde_backend", "")).strip().lower()
        if backend:
            try:
                dde.backend.set_default_backend(backend)
            except Exception:
                pass

        sen = bundle.sensors
        if "split" in sen.columns:
            tr = sen[sen["split"].astype(str) == "train"]
            te = sen[sen["split"].astype(str) == "test"]
        else:
            tr = sen
            te = sen.iloc[:0]

        if len(tr) == 0:
            raise RuntimeError("No training sensor rows found (split=='train').")

        y_cols: List[str] = [c for c in ["u", "v", "p"] if c in tr.columns]
        if not {"u", "v"}.issubset(set(y_cols)):
            raise RuntimeError("Sensors must include at least columns u and v.")

        X_train = tr[["x", "y"]].to_numpy().astype("float32")
        Y_train = tr[y_cols].to_numpy().astype("float32")

        def pad3(y: np.ndarray) -> np.ndarray:
            out = np.zeros((y.shape[0], 3), dtype=np.float32)
            for j, c in enumerate(y_cols):
                out[:, ["u", "v", "p"].index(c)] = y[:, j]
            return out

        Y_train3 = pad3(Y_train)

        X_test = te[["x", "y"]].to_numpy().astype("float32") if len(te) else None
        Y_test3 = pad3(te[y_cols].to_numpy().astype("float32")) if len(te) else None

        data = dde.data.DataSet(X_train=X_train, y_train=Y_train3, X_test=X_test, y_test=Y_test3)

        width = int(model_cfg.get("width", 256))
        depth = int(model_cfg.get("depth", 4))
        act = str(model_cfg.get("activation", "tanh"))
        initializer = str(model_cfg.get("initializer", "Glorot normal"))

        net = dde.maps.FNN([2] + [width] * depth + [3], act, initializer)
        model = dde.Model(data, net)

        lr = float(cfg.get("lr", 1e-3))
        epochs = int(cfg.get("epochs", 20000))

        model.compile("adam", lr=lr)
        losshistory, _ = model.train(epochs=epochs)

        def predict_fn(xy: np.ndarray) -> np.ndarray:
            xy = np.asarray(xy, dtype=np.float32)
            y = np.asarray(model.predict(xy), dtype=np.float32)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if y.shape[1] != 3:
                out = np.zeros((y.shape[0], 3), dtype=np.float32)
                out[:, : min(3, y.shape[1])] = y[:, : min(3, y.shape[1])]
                return out
            return y

        last_loss = float("nan")
        try:
            if losshistory.loss_train:
                last_loss = float(losshistory.loss_train[-1][0])
        except Exception:
            pass

        return {
            "predict_fn": predict_fn,
            "metrics": {
                "backend": "deepxde",
                "epochs": float(epochs),
                "lr": float(lr),
                "train_loss_last": float(last_loss),
            },
        }