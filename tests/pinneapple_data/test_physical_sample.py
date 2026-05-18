import numpy as np
import torch
import xarray as xr

from pinneapple_data.physical_sample import PhysicalSample

def test_physical_sample_xr_summary():
    ds = xr.Dataset(
        data_vars=dict(T2M=(("t","x"), np.random.randn(4,3).astype("float32"))),
        coords=dict(t=np.arange(4), x=np.arange(3)),
    )
    s = PhysicalSample(state=ds, domain={"type":"grid"})
    info = s.summary()
    assert info["domain_type"] in ("grid","Grid","GRID") or info["domain_type"] == "grid"
    assert "T2M" in info["state"]["vars"]

def test_physical_sample_to_train_dict_dict_state():
    state = {"a": torch.randn(5,2), "b": torch.randn(5,2)}
    s = PhysicalSample(state=state, domain={"type":"grid"})
    td = s.to_train_dict(x_vars=["a","b"])
    assert "x" in td
    x = td["x"]
    assert x.shape[-1] == 2  # stacked vars -> last dim
