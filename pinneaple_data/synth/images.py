"""Image reconstruction and inpainting synthetic generator."""
from __future__ import annotations

from typing import Any, Optional, Callable
import torch
import torch.nn as nn

from .base import SynthConfig, SynthOutput
from .pde import SimplePhysicalSample


def _as_tensor(img: Any, device, dtype) -> torch.Tensor:
    """
    Convert an input array-like object to a torch.Tensor on a target device/dtype.

    Parameters
    ----------
    img : Any
        Input data. May be a torch.Tensor or any array-like object convertible
        via `torch.tensor(...)`.
    device : Any
        Target torch device (e.g., torch.device("cpu"), torch.device("cuda")).
    dtype : Any
        Target torch dtype (e.g., torch.float32).

    Returns
    -------
    torch.Tensor
        Tensor representation of `img` moved to the specified device/dtype.
    """
    if isinstance(img, torch.Tensor):
        t = img
    else:
        t = torch.tensor(img)
    return t.to(device=device, dtype=dtype)


def _smooth2d(x: torch.Tensor) -> torch.Tensor:
    """
    Apply simple 2D smoothing via 4-neighbor averaging.

    The input may be either:
    - (H, W): single-channel image
    - (C, H, W): multi-channel image

    Smoothing is performed using a fixed 3x3 convolution kernel that averages
    the four direct neighbors (up, down, left, right). For multi-channel inputs,
    the convolution is applied depthwise (per-channel).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape (H, W) or (C, H, W).

    Returns
    -------
    torch.Tensor
        Smoothed tensor with the same shape as the input.

    Raises
    ------
    ValueError
        If `x` does not have 2 or 3 dimensions.
    """
    """
    Simple 2D smoothing via neighbor averaging.
    x: (H,W) or (C,H,W)
    """
    if x.ndim == 2:
        x2 = x[None, None, :, :]  # (1,1,H,W)
    elif x.ndim == 3:
        x2 = x[None, :, :, :]     # (1,C,H,W)
    else:
        raise ValueError("Expected (H,W) or (C,H,W)")

    # 4-neighbor averaging kernel
    k = torch.tensor(
        [[0.0, 0.25, 0.0],
         [0.25, 0.0, 0.25],
         [0.0, 0.25, 0.0]],
        device=x.device, dtype=x.dtype
    )[None, None, :, :]  # (1,1,3,3)

    if x2.shape[1] > 1:
        k = k.repeat(x2.shape[1], 1, 1, 1)  # depthwise
        y = torch.nn.functional.conv2d(x2, k, padding=1, groups=x2.shape[1])
    else:
        y = torch.nn.functional.conv2d(x2, k, padding=1)

    y = y[0]
    return y[0] if y.shape[0] == 1 else y


class ImageReconstructionSynthGenerator:
    """
    Synthetic generator for image reconstruction tasks.

    This generator produces corrupted inputs and corresponding reconstructions,
    packaged as PhysicalSample-like outputs.

    Supported modes
    ---------------
    - "inpaint_smooth":
        Performs iterative inpainting inside the missing mask region by repeatedly
        smoothing the current reconstruction and copying values into masked pixels.
    - "autoencoder":
        Uses a provided autoencoder model to reconstruct the corrupted image.

    Inputs
    ------
    img : (H, W) or (C, H, W)
        Image tensor/array.
    mask : (H, W) or (1, H, W) or (C, H, W), optional
        Mask indicating missing region. Values > 0.5 are treated as missing (True).
        If not provided, a deterministic rectangular missing block is created.
        Convention: mask == 1 means MISSING region to reconstruct.

    Output
    ------
    A `SimplePhysicalSample` with:
    - fields["img"]: original image
    - fields["img_corrupt"]: corrupted image (masked region zeroed)
    - fields["img_recon"]: reconstructed image
    - fields["mask"]: float mask in {0,1}
    and extras containing the original shape.

    Parameters
    ----------
    cfg : Optional[SynthConfig]
        Configuration controlling device/dtype/seed behavior.
    """
    """
    Image reconstruction / synthesis generator.

    MVP modes:
      - "inpaint_smooth": iterative smoothing inside masked region
      - "autoencoder": use provided AE model to reconstruct

    Input:
      img: (H,W) or (C,H,W)
      mask: same spatial shape (H,W) bool or 0/1
        mask==1 means MISSING region to reconstruct
    """
    def __init__(self, cfg: Optional[SynthConfig] = None):
        """
        Initialize the image reconstruction generator.

        Parameters
        ----------
        cfg : Optional[SynthConfig]
            Optional generator configuration. If not provided, defaults are used.
        """
        self.cfg = cfg or SynthConfig()

    @torch.no_grad()
    def generate(
        self,
        *,
        img: Any,
        mask: Optional[Any] = None,
        mode: str = "inpaint_smooth",
        steps: int = 200,
        ae_model: Optional[nn.Module] = None,
        post_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> SynthOutput:
        """
        Generate a corrupted image and a reconstruction according to the selected mode.

        Parameters
        ----------
        img : Any
            Input image as a torch.Tensor or array-like object. Expected shape
            is (H, W) or (C, H, W).
        mask : Optional[Any], optional
            Missing-region mask as a torch.Tensor or array-like object. Expected
            spatial shape (H, W) (optionally with leading channel dimension).
            Values > 0.5 indicate missing pixels. If None, a fixed rectangular
            missing block is created. Default is None.
        mode : str, optional
            Reconstruction mode: "inpaint_smooth" or "autoencoder".
            Default is "inpaint_smooth".
        steps : int, optional
            Number of smoothing iterations for "inpaint_smooth". Default is 200.
        ae_model : Optional[nn.Module], optional
            Autoencoder model used when mode="autoencoder". Must accept input of
            shape (1, C, H, W) (or (1, 1, H, W)) and return a matching output.
            Default is None.
        post_fn : Optional[Callable[[torch.Tensor], torch.Tensor]], optional
            Optional post-processing function applied to the reconstruction.
            Default is None.

        Returns
        -------
        SynthOutput
            Output containing one `SimplePhysicalSample` with original/corrupted/
            reconstructed images and the mask, plus extras containing the original
            image shape.

        Raises
        ------
        ValueError
            If an invalid mode is provided, or if `ae_model` is missing when
            mode="autoencoder".
        """
        device = torch.device(self.cfg.device)
        dtype = getattr(torch, self.cfg.dtype)

        x = _as_tensor(img, device, dtype)
        mode = mode.lower().strip()

        if mask is None:
            # default: random missing block
            H, W = (x.shape[-2], x.shape[-1]) if x.ndim == 3 else x.shape
            m = torch.zeros((H, W), device=device, dtype=torch.bool)
            h0 = int(0.3 * H)
            w0 = int(0.3 * W)
            m[h0:h0 + int(0.2 * H), w0:w0 + int(0.2 * W)] = True
        else:
            m = _as_tensor(mask, device, torch.float32) > 0.5
            if m.ndim == 3:
                m = m[0]  # assume (1,H,W) or (C,H,W) same mask for all

        # create corrupted
        x_cor = x.clone()
        if x.ndim == 2:
            x_cor[m] = 0.0
        else:
            x_cor[:, m] = 0.0

        if mode == "inpaint_smooth":
            rec = x_cor.clone()
            for _ in range(int(steps)):
                sm = _smooth2d(rec)
                if rec.ndim == 2:
                    rec[m] = sm[m]
                else:
                    rec[:, m] = sm[:, m]
        elif mode == "autoencoder":
            if ae_model is None:
                raise ValueError("ae_model is required for mode='autoencoder'")
            ae_model = ae_model.to(device).eval()
            inp = x_cor
            if inp.ndim == 2:
                inp = inp[None, None, :, :]
            elif inp.ndim == 3:
                inp = inp[None, :, :, :]
            rec = ae_model(inp)[0]
            # rec shape (C,H,W) or (1,H,W)
            if rec.shape[0] == 1 and x.ndim == 2:
                rec = rec[0]
        else:
            raise ValueError("mode must be inpaint_smooth | autoencoder")

        if post_fn is not None:
            rec = post_fn(rec)

        sample = SimplePhysicalSample(
            fields={"img": x, "img_corrupt": x_cor, "img_recon": rec, "mask": m.to(dtype=torch.float32)},
            coords={},
            meta={"mode": mode, "steps": int(steps)},
        )
        return SynthOutput(samples=[sample], extras={"shape": tuple(x.shape)})