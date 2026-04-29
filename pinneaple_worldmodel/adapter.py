"""NVIDIA Cosmos world foundation model adapter for PINNeAPPle.

How NVIDIA Cosmos actually works
---------------------------------
Cosmos is **not** a pip-installable Python package you can ``import cosmos``.
It is a family of large world foundation models (7B / 14B parameters) from
NVIDIA, and there are three legitimate ways to use it:

1. **NVIDIA NIM REST API** (recommended for cloud use):
   - Requires an NVIDIA API key from https://build.nvidia.com/
   - Set the environment variable ``COSMOS_API_KEY`` or pass ``api_key=``
   - Works via HTTP POST to NVIDIA's inference microservice endpoints
   - No local GPU required

2. **HuggingFace model weights** (local inference):
   - Models: ``nvidia/Cosmos-1.0-Diffusion-7B`` etc.
   - Requires accepting NVIDIA's license at https://huggingface.co/nvidia/Cosmos-1.0-Diffusion-7B
   - Requires ``transformers >= 4.40`` and a GPU with ~16 GB VRAM
   - ``from transformers import AutoModelForVideoGeneration``

3. **cosmos-tokenizer** (pip-installable, but only the tokenizer):
   - ``pip install cosmos-tokenizer``
   - Only encodes / decodes video tokens — does NOT generate frames
   - Useful if you want to tokenize video for training your own model

**What this module provides**
------------------------------
- A :class:`CosmosAdapter` that wraps routes 1 and 2 above.
- A ``_PhysicsVideoFallback`` that generates plausible frame sequences using
  a 2-D advection-diffusion PDE — **always available, no external deps**.
- By default (``use_api=False``, no API key) the physics fallback is used.
  All downstream code paths remain functional.
- :class:`SimToRealAdapter` for bridging simulation and real sensor data.
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WorldModelConfig:
    """Configuration for :class:`CosmosAdapter`.

    Attributes
    ----------
    model_name:
        Cosmos model variant.  For NIM API: ``"cosmos-1.0-diffusion-7b"`` etc.
        For HuggingFace: ``"nvidia/Cosmos-1.0-Diffusion-7B"``.
    physics_prior_weight:
        Weight applied to the physics-consistency loss term.
    guidance_scale:
        Classifier-free guidance scale for the diffusion model.
    n_frames:
        Number of video frames to generate per call.
    resolution:
        ``(H, W)`` frame resolution in pixels.
    use_api:
        ``True``  → attempt NVIDIA NIM REST API (requires ``COSMOS_API_KEY``).
        ``False`` → use the physics-based fallback (default, always works).
    use_huggingface:
        ``True`` → attempt loading weights from HuggingFace Hub (requires
        ``transformers`` and an accepted NVIDIA license).
    nim_base_url:
        Base URL for the NVIDIA NIM API endpoint.
    """
    model_name: str = "cosmos-1.0-diffusion-7b"
    physics_prior_weight: float = 0.1
    guidance_scale: float = 7.5
    n_frames: int = 16
    resolution: Tuple[int, int] = (256, 256)
    use_api: bool = False
    use_huggingface: bool = False
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------


def _nim_api_available(api_key: str) -> bool:
    """Return True if we have requests + a non-empty API key."""
    if not api_key:
        return False
    try:
        import requests  # noqa: F401
        return True
    except ImportError:
        return False


def _huggingface_available() -> bool:
    """Return True if transformers >= 4.40 is installed."""
    try:
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        return (major, minor) >= (4, 40)
    except (ImportError, ValueError):
        return False


def _cosmos_tokenizer_available() -> bool:
    """Return True if the cosmos-tokenizer pip package is installed."""
    try:
        import cosmos_tokenizer  # noqa: F401  # pip install cosmos-tokenizer
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Physics-based fallback video generator
# ---------------------------------------------------------------------------


class _PhysicsVideoFallback(nn.Module):
    """Pure-PyTorch advection-diffusion video generator (no external deps).

    Evolves a 2-D advection-diffusion field from *initial_state* to produce a
    plausible sequence of physics frames.  Used whenever Cosmos is unavailable.
    """

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config

    @torch.no_grad()
    def generate(
        self,
        initial_state: torch.Tensor,
        condition_text: str = "",
        n_frames: int = 16,
    ) -> torch.Tensor:
        """Generate ``n_frames`` video frames from *initial_state*.

        Parameters
        ----------
        initial_state : ``(C,)`` or ``(C, H, W)``
        condition_text : ignored in fallback mode.
        n_frames : number of frames to produce.

        Returns
        -------
        ``(T, 3, H, W)`` frames in ``[0, 1]``.
        """
        H, W = self.config.resolution
        device = initial_state.device

        if initial_state.ndim == 1:
            field_2d = initial_state[:1].mean().expand(1, H, W).clone()
        elif initial_state.ndim == 3:
            import torch.nn.functional as F
            field_2d = F.interpolate(
                initial_state[:1].unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        else:
            field_2d = torch.zeros(1, H, W, device=device)

        u = field_2d.clone().float()
        alpha, v_x, v_y = 0.05, 0.1, 0.0
        frames = []

        for _ in range(n_frames):
            laplacian = (
                torch.roll(u, 1, -1) + torch.roll(u, -1, -1)
                + torch.roll(u, 1, -2) + torch.roll(u, -1, -2)
                - 4.0 * u
            )
            du_dx = u - torch.roll(u, 1, -1)
            du_dy = u - torch.roll(u, 1, -2)
            u = (u + 0.1 * (alpha * laplacian - v_x * du_dx - v_y * du_dy)).clamp(0, 1)

            r = u.clamp(0, 1)
            g = (u * 0.5).clamp(0, 1)
            b = (1.0 - u).clamp(0, 1)
            frames.append(torch.cat([r, g, b], dim=0))

        return torch.stack(frames, dim=0)  # (T, 3, H, W)


# ---------------------------------------------------------------------------
# CosmosAdapter
# ---------------------------------------------------------------------------


class CosmosAdapter:
    """Adapter for the NVIDIA Cosmos world foundation model.

    Integration strategy (tried in order):
    1. **NVIDIA NIM REST API** — if ``config.use_api=True`` and an API key is
       set.  Sends a POST request to the NIM inference endpoint.
    2. **HuggingFace weights** — if ``config.use_huggingface=True`` and
       ``transformers >= 4.40`` is installed with accepted NVIDIA license.
    3. **Physics fallback** — always available; generates advection-diffusion
       frames locally with pure PyTorch (no GPU, no API key needed).

    Parameters
    ----------
    config : :class:`WorldModelConfig`
    api_key : NVIDIA API key.  Defaults to ``COSMOS_API_KEY`` env var.

    Notes
    -----
    NVIDIA Cosmos is **not** a pip package.  Do not ``import cosmos``.
    See the module-level docstring for the three correct integration paths.
    """

    def __init__(
        self,
        config: Optional[WorldModelConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.config = config or WorldModelConfig()
        self._api_key = api_key or os.environ.get("COSMOS_API_KEY", "")
        self._hf_model = None
        self._backend: str = "fallback"

        # Try NIM API
        if self.config.use_api:
            if _nim_api_available(self._api_key):
                self._backend = "nim"
            else:
                warnings.warn(
                    "CosmosAdapter: NIM API requested but `requests` is not installed "
                    "or COSMOS_API_KEY is empty. Falling back to physics generator.\n"
                    "  pip install requests\n"
                    "  export COSMOS_API_KEY=nvapi-...",
                    stacklevel=2,
                )

        # Try HuggingFace
        elif self.config.use_huggingface:
            if _huggingface_available():
                try:
                    self._hf_model = self._load_hf_model()
                    self._backend = "huggingface"
                except Exception as exc:
                    warnings.warn(
                        f"CosmosAdapter: HuggingFace loading failed: {exc}\n"
                        "  Make sure you accepted the NVIDIA license at "
                        "https://huggingface.co/nvidia/Cosmos-1.0-Diffusion-7B\n"
                        "  and have a GPU with ≥16 GB VRAM.  Falling back to physics generator.",
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    "CosmosAdapter: `transformers >= 4.40` not installed.\n"
                    "  pip install 'transformers>=4.40'\n"
                    "  Falling back to physics generator.",
                    stacklevel=2,
                )

        self._fallback = _PhysicsVideoFallback(self.config)

    def _load_hf_model(self):
        """Load Cosmos weights from HuggingFace Hub (requires license)."""
        from transformers import AutoProcessor, AutoModelForVideoGeneration  # type: ignore
        model_id = self.config.model_name
        if not model_id.startswith("nvidia/"):
            model_id = f"nvidia/{self.config.model_name}"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVideoGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16,
        )
        return {"model": model, "processor": processor}

    @property
    def using_cosmos(self) -> bool:
        """``True`` if a live Cosmos backend (NIM or HF) is active."""
        return self._backend in ("nim", "huggingface")

    @property
    def backend(self) -> str:
        """Active backend: ``'nim'`` | ``'huggingface'`` | ``'fallback'``."""
        return self._backend

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        initial_state: torch.Tensor,
        condition_text: str = "physical simulation",
        n_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate a physics video sequence.

        Parameters
        ----------
        initial_state : ``(C,)`` or ``(C, H, W)`` field snapshot.
        condition_text : text prompt (used by NIM / HF backends; ignored in fallback).
        n_frames : override frame count (defaults to ``config.n_frames``).

        Returns
        -------
        ``(T, 3, H, W)`` tensor in ``[0, 1]``.
        """
        T = n_frames or self.config.n_frames

        if self._backend == "nim":
            try:
                return self._nim_generate(initial_state, condition_text, T)
            except Exception as exc:
                warnings.warn(f"CosmosAdapter NIM call failed ({exc}); using physics fallback.")

        if self._backend == "huggingface" and self._hf_model is not None:
            try:
                return self._hf_generate(initial_state, condition_text, T)
            except Exception as exc:
                warnings.warn(f"CosmosAdapter HF call failed ({exc}); using physics fallback.")

        return self._fallback.generate(initial_state, condition_text, T)

    def _nim_generate(
        self,
        initial_state: torch.Tensor,
        condition_text: str,
        n_frames: int,
    ) -> torch.Tensor:
        """Call NVIDIA NIM REST API for video generation.

        NIM endpoint reference:
          POST {nim_base_url}/nvidia/cosmos-video-generation
          Headers: Authorization: Bearer {api_key}
          Body: { "prompt": str, "n_frames": int, "resolution": [H, W],
                  "guidance_scale": float }

        Note: the exact endpoint path and body schema depend on the NIM
        version deployed by NVIDIA.  Check https://build.nvidia.com/ for
        the current API specification.
        """
        import requests  # type: ignore
        import base64

        H, W = self.config.resolution
        url = f"{self.config.nim_base_url}/nvidia/cosmos-video-generation"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "prompt": condition_text,
            "num_frames": n_frames,
            "height": H,
            "width": W,
            "guidance_scale": self.config.guidance_scale,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        # Expected response: {"frames": [base64-encoded PNG strings]}
        frames = []
        import io
        from PIL import Image  # type: ignore

        for b64 in data.get("frames", []):
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((W, H))
            import torchvision.transforms.functional as TVF  # type: ignore
            frames.append(TVF.to_tensor(img))

        if not frames:
            raise ValueError("NIM API returned no frames.")
        return torch.stack(frames, dim=0)  # (T, 3, H, W)

    def _hf_generate(
        self,
        initial_state: torch.Tensor,
        condition_text: str,
        n_frames: int,
    ) -> torch.Tensor:
        """Generate frames using locally loaded HuggingFace Cosmos weights."""
        model = self._hf_model["model"]
        processor = self._hf_model["processor"]
        H, W = self.config.resolution

        inputs = processor(text=condition_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_frames=n_frames,
                height=H,
                width=W,
                guidance_scale=self.config.guidance_scale,
            )
        # outputs.frames: (1, T, 3, H, W) → (T, 3, H, W)
        return outputs.frames[0].float().clamp(0, 1)

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def extract_state(self, video_frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract approximate physical state from video frames.

        Uses luminance heuristics: temporal gradient → velocity proxy,
        spatial gradient magnitude → pressure proxy.

        Parameters
        ----------
        video_frames : ``(T, 3, H, W)``

        Returns
        -------
        dict with ``"velocity_proxy"`` and ``"pressure_proxy"``.
        """
        luma = (0.299 * video_frames[:, 0]
                + 0.587 * video_frames[:, 1]
                + 0.114 * video_frames[:, 2])  # (T, H, W)

        vel_proxy = luma[1:] - luma[:-1]  # (T-1, H, W)

        import torch.nn.functional as F
        grad_x = F.pad(luma[:, :, 1:] - luma[:, :, :-1], (0, 1))
        grad_y = F.pad(luma[:, 1:, :] - luma[:, :-1, :], (0, 0, 0, 1))
        press_proxy = (grad_x ** 2 + grad_y ** 2).sqrt()

        return {"velocity_proxy": vel_proxy, "pressure_proxy": press_proxy}

    # ------------------------------------------------------------------
    # Physics consistency loss
    # ------------------------------------------------------------------

    def physics_prior_loss(
        self,
        model_output: torch.Tensor,
        world_state: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between PINN output and world-model luminance field.

        Parameters
        ----------
        model_output : ``(T, C, H, W)`` or ``(N,)`` — PINN prediction.
        world_state  : ``(T, 3, H, W)`` — frames from :meth:`generate`.

        Returns
        -------
        Scalar loss tensor.
        """
        w = self.config.physics_prior_weight
        luma = (0.299 * world_state[:, 0]
                + 0.587 * world_state[:, 1]
                + 0.114 * world_state[:, 2])

        if model_output.shape != luma.shape:
            import torch.nn.functional as F
            try:
                if model_output.ndim == 4:
                    mo = F.interpolate(
                        model_output[:, :1], size=luma.shape[-2:],
                        mode="bilinear", align_corners=False,
                    ).squeeze(1)[:luma.shape[0]]
                else:
                    mo = model_output.reshape(-1)[:luma.numel()].reshape(luma.shape)
                luma = luma[:mo.shape[0]]
            except Exception:
                return torch.tensor(0.0, requires_grad=True)
        else:
            mo = model_output

        return w * torch.mean((mo - luma.detach()) ** 2)


# ---------------------------------------------------------------------------
# Sim-to-Real adapter
# ---------------------------------------------------------------------------


class SimToRealAdapter:
    """Bridge between a PINN simulation and real-world observations via WFM.

    Parameters
    ----------
    pinn_model : trained PINN ``nn.Module``.
    world_model : :class:`CosmosAdapter` instance.
    """

    def __init__(self, pinn_model: nn.Module, world_model: CosmosAdapter) -> None:
        self.pinn = pinn_model
        self.wm = world_model

    def align(
        self,
        real_observations: torch.Tensor,
        simulation_state: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian-smoothed additive alignment of simulation to real data.

        Parameters
        ----------
        real_observations : ``(T, C, H, W)`` or ``(N, d)``
        simulation_state  : same shape as real_observations

        Returns
        -------
        Corrected simulation tensor (same shape as *simulation_state*).
        """
        device = simulation_state.device

        if real_observations.shape == simulation_state.shape:
            residual = real_observations.detach() - simulation_state
        else:
            residual = torch.zeros_like(simulation_state)

        sigma = 2.0
        kernel_size = int(6 * sigma + 1) | 1
        x_k = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        gauss = torch.exp(-0.5 * (x_k / sigma) ** 2)
        gauss /= gauss.sum()

        flat = residual.reshape(-1, residual.shape[-1])
        import torch.nn.functional as F
        smoothed = F.conv1d(
            flat.unsqueeze(1),
            gauss.view(1, 1, -1),
            padding=kernel_size // 2,
        ).squeeze(1).reshape(residual.shape)

        return simulation_state + smoothed
