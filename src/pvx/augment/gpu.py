# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""GPU-accelerated augmentation transforms using PyTorch.

These transforms operate directly on ``torch.Tensor`` objects and can run
on CUDA, MPS, or CPU devices.  They avoid NumPy round-tripping and support
batched operation for maximum throughput during training.

.. warning::

   **Alpha release (0.1.0a1).**  This module is under active development.

Requirements
------------
``torch`` must be installed::

    pip install "pvx[torch]"

Usage
-----
>>> import torch
>>> from pvx.augment.gpu import TorchGainPerturber, TorchSpecAugment, TorchPipeline
>>>
>>> pipeline = TorchPipeline([
...     TorchGainPerturber(gain_db=(-6, 6), p=0.8),
...     TorchAddNoise(snr_db=(10, 30), noise_type="white", p=0.5),
...     TorchSpecAugment(freq_mask_param=27, time_mask_param=100, p=0.5),
... ])
>>>
>>> audio = torch.randn(16, 1, 48000)  # (batch, channels, samples)
>>> audio_aug = pipeline(audio, sr=16000)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for pvx.augment.gpu. "
            "Install with: pip install 'pvx[torch]'"
        ) from exc


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TorchTransform(ABC):
    """Base class for GPU-accelerated augmentation transforms.

    All transforms accept tensors of shape ``(batch, channels, samples)``
    or ``(channels, samples)`` or ``(samples,)`` and return the same shape.

    Parameters
    ----------
    p:
        Probability of applying this transform per batch element.
    """

    def __init__(self, p: float = 1.0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p!r}")
        self.p = float(p)

    def __call__(self, audio, sr: int, generator=None):
        """Apply transform with probability ``self.p``.

        Parameters
        ----------
        audio:
            ``torch.Tensor`` of shape ``(B, C, T)``, ``(C, T)``, or ``(T,)``.
        sr:
            Sample rate in Hz.
        generator:
            Optional ``torch.Generator`` for reproducibility.

        Returns
        -------
        torch.Tensor
            Augmented audio with the same shape as input.
        """
        torch = _require_torch()

        # Normalize to (B, C, T)
        original_ndim = audio.ndim
        if audio.ndim == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        elif audio.ndim == 2:
            audio = audio.unsqueeze(0)  # (1, C, T)

        B = audio.shape[0]

        if self.p < 1.0:
            mask = torch.rand(B, device=audio.device, generator=generator) < self.p
            if not mask.any():
                return self._restore_shape(audio, original_ndim)
            result = audio.clone()
            augmented = self.apply(audio[mask], sr, generator)
            result[mask] = augmented
        else:
            result = self.apply(audio, sr, generator)

        return self._restore_shape(result, original_ndim)

    @staticmethod
    def _restore_shape(audio, original_ndim: int):
        if original_ndim == 1:
            return audio.squeeze(0).squeeze(0)
        elif original_ndim == 2:
            return audio.squeeze(0)
        return audio

    @abstractmethod
    def apply(self, audio, sr: int, generator=None):
        """Apply transform to a ``(B, C, T)`` tensor."""

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({params})"


# ---------------------------------------------------------------------------
# TorchGainPerturber
# ---------------------------------------------------------------------------

class TorchGainPerturber(TorchTransform):
    """GPU-accelerated random gain perturbation.

    Parameters
    ----------
    gain_db:
        ``(min_db, max_db)`` range for random gain in decibels.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        gain_db: tuple[float, float] = (-6.0, 6.0),
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.gain_db_min = float(gain_db[0])
        self.gain_db_max = float(gain_db[1])

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        B = audio.shape[0]
        gain_db = torch.empty(B, 1, 1, device=audio.device).uniform_(
            self.gain_db_min, self.gain_db_max, generator=generator
        )
        scale = 10.0 ** (gain_db / 20.0)
        return audio * scale


# ---------------------------------------------------------------------------
# TorchAddNoise
# ---------------------------------------------------------------------------

class TorchAddNoise(TorchTransform):
    """GPU-accelerated additive noise injection.

    Supports ``"white"``, ``"pink"``, and ``"brown"`` noise types.

    Parameters
    ----------
    snr_db:
        ``(min_snr, max_snr)`` range in dB for the signal-to-noise ratio.
    noise_type:
        ``"white"``, ``"pink"``, or ``"brown"``.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        snr_db: tuple[float, float] = (10.0, 30.0),
        noise_type: str = "white",
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.snr_db_min = float(snr_db[0])
        self.snr_db_max = float(snr_db[1])
        if noise_type not in ("white", "pink", "brown"):
            raise ValueError(f"noise_type must be white/pink/brown, got {noise_type!r}")
        self.noise_type = noise_type

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        B, C, T = audio.shape

        # Generate noise
        if self.noise_type == "white":
            noise = torch.randn(B, C, T, device=audio.device, generator=generator)
        elif self.noise_type in ("pink", "brown"):
            # Spectral shaping in frequency domain on GPU
            noise = torch.randn(B, C, T, device=audio.device, generator=generator)
            N = T // 2 + 1
            freqs = torch.arange(1, N + 1, device=audio.device, dtype=torch.float32)
            if self.noise_type == "pink":
                spectrum_scale = 1.0 / torch.sqrt(freqs)  # 1/f^0.5
            else:  # brown
                spectrum_scale = 1.0 / freqs  # 1/f
            spectrum_scale[0] = 1.0  # DC component
            noise_fft = torch.fft.rfft(noise, dim=-1)
            noise_fft = noise_fft * spectrum_scale.unsqueeze(0).unsqueeze(0)
            noise = torch.fft.irfft(noise_fft, n=T, dim=-1)
        else:
            noise = torch.randn(B, C, T, device=audio.device, generator=generator)

        # Compute per-sample SNR
        snr_db = torch.empty(B, 1, 1, device=audio.device).uniform_(
            self.snr_db_min, self.snr_db_max, generator=generator
        )

        # Scale noise to desired SNR
        signal_power = (audio ** 2).mean(dim=-1, keepdim=True).clamp(min=1e-10)
        noise_power = (noise ** 2).mean(dim=-1, keepdim=True).clamp(min=1e-10)
        target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        scale = torch.sqrt(target_noise_power / noise_power)
        return audio + noise * scale


# ---------------------------------------------------------------------------
# TorchEQPerturber
# ---------------------------------------------------------------------------

class TorchEQPerturber(TorchTransform):
    """GPU-accelerated parametric EQ perturbation via frequency-domain shaping.

    Applies random gain at randomly chosen frequency bands in the STFT
    domain. This is a fast approximation of multi-band parametric EQ.

    Parameters
    ----------
    n_bands:
        Number of EQ bands to apply.
    gain_db_range:
        ``(min_db, max_db)`` for each band.
    q_range:
        ``(min_q, max_q)`` quality factor range (higher = narrower band).
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        n_bands: int = 4,
        gain_db_range: tuple[float, float] = (-6.0, 6.0),
        q_range: tuple[float, float] = (0.5, 4.0),
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.n_bands = int(n_bands)
        self.gain_db_min = float(gain_db_range[0])
        self.gain_db_max = float(gain_db_range[1])
        self.q_min = float(q_range[0])
        self.q_max = float(q_range[1])

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        B, C, T = audio.shape

        n_fft = 2048
        hop_length = n_fft // 4
        N = n_fft // 2 + 1

        # Build per-batch EQ curve
        freqs = torch.linspace(0, sr / 2, N, device=audio.device)
        eq_curve = torch.ones(B, N, device=audio.device)

        for _ in range(self.n_bands):
            center_hz = torch.empty(B, device=audio.device).uniform_(
                60.0, sr / 2 - 100, generator=generator
            )
            gain_db = torch.empty(B, device=audio.device).uniform_(
                self.gain_db_min, self.gain_db_max, generator=generator
            )
            q = torch.empty(B, device=audio.device).uniform_(
                self.q_min, self.q_max, generator=generator
            )
            bandwidth = center_hz / q.clamp(min=0.1)

            # Gaussian-shaped EQ band
            diff = (freqs.unsqueeze(0) - center_hz.unsqueeze(1)) / bandwidth.unsqueeze(1)
            band_shape = torch.exp(-0.5 * diff ** 2)
            band_gain = 10.0 ** (gain_db.unsqueeze(1) / 20.0)
            eq_curve = eq_curve * (1.0 + band_shape * (band_gain - 1.0))

        # Apply in frequency domain per channel
        result_channels = []
        for ch in range(C):
            spec = torch.stft(
                audio[:, ch, :], n_fft=n_fft, hop_length=hop_length,
                return_complex=True, window=torch.hann_window(n_fft, device=audio.device),
            )  # (B, N, frames)
            spec = spec * eq_curve.unsqueeze(-1)
            reconstructed = torch.istft(
                spec, n_fft=n_fft, hop_length=hop_length,
                length=T, window=torch.hann_window(n_fft, device=audio.device),
            )
            result_channels.append(reconstructed)

        return torch.stack(result_channels, dim=1)


# ---------------------------------------------------------------------------
# TorchSpecAugment
# ---------------------------------------------------------------------------

class TorchSpecAugment(TorchTransform):
    """GPU-accelerated SpecAugment (Park et al. 2019).

    Applies frequency and time masking in the STFT domain, entirely on GPU.

    Parameters
    ----------
    freq_mask_param:
        Maximum number of frequency bins to mask per mask.
    time_mask_param:
        Maximum number of time frames to mask per mask.
    num_freq_masks:
        Number of frequency masks to apply.
    num_time_masks:
        Number of time masks to apply.
    fill_value:
        Value to fill masked regions (0.0 or use ``"mean"``).
    n_fft:
        FFT size for STFT.
    hop_length:
        Hop length for STFT.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        fill_value: float | str = 0.0,
        n_fft: int = 1024,
        hop_length: int = 256,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)
        self.num_freq_masks = int(num_freq_masks)
        self.num_time_masks = int(num_time_masks)
        self.fill_value = fill_value
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        B, C, T = audio.shape

        window = torch.hann_window(self.n_fft, device=audio.device)

        result_channels = []
        for ch in range(C):
            spec = torch.stft(
                audio[:, ch, :], n_fft=self.n_fft, hop_length=self.hop_length,
                return_complex=True, window=window,
            )  # (B, freq_bins, time_frames)
            n_freq, n_time = spec.shape[1], spec.shape[2]

            mag = spec.abs()
            phase = spec.angle()

            fill = 0.0
            if self.fill_value == "mean":
                fill = mag.mean().item()
            elif isinstance(self.fill_value, (int, float)):
                fill = float(self.fill_value)

            # Frequency masking
            for _ in range(self.num_freq_masks):
                f = torch.randint(
                    0, self.freq_mask_param + 1, (B,),
                    device=audio.device, generator=generator,
                )
                f0 = torch.randint(
                    0, max(n_freq - self.freq_mask_param, 1), (B,),
                    device=audio.device, generator=generator,
                )
                for b in range(B):
                    f_end = min(int(f0[b] + f[b]), n_freq)
                    mag[b, int(f0[b]):f_end, :] = fill

            # Time masking
            for _ in range(self.num_time_masks):
                t = torch.randint(
                    0, self.time_mask_param + 1, (B,),
                    device=audio.device, generator=generator,
                )
                t0 = torch.randint(
                    0, max(n_time - self.time_mask_param, 1), (B,),
                    device=audio.device, generator=generator,
                )
                for b in range(B):
                    t_end = min(int(t0[b] + t[b]), n_time)
                    mag[b, :, int(t0[b]):t_end] = fill

            # Reconstruct
            spec_masked = torch.polar(mag, phase)
            reconstructed = torch.istft(
                spec_masked, n_fft=self.n_fft, hop_length=self.hop_length,
                length=T, window=window,
            )
            result_channels.append(reconstructed)

        return torch.stack(result_channels, dim=1)


# ---------------------------------------------------------------------------
# TorchNormalizer
# ---------------------------------------------------------------------------

class TorchNormalizer(TorchTransform):
    """GPU-accelerated peak or RMS normalization.

    Parameters
    ----------
    mode:
        ``"peak"`` or ``"rms"``.
    target_db:
        Target level in dBFS.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        mode: str = "peak",
        target_db: float = -1.0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if mode not in ("peak", "rms"):
            raise ValueError(f"mode must be 'peak' or 'rms', got {mode!r}")
        self.mode = mode
        self.target_db = float(target_db)

    def apply(self, audio, sr: int, generator=None):
        target = 10.0 ** (self.target_db / 20.0)
        if self.mode == "peak":
            current = audio.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        else:
            current = audio.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-10)
        scale = target / current
        return audio * scale


# ---------------------------------------------------------------------------
# TorchClippingSimulator
# ---------------------------------------------------------------------------

class TorchClippingSimulator(TorchTransform):
    """GPU-accelerated clipping simulation.

    Parameters
    ----------
    percentile:
        ``(min, max)`` percentile range for clipping threshold.
    mode:
        ``"hard"`` for rectangular clipping, ``"soft"`` for tanh saturation.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        percentile: tuple[float, float] = (90.0, 99.5),
        mode: str = "hard",
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.pct_min = float(percentile[0])
        self.pct_max = float(percentile[1])
        if mode not in ("hard", "soft"):
            raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")
        self.mode = mode

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        B = audio.shape[0]
        pct = torch.empty(B, device=audio.device).uniform_(
            self.pct_min / 100.0, self.pct_max / 100.0, generator=generator
        )
        # Compute per-sample threshold via quantile (one per batch element)
        flat = audio.abs().reshape(B, -1)
        thresholds = []
        for b in range(B):
            thresholds.append(torch.quantile(flat[b], pct[b]))
        th = torch.stack(thresholds).reshape(B, 1, 1)

        if self.mode == "hard":
            return audio.clamp(-th, th)
        else:
            return torch.tanh(audio / (th + 1e-8)) * th


# ---------------------------------------------------------------------------
# TorchPipeline
# ---------------------------------------------------------------------------

class TorchPipeline:
    """Sequential composition of GPU transforms.

    Parameters
    ----------
    transforms:
        Ordered list of :class:`TorchTransform` instances.
    seed:
        Optional seed for a shared ``torch.Generator``.

    Examples
    --------
    >>> pipeline = TorchPipeline([
    ...     TorchGainPerturber(gain_db=(-6, 6)),
    ...     TorchAddNoise(snr_db=(10, 30)),
    ...     TorchSpecAugment(freq_mask_param=27),
    ... ], seed=42)
    >>> audio_aug = pipeline(audio_batch, sr=16000)
    """

    def __init__(
        self,
        transforms: Sequence[TorchTransform],
        seed: int | None = None,
    ) -> None:
        self.transforms = list(transforms)
        self.seed = seed

    def __call__(self, audio, sr: int, seed: int | None = None):
        torch = _require_torch()
        s = seed if seed is not None else self.seed
        gen = None
        if s is not None:
            gen = torch.Generator(device=audio.device)
            gen.manual_seed(s)

        for t in self.transforms:
            audio = t(audio, sr, generator=gen)
        return audio

    def __repr__(self) -> str:
        inner = "\n  ".join(repr(t) for t in self.transforms)
        return f"TorchPipeline(\n  {inner}\n)"

    def __len__(self) -> int:
        return len(self.transforms)


# ---------------------------------------------------------------------------
# Adapter: wrap NumPy transforms for batched GPU usage
# ---------------------------------------------------------------------------

class NumpyTransformAdapter(TorchTransform):
    """Wrap any NumPy-based ``pvx.augment.Transform`` for use in a
    :class:`TorchPipeline`.

    This is a convenience adapter — it moves data to CPU, applies the
    NumPy transform per-sample, and moves back to the original device.
    Use native ``Torch*`` transforms for maximum GPU throughput.

    Parameters
    ----------
    transform:
        A ``pvx.augment.Transform`` instance.
    p:
        Override probability (uses the original transform's p if not set).
    """

    def __init__(self, transform, p: float | None = None) -> None:
        actual_p = p if p is not None else getattr(transform, "p", 1.0)
        super().__init__(p=actual_p)
        self.transform = transform

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        import numpy as np

        device = audio.device
        B, C, T = audio.shape
        results = []
        for b in range(B):
            np_audio = audio[b].cpu().numpy()
            seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
            aug_audio, _ = self.transform(np_audio, sr, seed=seed)
            results.append(torch.from_numpy(np.asarray(aug_audio, dtype=np.float32)))

        return torch.stack(results, dim=0).to(device)
