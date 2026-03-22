import numpy as np
import torch
from scipy import ndimage


def phase_stretch_transform(img_gray: np.ndarray,
                            lpf_sigma: float = 0.21,
                            phase_strength: float = 0.48,
                            warp_strength: float = 12.14,
                            morph_flag: bool = True,
                            min_thresh: float = -0.35,
                            max_thresh: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """
    Phase Stretch Transform (PST) — matches JalaliLab reference implementation.

    lpf_sigma is treated as FWHM (not sigma) of the Gaussian LPF, consistent
    with the original MATLAB/Python reference code.

    Returns:
        phase : raw PST phase output (float, -pi..pi)
        edges : binary or soft edge map (float 0..1)
    """
    H, W = img_gray.shape
    L = 0.5

    x = np.linspace(-L, L, W)
    y = np.linspace(-L, L, H)
    FX, FY = np.meshgrid(x, y)
    R = np.sqrt(FX ** 2 + FY ** 2)

    # ── Low-pass filter ───────────────────────────────────────────────────────
    # lpf_sigma is FWHM; convert to Gaussian sigma: FWHM = 2*sqrt(2*ln2)*sigma
    sigma = lpf_sigma / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    expo = np.fft.fftshift(np.exp(-R ** 2 / (2.0 * sigma ** 2)))

    # Stage 1: apply LPF → real filtered image (matches reference two-stage approach)
    img_f = np.fft.fft2(img_gray.astype(np.float64))
    img_filtered = np.real(np.fft.ifft2(img_f * expo))

    # ── PST kernel (Eq.5): K = exp(j * S * phi_w), phi_w = W*R/(1+W*R) ───────
    phi_w = warp_strength * R / (1.0 + warp_strength * R)   # warped phase ramp ∈ [0, 1)
    pst_kernel = np.exp(1j * phase_strength * phi_w)         # unit-amplitude pure-phase

    # Stage 2: FFT of filtered image → apply PST kernel → IFFT
    img_filtered_f = np.fft.fftshift(np.fft.fft2(img_filtered))
    img_pst = np.fft.ifft2(np.fft.fftshift(img_filtered_f * pst_kernel))

    # ── Phase angle ──────────────────────────────────────────────────────────
    phase = np.angle(img_pst)   # range: -π … π

    if morph_flag:
        edges = ((phase > max_thresh) | (phase < min_thresh)).astype(np.float64)
        edges = ndimage.binary_closing(edges, structure=np.ones((3, 3))).astype(np.float64)
        edges = ndimage.binary_opening(edges, structure=np.ones((3, 3))).astype(np.float64)
    else:
        # Soft edge map: |phase| with robust percentile normalization
        abs_phase = np.abs(phase)
        p99 = np.percentile(abs_phase, 99)
        edges = np.clip(abs_phase / (p99 + 1e-9), 0.0, 1.0)

    return phase, edges


class PhaseStretchTransformNode:
    """ComfyUI node – Phase Stretch Transform edge detector."""

    CATEGORY = "image/processing"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("edge_image", "phase_image")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lpf_sigma": ("FLOAT", {
                    "default": 0.21, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Gaussian LPF FWHM (same unit as reference code). "
                               "Higher = more smoothing before PST."
                }),
                "phase_strength": ("FLOAT", {
                    "default": 0.48, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "PST phase strength (S). Higher = stronger edge response."
                }),
                "warp_strength": ("FLOAT", {
                    "default": 12.14, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "PST warp strength (W). Controls frequency warping."
                }),
                "min_thresh": ("FLOAT", {
                    "default": -0.35, "min": -3.15, "max": 0.0, "step": 0.001,
                    "tooltip": "Lower phase threshold for binary edge map (morph_flag=True only)"
                }),
                "max_thresh": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 3.15, "step": 0.001,
                    "tooltip": "Upper phase threshold for binary edge map (morph_flag=True only)"
                }),
                "morph_flag": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "True = binary edges with morphological clean-up; "
                               "False = soft continuous edge map (|phase| normalised)"
                }),
                "output_mode": (["grayscale", "rgb_overlay"], {
                    "default": "grayscale",
                    "tooltip": "grayscale = white edges on black; "
                               "rgb_overlay = edges overlaid in red on original"
                }),
            },
        }

    def run(self, image: torch.Tensor,
            lpf_sigma: float,
            phase_strength: float,
            warp_strength: float,
            min_thresh: float,
            max_thresh: float,
            morph_flag: bool,
            output_mode: str):
        """image shape: (B, H, W, C) – ComfyUI convention, float32 0..1"""
        B, H, W, C = image.shape
        edge_frames = []
        phase_frames = []

        for b in range(B):
            frame = image[b].cpu().numpy()   # (H, W, C)

            # Grayscale conversion (luminance weights)
            if C == 1:
                gray = frame[..., 0]
            else:
                gray = (0.2126 * frame[..., 0] +
                        0.7152 * frame[..., 1] +
                        0.0722 * frame[..., 2])

            phase, edges = phase_stretch_transform(
                gray,
                lpf_sigma=lpf_sigma,
                phase_strength=phase_strength,
                warp_strength=warp_strength,
                morph_flag=morph_flag,
                min_thresh=min_thresh,
                max_thresh=max_thresh,
            )

            # ── edge output ──────────────────────────────────────────────────
            if output_mode == "rgb_overlay":
                overlay = frame.copy()
                mask = edges > 0.5
                overlay[mask] = np.array([1.0, 0.2, 0.2], dtype=np.float32)
                edge_out = overlay.astype(np.float32)
            else:
                edge_out = np.stack([edges] * 3, axis=-1).astype(np.float32)

            # ── phase_image: |phase| with percentile normalisation ────────────
            abs_phase = np.abs(phase)
            p99 = np.percentile(abs_phase, 99)
            phase_norm = np.clip(abs_phase / (p99 + 1e-9), 0.0, 1.0)
            phase_out = np.stack([phase_norm] * 3, axis=-1).astype(np.float32)

            edge_frames.append(torch.from_numpy(edge_out))
            phase_frames.append(torch.from_numpy(phase_out))

        edge_tensor = torch.stack(edge_frames, dim=0)    # (B, H, W, 3)
        phase_tensor = torch.stack(phase_frames, dim=0)  # (B, H, W, 3)

        return edge_tensor, phase_tensor
