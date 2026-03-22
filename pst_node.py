import numpy as np
import torch
from scipy import ndimage


def phase_stretch_transform(img_gray: np.ndarray,
                            lpf_sigma: float = 0.21,
                            phase_strength: float = 0.48,
                            warp_strength: float = 12.14,
                            morph_flag: bool = True,
                            min_thresh: float = -1.0,
                            max_thresh: float = 0.003) -> tuple[np.ndarray, np.ndarray]:
    """
    Phase Stretch Transform (PST)

    Returns:
        phase_image : raw phase output (float, roughly -pi..pi)
        edge_image  : binary or soft edge map (float 0..1)
    """
    H, W = img_gray.shape
    L = 0.5

    fx = np.linspace(-L, L, W)
    fy = np.linspace(-L, L, H)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX ** 2 + FY ** 2)          # radial frequency grid

    # ── Low-pass filter (Gaussian) ───────────────────────────────────────────
    lpf = np.exp(-0.5 * (R / (lpf_sigma + 1e-9)) ** 2)

    img_fft = np.fft.fftshift(np.fft.fft2(img_gray.astype(np.float64)))
    img_filtered = img_fft * lpf

    # ── PST kernel (warped phase ramp) ───────────────────────────────────────
    W_val = warp_strength
    S_val = phase_strength
    pst_kernel = (W_val * R / (1.0 + W_val * R)) * np.exp(-1j * S_val * R / (1.0 + R))

    img_pst = np.fft.ifft2(np.fft.ifftshift(img_filtered * pst_kernel))

    # ── Phase angle ──────────────────────────────────────────────────────────
    phase = np.angle(img_pst)          # range: -π … π

    if morph_flag:
        edges = ((phase > max_thresh) | (phase < min_thresh)).astype(np.float64)
        # small morphological clean-up
        edges = ndimage.binary_closing(edges, structure=np.ones((3, 3))).astype(np.float64)
        edges = ndimage.binary_opening(edges, structure=np.ones((3, 3))).astype(np.float64)
    else:
        # soft edge map – normalise phase to 0..1
        edges = (phase - phase.min()) / (phase.max() - phase.min() + 1e-9)

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
                    "tooltip": "Low-pass filter sigma – higher = more smoothing before PST"
                }),
                "phase_strength": ("FLOAT", {
                    "default": 0.48, "min": 0.0, "max": 3.0, "step": 0.01,
                    "tooltip": "PST phase strength (S)"
                }),
                "warp_strength": ("FLOAT", {
                    "default": 12.14, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "PST warp strength (W)"
                }),
                "min_thresh": ("FLOAT", {
                    "default": -1.0, "min": -3.15, "max": 0.0, "step": 0.001,
                    "tooltip": "Lower phase threshold for binary edge map"
                }),
                "max_thresh": ("FLOAT", {
                    "default": 0.003, "min": 0.0, "max": 3.15, "step": 0.001,
                    "tooltip": "Upper phase threshold for binary edge map"
                }),
                "morph_flag": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply morphological clean-up (binary edges); "
                               "False returns normalised phase image"
                }),
                "output_mode": (["grayscale", "rgb_overlay"], {
                    "default": "grayscale",
                    "tooltip": "grayscale = white edges on black; "
                               "rgb_overlay = edges overlaid on original"
                }),
            },
        }

    # ------------------------------------------------------------------
    def run(self, image: torch.Tensor,
            lpf_sigma: float,
            phase_strength: float,
            warp_strength: float,
            min_thresh: float,
            max_thresh: float,
            morph_flag: bool,
            output_mode: str):
        """
        image shape: (B, H, W, C)  – ComfyUI convention, float32 0..1
        """
        B, H, W, C = image.shape
        edge_frames = []
        phase_frames = []

        for b in range(B):
            frame = image[b].cpu().numpy()          # (H, W, C)

            # Convert to grayscale for PST (luminance weights)
            if C == 1:
                gray = frame[..., 0]
            else:
                gray = 0.2126 * frame[..., 0] + \
                       0.7152 * frame[..., 1] + \
                       0.0722 * frame[..., 2]

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
                edge_rgb = np.stack([edges] * 3, axis=-1).astype(np.float32)
                edge_out = edge_rgb

            # ── phase output (normalised to 0..1, RGB) ───────────────────────
            phase_norm = (phase - phase.min()) / (phase.max() - phase.min() + 1e-9)
            phase_out = np.stack([phase_norm] * 3, axis=-1).astype(np.float32)

            edge_frames.append(torch.from_numpy(edge_out))
            phase_frames.append(torch.from_numpy(phase_out))

        edge_tensor = torch.stack(edge_frames, dim=0)    # (B, H, W, 3)
        phase_tensor = torch.stack(phase_frames, dim=0)  # (B, H, W, 3)

        return edge_tensor, phase_tensor
