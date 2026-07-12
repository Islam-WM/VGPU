"""
Virtual GPU — Framebuffer
Color buffer (RGB float) + Depth buffer (Z-buffer).
"""
import numpy as np


class Framebuffer:
    """GPU output: color + depth buffers."""

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height
        # Float RGB [0,1] — same as GPU render target
        self.color = np.zeros((height, width, 3), dtype=np.float32)
        # Depth initialized to +inf (everything fails the test until written)
        self.depth = np.full((height, width), np.inf, dtype=np.float32)

    # ── Clear ────────────────────────────────────────────────────────────────

    def clear(self, color=(0.05, 0.07, 0.14)):
        """Clear color and depth buffers."""
        self.color[:] = color
        self.depth[:] = np.inf

    # ── Pixel write (vectorized, with depth test) ────────────────────────────

    def set_pixels_batch(
        self,
        xs:     np.ndarray,   # (N,) int — screen X
        ys:     np.ndarray,   # (N,) int — screen Y
        colors: np.ndarray,   # (N,3) float [0,1]
        depths: np.ndarray,   # (N,) float
    ):
        """
        Write N pixels at once.
        Only writes where depth < current depth (Z-buffer test).
        Pixels within one triangle are always unique (x,y), so simple
        array indexing is safe here.
        """
        # 1. Bounds check
        valid = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
        xs, ys, colors, depths = xs[valid], ys[valid], colors[valid], depths[valid]
        if xs.size == 0:
            return

        # 2. Depth test
        closer = depths < self.depth[ys, xs]
        xs, ys   = xs[closer],    ys[closer]
        colors   = colors[closer]
        depths   = depths[closer]
        if xs.size == 0:
            return

        # 3. Write
        self.depth[ys, xs] = depths
        self.color[ys, xs] = np.clip(colors, 0.0, 1.0)

    # ── Output ───────────────────────────────────────────────────────────────

    def to_uint8(self) -> np.ndarray:
        """Convert to uint8 with gamma correction (≈ sRGB)."""
        gamma = np.clip(self.color, 0.0, 1.0) ** (1.0 / 2.2)
        return (gamma * 255.0).astype(np.uint8)

    def save(self, path: str):
        """Save framebuffer as PNG."""
        img = self.to_uint8()
        _save_image(img, path)
        print(f"  → {path}")


# ── Internal image writer ────────────────────────────────────────────────────

def _save_image(arr: np.ndarray, path: str):
    """Save RGB uint8 array to PNG (PIL or matplotlib fallback)."""
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
        return
    except ImportError:
        pass
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.imsave(path, arr)
        return
    except ImportError:
        pass
    raise RuntimeError("Install Pillow or matplotlib to save PNG files.")
