"""
Virtual GPU — Texture Unit
2D textures with bilinear filtering and procedural generators.
"""
import numpy as np


class Texture:
    """
    2D texture: (H, W, 3) float32 array in [0, 1].
    Supports bilinear filtering and UV wrapping.
    """

    def __init__(self, data: np.ndarray):
        self.data   = np.asarray(data, dtype=np.float32)
        self.height = self.data.shape[0]
        self.width  = self.data.shape[1]

    # ── Procedural constructors ──────────────────────────────────────────────

    @classmethod
    def checkerboard(
        cls, size: int = 128, squares: int = 8,
        color0=(1.0, 1.0, 1.0), color1=(0.1, 0.1, 0.1)
    ) -> "Texture":
        """Classic checkerboard pattern."""
        data = np.zeros((size, size, 3), dtype=np.float32)
        sq   = size // squares
        yy   = (np.arange(size) // sq)[:, None]
        xx   = (np.arange(size) // sq)[None, :]
        mask = ((xx + yy) % 2 == 0)
        data[mask]  = color0
        data[~mask] = color1
        return cls(data)

    @classmethod
    def gradient(
        cls, size: int = 128,
        c0=(1.0, 0.2, 0.0), c1=(0.0, 0.3, 1.0)
    ) -> "Texture":
        """Horizontal gradient."""
        t    = np.linspace(0, 1, size, dtype=np.float32)
        row  = np.outer(np.ones(size), t)  # (size, size)
        data = (1 - row[:, :, None]) * np.array(c0) + row[:, :, None] * np.array(c1)
        return cls(data.astype(np.float32))

    @classmethod
    def uv_debug(cls, size: int = 128) -> "Texture":
        """UV debug texture: R=U, G=V, B=0."""
        u = np.tile(np.linspace(0, 1, size, dtype=np.float32), (size, 1))
        v = np.tile(np.linspace(0, 1, size, dtype=np.float32)[:, None], (1, size))
        data = np.stack([u, v, np.zeros((size, size), dtype=np.float32)], axis=-1)
        return cls(data)

    @classmethod
    def noise(cls, size: int = 128, scale: float = 8.0) -> "Texture":
        """Simple value-noise-like texture."""
        rng  = np.random.default_rng(42)
        data = np.zeros((size, size, 3), dtype=np.float32)
        # Octave 1
        n    = rng.random((size, size)).astype(np.float32)
        data[:, :, 0] = n
        data[:, :, 1] = np.roll(n, size // 3, axis=0)
        data[:, :, 2] = np.roll(n, size // 5, axis=1)
        return cls(np.clip(data, 0, 1))

    @classmethod
    def from_file(cls, path: str) -> "Texture":
        """Load from an image file (requires PIL/Pillow)."""
        from PIL import Image
        img  = Image.open(path).convert('RGB')
        data = np.asarray(img, dtype=np.float32) / 255.0
        return cls(data)

    # ── Sampling ─────────────────────────────────────────────────────────────

    def sample(self, uvs: np.ndarray) -> np.ndarray:
        """
        Bilinear texture sampling.
        uvs : (N, 2) float — UV coordinates (wraps at 0/1 boundary)
        returns: (N, 3) float RGB [0, 1]
        """
        u = uvs[:, 0] % 1.0
        v = uvs[:, 1] % 1.0

        # Pixel coordinates (float)
        fpx = u * (self.width  - 1)
        fpy = v * (self.height - 1)

        # Integer corners
        x0 = np.floor(fpx).astype(np.int32)
        y0 = np.floor(fpy).astype(np.int32)
        x1 = np.minimum(x0 + 1, self.width  - 1)
        y1 = np.minimum(y0 + 1, self.height - 1)

        # Fractional part for blending
        fx = (fpx - x0)[:, None]   # (N, 1)
        fy = (fpy - y0)[:, None]

        # Sample 4 corners
        c00 = self.data[y0, x0]   # (N, 3)
        c10 = self.data[y0, x1]
        c01 = self.data[y1, x0]
        c11 = self.data[y1, x1]

        # Bilinear blend
        return (
            c00 * (1 - fx) * (1 - fy) +
            c10 * fx       * (1 - fy) +
            c01 * (1 - fx) * fy       +
            c11 * fx       * fy
        )
