"""
Virtual GPU — Math Library
Vec3, Vec4, Mat4 using numpy for high performance.
"""
import numpy as np


# ── Vector helpers ──────────────────────────────────────────────────────────

def vec3(x=0., y=0., z=0.) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float32)

def vec4(x=0., y=0., z=0., w=1.) -> np.ndarray:
    return np.array([x, y, z, w], dtype=np.float32)

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v.copy()

def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b).astype(np.float32)


# ── Mat4 ────────────────────────────────────────────────────────────────────

class Mat4:
    """4×4 matrix factory for 3D transforms."""

    @staticmethod
    def identity() -> np.ndarray:
        return np.eye(4, dtype=np.float32)

    @staticmethod
    def translate(tx: float, ty: float, tz: float) -> np.ndarray:
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = [tx, ty, tz]
        return m

    @staticmethod
    def scale(sx: float, sy: float, sz: float) -> np.ndarray:
        return np.diag([sx, sy, sz, 1.0]).astype(np.float32)

    @staticmethod
    def rotate_x(a: float) -> np.ndarray:
        """Rotation around X axis (radians)."""
        c, s = float(np.cos(a)), float(np.sin(a))
        return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=np.float32)

    @staticmethod
    def rotate_y(a: float) -> np.ndarray:
        """Rotation around Y axis (radians)."""
        c, s = float(np.cos(a)), float(np.sin(a))
        return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=np.float32)

    @staticmethod
    def rotate_z(a: float) -> np.ndarray:
        """Rotation around Z axis (radians)."""
        c, s = float(np.cos(a)), float(np.sin(a))
        return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)

    @staticmethod
    def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Perspective projection matrix (right-hand, Y-up)."""
        f = 1.0 / np.tan(fov_y * 0.5)
        nf = 1.0 / (near - far)
        return np.array([
            [f / aspect,  0,                    0,  0],
            [0,           f,                    0,  0],
            [0,           0,  (far + near) * nf,  2 * far * near * nf],
            [0,           0,                   -1,  0],
        ], dtype=np.float32)

    @staticmethod
    def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Camera / view matrix."""
        eye    = np.asarray(eye,    dtype=np.float32)
        center = np.asarray(center, dtype=np.float32)
        up     = np.asarray(up,     dtype=np.float32)
        f = normalize(center - eye)
        r = normalize(np.cross(f, up).astype(np.float32))
        u = np.cross(r, f).astype(np.float32)
        return np.array([
            [ r[0],  r[1],  r[2], -float(r @ eye)],
            [ u[0],  u[1],  u[2], -float(u @ eye)],
            [-f[0], -f[1], -f[2],  float(f @ eye)],
            [    0,      0,     0,               1],
        ], dtype=np.float32)

    @staticmethod
    def normal_matrix(model: np.ndarray) -> np.ndarray:
        """Inverse-transpose of upper-left 3×3 (for transforming normals)."""
        try:
            return np.linalg.inv(model[:3, :3]).T.astype(np.float32)
        except np.linalg.LinAlgError:
            return model[:3, :3].copy()
