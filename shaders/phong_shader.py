"""
Virtual GPU — Phong Shader
Classic per-pixel Blinn-Phong lighting (ambient + diffuse + specular).

Vertex shader outputs:
  clip_pos      vec4  — clip-space position (required)
  world_pos     vec3  — world-space position
  world_normal  vec3  — world-space normal
  uv            vec2  — texture coordinates

Fragment shader uniforms:
  model       Mat4
  view        Mat4
  proj        Mat4
  normal_mat  Mat3  (optional, auto-computed)
  camera_pos  vec3
  lights      list of dicts:
                position  vec3
                color     vec3  (default white)
                intensity float (default 1.0)
  material    dict:
                ambient   float (default 0.15)
                diffuse   float (default 0.75)
                specular  float (default 0.5)
                shininess float (default 32)
                color     vec3  (default white)
  texture     Texture  (optional)
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



class PhongVertexShader:
    """Transform vertex to clip space; pass world-space attrs as varyings."""

    def __call__(self, attribs: dict, uniforms: dict) -> dict:
        pos    = np.asarray(attribs['position'], dtype=np.float32)
        normal = np.asarray(attribs.get('normal', [0, 1, 0]), dtype=np.float32)
        uv     = np.asarray(attribs.get('uv',     [0, 0]),    dtype=np.float32)

        model = uniforms['model']
        view  = uniforms['view']
        proj  = uniforms['proj']

        # World position
        pos4      = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float32)
        world_pos = (model @ pos4)[:3]

        # World normal (use inverse-transpose to handle non-uniform scaling)
        normal_mat  = uniforms.get('normal_mat', Mat4.normal_matrix(model))
        world_normal = normal_mat @ normal
        n = np.linalg.norm(world_normal)
        if n > 1e-10:
            world_normal /= n

        # Clip-space position
        clip_pos = proj @ view @ model @ pos4

        return {
            'clip_pos':     clip_pos,
            'world_pos':    world_pos,
            'world_normal': world_normal,
            'uv':           uv,
        }


class PhongFragmentShader:
    """
    Blinn-Phong per-pixel lighting — fully vectorized with numpy.
    Processes all pixels of a triangle in one call (like a GPU warp).
    """

    def __call__(self, varyings: dict, uniforms: dict, n_pixels: int) -> np.ndarray:
        world_pos    = varyings['world_pos']     # (N, 3)
        world_normal = varyings['world_normal']  # (N, 3)
        uv           = varyings.get('uv', np.zeros((n_pixels, 2), dtype=np.float32))

        # Re-normalize interpolated normals (they can drift after interpolation)
        nlen = np.linalg.norm(world_normal, axis=1, keepdims=True)
        nlen = np.where(nlen < 1e-10, 1.0, nlen)
        N = world_normal / nlen   # (N, 3)  surface normals

        # View direction (from surface toward camera)
        cam_pos = np.asarray(uniforms.get('camera_pos', [0, 0, 5]), dtype=np.float32)
        V_raw   = cam_pos - world_pos                          # toward camera
        V       = V_raw / (np.linalg.norm(V_raw, axis=1, keepdims=True) + 1e-10)

        # Material properties
        mat       = uniforms.get('material', {})
        ka        = float(mat.get('ambient',   0.15))
        kd        = float(mat.get('diffuse',   0.75))
        ks        = float(mat.get('specular',  0.5))
        shininess = float(mat.get('shininess', 32.0))

        # Base color: from texture or solid material color
        texture = uniforms.get('texture')
        if texture is not None:
            base_color = texture.sample(uv)                          # (N, 3)
        else:
            mc = np.asarray(mat.get('color', [1.0, 1.0, 1.0]), dtype=np.float32)
            base_color = np.broadcast_to(mc, (n_pixels, 3)).copy()

        # Ambient term
        color = base_color * ka     # (N, 3)

        # Accumulate each light
        lights = uniforms.get('lights', [
            {'position': [5, 5, 5], 'color': [1, 1, 1], 'intensity': 1.0}
        ])

        for light in lights:
            L_pos   = np.asarray(light['position'], dtype=np.float32)
            L_color = np.asarray(light.get('color', [1, 1, 1]), dtype=np.float32)
            intensity = float(light.get('intensity', 1.0))

            # Light direction (surface → light), per pixel
            L_raw = L_pos - world_pos                                # (N, 3)
            dist  = np.linalg.norm(L_raw, axis=1, keepdims=True)
            L     = L_raw / (dist + 1e-10)

            # ── Diffuse (Lambertian) ────────────────────────────────────────
            NdotL    = np.maximum(0.0, np.sum(N * L, axis=1))       # (N,)
            diffuse  = kd * NdotL[:, None] * base_color * L_color * intensity

            # ── Specular (Blinn-Phong half-vector) ──────────────────────────
            H        = L + V                                          # halfway vector
            H        = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-10)
            NdotH    = np.maximum(0.0, np.sum(N * H, axis=1))
            specular = ks * (NdotH[:, None] ** shininess) * L_color * intensity

            color = color + diffuse + specular

        return np.clip(color, 0.0, 1.0)


class FlatFragmentShader:
    """Flat shading: uniform color per triangle (no lighting calculation)."""

    def __call__(self, varyings: dict, uniforms: dict, n_pixels: int) -> np.ndarray:
        mat  = uniforms.get('material', {})
        color = np.asarray(mat.get('color', [0.7, 0.4, 0.2]), dtype=np.float32)
        return np.broadcast_to(color, (n_pixels, 3)).copy()


class NormalDebugFragmentShader:
    """Visualize world-space normals as RGB colors (great for debugging)."""

    def __call__(self, varyings: dict, uniforms: dict, n_pixels: int) -> np.ndarray:
        normals = varyings['world_normal']  # (N, 3) in [-1, 1]
        nlen    = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(nlen < 1e-10, 1.0, nlen)
        return np.clip(normals * 0.5 + 0.5, 0, 1)   # remap to [0,1]


class UVDebugFragmentShader:
    """Visualize UV coordinates as RG colors."""

    def __call__(self, varyings: dict, uniforms: dict, n_pixels: int) -> np.ndarray:
        uv = varyings.get('uv', np.zeros((n_pixels, 2), dtype=np.float32))
        colors = np.zeros((n_pixels, 3), dtype=np.float32)
        colors[:, 0] = uv[:, 0] % 1.0
        colors[:, 1] = uv[:, 1] % 1.0
        return colors
