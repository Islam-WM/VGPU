"""
Virtual GPU — Mesh Primitives
Procedural mesh generators: cube, UV sphere, torus, plane.

Every function returns a mesh dict:
  'vertices'  (N, 3) float32
  'normals'   (N, 3) float32
  'uvs'       (N, 2) float32
  'indices'   (M, 3) int32
"""
import numpy as np


def cube(size: float = 1.0) -> dict:
    """
    Axis-aligned unit cube centered at origin.
    24 vertices (4 per face, so normals are sharp at edges).
    12 triangles (2 per face).
    Winding: CCW when viewed from outside each face (right-hand, Y-up).
    """
    h = size * 0.5

    # Each face: [v0, v1, v2, v3] CCW from outside, outward normal, UV corners
    faces = [
        # (vertices_4,               normal,     )
        ([(-h,-h, h),(h,-h, h),(h,h, h),(-h,h, h)], ( 0, 0, 1)),  # +Z front
        ([(h,-h,-h),(-h,-h,-h),(-h,h,-h),(h,h,-h)], ( 0, 0,-1)),  # -Z back
        ([(h,-h, h),(h,-h,-h),(h,h,-h),(h,h, h)],   ( 1, 0, 0)),  # +X right
        ([(-h,-h,-h),(-h,-h,h),(-h,h,h),(-h,h,-h)],(-1, 0, 0)),  # -X left
        ([(-h,h, h),(h,h, h),(h,h,-h),(-h,h,-h)],   ( 0, 1, 0)),  # +Y top
        ([(-h,-h,-h),(h,-h,-h),(h,-h,h),(-h,-h,h)], ( 0,-1, 0)),  # -Y bottom
    ]
    uv_corners = [(0,0), (1,0), (1,1), (0,1)]

    vertices, normals, uvs, indices = [], [], [], []
    for face_verts, face_normal in faces:
        base = len(vertices)
        for v, uv in zip(face_verts, uv_corners):
            vertices.append(v)
            normals.append(face_normal)
            uvs.append(uv)
        indices += [[base,   base+1, base+2],
                    [base,   base+2, base+3]]

    return _make_mesh(vertices, normals, uvs, indices)


def uv_sphere(radius: float = 1.0, segments: int = 32, rings: int = 24) -> dict:
    """
    UV sphere.  segments = longitude subdivisions, rings = latitude.
    North/south poles are handled as regular quads (no fan, avoids UV seam issues).
    """
    verts, norms, uvs_list, idx = [], [], [], []

    for ring in range(rings + 1):
        phi = np.pi * ring / rings           # 0 (north) … π (south)
        v   = ring / rings                   # UV V

        for seg in range(segments + 1):
            theta = 2 * np.pi * seg / segments   # 0 … 2π
            u     = seg / segments               # UV U

            x = np.sin(phi) * np.cos(theta)
            y = np.cos(phi)
            z = np.sin(phi) * np.sin(theta)

            verts.append([radius * x, radius * y, radius * z])
            norms.append([x, y, z])
            uvs_list.append([u, v])

    for ring in range(rings):
        for seg in range(segments):
            a = ring       * (segments + 1) + seg
            b = a + 1
            c = (ring + 1) * (segments + 1) + seg
            d = c + 1
            idx += [[a, b, d], [a, d, c]]

    return _make_mesh(verts, norms, uvs_list, idx)


def torus(R: float = 0.7, r: float = 0.3,
          segments: int = 48, rings: int = 24) -> dict:
    """
    Torus.  R = major radius (centre of tube), r = minor radius (tube radius).
    segments = around the tube, rings = around the whole torus.
    """
    verts, norms, uvs_list, idx = [], [], [], [] 

    for ring in range(rings + 1):
        theta   = 2 * np.pi * ring / rings
        cos_t   = np.cos(theta)
        sin_t   = np.sin(theta)
        v_coord = ring / rings

        for seg in range(segments + 1):
            phi   = 2 * np.pi * seg / segments
            cos_p = np.cos(phi)
            sin_p = np.sin(phi)
            u_coord = seg / segments

            # Surface point
            x = (R + r * cos_p) * cos_t
            y = r * sin_p
            z = (R + r * cos_p) * sin_t

            # Surface normal (toward outside of tube)
            nx = cos_p * cos_t
            ny = sin_p
            nz = cos_p * sin_t

            verts.append([x, y, z])
            norms.append([nx, ny, nz])
            uvs_list.append([u_coord, v_coord])

    for ring in range(rings):
        for seg in range(segments):
            a = ring       * (segments + 1) + seg
            b = a + 1
            c = (ring + 1) * (segments + 1) + seg
            d = c + 1
            idx += [[a, b, d], [a, d, c]]

    return _make_mesh(verts, norms, uvs_list, idx)


def plane(size: float = 2.0, subdivisions: int = 1) -> dict:
    """
    Flat XZ plane centred at origin, facing +Y.
    subdivisions: how many quads per side.
    """
    n   = subdivisions
    verts, norms, uvs_list, idx = [], [], [], []

    for row in range(n + 1):
        for col in range(n + 1):
            x = -size * 0.5 + size * col / n
            z = -size * 0.5 + size * row / n
            verts.append([x, 0.0, z])
            norms.append([0.0, 1.0, 0.0])
            uvs_list.append([col / n, row / n])

    for row in range(n):
        for col in range(n):
            a = row * (n + 1) + col
            b = a + 1
            c = (row + 1) * (n + 1) + col
            d = c + 1
            idx += [[a, b, d], [a, d, c]]

    return _make_mesh(verts, norms, uvs_list, idx)


# ── Helper ───────────────────────────────────────────────────────────────────

def _make_mesh(vertices, normals, uvs, indices) -> dict:
    return {
        'vertices': np.array(vertices, dtype=np.float32),
        'normals':  np.array(normals,  dtype=np.float32),
        'uvs':      np.array(uvs,      dtype=np.float32),
        'indices':  np.array(indices,  dtype=np.int32),
    }
