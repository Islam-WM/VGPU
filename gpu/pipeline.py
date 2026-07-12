"""
Virtual GPU — Render Pipeline

Simulates the full GPU pipeline:


    1. Vertex Shader      (programmable, per-vertex)       
    2. Primitive Assembly (group into triangles)           
    3. Clipping           (near-plane, simplified)         
    4. Perspective Divide (clip → NDC)                     
    5. Viewport Transform (NDC → screen pixels)            
    6. Backface Culling                                    
    7. Rasterization      (barycentric, vectorized)        
    8. Attribute Interpolation (perspective-correct)       
    9. Fragment Shader    (programmable, per-pixel batch)  
   10. Depth Test + Framebuffer Write                      
  
"""
import numpy as np
from .framebuffer import Framebuffer


class RenderPipeline:

    def __init__(self, framebuffer: Framebuffer):
        self.fb = framebuffer

        # Programmable stages (assign your shader objects)
        self.vertex_shader   = None
        self.fragment_shader = None

        # Pipeline state
        self.cull_backfaces = True
        self.wireframe      = False   # future: draw edges only

    # ── Main draw call ───────────────────────────────────────────────────────

    def draw_mesh(self, mesh: dict, uniforms: dict = None) -> int:
        """
        Render a mesh.

        mesh keys:
          'vertices'  (N, 3) float32  — model-space positions
          'indices'   (M, 3) int      — triangle index triples
          'normals'   (N, 3) float32  — vertex normals  [optional]
          'uvs'       (N, 2) float32  — texture coords  [optional]

        uniforms: dict passed to both shaders (matrices, lights, …)

        Returns the number of triangles rasterized.
        """
        if self.vertex_shader is None:
            raise ValueError("No vertex_shader bound to pipeline.")
        if self.fragment_shader is None:
            raise ValueError("No fragment_shader bound to pipeline.")

        uniforms = uniforms or {}
        verts   = mesh['vertices']
        indices = mesh['indices']
        normals = mesh.get('normals')
        uvs     = mesh.get('uvs')

        W, H = self.fb.width, self.fb.height

        # ── Stage 1: Vertex Shader ───────────────────────────────────────────
        vs_out = []
        for i in range(len(verts)):
            attribs = {'position': verts[i]}
            if normals is not None:
                attribs['normal'] = normals[i]
            if uvs is not None:
                attribs['uv'] = uvs[i]
            vs_out.append(self.vertex_shader(attribs, uniforms))

        # ── Stages 2–10: per triangle ────────────────────────────────────────
        tri_count = 0
        for tri in indices:
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
            o0, o1, o2 = vs_out[i0], vs_out[i1], vs_out[i2]

            cp0 = np.asarray(o0['clip_pos'], dtype=np.float32)
            cp1 = np.asarray(o1['clip_pos'], dtype=np.float32)
            cp2 = np.asarray(o2['clip_pos'], dtype=np.float32)

            # ── Stage 3: Near-plane clip (discard if any vertex behind camera) ──
            if cp0[3] < 0.001 or cp1[3] < 0.001 or cp2[3] < 0.001:
                continue

            # ── Stage 4: Perspective divide → NDC [-1, 1] ──────────────────
            ndc0 = cp0[:3] / cp0[3]
            ndc1 = cp1[:3] / cp1[3]
            ndc2 = cp2[:3] / cp2[3]

            # Coarse frustum cull on a single axis
            xs_ndc = np.array([ndc0[0], ndc1[0], ndc2[0]])
            ys_ndc = np.array([ndc0[1], ndc1[1], ndc2[1]])
            if xs_ndc.min() > 1 or xs_ndc.max() < -1:
                continue
            if ys_ndc.min() > 1 or ys_ndc.max() < -1:
                continue

            # ── Stage 5: Viewport transform ────────────────────────────────
            #  NDC X [-1,1] → screen [0, W-1]
            #  NDC Y [ 1,-1] → screen [0, H-1]  (Y flip!)
            #  NDC Z kept for depth buffer
            def to_screen(ndc):
                sx = (ndc[0] + 1.0) * 0.5 * (W - 1)
                sy = (1.0 - ndc[1]) * 0.5 * (H - 1)
                return np.array([sx, sy, ndc[2]], dtype=np.float32)

            s0, s1, s2 = to_screen(ndc0), to_screen(ndc1), to_screen(ndc2)

            # ── Stage 6: Backface culling ────────────────────────────────────
            #  After Y-flip, front faces have NEGATIVE signed area in screen space.
            #  (CCW in world → CW in screen due to Y inversion)
            if self.cull_backfaces:
                e01 = s1[:2] - s0[:2]
                e02 = s2[:2] - s0[:2]
                signed_area = e01[0] * e02[1] - e01[1] * e02[0]
                if signed_area >= 0:   # positive = CW in screen = back-face
                    continue

            # ── Stages 7–10: Rasterize ───────────────────────────────────────
            self._rasterize(s0, s1, s2, cp0[3], cp1[3], cp2[3], o0, o1, o2, uniforms)
            tri_count += 1

        return tri_count

    # ── Rasterizer ───────────────────────────────────────────────────────────

    def _rasterize(self, s0, s1, s2, w0, w1, w2, vs0, vs1, vs2, uniforms):
        """
        Rasterize one triangle.

        Uses barycentric coordinates (edge function method) and
        perspective-correct attribute interpolation.
        All pixel operations are vectorized with numpy.
        """
        W, H = self.fb.width, self.fb.height

        # Bounding box (clamped to screen)
        xmin = max(0,   int(np.floor(min(s0[0], s1[0], s2[0]))))
        xmax = min(W-1, int(np.ceil( max(s0[0], s1[0], s2[0]))))
        ymin = max(0,   int(np.floor(min(s0[1], s1[1], s2[1]))))
        ymax = min(H-1, int(np.ceil( max(s0[1], s1[1], s2[1]))))

        if xmin > xmax or ymin > ymax:
            return

        # ── Pixel grid (centres at x+0.5, y+0.5) ────────────────────────────
        xs = np.arange(xmin, xmax + 1, dtype=np.float32) + 0.5
        ys = np.arange(ymin, ymax + 1, dtype=np.float32) + 0.5
        px, py = np.meshgrid(xs, ys, indexing='xy')
        px = px.ravel()
        py = py.ravel()

        # ── Edge functions (= 2× signed area of sub-triangle) ───────────────
        def ef(ax, ay, bx, by, px, py):
            return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

        area = ef(s0[0], s0[1], s1[0], s1[1], s2[0], s2[1])
        if abs(area) < 1.0:
            return

        e0 = ef(s1[0], s1[1], s2[0], s2[1], px, py)
        e1 = ef(s2[0], s2[1], s0[0], s0[1], px, py)
        e2 = ef(s0[0], s0[1], s1[0], s1[1], px, py)
        # Inside triangle (all edge functions same sign as area)
        if area < 0:
            mask = (e0 <= 0) & (e1 <= 0) & (e2 <= 0)
            area, e0, e1, e2 = -area, -e0, -e1, -e2
        else:
            mask = (e0 >= 0) & (e1 >= 0) & (e2 >= 0)

        if not np.any(mask):
            return

        # ── Barycentric coordinates ──────────────────────────────────────────
        b0 = e0[mask] / area
        b1 = e1[mask] / area
        b2 = e2[mask] / area

        pxi = px[mask].astype(np.int32)
        pyi = py[mask].astype(np.int32)

        # ── Depth (linear interpolation of NDC Z) ────────────────────────────
        depth = b0 * s0[2] + b1 * s1[2] + b2 * s2[2]

        # ── Perspective-correct attribute interpolation ──────────────────────
        #  Attributes must be divided by clip-w before interpolation, then
        #  divided by the interpolated 1/w to recover the correct value.
        #  This is how real GPUs handle perspective distortion.
        iw0 = 1.0 / w0 if abs(w0) > 1e-10 else 0.0
        iw1 = 1.0 / w1 if abs(w1) > 1e-10 else 0.0
        iw2 = 1.0 / w2 if abs(w2) > 1e-10 else 0.0

        interp_iw = b0 * iw0 + b1 * iw1 + b2 * iw2
        safe_iw   = np.where(np.abs(interp_iw) > 1e-12, interp_iw, 1e-12)

        # Corrected barycentric weights (perspective-correct)
        cb0 = (b0 * iw0) / safe_iw
        cb1 = (b1 * iw1) / safe_iw
        cb2 = (b2 * iw2) / safe_iw

        # ── Interpolate varyings ─────────────────────────────────────────────
        N = int(np.sum(mask))
        varyings = {}
        for key in vs0:
            if key == 'clip_pos':
                continue
            a0 = np.asarray(vs0[key], dtype=np.float32)
            a1 = np.asarray(vs1[key], dtype=np.float32)
            a2 = np.asarray(vs2[key], dtype=np.float32)
            if a0.ndim == 0:
                varyings[key] = cb0 * float(a0) + cb1 * float(a1) + cb2 * float(a2)
            else:
                varyings[key] = (
                    cb0[:, None] * a0 +
                    cb1[:, None] * a1 +
                    cb2[:, None] * a2
                )

        # ── Stage 9: Fragment Shader ─────────────────────────────────────────
        #  Called once for ALL pixels of this triangle simultaneously.
        #  Mirrors GPU warp/SIMD execution.
        colors = self.fragment_shader(varyings, uniforms, N)  # → (N, 3)

        # ── Stage 10: Depth test + write ─────────────────────────────────────
        self.fb.set_pixels_batch(pxi, pyi, colors, depth)
