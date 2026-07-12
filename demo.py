"""
Virtual GPU — Demo
Renders three scenes, each as a 36-frame rotation, saves PNGs + GIF.

  Scene 0: Textured cube    (checkerboard + Phong)
  Scene 1: Smooth sphere    (gradient texture + Phong)
  Scene 2: Torus            (UV-debug color + Phong)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from gpu.math3d      import Mat4, vec3
from gpu.framebuffer import Framebuffer
from gpu.pipeline    import RenderPipeline
from gpu.texture     import Texture
from shaders.phong_shader import PhongVertexShader, PhongFragmentShader
from meshes.primitives    import cube, uv_sphere, torus


# ── Config ───────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 640, 480
FRAMES        = 36        # one full rotation (10° steps)
OUTPUT_DIR    = 'output'


# ── Scene definitions ─────────────────────────────────────────────────────────

SCENES = [
    {
        'name':    'cube',
        'mesh':    lambda: cube(1.2),
        'texture': lambda: Texture.checkerboard(128, 8,
                       color0=(0.95, 0.35, 0.1), color1=(0.15, 0.2, 0.75)),
        'material': {
            'ambient': 0.12, 'diffuse': 0.8, 'specular': 0.7, 'shininess': 64.0
        },
        'eye': vec3(0, 1.4, 3.5),
    },
    {
        'name':    'sphere',
        'mesh':    lambda: uv_sphere(1.0, 48, 32),
        'texture': lambda: Texture.gradient(256,
                       c0=(0.9, 0.3, 0.05), c1=(0.1, 0.6, 0.9)),
        'material': {
            'ambient': 0.08, 'diffuse': 0.7, 'specular': 0.9, 'shininess': 96.0
        },
        'eye': vec3(0, 1.0, 3.2),
    },
    {
        'name':    'torus',
        'mesh':    lambda: torus(0.7, 0.3, 64, 32),
        'texture': lambda: Texture.checkerboard(128, 12,
                       color0=(0.1, 0.8, 0.5), color1=(0.8, 0.1, 0.5)),
        'material': {
            'ambient': 0.10, 'diffuse': 0.75, 'specular': 0.6, 'shininess': 48.0
        },
        'eye': vec3(0, 1.5, 3.8),
    },
]


# ── Lighting ──────────────────────────────────────────────────────────────────

LIGHTS = [
    {'position': [ 4,  6,  4], 'color': [1.00, 0.95, 0.88], 'intensity': 1.2},
    {'position': [-3,  2,  3], 'color': [0.35, 0.55, 1.00], 'intensity': 0.7},
    {'position': [ 0, -3,  2], 'color': [1.00, 0.40, 0.20], 'intensity': 0.3},
]


# ── Render loop ───────────────────────────────────────────────────────────────

def render_scene(scene: dict):
    name = scene['name']
    eye  = scene['eye']
    print(f"\n{'═'*50}")
    print(f"  Scene: {name}  ({FRAMES} frames, {WIDTH}×{HEIGHT})")
    print(f"{'═'*50}")

    os.makedirs(os.path.join(OUTPUT_DIR, name), exist_ok=True)

    # Pipeline setup
    fb       = Framebuffer(WIDTH, HEIGHT)
    pipeline = RenderPipeline(fb)
    pipeline.vertex_shader   = PhongVertexShader()
    pipeline.fragment_shader = PhongFragmentShader()

    # Scene resources
    mesh    = scene['mesh']()
    texture = scene['texture']()
    mat     = scene['material']

    n_verts = len(mesh['vertices'])
    n_tris  = len(mesh['indices'])
    print(f"  Mesh: {n_verts} vertices, {n_tris} triangles")

    # Camera (fixed)
    center = vec3(0, 0, 0)
    up     = vec3(0, 1, 0)
    view   = Mat4.look_at(eye, center, up)
    proj   = Mat4.perspective(np.radians(45), WIDTH / HEIGHT, 0.1, 100.0)

    frame_paths = []
    t0 = time.time()

    for i in range(FRAMES):
        angle = 2 * np.pi * i / FRAMES

        # Rotate model around Y (+ slight tilt for depth)
        model  = Mat4.rotate_y(angle) @ Mat4.rotate_x(np.radians(15))
        norm_m = Mat4.normal_matrix(model)

        uniforms = {
            'model':      model,
            'view':       view,
            'proj':       proj,
            'normal_mat': norm_m,
            'camera_pos': eye,
            'lights':     LIGHTS,
            'material':   mat,
            'texture':    texture,
        }

        fb.clear(color=(0.04, 0.06, 0.12))
        tris_drawn = pipeline.draw_mesh(mesh, uniforms)

        path = os.path.join(OUTPUT_DIR, name, f'frame_{i:03d}.png')
        fb.save(path)
        frame_paths.append(path)

        elapsed = time.time() - t0
        fps     = (i + 1) / elapsed
        print(f"  Frame {i+1:3d}/{FRAMES} | {tris_drawn:5d} tris drawn | {fps:5.1f} fps")

    total = time.time() - t0
    print(f"\n  ✓ {FRAMES} frames in {total:.1f}s  ({FRAMES/total:.1f} fps avg)")

    # Try to save animated GIF
    gif_path = os.path.join(OUTPUT_DIR, f'{name}_animation.gif')
    _save_gif(frame_paths, gif_path, fps=24)

    return frame_paths


def _save_gif(frame_paths, gif_path, fps=24):
    try:
        from PIL import Image
        imgs = [Image.open(p) for p in frame_paths]
        duration_ms = int(1000 / fps)
        imgs[0].save(
            gif_path,
            save_all=True, append_images=imgs[1:],
            duration=duration_ms, loop=0, optimize=False,
        )
        print(f"  → GIF saved: {gif_path}")
    except ImportError:
        print("  (Install Pillow for animated GIF output)")
    except Exception as e:
        print(f"  (GIF failed: {e})")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Which scenes to render (default: all)
    scene_filter = sys.argv[1:] if len(sys.argv) > 1 else [s['name'] for s in SCENES]
    scenes_to_run = [s for s in SCENES if s['name'] in scene_filter]

    if not scenes_to_run:
        print(f"Unknown scene. Available: {[s['name'] for s in SCENES]}")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════╗")
    print("║         Virtual GPU — Software Renderer          ║")
    print("║   Pipeline: VS → Clip → Raster → FS → Z-test    ║")
    print("╚══════════════════════════════════════════════════╝")

    all_t0 = time.time()
    for scene in scenes_to_run:
        render_scene(scene)

    total = time.time() - all_t0
    print(f"\n{'═'*50}")
    print(f"  All done in {total:.1f}s")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'═'*50}")
