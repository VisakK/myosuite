# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Review videos: MyoFullBody motion with the plant reference drawn as a ghost.

Each frame shows two views (side and front) of the subject MyoFullBody playing a
``qpos`` trajectory, with her plant reference's colliders overlaid in translucent
blue (green while that body touches the floor in the reference).

CLI::

    MUJOCO_GL=egl python -m myosuite.integrations.yoga.render --table subject_table.json \
        --plant-xml <plant.xml> --refs <ref dir> --clips-dir <retargeted clip dir> \
        --out <dir> --clips Crane Handstand
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np

from myosuite.integrations.yoga import subject_body as sb
from myosuite.integrations.yoga.reference_io import (
    PlantColliders,
    YogaReference,
    load_reference,
    plant_body_lowest,
    plant_colliders,
    quat_to_mat,
)

GHOST = np.array([0.25, 0.55, 1.0, 0.28], np.float32)
GHOST_CONTACT = np.array([0.1, 0.9, 0.3, 0.45], np.float32)


def _add_geom(
    scene: mujoco.MjvScene,
    gtype: int,
    size: np.ndarray,
    pos: np.ndarray,
    mat: np.ndarray,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        gtype,
        np.asarray(size, float),
        np.asarray(pos, float),
        np.asarray(mat, float).reshape(-1),
        rgba,
    )
    # mjv_initGeom leaves these uninitialised.
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    g.objtype, g.objid, g.segid, g.transparent, g.camdist = 0, -1, -1, 1, 0.0
    scene.ngeom += 1


def draw_reference(
    scene: mujoco.MjvScene,
    ref: YogaReference,
    frame: int,
    colliders: PlantColliders,
    contact: np.ndarray,
) -> None:
    """Overlay the plant colliders of one reference frame."""
    R = quat_to_mat(ref.body_quat[frame])
    for g, b in enumerate(colliders.body_index):
        pos = ref.body_pos[frame, b] + R[b] @ colliders.pos[g]
        mat = R[b] @ colliders.mat[g]
        rgba = GHOST_CONTACT if contact[b] else GHOST
        _add_geom(scene, int(colliders.geom_type[g]), colliders.size[g], pos, mat, rgba)


class ClipRenderer:
    """Two-view offscreen renderer of the subject model.

    Args:
        model: Subject MyoFullBody.
        width: Width of one view in pixels.
        height: Height in pixels.
    """

    def __init__(
        self, model: mujoco.MjModel, width: int = 640, height: int = 540
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        model.vis.headlight.ambient[:] = 0.45
        model.vis.headlight.diffuse[:] = 0.75
        model.tendon_width[:] = np.minimum(model.tendon_width, 0.0025)
        self.renderer = mujoco.Renderer(
            model, height=height, width=width, max_geom=6000
        )
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
        self.cams = []
        for az in (90.0, 0.0):
            c = mujoco.MjvCamera()
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.distance, c.elevation, c.azimuth = 2.9, -10.0, az
            self.cams.append(c)

    def frame(
        self,
        qpos: np.ndarray,
        draw: Callable[[mujoco.MjvScene], None] | None = None,
        lookat: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render one ``qpos``; *draw(scene)* may add geoms. Returns ``(H, 2W, 3)`` uint8."""
        d = self.data
        d.qpos[:] = qpos
        mujoco.mj_forward(self.model, d)
        if lookat is None:
            # Height: the middle of the body's vertical extent. The mean would sit near the
            # floor when inverted, dragged down by the many hand and finger bodies.
            z = d.xpos[1:, 2]
            center = np.r_[d.xpos[1:, :2].mean(axis=0), 0.5 * (z.min() + z.max())]
        else:
            center = lookat
        views = []
        for c in self.cams:
            c.lookat[:] = [center[0], center[1], max(0.5, center[2])]
            self.renderer.update_scene(d, camera=c, scene_option=self.opt)
            if draw is not None:
                draw(self.renderer.scene)
            views.append(self.renderer.render())
        return np.concatenate(views, axis=1)


def render_retarget(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qpos_hz: float,
    ref: YogaReference,
    colliders: PlantColliders,
    out: Path,
    fps: float = 30.0,
    title: str | None = None,
    start_s: float = 0.0,
) -> None:
    """Video of a ``qpos`` trajectory against its reference (which starts at ``start_s``)."""
    import imageio.v2 as imageio

    contact = plant_body_lowest(ref, colliders) < 0.015
    rr = ClipRenderer(model)
    n = int(min(len(qpos) / qpos_hz, ref.n_frames / ref.fps - start_s) * fps)
    with imageio.get_writer(
        out, fps=fps, codec="libx264", quality=7, macro_block_size=8
    ) as w:
        for k in range(n):
            t = k / fps
            i = min(int(round(t * qpos_hz)), len(qpos) - 1)
            j = min(int(round((t + start_s) * ref.fps)), ref.n_frames - 1)
            img = rr.frame(
                qpos[i], lambda s, j=j: draw_reference(s, ref, j, colliders, contact[j])
            )
            w.append_data(_caption(img, f"{title or ref.name}  t={t + start_s:5.1f}s"))


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((8, 6), text, fill=(20, 20, 20))
    return np.asarray(im)


def main() -> None:
    """Render retargeted clips against their references."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--plant-xml", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--clips-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = sb.compile_spec(sb.build_subject_spec(sb.SubjectTable.load(args.table)))
    cols = plant_colliders(args.plant_xml)
    for p in sorted(args.clips_dir.glob("*.npz")):
        if args.clips and not any(c in p.stem for c in args.clips):
            continue
        clip = np.load(p)
        ref = load_reference(args.refs / p.name)
        out = args.out / f"{p.stem}.mp4"
        render_retarget(
            model, clip["qpos"].astype(float), float(clip["frequency"]), ref, cols, out
        )
        print("wrote", out)


if __name__ == "__main__":
    main()
