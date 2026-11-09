# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Data-free checks of the yoga integration (subject scaling, couplings, geometry)."""

from __future__ import annotations

import importlib.util

import mujoco
import numpy as np
import pytest

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.skipif(
        importlib.util.find_spec("musclemimic_models") is None,
        reason="needs musclemimic_models",
    ),
]


@pytest.fixture(scope="module")
def base_model() -> mujoco.MjModel:
    from myosuite.integrations.yoga import subject_body as sb

    return sb.compile_spec(sb.base_fullbody_spec())


def _identity_table(model: mujoco.MjModel):
    from myosuite.integrations.yoga import subject_body as sb

    groups = sb.myo_body_groups(model)
    mass: dict[str, float] = {}
    for b in range(1, model.nbody):
        g = groups[model.body(b).name]
        mass[g] = mass.get(g, 0.0) + float(model.body_mass[b])
    return sb.SubjectTable(
        segment_scale={}, hip_width_scale=1.0, clavicle_scale={"l": 1.0, "r": 1.0}, head_lift=0.0,
        group_mass=mass, neck=False,
    )  # fmt: skip


def test_coupling_convention_matches_mujoco(base_model: mujoco.MjModel) -> None:
    from myosuite.integrations.yoga.kinematics import ReducedKinematics
    from myosuite.integrations.yoga.subject_body import sample_poses

    rk = ReducedKinematics(base_model)
    assert rk.n_indep == 37
    for q in sample_poses(base_model, 5, seed=1):
        assert rk.coupling_residual(q) < 1e-9


def test_velocity_map_matches_finite_differences(base_model: mujoco.MjModel) -> None:
    from myosuite.integrations.yoga.kinematics import ReducedKinematics
    from myosuite.integrations.yoga.subject_body import sample_poses

    rk = ReducedKinematics(base_model)
    q = sample_poses(base_model, 1, seed=2)[0]
    T = rk.velocity_map(q)
    eps = 1e-7
    for col, dof in enumerate(rk.indep_dof[6:12], start=6):
        jnt = int(np.nonzero(base_model.jnt_dofadr == dof)[0][0])
        q2 = q.copy()
        q2[base_model.jnt_qposadr[jnt]] += eps
        v = np.zeros(base_model.nv)
        mujoco.mj_differentiatePos(base_model, v, eps, q, rk.apply_couplings(q2))
        np.testing.assert_allclose(v, T[:, col], atol=1e-5)


def test_identity_table_keeps_the_model(base_model: mujoco.MjModel) -> None:
    from myosuite.integrations.yoga import subject_body as sb

    model = sb.compile_spec(sb.build_subject_spec(_identity_table(base_model)))
    np.testing.assert_allclose(
        model.body_mass, base_model.body_mass, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(model.body_pos, base_model.body_pos, atol=1e-12)
    np.testing.assert_allclose(model.site_pos, base_model.site_pos, atol=1e-12)
    np.testing.assert_allclose(
        model.actuator_lengthrange, base_model.actuator_lengthrange, atol=1e-12
    )


def test_segment_scale_moves_the_joint_and_its_muscles(
    base_model: mujoco.MjModel,
) -> None:
    from myosuite.integrations.yoga import subject_body as sb

    table = _identity_table(base_model)
    table.segment_scale = {"thigh_l": 1.1}
    model = sb.compile_spec(sb.build_subject_spec(table))
    d0, d1 = mujoco.MjData(base_model), mujoco.MjData(model)
    mujoco.mj_forward(base_model, d0)
    mujoco.mj_forward(model, d1)

    def thigh(m: mujoco.MjModel, d: mujoco.MjData, side: str) -> float:
        return float(
            np.linalg.norm(
                d.xpos[m.body(f"tibia_{side}").id] - d.xpos[m.body(f"femur_{side}").id]
            )
        )

    assert thigh(model, d1, "l") == pytest.approx(
        1.1 * thigh(base_model, d0, "l"), rel=1e-9
    )
    # The right thigh and the total mass of the (unchanged) mass groups are untouched.
    assert thigh(model, d1, "r") == pytest.approx(thigh(base_model, d0, "r"), rel=1e-12)
    assert model.body_mass.sum() == pytest.approx(base_model.body_mass.sum(), rel=1e-9)


def test_lowest_point_matches_mujoco_contact_depth() -> None:
    from myosuite.integrations.yoga.reference_io import lowest_point_z, quat_to_mat

    rng = np.random.default_rng(0)
    xml = """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .1"/>
      <body pos="0 0 0.05"><freejoint/><geom name="g" type="{t}" size="{s}"/></body></worldbody></mujoco>"""
    for t, s, code in (
        ("sphere", "0.05", mujoco.mjtGeom.mjGEOM_SPHERE), ("capsule", "0.03 0.1", mujoco.mjtGeom.mjGEOM_CAPSULE),
        ("box", "0.04 0.05 0.02", mujoco.mjtGeom.mjGEOM_BOX), ("ellipsoid", "0.04 0.06 0.03", mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    ):  # fmt: skip
        model = mujoco.MjModel.from_xml_string(xml.format(t=t, s=s))
        model.geom_margin[:] = 1.0
        data = mujoco.MjData(model)
        q = rng.normal(size=4)
        data.qpos[3:7] = q / np.linalg.norm(q)
        mujoco.mj_forward(model, data)
        g = model.geom("g").id
        low = lowest_point_z(
            np.array([code]),
            data.geom_xpos[[g]],
            quat_to_mat(data.xquat[[1]]),
            model.geom_size[[g]],
        )[0]
        dist = min(c.dist for c in data.contact[: data.ncon])
        assert low == pytest.approx(dist, abs=2e-4), t


def test_per_site_termination_threshold() -> None:
    from myosuite.terms.mimic_obs import mimic_termination_mask

    ref = np.zeros((2, 4, 3))
    cur = ref.copy()
    cur[1, 0, 0] = 0.6  # one site 0.6 m off, mean 0.15 m
    base = mimic_termination_mask(np, cur, ref, None, None, site_err_threshold=1.0)
    strict = mimic_termination_mask(
        np, cur, ref, None, None, site_err_threshold=1.0, max_site_err_threshold=0.5
    )
    assert not base.any()
    assert strict.tolist() == [False, True]


def test_passive_scale_touches_only_passive_force(base_model: mujoco.MjModel) -> None:
    from myosuite.integrations.yoga import subject_body as sb

    table = _identity_table(base_model)
    name = base_model.actuator(0).name
    table.passive_scale = {name: 0.5}
    model = sb.compile_spec(sb.build_subject_spec(table))
    a = model.actuator(name).id
    assert model.actuator_gainprm[a, 7] == pytest.approx(
        0.5 * base_model.actuator_gainprm[a, 7]
    )
    assert model.actuator_biasprm[a, 7] == pytest.approx(
        0.5 * base_model.actuator_biasprm[a, 7]
    )
    np.testing.assert_array_equal(
        model.actuator_gainprm[a, :7], base_model.actuator_gainprm[a, :7]
    )


def test_self_collision_replaces_explicit_pairs(base_model: mujoco.MjModel) -> None:
    from myosuite.integrations.yoga import subject_body as sb

    assert base_model.npair > 0
    table = _identity_table(base_model)
    table.colliders = [
        {"name": "c_femur", "body": "femur_l", "type": int(mujoco.mjtGeom.mjGEOM_CAPSULE),
         "size": [0.05, 0.15, 0.0], "pos": [0.0, -0.2, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
        {"name": "c_tibia", "body": "tibia_l", "type": int(mujoco.mjtGeom.mjGEOM_CAPSULE),
         "size": [0.05, 0.15, 0.0], "pos": [0.0, -0.2, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
    ]  # fmt: skip
    table.self_collision = True
    model = sb.compile_spec(sb.build_subject_spec(table))
    assert model.npair == 0
    assert model.nexclude == base_model.nexclude + len(sb.ADJACENT_HOSTS)
    colliding = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] > 0
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    assert sorted(model.geom(g).name for g in colliding) == ["c_femur", "c_tibia"]
    assert all(model.geom_margin[g] == 0.0 for g in colliding)


def test_start_sweep_cuts_and_names_each_start() -> None:
    from pathlib import Path

    from myosuite.core.trajectory_io import MotionClip
    from myosuite.integrations.yoga.task import clip_names, start_sweep

    frames = np.arange(550, dtype=float)[:, None]  # 5.5 s at 100 Hz
    clip = MotionClip(
        qpos=frames.repeat(3, 1), qvel=frames.repeat(2, 1), site_xpos=None, site_names=None,
        frequency_hz=100.0, source_path=Path("/clips/pose.npz"),
    )  # fmt: skip
    cuts = start_sweep((clip,), step_s=2.0)  # a start at 6 s would leave < 1 s
    assert clip_names(cuts) == ["pose@0s", "pose@2s", "pose@4s"]
    assert [c.qpos[0, 0] for c in cuts] == [0.0, 200.0, 400.0]
    assert cuts[2].qvel.shape == (150, 2) and cuts[2].qvel[0, 0] == 400.0


def test_noise_matched_action_matches_the_noisy_mean_excitation() -> None:
    torch = pytest.importorskip("torch")
    from myosuite.integrations.yoga.evaluate import noise_matched_action

    class _Dist:
        std_type = "scalar"
        std_range = (1e-6, 1e6)
        std_param = torch.tensor([0.77, 0.5, 0.3, 0.6])

    class _Policy:
        distribution = _Dist()

        def __call__(self, obs):
            return obs

    mu = torch.tensor([[-0.5, 0.0, 0.9, 0.25]])
    out = noise_matched_action(_Policy(), mu, n_muscles=3)
    gen = torch.Generator().manual_seed(0)
    noise = torch.randn(400_000, 3, generator=gen) * _Dist.std_param[:3]
    expected = torch.sigmoid(5.0 * (mu[:, :3] + noise - 0.5)).mean(0)
    assert torch.allclose(torch.sigmoid(5.0 * (out[0, :3] - 0.5)), expected, atol=0.012)
    assert out[0, 3] == mu[0, 3]


def test_pelvis_frame_error_separates_posture_from_whole_body_motion() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    from mjlab.utils.lab_api.math import quat_apply

    from myosuite.integrations.yoga.evaluate import _pelvis_frame_error

    gen = torch.Generator().manual_seed(0)
    targets = torch.randn(2, 5, 3, generator=gen)
    ref_root = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]] * 2)
    # Env 0: the whole body tipped 40 deg about x and slid 0.3 m; env 1: one site moved 0.4 m.
    half = float(np.deg2rad(40.0) / 2)
    q = torch.tensor([np.cos(half), np.sin(half), 0.0, 0.0], dtype=torch.float32)
    pos = torch.tensor([0.3, 0.0, 1.0])
    tipped = quat_apply(q.expand(5, 4), targets[0] - ref_root[0, :3]) + pos
    moved = targets[1].clone()
    moved[2, 2] += 0.4
    sites = torch.stack([tipped, moved])
    root_pos = torch.stack([pos, ref_root[1, :3]])
    root_quat = torch.stack([q, ref_root[1, 3:]])
    err = _pelvis_frame_error(sites, targets, root_pos, root_quat, ref_root)
    assert np.allclose(err[0], 0.0, atol=1e-5)
    assert np.allclose(err[1], [0.0, 0.0, 0.4, 0.0, 0.0], atol=1e-5)
