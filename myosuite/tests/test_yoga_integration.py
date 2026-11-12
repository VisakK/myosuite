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


@pytest.fixture(scope="module")
def neck_model(base_model: mujoco.MjModel) -> mujoco.MjModel:
    import dataclasses

    from myosuite.integrations.yoga import subject_body as sb

    table = dataclasses.replace(_identity_table(base_model), neck=True)
    return sb.compile_spec(sb.build_subject_spec(table))


def _coordination(model: mujoco.MjModel):
    import torch

    from myosuite.integrations.yoga import coordination as co
    from myosuite.integrations.yoga.subject_body import sample_poses

    tb = co.build_tables(model, co.CoordinationCfg(), sample_poses(model, 4, seed=3))
    return co, tb, co.torch_tables(tb, "cpu", torch.float64)


def test_coordination_muscle_forces_match_mujoco(neck_model: mujoco.MjModel) -> None:
    import torch

    from myosuite.integrations.yoga.subject_body import sample_poses

    co, tb, t = _coordination(neck_model)
    rng = np.random.default_rng(4)
    d = mujoco.MjData(neck_model)
    ma = tb.muscle_actuators
    for q in sample_poses(neck_model, 3, seed=5):
        d.qpos[:] = q
        d.qvel[:] = rng.normal(0.0, 1.0, neck_model.nv)
        d.act[:] = rng.uniform(0.0, 1.0, neck_model.na)
        mujoco.mj_forward(neck_model, d)
        gain, bias = co.muscle_gain_bias(
            torch.as_tensor(d.actuator_length[None, ma]), torch.as_tensor(d.actuator_velocity[None, ma]), t
        )  # fmt: skip
        force = (gain * torch.as_tensor(d.act[None, ma]) + bias)[0].numpy()
        np.testing.assert_allclose(force, d.actuator_force[ma], atol=1e-8)


def test_coordination_matrix_matches_projected_muscle_forces(
    neck_model: mujoco.MjModel,
) -> None:
    import torch

    from myosuite.integrations.yoga.kinematics import ReducedKinematics
    from myosuite.integrations.yoga.subject_body import sample_poses

    co, tb, t = _coordination(neck_model)
    rk = ReducedKinematics(neck_model)
    col = {int(dof): i for i, dof in enumerate(rk.indep_dof)}
    rows = [col[int(tb.hinge_dof[h])] for h in tb.muscle_rows]
    rng = np.random.default_rng(6)
    d = mujoco.MjData(neck_model)
    ma = tb.muscle_actuators
    dense = np.zeros((neck_model.nu, neck_model.nv))
    for q in sample_poses(neck_model, 3, seed=7):
        d.qpos[:] = q
        d.qvel[:] = 0.0
        d.act[:] = rng.uniform(0.0, 1.0, neck_model.na)
        mujoco.mj_forward(neck_model, d)
        length = torch.as_tensor(d.actuator_length[None, ma])
        gain, bias = co.muscle_gain_bias(length, torch.zeros_like(length), t)
        a_mat, b = co.muscle_matrix(
            torch.as_tensor(d.ten_J[None]), torch.as_tensor(d.qpos[None]), gain, bias, t
        )
        ours = (a_mat[0] @ torch.as_tensor(d.act[ma]) + b[0]).numpy()
        mujoco.mju_sparse2dense(
            dense, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind
        )
        projected = rk.velocity_map(d.qpos).T @ (dense[ma].T @ d.actuator_force[ma])
        np.testing.assert_allclose(
            ours, projected[rows], atol=1e-9 * max(1.0, np.abs(projected).max())
        )


def test_static_optimisation_meets_feasible_torques(neck_model: mujoco.MjModel) -> None:
    import torch

    co, tb, t = _coordination(neck_model)
    d = mujoco.MjData(neck_model)
    mujoco.mj_forward(neck_model, d)
    n = 16
    ma = tb.muscle_actuators
    length = torch.as_tensor(d.actuator_length[None, ma]).expand(n, -1)
    gain, bias = co.muscle_gain_bias(length, torch.zeros_like(length), t)
    a_mat, b = co.muscle_matrix(
        torch.as_tensor(d.ten_J[None]).expand(n, -1), torch.as_tensor(d.qpos[None]).expand(n, -1), gain, bias, t
    )  # fmt: skip
    rng = np.random.default_rng(8)
    a_true = torch.as_tensor(rng.uniform(0.0, 0.4, (n, len(ma))))
    tau = torch.bmm(a_mat, a_true[..., None])[..., 0] + b
    scale = torch.as_tensor(tb.capacity).clamp(min=1e-6)
    keep = torch.as_tensor(tb.active_rows, dtype=torch.float64)
    a_n = a_mat / scale[None, :, None] * keep[None, :, None]
    r_n = (tau - b) / scale * keep
    a = co.static_optimisation(a_n, r_n, torch.full_like(a_true, 0.03), 1e4, 4)
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0
    resid = (torch.bmm(a_n, a[..., None])[..., 0] - r_n).abs()
    assert float(resid.max()) < 1e-2
    assert float(a.pow(2).sum()) < float(
        a_true.pow(2).sum()
    )  # cheaper than the generating pattern


def test_excitation_reaches_the_target_activation() -> None:
    import torch

    from myosuite.integrations.yoga.coordination import excitation_for

    rng = np.random.default_rng(9)
    act = torch.as_tensor(rng.uniform(0.2, 0.6, 200))
    target = act + torch.as_tensor(rng.uniform(-0.05, 0.05, 200))
    tau_a, tau_d, h, n = 0.01, 0.04, 0.002, 5
    u = excitation_for(
        target, act, torch.full_like(act, tau_a), torch.full_like(act, tau_d), h, n
    )
    a = act.clone()
    tau = torch.where(u > act, tau_a * (0.5 + 1.5 * act), tau_d / (0.5 + 1.5 * act))
    for _ in range(n):  # MuJoCo's Euler step with tau frozen at the starting activation
        a = a + h * (u - a) / tau
    reachable = (u > 0.0) & (u < 1.0)
    assert bool(reachable.any())
    np.testing.assert_allclose(
        a[reachable].numpy(), target[reachable].numpy(), atol=1e-9
    )


def _wrist_moment_arms(
    model: mujoco.MjModel, names: list[str], angle_deg: float
) -> np.ndarray:
    """Flexion moment arms (m, + = flexor) of *names* about the right wrist, elbow 90°, pronated."""
    d = mujoco.MjData(model)
    for j, v in (("elbow_flex_r", 90.0), ("pro_sup_r", 90.0), ("flexion_r", angle_deg)):
        d.qpos[model.joint(j).qposadr[0]] = np.radians(v)
    mujoco.mj_forward(model, d)
    moment = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(
        moment, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind
    )
    dof = model.joint("flexion_r").dofadr[0]
    return np.array([-moment[model.actuator(n).id, dof] for n in names])


def test_finger_flexors_are_kept_as_welded_wrist_muscles(
    base_model: mujoco.MjModel,
) -> None:
    from myosuite.integrations.yoga import subject_body as sb

    model = sb.compile_spec(sb.base_fullbody_spec(finger_flexors=True))
    extra = {model.actuator(i).name for i in range(model.nu)} - {
        base_model.actuator(i).name for i in range(base_model.nu)
    }
    assert extra == {n for k in sb.FINGER_FLEXORS for n in (k, f"{k}_left")}
    assert (model.nq, model.nv, model.nbody) == (
        base_model.nq,
        base_model.nv,
        base_model.nbody,
    )
    np.testing.assert_allclose(model.body_mass, base_model.body_mass)
    assert (_wrist_moment_arms(model, list(sb.FINGER_FLEXORS), 0.0) > 0.005).all()


def test_wrist_obstacles_keep_the_moment_arms_beyond_the_stock_range(
    base_model: mujoco.MjModel,
) -> None:
    import dataclasses

    from myosuite.integrations.yoga import subject_body as sb

    table = dataclasses.replace(
        _identity_table(base_model),
        wrist_obstacles={k: list(v) for k, v in sb.WRIST_OBSTACLES.items()},
    )
    model = sb.compile_spec(sb.build_subject_spec(table))
    flexors, extensors = ["FCR", "FCU", "PL"], ["ECRL", "ECRB", "ECU"]
    for angle in (-95.0, -110.0):
        assert (_wrist_moment_arms(model, flexors, angle) > 0.009).all()
        assert (_wrist_moment_arms(model, extensors, angle) < 0.0).all()
    for angle in (
        -45.0,
        0.0,
        45.0,
    ):  # MyoArm's own range: paths change by millimetres at most
        diff = _wrist_moment_arms(
            model, flexors + extensors, angle
        ) - _wrist_moment_arms(base_model, flexors + extensors, angle)
        assert np.abs(diff).max() < 0.004
        assert np.abs(diff[1:3]).max() < 1e-9  # FCU and PL: the obstacle stays clear


def test_strength_scale_and_stiff_limits(base_model: mujoco.MjModel) -> None:
    import dataclasses

    from myosuite.integrations.yoga import subject_body as sb

    table = dataclasses.replace(
        _identity_table(base_model), strength_scale={"SUPSP": 2.0},
        limit_solref={"flexion_r": list(sb.STIFF_LIMIT_SOLREF)},
        limit_solimp={"flexion_r": list(sb.STIFF_LIMIT_SOLIMP)},
    )  # fmt: skip
    model = sb.compile_spec(sb.build_subject_spec(table))
    a = model.actuator("SUPSP").id
    assert model.actuator_gainprm[a, 2] == pytest.approx(
        2.0 * base_model.actuator_gainprm[a, 2]
    )
    np.testing.assert_array_equal(
        model.actuator_biasprm[a], base_model.actuator_biasprm[a]
    )
    j = model.joint("flexion_r").id
    np.testing.assert_allclose(model.jnt_solref[j], sb.STIFF_LIMIT_SOLREF)
    np.testing.assert_allclose(model.jnt_solimp[j], sb.STIFF_LIMIT_SOLIMP)
    assert not np.allclose(
        model.jnt_solref[model.joint("flexion_l").id], sb.STIFF_LIMIT_SOLREF
    )


def test_statics_limit_reactions_follow_the_joint_set(
    base_model: mujoco.MjModel,
) -> None:
    from myosuite.integrations.yoga import statics
    from myosuite.integrations.yoga.kinematics import ReducedKinematics

    rk = ReducedKinematics(base_model)
    q = base_model.qpos0.copy()
    j = base_model.joint("flexion_r").id
    q[base_model.jnt_qposadr[j]] = base_model.jnt_range[j, 0]

    def count(lim: bool | set[str]) -> int:
        return statics.hold_statics(base_model, rk, q, use_limits=lim)["limits"]

    assert count(True) >= 1
    assert count({"flexion_r"}) == 1
    assert count({"flexion_l"}) == 0
    assert count(False) == 0


def test_dynamic_statics_needs_no_effort_in_free_fall(
    base_model: mujoco.MjModel,
) -> None:
    from myosuite.integrations.yoga import statics
    from myosuite.integrations.yoga.kinematics import ReducedKinematics

    rk = ReducedKinematics(base_model)
    q = base_model.qpos0.copy()
    q[2] += 1.0
    fall = np.zeros(base_model.nv)
    fall[2] = -9.81
    assert statics.hold_statics(base_model, rk, q)["s_star"] is None
    free = statics.hold_statics(base_model, rk, q, qacc=fall, passive_scale=0.0)
    assert free["s_star"] == pytest.approx(0.0, abs=1e-9)


def test_planted_inertia_is_the_body_about_the_planted_hinge() -> None:
    from myosuite.integrations.yoga.feedforward import planted_hinge_inertia

    m_top, h, r = 5.0, 0.5, 0.1
    xml = f"""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <geom type="plane" size="1 1 0.1"/>
      <body name="top" pos="0 0 {0.3 + h}"><freejoint/>
        <geom type="sphere" size="{r}" mass="{m_top}" contype="0" conaffinity="0"/>
        <body name="foot" pos="0 0 {-h}"><joint name="ankle" type="hinge" axis="1 0 0"/>
          <geom type="box" size="0.1 0.1 0.15" pos="0 0 -0.15" mass="1"/>
        </body>
      </body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    model.geom_margin[:] = 0.01
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon >= 3
    full_m = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, full_m, data.qM)
    tmap = np.eye(model.nv)
    free = np.array([full_m[6, 6]])
    out = planted_hinge_inertia(
        model,
        data,
        tmap,
        full_m,
        np.array([6]),
        np.array([model.body("foot").id]),
        free,
        np.arange(6),
    )
    expected = m_top * (
        0.4 * r**2 + h**2
    )  # the sphere rotating about the planted ankle
    assert out[0] == pytest.approx(expected, rel=1e-3)
    assert out[0] > 10 * free[0]


def test_center_of_pressure_balances_the_moments() -> None:
    from myosuite.integrations.yoga.dynamics_viz import center_of_pressure

    pts = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    assert center_of_pressure(
        pts, np.array([[0, 0, 300.0], [0, 0, 100.0]])
    ) == pytest.approx([0.1, 0.0])
    # A force (50, 0, 400) N applied 0.1 m up meets the floor 1.25 cm behind its point.
    cop = center_of_pressure(
        np.array([[0.0, 0.0, 0.1]]), np.array([[50.0, 0.0, 400.0]])
    )
    assert cop == pytest.approx([-0.0125, 0.0])
    assert np.isnan(center_of_pressure(np.zeros((0, 3)), np.zeros((0, 3)))).all()


def test_muscle_layout_pairs_sides_and_regions(base_model: mujoco.MjModel) -> None:
    from myosuite.integrations.yoga.dynamics_viz import REGIONS, muscle_layout

    muscles = np.flatnonzero(base_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
    lay = muscle_layout(base_model, muscles, base_model.qpos0[None])
    assert len(lay.names) == len(muscles) // 2
    assert sorted(np.r_[lay.right, lay.left]) == list(range(len(muscles)))
    region = dict(zip(lay.names, lay.regions))
    expected = {"FCU": "Wrist & hand", "BRA": "Elbow & forearm", "DELT1": "Shoulder",
                "rect_abd": "Trunk", "glmax1": "Hip", "vaslat": "Knee", "soleus": "Ankle & foot"}  # fmt: skip
    assert {k: region[k] for k in expected} == expected
    assert [REGIONS.index(r) for r in lay.regions] == sorted(
        REGIONS.index(r) for r in lay.regions
    )


def test_replay_floor_reaction_carries_the_weight(base_model: mujoco.MjModel) -> None:
    from myosuite.integrations.yoga.dynamics_viz import replay

    d = mujoco.MjData(base_model)
    keys = ("qpos", "qvel", "act", "ctrl")
    state: dict = {k: [] for k in keys}
    for i in range(2000):  # 4 s at zero activation: the body falls and comes to rest
        mujoco.mj_step(base_model, d)
        if i >= 1500 and i % 10 == 0:
            for k in keys:
                state[k].append(getattr(d, k).copy())
    state = {k: np.array(v) for k, v in state.items()} | {"start_s": 0.0, "hz": 100.0}
    dyn = replay(base_model, state)
    fz = dyn["group_force"][..., 2].sum(axis=1) / dyn["weight"]
    assert fz.mean() == pytest.approx(1.0, abs=0.03)
    assert dyn["eq_leak"].max() < 1e-6
    pts = dyn["points"]
    assert np.all(
        (dyn["cop"] >= pts.min(axis=0) - 1e-6) & (dyn["cop"] <= pts.max(axis=0) + 1e-6)
    )
