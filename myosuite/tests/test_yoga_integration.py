# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Data-free checks of the yoga integration (subject scaling, couplings, geometry)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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


def test_action_noise_is_copied_scaled_capped_and_frozen(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    from rsl_rl.modules.distribution import GaussianDistribution

    from myosuite.integrations.yoga.train import set_action_noise

    src = torch.tensor([0.05, 0.1, 0.3])
    torch.save({"actor_state_dict": {"distribution.std_param": src}}, tmp_path / "a.pt")
    dist = GaussianDistribution(3, init_std=0.2, std_range=(1e-6, 0.5))
    set_action_noise(dist, tmp_path / "a.pt", scale=2.0, max_std=0.5, freeze=True)
    assert torch.allclose(dist.std_param, torch.tensor([0.1, 0.2, 0.5]))
    assert not dist.std_param.requires_grad
    # Without options the noise is left as it is, and stays learnable.
    dist = GaussianDistribution(3, init_std=0.2)
    set_action_noise(dist)
    assert torch.allclose(dist.std_param, torch.full((3,), 0.2))
    assert dist.std_param.requires_grad


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


_MUJOCO4 = mujoco.mj_version() >= 4000000


def _wrist_lengths(
    model: mujoco.MjModel, names: list[str], angles_deg: np.ndarray
) -> np.ndarray:
    """Tendon lengths (m) of *names* over right wrist flexion angles, elbow 90°, pronated."""
    d = mujoco.MjData(model)
    for j, v in (("elbow_flex_r", 90.0), ("pro_sup_r", 90.0)):
        d.qpos[model.joint(j).qposadr[0]] = np.radians(v)
    tendons = [int(model.actuator(n).trnid[0]) for n in names]
    out = []
    for angle in angles_deg:
        d.qpos[model.joint("flexion_r").qposadr[0]] = np.radians(angle)
        mujoco.mj_kinematics(model, d)
        mujoco.mj_tendon(model, d)
        out.append(d.ten_length[tendons].copy())
    return np.array(out)


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
    # MuJoCo 3 kept ECRL and ECU extensors out here by jumping them onto the obstacle
    # (31-50 mm of tendon). MuJoCo 4 has no jump: they cross the axis near -102° and -90°.
    if _MUJOCO4:
        assert (_wrist_moment_arms(model, ["ECRL", "ECRB"], -95.0) < 0.0).all()
    else:
        for angle in (-95.0, -110.0):
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


@pytest.mark.skipif(not _MUJOCO4, reason="round 5 places side sites for MuJoCo 4")
def test_round5_paths_keep_their_side_without_jumps(base_model: mujoco.MjModel) -> None:
    import dataclasses

    from myosuite.integrations.yoga import subject_body as sb

    table = sb.round5_table(
        dataclasses.replace(
            _identity_table(base_model),
            finger_flexors=True,
            wrist_obstacles={k: list(v) for k, v in sb.WRIST_OBSTACLES.items()},
        )
    )
    model = sb.compile_spec(sb.build_subject_spec(table))
    fingers = [f"FD{k}{i}" for k in "SP" for i in range(2, 6)]
    for angle in (-95.0, -110.0):
        assert (_wrist_moment_arms(model, fingers, angle) > 0.009).all()
    lengths = _wrist_lengths(
        model, list(sb.WRIST_OBSTACLES), np.arange(-110.0, 88.5, 0.5)
    )
    assert np.abs(np.diff(lengths, axis=0)).max() < 0.001  # per half degree
    assert int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    assert model.opt.impratio == sb.R5_FRICTION[1]
    # teres major and latissimus 2 go round the humerus cylinder (about 140° of arc)
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    cylinder = model.geom("TMAJ_LAThum_cylinder")
    obj, xpos = d.wrap_obj.reshape(-1), d.wrap_xpos.reshape(-1, 3)
    for muscle in ("TMAJ", "LAT2"):
        tid = int(model.actuator(muscle).trnid[0])
        adr, num = int(d.ten_wrapadr[tid]), int(d.ten_wrapnum[tid])
        on = np.flatnonzero(obj[adr : adr + num] == cylinder.id) + adr
        assert len(on) == 2
        assert np.linalg.norm(xpos[on[0]] - xpos[on[1]]) > 1.5 * cylinder.size[0]


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


def test_com_forward_moves_only_the_mass_centre(base_model: mujoco.MjModel) -> None:
    import dataclasses

    from myosuite.integrations.yoga import subject_body as sb

    table = dataclasses.replace(
        _identity_table(base_model), com_forward={"torso": 0.03}
    )
    model = sb.compile_spec(sb.build_subject_spec(table))
    d0, d1 = mujoco.MjData(base_model), mujoco.MjData(model)
    mujoco.mj_forward(base_model, d0)
    mujoco.mj_forward(model, d1)
    b = model.body("torso").id
    np.testing.assert_allclose(
        d1.xipos[b] - d0.xipos[b], 0.03 * sb.MYO_FORWARD, atol=1e-9
    )
    others = np.arange(model.nbody) != b
    np.testing.assert_allclose(d1.xipos[others], d0.xipos[others], atol=1e-12)
    np.testing.assert_allclose(model.body_mass, base_model.body_mass, rtol=1e-9)
    np.testing.assert_allclose(model.body_inertia, base_model.body_inertia, rtol=1e-9)
    share = base_model.body_mass[b] / base_model.body_mass.sum()
    np.testing.assert_allclose(
        d1.subtree_com[1] - d0.subtree_com[1], 0.03 * share * sb.MYO_FORWARD, atol=1e-9
    )


def test_wrist_targets_follow_the_markers_only_with_the_hand_down() -> None:
    from myosuite.integrations.yoga import retarget as rt

    fps, T = 60.0, 600
    names = [lm.name for lm in rt.landmark_set()]
    tg = np.zeros((T, len(names), 3))
    contact = np.zeros((T, len(rt.PLANT_BODIES)), bool)
    contact[200:400, rt.PI["L_Wrist"]] = True
    contact[:, rt.PI["R_Hand"]] = True
    markers = np.tile([0.02, -0.01, 0.05], (T, 2, 1))
    markers[250:260, 0] = np.nan  # a short gap is bridged, without weight
    markers[:, 1] = np.nan  # never seen: the right wrist stays
    out, gate = rt.pin_wrist_targets(tg, names, contact, markers, fps)
    wl, wr, dors = (names.index(n) for n in ("wrist_l", "wrist_r", "farm_dors_l"))
    np.testing.assert_allclose(out[320, wl], [0.02, -0.01, 0.0])  # height kept
    np.testing.assert_allclose(out[320, dors], out[320, wl])
    np.testing.assert_array_equal(out[:150, wl], 0.0)
    np.testing.assert_array_equal(out[:, wr], 0.0)
    assert gate[320, 0] == pytest.approx(1.0) and gate[255, 0] < 1.0
    assert (gate[:, 1] == 0.0).all() and (gate[:150, 0] == 0.0).all()
    moved = np.setdiff1d(np.nonzero(np.abs(out - tg).sum((0, 2)))[0], [wl, dors])
    assert moved.size == 0
    # The target moves by at most the ramp's step per frame.
    step = np.abs(np.diff(out[:, wl, 0])).max()
    assert step <= 0.02 / (rt.WRIST_MARKER_RAMP_S * fps) + 1e-12


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
    mujoco.mju_sym2dense(full_m, data.M, model.M_rownnz, model.M_rowadr, model.M_colind)
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


def _levels_head(**kw):
    import torch
    from torch import nn

    from myosuite.integrations.yoga.levels import LevelsDistribution

    torch.manual_seed(0)
    dist = LevelsDistribution(output_dim=8, levels=5, n_gauss=2, **kw)
    mlp = nn.Sequential(nn.Linear(6, 16), nn.ELU(), nn.Linear(16, dist.input_dim))
    return dist, mlp


@pytest.mark.parametrize("ordinal", [False, True])
def test_levels_head_starts_near_rest_and_samples_on_the_grid(ordinal: bool) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    dist, mlp = _levels_head(p_rest=0.9, ordinal=ordinal)
    dist.init_mlp_weights(mlp)
    out = mlp(torch.randn(4096, 6)).detach()
    dist.update(out)
    p = dist.params[0].exp().reshape(4096, 6, 5)
    assert torch.allclose(p[..., 0], torch.full((4096, 6), 0.9), atol=0.01)
    assert torch.allclose(p[..., 1:], torch.full((4096, 6, 4), 0.025), atol=0.005)
    a = dist.sample()
    assert a.shape == (4096, 8)
    assert torch.equal(a[:, :6] * 4, torch.round(a[:, :6] * 4))
    assert a[:, :6].min() == 0.0 and a[:, :6].max() == 1.0
    # Sampled frequencies follow the probabilities.
    assert (a[:, :6] == 0).float().mean() == pytest.approx(0.9, abs=0.01)
    assert dist.mean[:, :6].mean() == pytest.approx(0.0625, abs=0.003)
    assert dist.stats()["p_rest"] == pytest.approx(0.9, abs=0.01)


@pytest.mark.parametrize("ordinal", [False, True])
def test_levels_head_matches_torch_distributions(ordinal: bool) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    from torch.distributions import Categorical, Normal, kl_divergence

    from myosuite.integrations.yoga.levels import level_logits

    dist, mlp = _levels_head(init_std=0.3, ordinal=ordinal)
    out, out2 = mlp(torch.randn(64, 6)).detach(), mlp(torch.randn(64, 6)).detach()
    logits = level_logits(out[:, :30].reshape(64, 6, 5), ordinal)
    cat, gauss = Categorical(logits=logits), Normal(out[:, 30:], 0.3)
    dist.update(out)
    a = dist.sample()
    idx = torch.round(a[:, :6] * 4).long()
    expected = cat.log_prob(idx).sum(-1) + gauss.log_prob(a[:, 6:]).sum(-1)
    assert torch.allclose(dist.log_prob(a), expected, atol=1e-5)
    entropy = cat.entropy().sum(-1) + gauss.entropy().sum(-1)
    assert torch.allclose(dist.entropy, entropy, atol=1e-5)
    old = tuple(p.detach() for p in dist.params)
    assert dist.kl_divergence(old, old).abs().max() < 1e-6
    dist.update(out2)
    cat2 = Categorical(logits=level_logits(out2[:, :30].reshape(64, 6, 5), ordinal))
    kl = kl_divergence(cat, cat2).sum(-1) + kl_divergence(
        gauss, Normal(out2[:, 30:], 0.3)
    ).sum(-1)
    assert torch.allclose(dist.kl_divergence(old, dist.params), kl, atol=1e-5)
    if ordinal:  # output k is the log-odds of level k against level k - 1
        logp = dist.params[0].reshape(64, 6, 5)
        steps = out2[:, :30].reshape(64, 6, 5)[..., 1:]
        assert torch.allclose(logp[..., 1:] - logp[..., :-1], steps, atol=1e-5)


def test_levels_head_deterministic_outputs() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    dist, mlp = _levels_head()
    out = mlp(torch.randn(32, 6)).detach()
    p = out[:, :30].reshape(32, 6, 5).softmax(-1)
    values = torch.linspace(0.0, 1.0, 5)
    mean = dist.deterministic_output(out)
    assert torch.allclose(mean[:, :6], (p * values).sum(-1), atol=1e-6)
    assert torch.equal(mean[:, 6:], out[:, 30:])
    assert torch.allclose(dist.as_deterministic_output_module()(out), mean)
    dist.det = "mode"
    mode = dist.deterministic_output(out)
    assert torch.equal(mode[:, :6], values[p.argmax(-1)])


def _random_bodies(torch, n: int, b: int, s: int):
    from mjlab.utils.lab_api.math import matrix_from_quat

    gen = torch.Generator().manual_seed(3)

    def rand(*shape):
        return torch.randn(*shape, generator=gen)

    rot = matrix_from_quat(torch.nn.functional.normalize(rand(n, b, 4), dim=-1))
    ref_rot = matrix_from_quat(torch.nn.functional.normalize(rand(n, s, b, 4), dim=-1))
    ref = torch.cat(
        [rand(n, s, b, 3), ref_rot[..., 0], ref_rot[..., 2], rand(n, s, b, 6)], dim=-1
    )
    return rand(n, b, 3), rot, rand(n, b, 3), rand(n, b, 3), ref


def test_body_observation_ignores_place_and_heading() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    from myosuite.integrations.yoga.body_obs import (
        body_state_obs,
        preview_obs,
        target_pose_obs,
    )

    n, b, s = 5, 7, 3
    pos, rot, lin, ang, ref = _random_bodies(torch, n, b, s)
    state = body_state_obs(pos, rot, lin, ang)
    target = target_pose_obs(pos, rot, lin, ang, ref)
    preview = preview_obs(pos, rot, ref[..., :3])
    assert state.shape == (n, 1 + 3 * (b - 1) + 12 * b)
    assert target.shape == (n, s * 24 * b)
    assert preview.shape == (n, s * 3 * b)
    # The same motion somewhere else on the floor, facing another way.
    yaw = torch.tensor([0.3, -2.0, 3.1, 1.2, -0.7])
    c, s_ = yaw.cos(), yaw.sin()
    zero, one = torch.zeros(n), torch.ones(n)
    turn = torch.stack(
        [torch.stack([c, -s_, zero], -1), torch.stack([s_, c, zero], -1), torch.stack([zero, zero, one], -1)], -2
    )  # fmt: skip
    shift = torch.tensor([[1.5, -4.0, 0.0]])

    def moved(v, is_pos=False):
        t = turn.reshape(n, *([1] * (v.dim() - 2)), 3, 3)
        out = (t @ v.unsqueeze(-1)).squeeze(-1)
        return out + shift.reshape(1, *([1] * (v.dim() - 2)), 3) if is_pos else out

    rot2 = turn[:, None] @ rot
    ref2 = torch.cat(
        [moved(ref[..., 0:3], True), moved(ref[..., 3:6]), moved(ref[..., 6:9]),
         moved(ref[..., 9:12]), moved(ref[..., 12:15])], dim=-1
    )  # fmt: skip
    args = (moved(pos, True), rot2, moved(lin), moved(ang))
    assert torch.allclose(body_state_obs(*args), state, atol=1e-5)
    assert torch.allclose(target_pose_obs(*args, ref2), target, atol=1e-5)
    assert torch.allclose(preview_obs(args[0], rot2, ref2[..., :3]), preview, atol=1e-5)
    # A reference equal to the present state reads as zero error.
    here = torch.cat([pos, rot[..., 0], rot[..., 2], lin, ang], dim=-1)[:, None]
    same = target_pose_obs(pos, rot, lin, ang, here)
    assert same[:, 3 * b : 6 * b].abs().max() < 1e-6  # position error
    assert same[:, 18 * b :].abs().max() < 1e-6  # velocity differences
    local = same[:, 12 * b : 18 * b].reshape(n, b, 6)  # target axes in the body frames
    unit = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(local, unit.expand(n, b, 6), atol=1e-5)


def test_body_observation_heading_is_continuous_through_an_inversion() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    from myosuite.integrations.yoga.body_obs import heading_cos_sin

    # The root pitches about its lateral axis from upright to upside down (a fold, then a
    # handstand). Its frame: x forward, y up, z right; the subject faces world +x.
    pitch = torch.linspace(0.0, torch.pi, 181)
    c, s = pitch.cos(), pitch.sin()
    zero = torch.zeros_like(pitch)
    forward = torch.stack([c, zero, -s], -1)
    lateral = torch.stack([zero, -torch.ones_like(pitch), zero], -1)
    cos, sin = heading_cos_sin(lateral)
    assert torch.allclose(cos, torch.ones_like(cos)) and sin.abs().max() < 1e-6
    # The forward axis has no horizontal part at 90 degrees and points backwards after it.
    assert forward[90, :2].norm() < 1e-6 and forward[180, 0] < -0.99


def test_reference_bodies_match_mujoco(base_model: mujoco.MjModel) -> None:
    from myosuite.integrations.yoga import subject_body as sb
    from myosuite.integrations.yoga.body_obs import OBS_BODIES, body_kinematics

    rng = np.random.default_rng(4)
    qpos = sb.sample_poses(base_model, 3, seed=5)
    qvel = rng.normal(0.0, 1.0, (3, base_model.nv))
    ids = np.array([base_model.body(n).id for n in OBS_BODIES])
    rows = body_kinematics(base_model, qpos, qvel, ids)
    assert rows.shape == (3, len(OBS_BODIES), 15)
    data = mujoco.MjData(base_model)
    vel = np.zeros(6)
    for f in range(3):
        data.qpos[:], data.qvel[:] = qpos[f], qvel[f]
        mujoco.mj_forward(base_model, data)
        for j, b in enumerate(ids):
            mat = data.xmat[b].reshape(3, 3)
            mujoco.mj_objectVelocity(base_model, data, mujoco.mjtObj.mjOBJ_XBODY, b, vel, 0)  # fmt: skip
            expected = np.r_[data.xpos[b], mat[:, 0], mat[:, 2], vel[3:], vel[:3]]
            np.testing.assert_allclose(rows[f, j], expected, atol=2e-5)
    # At rest when no velocities are given.
    assert np.abs(body_kinematics(base_model, qpos, None, ids)[..., 9:]).max() == 0.0


def test_muscle_scales_normalise_to_the_muscle_model(
    base_model: mujoco.MjModel,
) -> None:
    from myosuite.integrations.yoga.body_obs import muscle_scales

    sc = muscle_scales(base_model)
    n = int((base_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE).sum())
    assert len(sc["ids"]) == n and n > 300
    lr = base_model.actuator_lengthrange[sc["ids"]]
    gp = base_model.actuator_gainprm[sc["ids"]]
    # The ends of the tendon length range map onto the ends of the operating range.
    np.testing.assert_allclose(
        sc["range0"] + (lr[:, 1] - sc["lo"]) / sc["l0"], gp[:, 1]
    )
    assert sc["f0"].min() > 0 and sc["vmax"].min() > 0


def test_levels_head_without_gaussian_outputs() -> None:
    """With the neck on a servo every output is a muscle: no Gaussian part."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    from torch import nn
    from torch.distributions import Categorical

    from myosuite.integrations.yoga.levels import LevelsDistribution

    torch.manual_seed(0)
    dist = LevelsDistribution(output_dim=6, levels=5, n_gauss=0)
    assert dist.input_dim == 30 and dist.std_param.numel() == 0
    out = nn.Linear(4, 30)(torch.randn(16, 4)).detach()
    dist.update(out)
    a = dist.sample()
    assert a.shape == (16, 6)
    cat = Categorical(logits=out.reshape(16, 6, 5))
    idx = torch.round(a * 4).long()
    assert torch.allclose(dist.log_prob(a), cat.log_prob(idx).sum(-1), atol=1e-5)
    assert torch.allclose(dist.entropy, cat.entropy().sum(-1), atol=1e-5)
    old = tuple(p.detach() for p in dist.params)
    assert dist.kl_divergence(old, old).abs().max() < 1e-6
    assert dist.deterministic_output(out).shape == (16, 6)
    assert "neck_std" not in dist.stats()


def test_neck_servo_gains_are_damped_on_each_hinge(
    neck_model: mujoco.MjModel,
) -> None:
    from myosuite.integrations.yoga import subject_body as sb
    from myosuite.integrations.yoga.neck import NeckServoActionCfg, neck_servo_gains

    kp, kd = neck_servo_gains(neck_model, 100.0, 1.0)
    inertia = kd**2 / (4.0 * kp)
    assert np.all(kp == 100.0)
    # The head turns about its long axis far more easily than it nods or tilts.
    assert inertia[2] < 0.3 * inertia[0] and inertia[2] < 0.3 * inertia[1]
    assert 0.05 < inertia[0] < 0.4
    # A held PD is stable only below kd dt / I = 2.
    assert np.all(kd * neck_model.opt.timestep / inertia < 0.5)
    cfg = NeckServoActionCfg(entity_name="e", mj_model=neck_model, clips=(), ctrl_dt=0.01)  # fmt: skip
    assert [neck_model.joint(j).qposadr[0] for j in sb.NECK_JOINTS] == cfg.qadr
    assert cfg.gear == [30.0, 30.0, 30.0]


def _start_sampler(**kw):
    from myosuite.envs.myo.backends.mjlab.start_sampling import (
        AdaptiveStartsCfg,
        FailureWeightedStarts,
    )

    cfg = AdaptiveStartsCfg(**{"min_hazard": 1e-9, "prior_visits": 0.0, "lead": 0, **kw})  # fmt: skip
    return FailureWeightedStarts([250, 400], 100, 0.01, 8, cfg, "cpu")


def _episodes(torch, rows):
    """``(env, clip, start, end, failed)`` rows as the sampler's arguments."""
    env, clip, start, end, failed = (torch.tensor(c) for c in zip(*rows))
    return env, clip, start, end, failed.bool()


def test_failure_weighted_starts_are_uniform_without_failures() -> None:
    torch = pytest.importorskip("torch")
    s = _start_sampler()
    p = s.start_probabilities()
    # A clip uniformly, then one of its starts uniformly: 3 and 4 starts.
    assert torch.allclose(p, torch.tensor([1 / 6] * 3 + [1 / 8] * 4))
    clips, frames = s.sample(4000)
    assert set(frames[clips == 0].tolist()) == {0, 100, 200}
    assert set(frames[clips == 1].tolist()) == {0, 100, 200, 300}
    assert (clips == 0).float().mean() == pytest.approx(0.5, abs=0.04)
    # Episodes that reach their time limit are visits, not failures.
    s.observe(1, *_episodes(torch, [(0, 1, 0, 400, False), (1, 0, 100, 250, False)]))
    assert torch.equal(s.visit > 0, torch.tensor([0, 1, 1, 1, 1, 1, 1]).bool())
    assert float(s.fail.sum()) == 0.0 and s.stats()["weighted_share"] == 0.0
    assert s.stats()["clip_pass_estimate"] == pytest.approx(1.0)


def test_failure_weighted_starts_follow_the_failure_rate_not_the_count() -> None:
    torch = pytest.importorskip("torch")
    s = _start_sampler(uniform=0.3, kernel=3, decay=0.8)
    # Clip 1: four episodes from frame 0; one ends early in its second bin, the others
    # go on and all three end early in its last bin.
    rows = [(0, 1, 0, 150, True)] + [(e, 1, 0, 350, True) for e in (1, 2, 3)]
    # Clip 0: one episode passes all of it.
    rows.append((4, 0, 0, 250, False))
    s.observe(10, *_episodes(torch, rows))
    hazard = s.hazard()
    assert torch.allclose(hazard, torch.tensor([0, 0, 0, 0, 0.25, 0, 1.0]))
    assert s.clip_pass_estimate().tolist() == pytest.approx([1.0, 0.0], abs=1e-5)
    w = torch.tensor([1.0, 0.8, 0.64]) / 2.44
    spread = torch.zeros(7)
    # The clip's first start also takes the weight that would fall before the clip.
    spread[3] += 0.25 * (w[1] + w[2])
    spread[4] += 0.25 * w[0] + 1.0 * w[2]
    spread[5] += 1.0 * w[1]
    spread[6] += 1.0 * w[0]
    uniform = torch.tensor([1 / 6] * 3 + [1 / 8] * 4)
    share = s.weighted_share()
    assert share == pytest.approx(0.7, abs=1e-6)
    expected = share * spread / spread.sum() + (1 - share) * uniform
    assert torch.allclose(s.start_probabilities(), expected, atol=1e-6)
    assert s.stats()["top_start_s"] == 3.0
    # Nothing of clip 1's failures reaches clip 0.
    assert torch.allclose(s.start_probabilities()[:3], 0.3 * uniform[:3], atol=1e-6)
    clips, frames = _start_sampler(uniform=0.0).sample(10)
    assert bool((frames % 100 == 0).all())
    # With a lead the starts are before the failing second: here all on the first frame.
    led = _start_sampler(uniform=0.0, kernel=2, lead=1)
    led.observe(10, *_episodes(torch, [(0, 1, 0, 150, True)]))
    assert torch.allclose(led.start_probabilities(), torch.tensor([0, 0, 0, 1.0, 0, 0, 0]))  # fmt: skip
    led.observe(10, *_episodes(torch, [(1, 1, 0, 350, True), (2, 0, 0, 250, False)]))
    p = led.start_probabilities()
    assert float(p[:3].sum()) < 1e-6 and float(p[6]) < 1e-6 and float(p[5]) > 0.1


def test_failure_weighted_starts_ignore_early_falls_and_forget() -> None:
    torch = pytest.importorskip("torch")
    s = _start_sampler(min_fail_s=0.75, alpha=0.01)
    # 0.5 s after its start: the start state is to blame, not the policy.
    s.observe(5, *_episodes(torch, [(0, 0, 100, 150, True)]))
    assert float(s.fail.sum()) == 0.0 and float(s.visit.sum()) == 0.0
    assert s.start_quality().tolist() == pytest.approx([1, 0, 1, 1, 1, 1, 1])
    # A start that cannot be survived gets none of the weighted share.
    dead = _start_sampler(uniform=0.0, kernel=2, lead=0)
    rows = [(0, 1, 100, 150, True), (1, 1, 0, 180, True), (2, 1, 0, 180, True)]
    dead.observe(5, *_episodes(torch, rows))
    assert dead.hazard().tolist() == pytest.approx([0, 0, 0, 0, 1, 0, 0])
    assert torch.allclose(dead.start_probabilities(), torch.tensor([0, 0, 0, 1.0, 0, 0, 0]))  # fmt: skip
    # From the clip's first frame every fall counts.
    first = _start_sampler(min_fail_s=0.75, alpha=0.01)
    first.observe(5, *_episodes(torch, [(0, 0, 0, 50, True)]))
    assert first.fail.tolist() == pytest.approx([0.01, 0, 0, 0, 0, 0, 0])
    s.observe(6, *_episodes(torch, [(0, 0, 100, 180, True)]))
    assert s.fail.tolist() == pytest.approx([0, 0.01, 0, 0, 0, 0, 0])
    s.observe(106, *_episodes(torch, [(0, 0, 0, 0, False)]))
    assert float(s.fail.sum()) == pytest.approx(0.01 * 0.99**100)
    # An episode flagged before it ended (shifted clocks) is left out, once.
    s2 = _start_sampler()
    s2.skip_running_episodes()
    s2.observe(1, *_episodes(torch, [(3, 0, 0, 180, True)]))
    assert float(s2.visit.sum()) == 0.0
    s2.observe(2, *_episodes(torch, [(3, 0, 0, 180, True)]))
    assert float(s2.fail.sum()) > 0.0
    # With few failures the starts return to uniform; few visits damp a bin's rate.
    rare = _start_sampler(min_hazard=0.05, uniform=0.3)
    rows = [(0, 0, 0, 180, True)] + [(e, 0, 0, 250, False) for e in range(1, 8)] * 20
    rare.observe(1, *_episodes(torch, rows))
    assert rare.hazard().max() == pytest.approx(1 / 141)
    assert rare.weighted_share() < 0.05
    damped = _start_sampler(prior_visits=20.0)
    damped.observe(1, *_episodes(torch, [(0, 0, 0, 180, True)]))
    assert float(damped.hazard().max()) == pytest.approx(1 / 21)
