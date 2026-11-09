<!--
  Copyright (c) MyoSuite Authors. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Muscle-driven yoga teacher on MyoFullBody

Can the yoga teacher trained in ProtoMotions (IsaacLab/PhysX, a 24-body PD-actuated plant
built on the performer's skeleton) be reproduced with a **muscle-actuated** humanoid? This package
is the investigation and the first working pipeline: MyoFullBody scaled to the performer, her
curated yoga references retargeted onto it, a static muscle-feasibility analysis, and a GPU
(mjlab / MuJoCo-Warp) PPO task that tracks the 56-clip corpus with 354 muscles.

The motion data (MOYO capture, AMASS-licensed SMPL-X fits) is not redistributable and is never
written inside this package; generated data lives in a directory you choose (below:
`$Y = /home/visakii/Documents/MovementGym/yoga_muscle`).

**Status (2026-11-04): closed.** All runs are stopped. The consolidated record, including why
this approach stalls and the directions that would change it, is
`<ProtoMotions>/expert_revist/muscle_teacher_myosuite_2026_11/README.MD`.

## 1. Short answer

* **The simulation stack already exists in MyoSuite.** MuscleMimic's `myoMimicFullbody-v0` runs
  MyoFullBody (354 muscles, fingers removed) on mjlab with clip banks, a DeepMimic reward,
  reference-state initialisation and rsl_rl PPO. On one RTX A5000 the yoga task runs at
  9.5–12k env-steps/s with 2048 envs (~4–6 GB). Nothing had to be rewritten; the work is in the
  body, the references and the task.
* **"Fix the skeleton or retarget the motion?" Both, and the skeleton is the smaller part.**
  Her segment lengths differ from MyoFullBody's by only 0–5 % (hip and shoulder width ~9 %), so
  scaling fixes them exactly. The larger mismatches are the **DOF structure** (37 independent
  DOFs vs the plant's 69: no neck, rigid thorax, coupled shoulder girdle and knee), the **joint
  ranges** (normative clinical ranges; a yogi needs ~90° wrist extension, 155° hip flexion, 160°
  knee flexion), the **contact geometry** (MyoFullBody's welded, curled fingers cannot put a
  palm on the floor; its self-collision misses braces such as crow's shins on the upper arms) and
  the **passive tissue** (her flexibility). Angles cannot be copied across those differences, so a
  contact-aware IK retarget is required anyway; scaling first makes its residuals small (median
  joint-centre error 1.0 cm over 81k frames).
* **Can the muscles hold the poses?** A static LP at the 270 annotated hold exemplars (minimum
  peak activation for static equilibrium with floor and brace contact forces). Each fix to the
  body moved it: stock passive muscles, floor contacts only **180 (67 %)** holdable → with her
  flexibility (§2.6) 204 → with brace forces on MyoFullBody's own pairs 212 → with the plant's
  self-collision (§2.4) **224 (83 %)**, 5 without equilibrium, median peak activation 0.47. The
  remaining 41 exceed maximal activation, mostly in the lumbar/thoracolumbar trunk musculature,
  which must do the work of a spine whose thorax is rigid, and in arm balances' shoulders.
* **Training**: `v4` (subject body + her flexibility + the plant's contact model, all 56 clips,
  2048 envs) and a two-clip side run (Tree, Downward Dog) restarted 2026-11-02 23:45 after a
  learning-rate bug (§5). At iteration 1200 both were branched into `v5_std03` /
  `side_v5_std03` with the action noise capped. The uncapped policy had learnt to rely on its own
  exploration noise, which through the muscle sigmoid acts as free co-contraction (§5), so its
  deterministic controller survived 33–40 % less than the noisy one on the 56 clips (68 % less
  on Tree). Capping closed most of that gap within 200 iterations. Earlier runs are kept as short records (§5). A walker on this stack
  needs ~2B steps (MuscleMimic baseline); an overnight run is ~0.3B, so expect an early-stage
  tracker, not a replicated teacher. Even so, with the reference frozen at a pose, the 56-clip
  policy's deterministic controller **holds five poses for 10 s** at iteration 1700
  (~84M env steps, the last 500 iterations with the cap): downward dog, dolphin, chaturanga,
  plow and a supported headstand. Survival ranks
  with the statics: multi-support poses and supported inversions first, arm balances and
  single-leg balances last (§5).
* **One clip, the handstand (`handstand_v1`, 8192 envs from scratch, overnight 11-03/04, stopped
  at 416M steps, §5)**: it tracks the fold and the one-leg-up downward dogs between kick-ups for
  3–5 s,
  twice what the 56-clip run manages on that clip, but it holds no handstand phase (mean action
  < 1 s from every start inside one, 0 of 16 frozen holds): the inverted balance is not learnt
  yet.
* **Baseline (both runs stopped 2026-11-03 21:01, §5 "Final state")**: the 56-clip run
  plateaued. From iteration ~3400 (~170M env steps) it holds 6–9 poses for 10 s in all
  4 replicas, and no more by its end at 6927 (~340M). Its training reward still crept up. No
  clip is completed from its first frame. The two-clip specialist learnt Tree late: it completes
  the 20 s Tree clip at iterations 8200, 8700 and 9500 (~200M steps on), where the 56-clip
  run's single-leg holds last 0.75 s.

## 2. How MyoFullBody differs from her body

Her body is plant v2's skeleton: the SMPL-X joint centres of her personal template
(`stagei_debug_details.v_template` of any of her MoSh++ fits, regressed with
`SMPLX_FEMALE.npz`), 1.748 m, 74.0 kg with per-segment masses from her template volume.

### 2.1 Segment lengths and mass

| segment | her (m) | MyoFullBody (m) | scale applied |
|---|---|---|---|
| hip width (joint centres) | 0.1415 | 0.1545 | pelvis/sacrum lateral × 0.916 |
| shoulder width (GH centres) | 0.3105 | 0.3400 | clavicles × 0.891 |
| thigh L / R | 0.4039 / 0.3999 | 0.4044 | 0.999 / 0.989 |
| shank L / R | 0.4150 / 0.4038 | 0.4001 | 1.037 / 1.009 |
| upper arm L / R | 0.3045 / 0.2926 | 0.2907 | 1.047 / 1.006 |
| elbow → middle fingertip L / R | 0.448 / 0.450 | 0.440 | forearm and hand 1.019 / 1.021 |
| foot heel → big toe L / R | 0.269 / 0.265 | 0.279 | 0.956 |
| crown above shoulders | 0.314 | 0.302 | head +1.2 cm |
| mass | 74.0 kg | 84.3 kg | per segment group, to hers |

Two definitional traps, both measured: SMPL-X's wrist joint sits ~12 cm proximal of MCP3 (a
typical anatomical wrist is 8–9 cm), so "wrist → hand" would have scaled the hand by 1.39; and
SMPL-X's "toe" joint is not the MTP joint. Those segments are scaled from **surface** distances
on her template instead (elbow → middle fingertip, heel → big toe, crown height); every
joint-centre segment matches hers to 0.0 mm after scaling. Her mass distribution differs from
MyoFullBody's (her head+neck 5.1 kg vs 2.4 kg; trunk 35.5 kg vs 47.5 kg), and the masses are
moved to hers group by group with inertias following `m r²`.

### 2.2 DOF structure

MyoFullBody drives 37 DOFs independently; 51 joint-equality polynomials slave the vertebrae
(to 3 lumbar DOFs), the clavicle/scapula (to shoulder elevation), shoulder1_r2 (to elevation
angle) and the knee translations/rotations and patella (to knee flexion).

* **No neck DOF**: the head is welded to the thorax. Shoulderstand and plow need ~90° of neck
  flexion; her corpus needs neck flexion −69°…+62° (the head would otherwise go through the
  floor). Added: a 3-hinge neck at the cervical base with three **torque motors** (30 N·m,
  passive spring 2 N·m/rad). These are the only non-muscle actuators.
* **Rigid thorax**: all spinal motion is lumbar. Spine-on-floor poses (bridge, plow,
  shoulderstand, legs-up-the-wall) are where the retarget is worst (p95 3.4–4.6 cm, supports up
  to 4 cm off the floor on 11–22 % of frames).
* **Coupled shoulder girdle**: no independent shrug/protraction; shoulder joint centres differ
  from hers by up to ~3 cm in her calibration T-pose.

### 2.3 Joint ranges (the range audit)

All 56 clips were retargeted with a deliberately wide box; the table compares MyoFullBody's
range, the corpus p0.5/p99.5 that her motion needs, the range adopted (`YOGA_RANGES_DEG`), and
how many clips leave MyoFullBody's range (degrees; + = flexion/dorsiflexion/extension of the
lumbar per the sign conventions in `subject_body.py`):

| joint | MyoFullBody | her p0.5 / p99.5 | adopted | clips beyond |
|---|---|---|---|---|
| wrist `flexion_l` | −45 / 45 | −100 / 85 | −95 / 88 | 42 |
| `hip_flexion_l` | −30 / 120 | −55 / 160 | −55 / 155 | 21 |
| `knee_angle_l` | 0 / 120 | 0 / 160 | 0 / 160 | 27 |
| `ankle_angle_l` | −40 / 30 | −70 / 48 | −70 / 55 | 44 |
| `mtp_angle_l` (− = toe extension) | −30 / 30 | −80 / −4 | −80 / 45 | 55 |
| `hip_adduction_l` | −50 / 30 | −74 / 25 | −75 / 32 | 4 |
| `hip_rotation_r` | −40 / 40 | −69 / 78 | −70 / 75 | 12 |
| `elbow_flex_r` | 0 / 130 | 0 / 139 | 0 / 145 | 2 |
| lumbar `flex_extension` | −77 / 43 | −65 / 53 | −77 / 58 | 3 |
| lumbar `axial_rotation` | −45 / 45 | −66 / 56 | −70 / 62 | 5 |

With MyoFullBody's own ranges the crow clip's lead-in alone (fold forward, hands flat) had
joint-centre p95 8.9 cm and elbows/wrists 10–12 cm off; with the adopted ranges 3.4 cm.
Twist and foot-roll DOFs (pronation, shoulder rotation, subtalar, MTP flexion) hit the audit's
own box: that was slack, not need — SMPL-X keeps some humeral/femoral axial rotation in the
elbow/knee frames, so twist landmarks on the plant's upper-arm/thigh frames are down-weighted
and a weak neutral prior holds those DOFs. Knee couplings were fitted on 0–120°; past that the
patella polynomial diverges, so every knee coupling is continued C1-linearly to 160°.
The stock model's vertebral limits also contradict its own couplings (they cap lumbar flexion
near −58° although the joint allows −77°); dependent ranges are widened to cover their
couplings' values.

### 2.4 Contact geometry: why the plant's colliders were transplanted

The static analysis first found 48 holds without equilibrium, and the cause was geometric: in
crow MyoFullBody touches the floor **only with the thumb and little finger**. With fingers
removed, its fingers are welded in a relaxed curl and the hand has no palm collider, so the heel
of the hand sits 1–2 cm up while her COM is over it; forearm stands fail the same way (forearm
capsules 1.5–3 cm radius vs the plant's 4 cm). Following the plant-v2 decision (keep the shipped
primitives re-spanned to her segments; do not fit colliders to the skin), the plant's 24
primitives are **transplanted onto MyoFullBody's bones** through the registration T-pose and
MyoFullBody's own colliders become visual-only. In the T-pose the transplanted primitives
coincide with the plant's to 3e-16 m, so the muscle model touches the floor where the PhysX
teacher did — the geometry its references were contact-repaired for. Retarget quality improved
with it (spine-on-floor clips 8–9.5 → 3.4–4.6 cm p95).

**Self-collision and braces.** Stock MyoFullBody has 75 explicit contact pairs on its own geometry
(arms vs thorax, forearms vs thighs/shanks, legs vs legs, feet vs feet) and no others; the PhysX
plant let every non-adjacent body pair collide. Braces carry load: in crow the binding joint is
the shoulder (+80 N·m of elevation torque needed, +59 N·m available at that angle) unless the
shins rest on the upper arms, and shoulder-pressing needs 62× maximal activation with its legs
held in the air by the hip flexors. With `self_collision` the transplanted primitives collide with
each other under the plant's rule (every pair except anatomically adjacent segments,
`ADJACENT_HOSTS`, mapped onto MyoFullBody bones because its tree has phantom bodies between
segments) and the explicit pairs are removed. The references' braces then overlap by median
0.5 cm (p95 3.0 cm, worst 9.8 cm in a supported headstand's hands around the head), against
1.1 / 4.7 cm on MyoFullBody's pairs. mujoco-warp's multi-CCD box–box path requires zero geom
margin, so the primitives use margin 0. Throughput cost: ~2 % (measured side by side).

### 2.5 Muscles under scaling

The muscles are MyoConverter-calibrated: an explicit `lengthrange` with a per-muscle
normalised `range`, from which MuJoCo derives optimal fibre length and tendon slack length.
MuJoCo does not recompute either after a geometry change, so each muscle's `lengthrange` is
multiplied by its measured musculotendon-length ratio (scaled vs stock, least squares over 400
random poses; 0.93–1.05), which is OpenSim's Scale-tool rule. Normalised fibre lengths change by
a median 0.002 (p99 0.05). Peak isometric forces are unchanged (strength-to-weight +14 %).

### 2.6 Passive tissue: her flexibility

Joint ranges are one half of flexibility; the other is how hard stretched muscles pull back.
At her holds, the muscles of MyoFullBody reach a median 1.40× (p90 1.73×) optimal fibre length,
where MuJoCo's passive curve gives 0.45–1.34 × peak isometric force (§3.2). `fit_passive_compliance`
(in `statics.py`) scales `fpmax` (passive force only; the active force-length curve is
unchanged) of every muscle whose corpus p99 stretch exceeds 1.25 so that her p99 stretch costs
what 1.25 costs the stock model. 180 of 354 muscles are softened, median factor 0.39; the
strongest are the obliques (0.04–0.09, twists and side bends), hamstrings (~0.11), psoas
fascicles and piriformis. Stored as `SubjectTable.passive_scale`
(`$Y/model/subject_table_compliant_selfcol.json` is the final body, used by `v4`).

## 3. What was built

| module | role |
|---|---|
| `reference_io.py` | reads ProtoMotions `.motion` files with a stub unpickler (no ProtoMotions needed; quaternions verified xyzw by FK to 2e-7 m); plant collider geometry and floor states |
| `subject_body.py` | skeleton audit, `SubjectTable` (JSON), `build_subject_spec` (scaling, masses, muscle lengthranges, ranges and coupling continuation, neck, collider transplant) |
| `kinematics.py` | independent coordinates of a model with joint couplings (exact couplings, velocity map) |
| `retarget.py` | registered marker IK (T-pose registration, projected LM, floor terms), 60 Hz solve → 100 Hz MuscleMimic clips, metrics; the collider transplant |
| `statics.py` | static muscle-feasibility LP per hold |
| `task.py` | mjlab task `myoMimicYoga-v0`: subject body, 56-clip bank, neck action term, effort penalty, 0.5 m per-site termination |
| `train.py` / `evaluate.py` / `render.py` | PPO training (resume, std cap); per-clip evaluation from frame 0, from the hold (or a chosen time), against a frozen hold reference, or from every few seconds of a clip (start-time sweep), with the mean, sampled or noise-matched controller, replicas and the termination rule that ended each rollout; review videos with the reference as a ghost |

Shared-code changes (backward compatible, defaults unchanged):
`terms/mimic_obs.mimic_termination_mask(max_site_err_threshold=None)` (a per-site deviation
rule, the PhysX teacher's "any body more than 0.5 m from its reference"), plumbed through
`_mimic_early_termination` / `_make_mimic_env_cfg`; and `set_mimic_entity_model` in
`mimic_mjlab_env.py`, because the mimic closures rebuilt the *stock* body to look up sites and
expand clips (an edited body with extra joints failed with a qpos-width error).

### 3.1 Retarget quality (final model, all 56 clips, 81.5k frames at 60 Hz)

| group | clips | joint-centre median | joint-centre p95 | support float p95 | frames with a support > 2 cm up | max penetration |
|---|---|---|---|---|---|---|
| arm balance | 9 | 0.9 cm | 3.9 cm | 0.3 cm | 0.0 % | 1.0 cm |
| connective | 21 | 1.1 cm | 3.7 cm | 0.2 cm | 0.8 % | 1.6 cm |
| inversion | 13 | 0.9 cm | 4.3 cm | 0.7 cm | 5.8 % | 2.4 cm |
| single leg | 13 | 0.9 cm | 3.3 cm | 0.2 cm | 0.0 % | 0.6 cm |

(Medians over clips of the per-clip statistic; "support" = a plant body within 1.5 cm of the floor
in the reference.) Worst p95: Scorpion -a 5.8, Peacock 5.7, Feathered Peacock 5.1–5.5 cm.
77 of 81.5k frames needed a cold restart of the warm-started solver.

**Brace-aware option (`--brace-allowance 0.01`, for the next run).** With self-collision on, the
retargeted braces overlap (the IK ignores body–body contact). An extra residual pushes two
primitives apart once they overlap by more than 1 cm ("soft tissue"). Measured on all 56 clips
against the same solve without it: per-clip worst overlap median 2.1 → 1.1 cm (p90 4.3 → 2.7 cm,
max 7.0 → 4.6 cm), per-clip p95 overlap p90 3.3 → 1.8 cm, at a cost of +0.06 cm median
joint-centre error; floor contact unchanged. Output in `$Y/clips_selfcol_brace/`; `v4` trains on
the clips without it.

### 3.2 Muscle feasibility of the holds (2 cm contact rule, as the plant's statics LP)

Holdable = minimum peak activation s* ≤ 1. Columns add one body fix at a time.

| group | holds | stock passive, floor only | + her flexibility (§2.6) | + braces on MyoFullBody's pairs | + plant self-collision (final body) | final: no equilibrium |
|---|---|---|---|---|---|---|
| arm balance | 60 | 32 | 34 | 40 | 42 | 0 |
| connective | 87 | 69 | 74 | 76 | 78 | 1 |
| inversion | 69 | 45 | 57 | 57 | 58 | 2 |
| single leg | 54 | 34 | 39 | 39 | 46 | 2 |
| all | 270 | 180 | 204 | 212 | **224** | 5 |

Without any passive muscle force the final body would hold 232. The remaining 41 with s* > 1 are
trunk-limited (thoracolumbar extensors, quadratus lumborum, psoas, obliques) or, in arm
balances, shoulder-limited.

With a 1 cm contact rule 31 holds lack an equilibrium instead of 12 — the reference itself
often rests the heel of the hand or a forearm 1–2 cm up (her plant's COM is outside the same
1 cm support polygon by the same margin), which is why the plant's own LP used 2 cm.

The 34 holds that only removing passive force rescues (downdog, standing split, low lunge,
warrior II, revolved poses, side plank, handstand, headstand, shoulderstand, upward plank,
Eagle, …) are loaded by exactly the tissues yogis train to be compliant: long head of triceps
(mean 479 N), gluteus maximus (448 N), hamstrings (biceps femoris long head 392 N,
semimembranosus 398 N), rectus abdominis, psoas, soleus, tibialis anterior and the deltoids,
stretched to a median 1.40× optimal fibre length (p90 1.73×). In three holds iliacus is driven
past its calibrated range into ~6 kN of passive force, a model artefact. MyoFullBody's passive
force-length curves describe average flexibility, not hers.

The LP is checked against the simulator: holding its activations keeps the body within 1.2–3.7 cm
for 0.3 s where zero activation collapses 21–33 cm. Brace forces act at body–body contacts within
1 cm (the plant LP's pair rule), equal and opposite, in MuJoCo's contact frame. Caveats: s* is
the min-max activation (a demanding criterion), contacts only at the transplanted primitives,
static only, and the brace forces are those the geometry allows, not measured ones. Standing itself needs s* ≈ 0.6 even on **stock** MyoFullBody: at its default pose
several muscles carry large passive forces (rectus femoris 555 N, middle deltoid 560 N, rectus
abdominis 308 N) that the back extensors must cancel — a property of the model, not of the
scaling.

## 4. Decisions a reviewer should know about

* Three neck torque motors are the only non-muscle actuators (MyoFullBody has no neck muscles).
* Contacts are the PhysX plant's 24 primitives on MyoFullBody's bones, colliding with the floor
  and with each other (all non-adjacent pairs); MyoFullBody's own colliders and 75 explicit pairs
  are inactive (visual geoms only). Fingers stay welded (MuscleMimic's default).
* MyoFullBody's passive muscle force is reduced for the 180 muscles her corpus over-stretches
  (§2.6), a fit to her data, not a measured property.
* Ranges beyond the OpenSim calibration: muscle paths/wraps there are not validated; the passive
  forces they produce are real in the simulation (see §3.2).
* The task is the dense-tracking (DeepMimic/MuscleMimic) formulation, i.e. ProtoMotions'
  Stage-1 tracker, not the paper's goal-conditioned 2-slot teacher, and without its contact
  penalties or AMP. Control is 100 Hz (MuscleMimic default) vs 30 Hz; clips are resampled from
  60 Hz (hinges linear, root slerp, couplings re-applied).
* `myoMimicYoga-v0` is registered at runtime (`register_yoga_task`) because its data is local. Like
  the stock clip-mode tasks (`register_mimic_mjlab_tasks_with_clip`), it is an mjlab special task
  with no registered CPU `MyoGymnasiumEnv` twin; evaluation and rendering use the mjlab env plus
  CPU MuJoCo playback. The CPU clip env, `MuscleMimicClipEnvV0`, tracks a single clip on the
  stock body. A yoga twin would generalise it to an injected subject model, a clip bank, the
  neck actions and the per-site termination, with a one-step parity test like
  `tests/test_mimic_cpu_mjlab_twin.py` (§6).

## 5. Training runs

`$Y/runs/v4` (tensorboard: `tensorboard --logdir $Y/runs`) is the overnight run, on the full
body (`subject_table_compliant_selfcol.json`): 2048 envs × 24 steps, actor and critic
1024-512-256 ELU with observation normalisation, init std 0.5, entropy 1e-3, adaptive LR from
3e-4 with KL target 0.3, rewards 5 × DeepMimic composite − mean squared activation, episodes ≤ 10 s,
termination when any of the 17 tracked sites is more than 0.5 m from its target (added here,
the PhysX teacher's rule), when the pelvis is more than 0.3 m from the reference root or the mean
site error exceeds 1 m (MuscleMimic's stock rules, kept). The same
configuration ran briefly on the earlier bodies, each stopped when the next statics result came
in: `v1_stock_passive` (136 iterations), `v2_compliant_explicitpairs` (100 iterations).
**Learning-rate trap (found at the first evaluation, 23:43).** rsl_rl's adaptive schedule
compares the KL *summed over action dimensions* with `desired_kl`; at the usual 0.01 and 357
actions (std 0.5) every useful update overshoots, and the learning rate fell from 3e-4 to its
1e-5 floor within the first iteration and stayed there (`Loss/learning_rate` in the logs of
`v3_lr_floor`, `side_tree_downdog_lr_floor`, which ran ~45 min that way and are kept as
records). `v4` and `side_v4` (restarted 23:45) scale the target to the action count
(`--desired-kl 0.3`, ≈0.0008 per dimension). MuscleMimic's own trainer sidesteps this with a
fixed, annealed LR and a very large exploration std.

`$Y/runs/side_v4` is the same configuration on two clips only, Tree -a
and Downward-Facing Dog -a (statics: holds at peak activation 0.24–0.41 and 0.11), with 1024 envs:
a direct test of whether this muscle body can hold a yoga pose, echoing the expert-60 finding
that single-motion experts learn far more cleanly. The two runs share the GPU (main ~5.0k, side
~3.4k steps/s, against ~9.3k for the main run alone). A watcher
(`$Y/tools/watch_eval.sh`) evaluates the newest checkpoint every 45 minutes, twice: every clip
from its first frame to its end (`<checkpoint>/full/metrics.json`), and every clip for 10 s from
its quietest mid-clip frame, which lands inside the main hold (crow starts at 7.32 s, her
annotated exemplar), with 12 clips rendered (`<checkpoint>/hold/*.mp4`). One line per checkpoint
lands in `$Y/renders/policy/summary.txt`. The state in the morning is recorded in `$Y/README.md`.

**Exploration-noise trap (found at the 02:10 evaluation).** Training episodes kept getting
longer while the deterministic evaluations stalled, and the policy's action std grew steadily
from 0.50 to 0.77 (main) and 0.70 (side) with no sign of levelling off (`Policy/mean_std`;
third panel of `$Y/renders/training_curves.png`). The muscle mapping is
`excitation = sigmoid(5 (a − 0.5))`, so Gaussian noise on `a` raises the *mean* excitation of
every quiet muscle: at std 0.77 a muscle the mean action keeps at 0.007 averages 0.12. That is
free co-contraction (joint stiffness) that the mean action does not have. The policy gradient
sees the benefit and keeps growing the std. Measured from inside each clip's hold, 10 s
horizon, 4 replicas per clip (`$Y/renders/noise_probe/`):

| checkpoint | mean action | sampled (as in training) | noise-matched mean |
|---|---|---|---|
| main `model_800`, 56 clips, mean survival | 0.61 s | 1.01 s | 0.96 s |
| side `model_1100`, downward dog -a | 2.53 s | 3.98 s | 2.54 s |
| side `model_1100`, Tree -a | 0.39 s | 1.22 s | 0.73 s |

The noise-matched controller is deterministic: the mean action with the muscle sigmoid
flattened by `1/sqrt(1 + π·25·std²/8)`, so each excitation equals its expectation under the
training noise (`evaluate.py --action matched`). It recovers most of the gap for the 56-clip
policy, so the static mean co-activation is the main mechanism. On the specialist's own clips
it recovers only part (Tree) or none (downward dog). Plausibly the rest is the asymmetric
activation dynamics (activation τ 10 ms, deactivation 40 ms), which lift mean activation
further under noisy excitation; this was not measured. The deterministic
evaluations therefore understate what the training curves show, and a teacher that will be
distilled should not depend on its noise. The response is `$Y/runs/side_v5_std03`: the side
run resumed at iteration 1200 with the std capped at 0.3 and no entropy bonus
(`train.py --max-std 0.3 --entropy-coef 0`), so the mean action has to supply the
co-contraction itself; its watcher (`$Y/tools/watch_eval2.sh`) scores both controllers at every
checkpoint. Its training episode length was back at the uncapped run's level within ~150
iterations, at 27 % less muscle effort per step. After 200 capped iterations the mean action held
Tree -a for 0.73 s (0.36 s at the branch point) and downward dog -a for 3.52 s (2.88 s). Within
the uncapped run, the mean-action Tree never improved over 800 iterations (0.47 → 0.55 → 0.39 →
0.36 s at iterations 400, 700, 1100, 1200). The main run was branched the same way at 03:10
(`$Y/runs/v5_std03` from `v4/model_1200`; `v4` stopped there). Its first evaluation, 200 capped
iterations later, confirms it on all 56 clips (10 s from each hold; per-family tables from
`$Y/tools/family_report.py`, which pairs every clip with the statics hold nearest its start):

| 56 clips, 10 s from the hold | `v4/model_1200` (branch point; 4 replicas) | `v5_std03/model_1400` (capped; mean action 1 replica, sampled 4) |
|---|---|---|
| mean action, mean completed | 0.091 | **0.119** |
| sampled actions, mean completed | 0.136 | 0.134 |
| mean action, multi-support ("connective", 21 clips) | 0.94 s | 1.37 s |
| mean action, single-leg (13) / arm balance (9) / inversion (13) | 0.39 / 0.81 / 0.67 s | 0.55 / 0.77 / 0.80 s |
| rank correlation of survival with statics feasibility (−s*) | +0.36 | +0.53 |

The deterministic controller now nearly matches the noisy one, and survival tracks the statics
more closely: holds the muscles can carry statically (s* ≤ 1, 37 clips) last 1.10 s on average,
the rest 0.68 s. The cap does not remove the effect entirely. At side iteration 1900 (std 0.3,
4 replicas each) downward dog lasts 3.25 s with the mean action, 4.30 s noise-matched and
4.18 s sampled, so what the noise still adds there is static co-activation. Tree lasts 0.58,
0.92 and 1.99 s. Its deterministic replicas are bimodal: they fall within 0.2 s or last about
1.6 s, while sampled ones last 1.65–2.21 s. Single-leg balance therefore still gets a dynamic
benefit from the noise, plausibly the asymmetric activation dynamics. A mean-preserving
bounded noise in excitation space (e.g. a Beta distribution) would remove the incentive at its
source (§6). The watchers score the mean action with 4 replicas from iteration 1900 on
(`watch_eval3.sh`).

**What ends an episode.** `evaluate.py` now records, for every rollout that ends before its
horizon, which rule fired (the largest margin at the last surviving step). When it is the
per-site limit, it also records whether that site's error persists in the pelvis frame. If it
does, the posture broke; if it doesn't, the body moved as a whole: an extremity is the first
point past 0.5 m when the whole body tips about its support. At `v5_std03/model_1500` (4
replicas, 224 rollouts from the holds), **68 % end in a posture break**, 13 % in whole-body
motion that carries an extremity past the limit, and 18 % with the pelvis 0.3 m off the
reference root; the mean-error rule never fires first. Single-leg poses break posture in 47 of
52 ends (the raised foot, the overhead hands), and inversions lose their legs (left toes first
in 31 of 52). Multi-support poses mostly lose a hand, or the root
(`$Y/renders/end_causes/v5_model_1500_report.md`). So at this stage the policy fails mainly by
letting a long limb drift out of the pose, not by falling over. Holding the limbs in place is
the binding skill so far, ahead of whole-body balance.

**Can it hold a pose? (frozen-reference test).** The "10 s from the hold" numbers above mix
holding with tracking. In these clips the reference starts moving 1.5–2 s after the quietest
frame (downward dog's joint speed reaches 7.7 rad/s at +2 s), so a 4 s "survival" there is
~2 s of hold plus part of the exit transition. `evaluate.py --start frozen` instead holds the
reference still at that frame for 10 s (zero velocity), which asks directly whether the
body can hold the pose. With the mean action and 4 replicas:

* `v5_std03/model_1700` (56 clips): **5 poses held for the full 10 s in all 4 replicas**:
  Downward Dog -a, Dolphin -a, Chaturanga -a, Plow -b and Supported Headstand -a (Plow -a
  5.5 s). Family means: multi-support 2.70 s, inversion 2.52 s, arm balance 0.82 s,
  single-leg 0.60 s. Statically holdable clips survive 2.41 s against 0.82 s for the rest
  (Spearman +0.59; `$Y/renders/frozen/v5_model_1700_report.md`).
* `side_v5_std03/model_2300` (trained on Tree and downward dog only): downward dog holds 10 s in
  3 of 4 replicas (twice reproduced); Tree lasts 1.5–2.1 s, ending in posture breaks of the
  raised foot and the hands. The video (`$Y/renders/frozen/side_model_2300/`, longest replica)
  shows the hips sagging behind the reference in the last seconds of the downward-dog hold,
  inside the termination limits.

The watchers ran this test at every checkpoint from 05:10 until the runs were stopped
(`watch_eval4.sh`, `frozen/`). Replicas of the *mean* action
differ (one clip: 2.16–2.53 s) because the GPU simulation is not bitwise deterministic across
environments. A single-replica number therefore carries roughly ±10–20 %.

**Final state (both runs stopped 2026-11-03 21:01).** `v5_std03` stopped at iteration 6927
(~340M env steps, last checkpoint `model_6900`). `side_v5_std03` stopped at 9739 (~240M, last
checkpoint `model_9700`). The last evaluated checkpoints are `model_6700` and `model_9500`. All
numbers are for the mean action, with 4 replicas unless marked (`$Y/renders/policy*_v5_std03/summary.txt`):

| 56-clip run | `model_2000` | `model_4000` | `model_6700` |
|---|---|---|---|
| frozen 10 s hold, mean completed | 0.231 | 0.301 | 0.278 |
| frozen: poses held 10 s in 4 of 4 replicas | 5 | 8 | 7 |
| 10 s from the hold, mean completed (sampled actions) | 0.150 (0.153) | 0.168 (0.180) | 0.190 (0.201) |
| every clip from its first frame, mean completed (1 replica) | 0.081 | 0.091 | 0.106 |
| training: mean episode length, tracking reward per episode | 123, 26.0 | 146, 32.8 | 163, 38.4 |

* **The frozen-hold test plateaued from iteration ~3400.** Since then it reads 0.26–0.30, with
  6–9 poses held in 4 of 4 replicas (9 at 3400, 3600 and 5400). The other measures still rose
  slowly, by 12–17 % over the last 2700 iterations. No clip was ever completed from its
  first frame. At `model_6700` the held poses are Dolphin -a, Downward Dog -a, Chaturanga -a,
  Plow -a and -b, Standing Forward Bend -a and Supported Headstand -a. Bridge -a and Plank -a hold
  in 3 of 4 replicas, Cobra -a in 2. Family means: multi-support 4.37 s, inversion 3.32 s, arm
  balance 1.10 s, single-leg 0.75 s. Statically holdable clips last 3.49 s against 1.34 s for the
  rest (Spearman +0.54; `$Y/renders/frozen/v5_model_2000_4000_6700_report.md`).
* **Limb drift still ends most rollouts.** Of the 188 frozen rollouts at `model_6700` that ended
  early, 60 % were posture breaks, 18 % whole-body motion and 22 % the pelvis leaving the root.
  Single-leg poses break posture in 42 of 52 ends, and inversions lose the left toes first in 23
  of 40.
* **The cap reduced the noise dependence but did not remove it.** Sampled actions beat the mean
  action at every checkpoint from 2900 on, by 1.5–15 % (median 8.5 %).
* **The two-clip specialist learnt Tree, late.** Tree -a, from its first frame (1 replica):
  1.0–2.6 s up to iteration 4200, 2.7–5.2 s from 4600 to 7400, then the full 20.2 s clip at
  8200, 8700 and 9500 (19.0–19.2 s at 7800 and 9100). Its frozen Tree held 10 s in 4 of 4
  replicas at 8200 and in 2 of 4 at 9100 and 9500.
  Of the 16 checkpoints from 3500 on, frozen downward dog held in 4 of 4 replicas at 7 and in at
  least 3 at 11 (3 at 9500). The way into it is never learnt: the hold starts 6.3 s into the
  13.4 s clip, and from the first frame every checkpoint ends after 1.2–2.7 s. Training episode
  length levelled off at ~500 steps from iteration ~6000. The specialist's Tree -b, an
  untrained clip of the same pose, still lasts 0.7–1.4 s frozen.

**Single-clip handstand run (`$Y/runs/handstand_v1`, 2026-11-03 22:00 to 11-04 09:13).** Asked after
the baseline: how well does this body learn one clip on its own? The clip,
`220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a`, is the corpus's only handstand. It lasts
49 s: a T-pose start, a forward fold to the hands (2–4 s), then four kick-ups from a one-leg-up
downward dog: a brief handstand (9–10 s), a long hold (14–24 s), another brief one (29–30 s)
and a second long hold (36–45 s). After each of the first three she comes back down to the
dog; after the last she stands up (46–49 s). The statics put the first
hold's exemplar (19.9 s) at s* 0.60, within the muscles, and the second hold's (38.6 s) at
s* 2.93, beyond them. Training is from scratch with the capped baseline's settings (std 0.3
from the start, so the cap is a real upper bound, entropy 0, KL target 0.3), observations and
reward unchanged. It uses 8192 environments, the most that fit beside the evaluator: 19.1 GB of
the A5000's 24 GB, against ~38 GB for 16384. The GPU is saturated by ~2048 environments (8192
gives 10.9–11.7k steps/s, about what 2048 gave alone). The extra environments therefore buy
4× larger PPO batches (196,608 samples per iteration, 4 minibatches of 49k), not more samples
per hour. Iterations take ~17–18 s.

`$Y/tools/watch_handstand.sh` evaluates the newest checkpoint every 30 min
(`$Y/renders/policy_handstand_v1/model_<N>/`, one line each in `summary.txt`). Each pass runs the
frozen test at both holds (16 replicas, the longest rendered), the start-time sweep (10 s from
every 2 s of the clip, 4 replicas: which parts are learnt) and the whole clip from its first
frame (4 replicas, the longest rendered). The default "hold" frame of this clip, its quietest
mid-clip frame (12.0 s), is a one-leg-up downward dog between kick-ups, not a handstand. Hence
the `--at-s` option, which freezes the reference at a chosen time.

**Result (stopped 2026-11-04 09:13 at iteration 2116, ~416M env steps, 11 h 12 min; last
checkpoint `model_2100`, last evaluated `model_2050`).** Stopped by the user: the handstand was
not being learnt. Figure: `$Y/renders/policy_handstand_v1/progress.png`.

| `handstand_v1`, 10 s from every 2 s of the clip | `model_0` | `model_400` | `model_750` | `model_1050` | `model_1500` | `model_2050` |
|---|---|---|---|---|---|---|
| mean action: starts inside the two long holds (10 starts) | 0.30 s | 0.47 s | 0.60 s | 0.68 s | 0.75 s | 0.84 s |
| mean action: all other starts (14) | 0.32 s | 0.46 s | 1.67 s | 1.73 s | 2.09 s | 2.23 s |
| noise-matched / sampled, other starts | – | 0.96 / 1.51 s | 1.85 / 1.87 s | 2.51 / 2.47 s | 2.49 / 2.66 s | 2.60 / 3.12 s |
| frozen first hold (19.9 s), mean action, 16 replicas | 0.31 s | 0.50 s | 0.66 s | 0.68 s | 0.70 s | 1.00 s |

* **It learns the parts around the handstand, not the handstand.** From starts in the fold and
  the one-leg-up downward dogs it tracks for 3–5 s with the mean action (up to 6 s with the
  matched or sampled controllers), and the rollouts end at the next kick-up: from 4, 26 and 32 s
  they last to ~7.7, 29.6 and 35.6 s. In the start at 32 s, the policy follows the dog and the
  preparation and fails at the kick-up into the second hold
  (`$Y/renders/policy_handstand_v1/model_1500/sweep_videos/`, starts at 4, 14, 26 and 32 s).
  No start inside a handstand phase survives 1 s on average, and the frozen handstand is never
  held (0 of 16 replicas at any checkpoint, ≤ 1.1 s with any controller). Over the run the hold
  starts went from 0.30 to 0.84 s while the other starts went from 0.32 to 2.23 s. The statically infeasible second
  hold (38.6 s) fails the same way (0.7–0.9 s). Of the frozen first-hold rollouts at iterations
  950–1300 (3 controllers × 16 replicas each), about half end with the legs leaving the pose (a
  toe or knee 0.5 m off, the error persisting in the pelvis frame) and half with the body tipping
  as a whole (the pelvis 0.3 m off the reference root, roughly 15° of lean about the hands). So
  the inverted balance itself is not learnt, not only the leg posture.
* **The whole clip from its first frame ends within 0.9 s**, in the opening T-pose stand. Few
  training episodes start there (the first 2 s are 4 % of the clip); the 56-clip run, which
  stands in many clips, lasts 2.7 s from the same start.
* **The noise dependence reappears from scratch and then closes.** With the std capped at 0.3
  from iteration 0, sampled actions survived 2.4× longer than the mean action at iteration 400
  (sweep means 1.13 vs 0.47 s). The noise-matched controller recovered about half of that gap at
  400 and nearly all of it from ~1050 on (1.77 vs 1.78 s at 1050, 1.79 vs 1.89 s at 1500), so
  the noise's benefit became mostly average co-contraction, which a deterministic controller
  reproduces. The mean action still trails by
  ~20–25 % (1.53 vs 1.89 s at 1500, 1.65 vs 2.19 s at 2050).
* **Against the 56-clip run on this clip** (its last checkpoint, `v5_std03/model_6900`, ~340M
  steps over 56 clips, same evaluations): sweep mean 0.72 s with the mean action (0.73 s
  matched) against the specialist's 1.53 s (1.79 s), and frozen first hold 0.47 s against 0.70 s.
  The specialist is ~2× better on its own clip, but neither holds a handstand. (The 56-clip
  run's earlier frozen result for this clip, in §5 "Final state", froze its 12.0 s one-leg dog,
  not a handstand; see the `--at-s` note above.)
* Training had not plateaued when stopped (episode length ~2.4 s, sampled), but the mean-action
  sweep had flattened at 1.57–1.71 s over its last four evaluations (1700–2050).

## 6. Next steps, in order

1. **Change how this body learns; the capped runs are the baseline** (§5 "Final state"). Judge
   every change by the mean action, per family (`tools/family_report.py`), with the frozen-hold
   test. The 56-clip run plateaued at 6–9 held poses, with single-leg balances last (0.75 s).
   The two-clip specialist did learn Tree, after ~8000 iterations, so the pose is learnable on
   this body. The 56-clip run spread 2048 environments over 56 clips, against the specialist's
   1024 over 2, so samples per clip, interference between clips and capacity are open
   explanations. The std cap at 0.3 was a first setting, and the noise still helps by ~8 %. Candidate knobs: an annealed std, and noise
   that preserves the mean excitation (a Beta distribution on excitation in [0, 1], or noise on
   the activation itself) so the policy gains nothing from it.
2. **Next run on the brace-aware clips** (`$Y/clips_selfcol_brace`, §3.1): fewer jolts at reset
   for the same pose fidelity; the remaining overlaps (≤ 4.6 cm in firefly and shoulder-pressing)
   are tight braces the IK cannot open without moving the pose.
3. **Trunk**: the s* > 1 holds are trunk-limited. Options: a thoracic joint, check MyoTorso's
   fascicle strengths, or accept and treat those holds as beyond the plant (as plant v2 did for
   Scorpion -b's wrists).
4. **Passive tissue**: §2.6 is a first, uniform rule (one stretch cap for every muscle). A
   better one would calibrate against measured passive joint torques of flexible subjects, and
   the iliacus artefact (driven past its calibrated range) needs its path checked.
5. **Teacher parity**: the teacher's reward stack (body-position terms, unwanted-support EMA
   penalty, lean term), then AMP, then the goal-conditioned 2-slot interface.
6. **CPU twin** (§4): generalise `MuscleMimicClipEnvV0` and add a one-step CPU ↔ mjlab parity
   test, for CPU playback, debugging and the repo's golden rule.

## 7. Reproduce

```bash
cd myosuite   # repo root, uv env: uv sync -p 3.10 --extra dev --extra mjlab
Y=/path/to/yoga_muscle
REL=<ProtoMotions>/data/smpl/reference_curation/holds_repaired_ftC_posefix.release_v2.a2dda5d2ac
PLANT=<ProtoMotions>/data/assets/smpl/smpl_yogi03596_v2.xml
PKL=<moyo_toolkit>/data/mosh/train/220923_yogi_body_hands_03596_Crane_(Crow)_Pose_or_Bakasana_-a_stageii.pkl
SMPLX=<moyo_toolkit>/data/body_models/smplx/SMPLX_FEMALE.npz
python -m myosuite.integrations.yoga.reference_io --release $REL --out $Y/refs                    # 56 clips, s
python -m myosuite.integrations.yoga.subject_body --plant-xml $PLANT --mosh-pkl "$PKL" --smplx $SMPLX \
    --out $Y/model/subject_table.json --xml-out $Y/model/myofullbody_subject.xml                 # ~5 s (plant colliders + self-collision)
OMP_NUM_THREADS=1 python -m myosuite.integrations.yoga.retarget --table $Y/model/subject_table.json \
    --mosh-pkl "$PKL" --smplx $SMPLX --plant-xml $PLANT --refs $Y/refs --out $Y/clips --workers 20  # ~2 min
python -m myosuite.integrations.yoga.statics --table $Y/model/subject_table.json --clips-dir $Y/clips \
    --holds $REL/release_holds.yaml --out $Y/model/statics.json                                  # ~5 s
python -m myosuite.integrations.yoga.statics --table $Y/model/subject_table.json --clips-dir $Y/clips \
    --holds $REL/release_holds.yaml --out $Y/model/statics_compliant.json \
    --write-compliant $Y/model/subject_table_compliant_selfcol.json                             # her flexibility, ~1 min
python -m myosuite.integrations.yoga.train --table $Y/model/subject_table_compliant_selfcol.json --clips-dir $Y/clips \
    --log-dir $Y/runs/v4 --num-envs 2048 --iterations 20000 --init-std 0.5 --desired-kl 0.3
python -m myosuite.integrations.yoga.train --table $Y/model/subject_table_compliant_selfcol.json --clips-dir $Y/clips \
    --log-dir $Y/runs/side_v5_std03 --num-envs 1024 --iterations 20000 --init-std 0.5 --desired-kl 0.3 \
    --include Tree_Pose_or_Vrksasana_-a Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a \
    --resume $Y/runs/side_v4/model_1200.pt --max-std 0.3 --entropy-coef 0                       # the capped-noise arm
MUJOCO_GL=egl python -m myosuite.integrations.yoga.evaluate --run $Y/runs/v4 --checkpoint $Y/runs/v4/model_N.pt \
    --refs $Y/refs --plant-xml $PLANT --out $Y/renders/policy/model_N --render Tree Crane Handstand
MUJOCO_GL=egl python -m myosuite.integrations.yoga.evaluate --run $Y/runs/v4 --checkpoint $Y/runs/v4/model_N.pt \
    --refs $Y/refs --plant-xml $PLANT --out $Y/renders/noise_probe/model_N/sampled --start hold --horizon-s 10 \
    --action sampled --repeats 4                                                                 # or --action mean / matched
MUJOCO_GL=egl python -m myosuite.integrations.yoga.evaluate --run $Y/runs/v5_std03 --checkpoint $Y/runs/v5_std03/model_N.pt \
    --refs $Y/refs --plant-xml $PLANT --out $Y/renders/frozen/model_N --start frozen --horizon-s 10 \
    --repeats 4 --render-best --render Downward Tree                                            # can it hold the pose?
python $Y/tools/family_report.py $Y/renders/frozen/model_N/metrics.json                         # per family, vs the statics
python -m myosuite.integrations.yoga.train --table $Y/model/subject_table_compliant_selfcol.json --clips-dir $Y/clips \
    --log-dir $Y/runs/handstand_v1 --num-envs 8192 --iterations 20000 --save-interval 50 \
    --include 220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a --init-std 0.3 --max-std 0.3 --entropy-coef 0 \
    --desired-kl 0.3                                                                             # one clip, from scratch
MUJOCO_GL=egl python -m myosuite.integrations.yoga.evaluate --run $Y/runs/handstand_v1 \
    --checkpoint $Y/runs/handstand_v1/model_N.pt --refs $Y/refs --plant-xml $PLANT --include Handstand \
    --out $Y/renders/policy_handstand_v1/model_N/sweep --start sweep --horizon-s 10 --repeats 4   # 10 s from every 2 s
MUJOCO_GL=egl python -m myosuite.integrations.yoga.evaluate --run $Y/runs/handstand_v1 \
    --checkpoint $Y/runs/handstand_v1/model_N.pt --refs $Y/refs --plant-xml $PLANT --include Handstand \
    --out $Y/renders/policy_handstand_v1/model_N/frozen_19.9s --start frozen --at-s 19.88 --horizon-s 10 \
    --repeats 16 --render-best --render Handstand                                               # hold at a chosen time
MUJOCO_GL=egl python -m myosuite.integrations.yoga.render --table $Y/model/subject_table.json --plant-xml $PLANT \
    --refs $Y/refs --clips-dir $Y/clips --out $Y/renders/retarget --clips Crane Handstand
```

Data-free tests: `pytest myosuite/tests/test_yoga_integration.py` (11: coupling convention,
velocity map, identity table, segment scaling, passive scaling, self-collision, primitive lowest
points vs MuJoCo, per-site termination, the noise-matched controller against Monte Carlo, the
pelvis-frame split of termination causes, and the start-time sweep's cuts). The full tier-1 suite (1426 tests), the mimic and the
mjlab mimic suites pass with the shared-code changes.
