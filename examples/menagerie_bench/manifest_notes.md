# Menagerie manifest notes

Menagerie clone: `examples/mujoco_menagerie`, read-only.

`MENAGERIE_COMMIT` verified with `git -C examples/mujoco_menagerie rev-parse HEAD`:
`8161bba264d7fa7c99ca301e91e7fb44737676ad` (reclone of 2026-09-08; bumped from `71f066a`, whose 8 newer commits touch only non-kinematic attributes of manifest xmls and add `seeed_rebot_devarm`).

## Selection rules (as agreed)

1. Never `scene*.xml` (floor/lights wrappers), never `mjx_*` / `*_mjx*` (simplified MJX
   variants), never actuator-only variants (`*_motor` / `*_position` / `*_velocity` /
   `*_pos`), never `keyframes.xml` or other non-robot xmls (test fixtures,
   cartesian/joint actuator configs).
2. Base variant WITHOUT optional attachments when the dir offers a with/without split
   (nohand / no_gripper / no-arm over the augmented variant).
3. But the FULL robot when the alternative is a strict sub-module
   (tidybot over its mobile base; aloha dual-arm is the robot).
4. `left_hand.xml` for handed pairs.
5. Skip only models that are not robots at all: zero-articulation sensors/cameras.

## Verification

Every chosen xml was loaded with `mujoco.MjModel.from_xml_path` using
`/home/john/.conda/envs/cusadi2/bin/python` (MuJoCo import from the cusadi2 env).
All 67 candidate xmls load without error. `realsense_d435i/d435i.xml` was also
loaded to confirm nq=0, nv=0 before marking it skipped.

| dir | chosen xml | nq | nv | nbody | floating | notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| agilex_piper | piper.xml | 8 | 8 | 10 | no | |
| agility_cassie | cassie.xml | 35 | 32 | 26 | yes | quaternion joints in shins/tarsus |
| aloha | aloha.xml | 16 | 16 | 21 | no | Rule 3: dual-arm IS the robot. Other xmls are actuator/keyframe fixtures (rule 1). |
| anybotics_anymal_b | anymal_b.xml | 19 | 18 | 14 | yes | |
| anybotics_anymal_c | anymal_c.xml | 19 | 18 | 14 | yes | not anymal_c_mjx.xml (rule 1) |
| apptronik_apollo | apptronik_apollo.xml | 39 | 38 | 37 | yes | |
| arx_l5 | arx_l5.xml | 8 | 8 | 10 | no | |
| berkeley_humanoid | berkeley_humanoid.xml | 19 | 18 | 14 | yes | |
| bitcraze_crazyflie_2 | cf2.xml | 7 | 6 | 2 | yes | AMBIGUOUS: free joint only, zero articulated joints. Kept — it is a robot (quadrotor free-flyer), not a sensor; rule 5 skip reserved for sensors/cameras. |
| booster_t1 | t1.xml | 30 | 29 | 25 | yes | |
| boston_dynamics_spot | spot.xml | 19 | 18 | 14 | yes | Rule 2: spot.xml over spot_arm.xml (arm is optional attachment) |
| dynamixel_2r | dynamixel_2r.xml | 2 | 2 | 4 | no | minimal 2R educational arm; still a robot, kept |
| flexiv_rizon4 | flexiv_rizon4.xml | 7 | 7 | 9 | no | new dir; only robot xml + scene.xml, no attachment variants |
| flexiv_rizon4s | flexiv_rizon4s.xml | 7 | 7 | 9 | no | new dir; same layout as rizon4 |
| flybody | fruitfly.xml | 109 | 108 | 68 | yes | biomechanical fly model, largest nv in the set |
| fourier_n1 | n1.xml | 30 | 29 | 30 | yes | |
| franka_emika_panda | panda_nohand.xml | 7 | 7 | 10 | no | Rule 2/3: nohand base; hand.xml is a detached gripper; mjx_* excluded (rule 1) |
| franka_fr3 | fr3.xml | 7 | 7 | 10 | no | |
| franka_fr3_v2 | fr3v2.xml | 7 | 7 | 11 | no | |
| google_barkour_v0 | barkour_v0.xml | 19 | 18 | 14 | yes | not barkour_v0_mjx.xml (rule 1) |
| google_barkour_vb | barkour_vb.xml | 19 | 18 | 16 | yes | not barkour_vb_mjx.xml (rule 1) |
| google_robot | robot.xml | 9 | 9 | 13 | no | |
| hello_robot_stretch | stretch.xml | 24 | 23 | 27 | yes | Stretch 2 |
| hello_robot_stretch_3 | stretch.xml | 27 | 26 | 38 | yes | same filename as stretch 2, different dir |
| i2rt_yam | yam.xml | 8 | 8 | 14 | no | |
| iit_softfoot | softfoot.xml | 92 | 92 | 50 | no | single foot module with many passive/tendon joints; articulated, kept |
| kinova_gen3 | gen3.xml | 7 | 7 | 9 | no | |
| kuka_iiwa_14 | iiwa14.xml | 7 | 7 | 9 | no | |
| leap_hand | left_hand.xml | 16 | 16 | 18 | no | Rule 4 |
| low_cost_robot_arm | low_cost_robot_arm.xml | 6 | 6 | 8 | no | |
| ms_human_700 | MS-Human-700.xml | 85 | 85 | 81 | no | new dir; full-body musculoskeletal base model. Locomotion variant simplifies the upper body; Manipulation variant adds a graspable bottle (extra object). Full-body is canonical. |
| pal_talos | talos.xml | 51 | 50 | 46 | yes | talos_motor/_position are actuator-only variants (rule 1) |
| pal_tiago | tiago.xml | 29 | 28 | 24 | yes | tiago_motor/_position/_velocity excluded (rule 1) |
| pal_tiago_dual | tiago_dual.xml | 32 | 31 | 27 | yes | dual-arm IS this product (rule 3); actuator variants excluded (rule 1) |
| pndbotics_adam_lite | adam_lite.xml | 32 | 31 | 27 | yes | |
| rainbow_robotics_rby1 | rby1a_1.2_no_gripper.xml | 31 | 30 | 27 | yes | AMBIGUOUS, new dir. Rule 2 gives no_gripper, but the rules do not decide RBY1-A (2-wheel diff base) vs RBY1-M (4 mecanum wheels), nor v1.2 vs v1.3 (1.3 exists only for M and only with grippers, so no rule-2-compliant 1.3 option). Picked A: listed first in the README as the base configuration. Human review welcome. |
| realsense_d435i | SKIP (zero_dof_camera) | 0 | 0 | 2 | no | d435i.xml loads with nq=0, nv=0 — camera, not a robot (rule 5) |
| rethink_robotics_sawyer | sawyer.xml | 7 | 7 | 10 | no | |
| seeed_rebot_devarm | seeed_rebot_devarm.xml | 8 | 8 | 11 | no | added at 8161bba; scene.xml is a wrapper (rule 1) |
| robot_soccer_kit | robot_soccer_kit.xml | 71 | 70 | 66 | yes | single omniwheel robot; 3 driven wheels x 20 passive rollers explains the surprising nv=70 |
| robotiq_2f85 | 2f85.xml | 8 | 8 | 15 | no | gripper with articulation — a robot module, kept (skip is for zero-dof sensors only) |
| robotiq_2f85_v4 | 2f85.xml | 6 | 6 | 12 | no | not mjx_2f85.xml (rule 1) |
| robotis_op3 | op3.xml | 27 | 26 | 22 | yes | |
| robotstudio_so101 | so101.xml | 6 | 6 | 9 | no | new dir; scene.xml/scene_box.xml are wrappers (rule 1) |
| shadow_dexee | shadow_dexee.xml | 12 | 12 | 20 | no | |
| shadow_hand | left_hand.xml | 24 | 24 | 26 | no | Rule 4; keyframes.xml excluded (rule 1) |
| sharpa_wave | left_hand.xml | 22 | 22 | 24 | no | new dir; handed pair, rule 4 |
| skydio_x2 | x2.xml | 7 | 6 | 2 | yes | AMBIGUOUS: same situation as crazyflie — free joint only, zero articulated joints; kept as a robot (quadrotor free-flyer) |
| stanford_tidybot | tidybot.xml | 18 | 18 | 23 | no | Rule 3: base.xml is just the holonomic mobile base |
| tetheria_aero_hand_open | left_hand.xml | 16 | 16 | 23 | no | new-ish dir; rule 4. Only scene_right.xml exists but left_hand.xml loads fine standalone. |
| toddlerbot_2xc | toddlerbot_2xc.xml | 51 | 50 | 46 | yes | not _pos (actuator variant) or _mjx (rule 1) |
| toddlerbot_2xm | toddlerbot_2xm.xml | 51 | 50 | 46 | yes | not _pos or _mjx (rule 1) |
| trossen_vx300s | vx300s.xml | 8 | 8 | 11 | no | |
| trossen_wx250s | wx250s.xml | 8 | 8 | 10 | no | |
| trossen_wxai | wxai_follower.xml | 8 | 8 | 14 | no | new dir. Single arm over wxai_follower_biarm.xml: the biarm is a script-generated composite of two arms on a platform (per README), i.e. an augmented variant (rule 2), not the product itself (unlike aloha). |
| trs_so_arm100 | so_arm100.xml | 6 | 6 | 8 | no | |
| ufactory_lite6 | lite6.xml | 6 | 6 | 8 | no | Rule 2: plain lite6 over lite6_gripper_narrow/_wide |
| ufactory_xarm7 | xarm7_nohand.xml | 7 | 7 | 9 | no | Rule 2: nohand; hand.xml is a detached gripper |
| umi_gripper | umi_gripper.xml | 8 | 8 | 4 | no | articulated gripper, kept |
| unitree_a1 | a1.xml | 19 | 18 | 14 | yes | |
| unitree_g1 | g1.xml | 36 | 35 | 31 | yes | Rule 2: g1 over g1_with_hands; g1_mjx excluded (rule 1) |
| unitree_go1 | go1.xml | 19 | 18 | 14 | yes | |
| unitree_go2 | go2.xml | 19 | 18 | 14 | yes | not go2_mjx.xml (rule 1) |
| unitree_h1 | h1.xml | 26 | 25 | 21 | yes | |
| unitree_z1 | z1.xml | 6 | 6 | 8 | no | Rule 2: z1 over z1_gripper |
| universal_robots_ur10e | ur10e.xml | 6 | 6 | 8 | no | |
| universal_robots_ur5e | ur5e.xml | 6 | 6 | 8 | no | |
| wonik_allegro | left_hand.xml | 16 | 16 | 22 | no | Rule 4 |

## AMBIGUOUS flags (human review requested)

- **rainbow_robotics_rby1** → `rby1a_1.2_no_gripper.xml`. no_gripper follows rule 2,
  but A-vs-M base and 1.2-vs-1.3 are not covered by the rules. RBY1-M v1.3 is the
  newest model but ships only with grippers.
- **bitcraze_crazyflie_2** → `cf2.xml` and **skydio_x2** → `x2.xml`. Zero articulated
  joints (free joint only, nv=6, nbody=2). Kept because rule 5 reserves skip for
  non-robots (sensors/cameras); a free-flying rigid body is still a valid dynamics
  benchmark. If the harness wants "articulated only", these are the two to drop.

## Totals

- 67 entries (one per model directory; assets/test/loose files excluded).
- 66 benchmarked, 1 skipped (`realsense_d435i`, zero_dof_camera).
- 0 load failures in MuJoCo.
