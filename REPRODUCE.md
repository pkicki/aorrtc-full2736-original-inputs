# Reproduction and interpretation

## Download and verify

```bash
git clone https://github.com/pkicki/aorrtc-full2736-original-inputs.git
cd aorrtc-full2736-original-inputs
git lfs install
git lfs pull
python verify_release.py
```

Every path is a new-path request: include path-specific preparation, dynamics/coefficient evaluation, optimization and output materialization in complete-request latency. Certification is separate. Do not cache path-specific coefficients for the headline comparison.

## Frozen methods

The release pins the source project revision `07ab69fc46436c59e4e355f69193bdb13f203b4e`, VAMP revision `f6de6a72e0725ba08e0829f0a34261a3f465aed7`, the qualified fast_v1 source commit `98e9f780bebc6f94b62a4ee8072da230dcecd1fd`, and exact per-path outputs under `evidence/`. Original compact Kaserer certified 2703/2736; analytical Kaserer_fixed and fast_v1 certified 2713/2736. All 2736 rows, including the 23 no-profile cases, are retained.

## Executable published frozen Kaserer code and verified independent rerun

The solver source of the **compact Kaserer DP3 and analytical-rate
`Kaserer_fixed`/`dp3_fixed_fast`** is now included under
[`reproduction/src/`](reproduction/src/curobo_timeopt/baselines/kaserer_dp3.py),
with its C++ DP extension and exact Pinocchio coefficient code. The
[numerical source blob SHA inventory](reproduction/frozen_source_git_blobs.json)
proves exact identity with the private frozen source commit.
The [independent run guide](reproduction/README.md) includes complete
Linux CPU installation and replay commands on the pinned original Panda path.

A **fresh independent GitHub Actions CPU replay** succeeded:
[CI #40135314791](https://github.com/pkicki/aorrtc-full2736-original-inputs/actions/runs/40135314791).
The published native solver returned duration
`1.4638021000955588` s versus frozen `1.463802100095559` s,
with all original velocity, acceleration, jerk, torque and analytical
torque-rate limits independently certified at S8193 and S32768.
The [machine-readable receipt](reproduction/independent_replay_receipt.json)
and [scientific decision](reproduction/REPLAY_STATUS.md) are committed.

The **remaining private project source** contains many other solvers,
including CUDA PDF q1/q5. This minimal public replay is sufficient to
reproduce one original Kaserer full-C3 problem without access to that
private project, but it is **not** a complete public executable
reproduction of the entire T-RO method comparison.

The frozen full-C3 contract uses 0.1 s jerk and torque-rate rise times; sensitivity contracts use 0.05/0.2 s. UR5 acceleration is explicitly assumed at 5 rad/s^2. Exact historical derived JSON records were not recovered; the corresponding projection files identify themselves as regenerated and cite the internal source receipt hash.
