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

The source project repository is private. Researchers with access can check out the pinned commit, install its documented runtime, set `VAMP_REPO` to a checkout at the pinned VAMP revision, and replay the source-ordered IDs with its archived runner/configuration. This public release provides exact input bytes, configs and results but cannot by itself provide that private solver source. Consequently it enables independent data/result verification, while a fully independent solver rerun requires authorized access to the source repository.

The frozen full-C3 contract uses 0.1 s jerk and torque-rate rise times; sensitivity contracts use 0.05/0.2 s. UR5 acceleration is explicitly assumed at 5 rad/s^2. Exact historical derived JSON records were not recovered; the corresponding projection files identify themselves as regenerated and cite the internal source receipt hash.
