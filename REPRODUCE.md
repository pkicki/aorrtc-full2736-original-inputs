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


## Original export record and physical conventions

The original export attempted 700 MBM Panda problems, 1,800 MPINets Panda problems and 700 MBM UR5 problems. The exported order is frozen in each `exported_ids.txt` and is aligned position-for-position with the cache's `records` list. Attempt metadata explains attempted versus exported populations. The cache and metadata are original bytes; do not regenerate or reorder the paths when comparing solver results.

Panda's joint-space YAML and the full-C3 derived vectors are in `limits/`. Jerk and torque-rate scales use 0.1 s rise times for the primary contract and 0.05/0.2 s for sensitivity. UR5 acceleration 5 rad/s^2 is an assumed planning limit, not a manufacturer limit. VAMP source commit `f6de6a72e0725ba08e0829f0a34261a3f465aed7` contains the upstream robot models; the URDF files themselves are intentionally not bundled.

The exporter and retiming source at project commit `07ab69fc46436c59e4e355f69193bdb13f203b4e` is private. Thus independent users can verify the complete source inputs and archived per-path numerical evidence, but cannot reproduce a new solver run without authorized access to that code.
