# Reproduce one frozen Full-C3 Kaserer retiming result independently

This directory publishes the **actual frozen solver implementation**, not
a reimplementation: the 21 numerical Python/C++ source modules under
`src/curobo_timeopt/` have Git blob SHA hashes **identical** to
`pkicki/time_optimal_curobo@98e9f780bebc6f94b62a4ee8072da230dcecd1fd`.
Only package `__init__` modules were simplified to avoid unrelated eager
imports; `run_one.py` is publication-specific orchestration derived from the
frozen original runner (it is not claimed to be byte-identical). Solver,
native C++ DP and analytical Pinocchio numerical operations are unmodified.

The default replay uses original source path
`motion_benchmaker/panda/bookshelf_small/0008`, source fingerprint
`db4bfc2faad5ddb9ecddf94df7979d88c3ae59f778d54af7fec9899f0fe9f645`.
The archived `dp3_fixed_fast` solution duration was **1.463802100095559 s**
and passed original S8193 and S32768 five-family independent audits.
This public smoke **re-solves the original path** and independently recomputes
velocity, acceleration, jerk, inverse-dynamics torque and *analytical
RNEA-JVP torque rate* at both grids. It checks zero endpoint speeds, C3
slope continuity and exact archived duration within 1e-8 s.

## Linux CPU procedure

Requires Git, Git LFS, Python 3.11, C++ build tools, ~4 GB memory and internet
to obtain the pinned VAMP model. From the root of the public dataset checkout:

```bash
git lfs install
git lfs pull --include='inputs/mbm_panda/aorrtc_all.pt'

python3.11 -m venv .venv-replay
source .venv-replay/bin/activate
pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
pip install -r reproduction/requirements-cpu.txt

git init /tmp/vamp-kaserer-pinned
git -C /tmp/vamp-kaserer-pinned fetch --depth=1 https://github.com/KavrakiLab/vamp.git f6de6a72e0725ba08e0829f0a34261a3f465aed7
git -C /tmp/vamp-kaserer-pinned checkout --detach FETCH_HEAD

export PYTHONPATH="$PWD/reproduction/src"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
python reproduction/run_one.py --vamp /tmp/vamp-kaserer-pinned \
  --output /tmp/kaserer-public-replay.json
```

Successful completion requires `five_family_certified: true` and duration
matching the archived original outcome. The new request's **wall-clock time
is not expected to match the historical Xeon CPU distribution**. Native
compilation and model loading occur before the online timer. The independent
audits are timed outside the solver request boundary.

`--method kaserer_dp3` evaluates the compact clean-room Kaserer
reconstruction with checkpoint finite-difference torque rate.
`--method kaserer_fixed` evaluates our analytical-rate adaptation with
reference precompute. `--method dp3_fixed_fast` is the opt-in accelerated
precomputation of the analytical-rate adaptation.

## Scope and remaining requirements

This is one independent original-geometry CPU replay of the full-C3
Kaserer method, not an independent reproduction of every PDF q1/q5 CUDA
pipeline and all external baseline methods across 2,736 paths. The latter
requires separately publishing and qualifying their larger solver stacks.
The dataset's two historical derived identity/limit JSON files are still
not recovered byte-for-byte; this replay uses the publicly documented
0.1 s rate contract explicitly marked as a regenerated public projection.
It independently verifies the original geometric fingerprint and URDF SHA.

Do not label the full T-RO campaign `PUBLIC_REPRODUCIBLE` solely because this
one-path replay passes. Record any drift, dependency incompatibility or failed
certification in a new machine-readable receipt.
