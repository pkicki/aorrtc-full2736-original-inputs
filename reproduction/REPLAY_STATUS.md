# Independent frozen-source CPU replay qualification — PASSED

**Scientific scope:** one independent original-path Kaserer fast_v1 full-C3
retiming solve and event-aware five-family physical certification, NOT a
complete recreation of the PDF/C3 GPU pipeline or every T-RO method.

**Evidence:** [public successful GitHub Actions run #40135314791](
https://github.com/pkicki/aorrtc-full2736-original-inputs/actions/runs/40135314791)
on source commit `f3fa50af1ec6f9cf774c654239a00ca84da304d5`.
Archived machine-readable [result receipt](independent_replay_receipt.json).
The original dataset release is v1.0.3 commit
`6268f7635b5e8781a6b01a721fcf3c21b66b1db4`.

**Source authenticity:** original frozen numerical Python and native C++
Git blobs are identical to private research commit
`98e9f780bebc6f94b62a4ee8072da230dcecd1fd`.
Only minimal package initializers and the new independent public runner
were authored for the release; solver/physical limits were unchanged.
The public Linux runner used Torch 2.6.0+cpu, NumPy 2.3.3 and Pinocchio 3.9.0
with ABI-pinned cmeel-urdfdom 4.0.1 / cmeel-tinyxml2 10.0.0.

- Exact verified path: `motion_benchmaker/panda/bookshelf_small/0008`,
  fingerprint `db4bfc2faad5ddb9ecddf94df7979d88c3ae59f778d54af7fec9899f0fe9f645`.
- **Reproduced duration:** 1.4638021000955588 s.
- **Frozen reference duration:** 1.463802100095559 s.
- **Absolute duration difference:** 2.22e-16 seconds.
- Native C++ Kaserer solution: success, `solved`.
- **S8193:** PASS, 8,382 event-aware samples.
- **S32768:** PASS, 32,957 event-aware samples.
- S32768 peak utilizations: velocity 0.984525,
  acceleration 0.946750, jerk 0.969615,
  torque 0.795266, torque rate 0.987530.
- Both endpoints have zero speed; max C3 slope jump 2.58e-14.

**Result:** `INDEPENDENT_ORIGINAL_PATH_KASERER_FULL_C3_REPRODUCED`.

**Remaining work for complete paper-wide execution:** PDF q1/q5 GPU pipeline
source/CPU-GPU source model compatibility and the other external baseline
dependencies/results have not all been republished or freshly rerun under
the public environment. Two historical source/limit JSON bytes are still
missing; the current single-path independent run uses the
original-model-SHA-verified public projection of the frozen 0.1 s policy.
Consequently the full T-RO campaign is **not** yet
`PUBLIC_REPRODUCIBLE` across all methods, despite this successful
independent frozen Kaserer reproduction.
