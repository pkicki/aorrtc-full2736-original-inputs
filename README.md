# Frozen AORRTC Full2736 data and benchmark evidence

This dataset consists of the **original, byte-for-byte** exported quintic geometric paths and source-order IDs: Panda MBM (650), Panda MPINets (1519), UR5 MBM (567), with attempt/export metadata, frozen physical-limit configurations, source identity, full-C3 contracts and matched benchmark results. The paths are owner-authorized.

**Dataset license:** CC-BY-4.0. The license applies only to the bundled original path and author-owned metadata; it does not grant rights to third-party upstream VAMP robot descriptions or software.

Use `git lfs install && git lfs pull` after cloning. Run `sha256sum -c SHA256SUMS.txt` to verify exact original bytes.

Frozen source revision: `07ab69fc46436c59e4e355f69193bdb13f203b4e`; VAMP revision: `f6de6a72e0725ba08e0829f0a34261a3f465aed7`. Results refer to `dp3_fixed_fast` frozen commit `98e9f780bebc6f94b62a4ee8072da230dcecd1fd`, not a rerun at release time. Model URDFs are obtained from the pinned upstream VAMP checkout.

The package includes raw per-path Kaserer, Kaserer_fixed, fast_v1 and V2 archives, PDF q1/q5 and external-baseline Full2736 records, certification diagnostics, experiment configurations and reports. `manifest.json` and `SHA256SUMS.txt` pin every shipped byte.

The exact original historical derived identity and rate-limit JSON files were not present in the archived repository. Their public projections are explicitly labeled regenerated, not historical originals. The hash-only source identity receipt is included byte-exact and retains its historical evidence label. The project source commit is identified but its private solver code is not redistributed. UR5 acceleration 5 rad/s^2 is an assumption, not a manufacturer limit. See `REPRODUCE.md`.
