# Frozen AORRTC Full2736 data and benchmark evidence

This dataset consists of the **original, byte-for-byte** exported quintic geometric paths and source-order IDs: Panda MBM (650), Panda MPINets (1519), UR5 MBM (567), with attempt/export metadata, frozen physical-limit configurations, source identity, full-C3 contracts and matched benchmark results. The paths are owner-authorized.

**Dataset license:** CC-BY-4.0. The license applies only to the bundled original path and author-owned metadata; it does not grant rights to third-party upstream VAMP robot descriptions or software.

Use `git lfs install && git lfs pull` after cloning. Run `sha256sum -c SHA256SUMS.txt` to verify exact original bytes.

Frozen source revision: `07ab69fc46436c59e4e355f69193bdb13f203b4e`; VAMP revision: `f6de6a72e0725ba08e0829f0a34261a3f465aed7`. Results refer to `dp3_fixed_fast` frozen commit `98e9f780bebc6f94b62a4ee8072da230dcecd1fd`, not a rerun at release time. Model URDFs are obtained from the pinned upstream VAMP checkout.

The package includes raw per-path Kaserer, Kaserer_fixed, fast_v1 and V2 archives, PDF q1/q5 and external-baseline Full2736 records, certification diagnostics, experiment configurations and reports. `manifest.json` and `SHA256SUMS.txt` pin every shipped byte.

The exact original historical derived identity and rate-limit JSON files were not present in the archived repository. Their public projections are explicitly labeled regenerated, not historical originals. The project source commit is identified but its private solver code is not redistributed. UR5 acceleration 5 rad/s^2 is an assumption, not a manufacturer limit. See `REPRODUCE.md`.


## Collection and export provenance

The original source populations are MotionBenchMaker Panda (650 exported from 700 attempts), MPINets Panda (1,519 exported from 1,800 attempts), and MotionBenchMaker UR5 (567 exported from 700 attempts). The four files under each `inputs/<track>/` directory are the byte-exact original `aorrtc_all.pt` cache, source-order `exported_ids.txt`, `.attempts.json` metadata, and `.export_stats.json`. The cache is a PyTorch format-version-1 mapping with a `records` list; the ID at each list position identifies that record. No paths were regenerated or reordered for this release.

The historical collector/export implementation is pinned to project source commit `07ab69fc46436c59e4e355f69193bdb13f203b4e` and VAMP source/model commit `f6de6a72e0725ba08e0829f0a34261a3f465aed7`. The project source repository is private, so the executable exporter is not redistributed here. The public package preserves the exact outputs and the metadata needed to inspect and verify them; reproducing the historical export operation itself requires authorized access to that source commit. For third-party model files, use the public VAMP checkout and do not copy them into this CC BY dataset package.
