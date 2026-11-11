# PepLab Bench

**[Read the paper](https://doi.org/10.5281/zenodo.24143298) · [Complete evidence](https://doi.org/10.5281/zenodo.24142178)**

## Better peptide decisions by combining complementary information

**PepLab brings sequence, chemistry and structural information together to help choose the next experimental batch.** This repository contains the code, peptide records, saved comparisons and model integration interfaces.

**A measured example:** with the same GP predictor, greedy selector and 32-measurement-reveal budget, adding contact features to sequence + chemistry increased mean top-eight discoveries from **2.815 to 3.265 — 16% more** in the saved HLA development study. Source combinations matter; adding every source was not uniformly better.

Read the [PepLab overview](docs/PepLab-overview.pdf) or the [workflow and evidence summary](docs/PRODUCT_OVERVIEW.md). Cost-aware candidate/layer allocation is the next development step; this benchmark provides retrospective source-complementarity evidence.

**Which information source helps which model and selection rule?** A transparent peptide-selection benchmark with plug-in interfaces and a complete saved development study.

**160 HLA-A*02:01 nine-mers · 151 configurations · 200 shared starts · 30,200 episodes.**

Each episode starts with eight measured candidates, then selects three batches of eight. All methods share the same initial eight per start. Measurements are archived experimental IC50 values, revealed from a cached oracle. They are not new assays.

## Start in two minutes

Python 3.10+; viewing and contract tests require no numerical dependencies:

```bash
python -m pip install -e .
peplab-bench verify --data data
peplab-bench results --data data --html results.html
python -m unittest discover -s tests -v
```

Open `results.html` to compare every method. The command reads saved summaries and performs no fitting. The [complete versioned evidence archive](https://doi.org/10.5281/zenodo.24142178) is public on Zenodo: all 30,200 episodes, original selection journals, raw numerical-qualification outputs, accepted structures and serialized systems, cohort-selection provenance and archived source. The full 382,822,111-byte download and all 12 component hashes were verified without authentication on 11 November 2026. See [the release index](docs/EVIDENCE_RELEASE.json).

## Findings at a glance

| Source / predictor / selector | Mean top-8 hits after 32 reveals |
|---|---:|
| Sequence + chemistry + contacts / GP / greedy | **3.265** |
| Chemistry + direct energy / XGB bootstrap / greedy | 2.970 |
| Chemistry / XGB bootstrap / greedy | **2.920** |
| Sequence + chemistry / GP / greedy | 2.815 |
| Sequence + chemistry + contacts + energy / GP / greedy | **2.765** |
| Chemistry / XGB bootstrap / top-8 utility | 2.795 |
| Random | 1.780 |

The observed leader recovered 1.485 more hits than random and 0.345 more than the best cheap-only combination. It was selected on development results; this is not a prospective superiority estimate. Adding a source sometimes helps and sometimes harms. All 151 summaries and 280 matched source-addition contrasts are included, not just the leader.

The study was extended from 20 to 200 starts after inspecting the original results. The original 3,020 episodes remain unchanged; 27,180 were added. Standard deviations describe initialization variability and MCSE describes precision over random starts on this same dataset. These are not 200 independent biological experiments.

The revised four-page balanced atlas in the [evidence release](https://doi.org/10.5281/zenodo.24142178) shows the strongest cheap comparator alongside the +0.450 matched contact gain, +0.345 descriptive best-versus-best difference and −0.500 adverse energy addition. Earlier figures remain archived in this repository as historical presentation evidence.

## Three independent axes

| Axis | Options |
|---|---|
| Source | S: 320 fixed ESM2 features; C: 2,058 chemistry features; K: 36 prepared contact features; E: 2 direct interaction-energy features |
| Predictor | GP; XGBoost with 16 bootstrap members |
| Selector | Greedy; expected improvement; UCB; Thompson sampling; top-8 utility |

All 15 nonempty source subsets × 2 predictors × 5 selectors, plus random. **Greedy is a selector, not an information source.** Contacts and energies are distinct source blocks. The structural inputs are template-informed prepared poses; earlier blind-docking recovery failures are part of the provenance. Direct energies are proxies, not binding free energies.

## Bring your model

Implement a predictor accepting `visible` labels, unlabeled feature blocks and ordered query IDs. Return `Prediction(ids, means, covariance, observation_noise, uncertainty_label)`. A mean-only model supports greedy selection. Bayesian acquisitions are unavailable until a valid covariance is supplied. Uncertainty must be described honestly; the reference GP and finite-bootstrap uncertainties are uncalibrated.

See [integration guide](docs/INTEGRATION.md), [methods](docs/METHODS.md), [data dictionary](docs/DATA.md) and [release provenance](docs/PROVENANCE.md). Replace a predictor, selector or source independently. A mock integration example is in `examples/custom_predictor.py`.

## Opt in to a fresh single-episode replay

This performs actual CPU model fits and is separate from viewing the completed study:

```bash
python -m pip install -e '.[models]'
peplab-bench replay --data data --sources SCK --predictor GP \
  --selector greedy --seed 2026110420 --out my-new-episode
```

The numerical functions retain the original ASTs; the portable wrapper and adapter layer are new. They have contract tests and the bounded numerical qualification below; the entire 30,200-episode study was not rerun. A new episode never appends to the frozen results. Outputs use exclusive creation and record selections before reveals. Cached values are isolated by software flow, not by an OS security boundary.

To run the complete factorial for one new replication start, use `peplab-bench benchmark --data data --starts 1 --out my-new-suite`. Set `--starts 200` and an explicit `--max-seconds` only when you intend the full workload. This new portable runner seals every planned episode before asking its separate oracle for the full response table, fails closed on an incomplete suite and writes a separate replication table. It does not launch cloud resources or overwrite the archived study. Its full-suite orchestration has a mocked global-barrier contract test. The 12-episode qualification covers the installed episode path, not a fresh full-suite numerical rerun.

## Numerical qualification and complete evidence

A clean, noneditable installation of public commit `29d4f50897b2642a9383a9e28eee61c9716f88bc` reproduced **12 historical episodes and 36 round contexts**, covering both learners, all five selectors and all four source blocks. Selected identities, batch order and final recovery counts matched exactly; historical mean and marginal-SD differences were below the declared 1e-10 tolerance. Full covariance matched a separate-process invocation of the original core; historical full covariance was not recorded. This is a bounded portability qualification, not a rerun of all 30,200 episodes.

The [Zenodo archive](https://doi.org/10.5281/zenodo.24142178) includes `PepLab-numerical-qualification-output.zip`, the qualification scripts/fixtures, the four-page balanced comparison atlas v4, and original 20/additional 180/combined 200 tables. The historical six-figure atlas is retained separately. The evidence archive and manuscript revision 6.4 are public. Author declarations use Metastate Bio. Journal submission is separate.

## Scope

This release supports reproducible retrospective source/model/selector comparison on one finite, previously exposed HLA cohort. Source features were available upfront. It does not establish cost-aware allocation across evaluation layers, affinity prediction for arbitrary peptides, prospective wet-lab benefit, trillion-candidate search performance, or calibrated uncertainty. The reserved 40 outcomes are excluded from every release file. Their broader historical exposure is not certified by this extension.

## Citation and reuse

Author: **Oğuzcan Unver**. Original code: Apache-2.0. Original manuscript/documentation: CC BY 4.0. IEDB-derived data: attribution and source terms in [third-party notices](THIRD_PARTY_NOTICES.md). Cite the original experiment (DOI `10.4049/jimmunol.0901607`), IEDB and the versioned PepLab release. No pretrained weights or third-party runtime binaries are bundled.

AI tools assisted software development, execution support, analysis presentation and manuscript drafting. Human author review and responsibility are required. Publication and peer-review status are stated in `docs/PUBLICATION.md`.
