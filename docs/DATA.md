# Data dictionary

* `peptides160.csv`: stable candidate ID, sequence, target, archived IC50[nM], pIC50, equality qualifier, assay method/source DOI, tie-aware top8 status, prepared PDB hash and direct-energy proxies. No new wet-lab outcomes.
* `sequence-chemistry.npz`: S and C arrays; exact original `.npy` bytes, renamed entries. No pretrained weights or reserved candidates.
* `structural-energy.json`: ordered IDs; K160×36, E160×2; units and prepared-PDB hash linkage. Serialized original full-precision floats; no recalculated feature values.
* `development-labels.json`: response mapping used only by the oracle during a fresh replay. Intentionally public archived development data; not a hidden external validation set.
* `cohort.json`: order and dimensions. `protocol.json`: scientific protocol allowlist, with hash of original private protocol. Infrastructure/identity/financial metadata and all reserved IDs/responses are excluded.
* `all151-methods.csv`: 200-start mean/SD/MCSE, recall, four-budget mean curves, old20 means and runtime/cache-affected fit counts.
* `all280-matched-source-additions.csv`: same predictor/selector, one added source, within-start contrasts, wins/ties/losses and paired MCSE.
* `predeclared-comparisons.csv`: four previously fixed report contrasts.
* `checksums.json`: SHA256 for every public data file except itself.

The larger dataset archive contains the 30,200 episode table, 24,160 peptide×method selection-frequency rows, the exact first-new-start journal and its score table (64,800 rows). Random actions are separate and have no fabricated prediction scores. The original full-precision final result, all 180 new seed journals and seals, original20 provenance, and prepared structures are archived separately for audit.

Numerical tables contain published experimental data, computational predictions and derived features in distinct columns. A label joined to a saved prediction after evaluation is not evidence it was visible during selection. Check the pre-reveal journals and seals.

## Complete evidence deposit

All larger archives are now available together in [Zenodo record 24142178](https://zenodo.org/records/24142178), DOI [10.5281/zenodo.24142178](https://doi.org/10.5281/zenodo.24142178). The 382,822,111-byte ZIP has SHA256 `86d4267564d1bf47f1b8c7346e248d61a875ca6d55b26e8f72e773d05f8e5be3`; its top-level `MANIFEST.json` lists the 12 component sizes and hashes. Public download and all component hashes were verified on 11 November 2026.

In addition to the episode/data and original-journal archives, it contains the separate raw numerical-qualification output ZIP; the 160 serialized OpenMM systems, topology maps and full-precision accepted coordinates; scientific preparation/energy source; and the public cohort-selection provenance export. The export retains development identities and sequence-hash selection rules while excluding reserved identities and measurements. The four-page balanced atlas v4 and historical six-figure atlas v2 are distinct preserved presentation documents.
