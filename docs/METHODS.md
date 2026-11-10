# Frozen study methods

## Cohort and endpoint

160 previously exposed canonical nine-mers, HLA-A*02:01. Published competitive purified-MHC radioactivity IC50, equality observations in nM, IEDB reference 1022551 / PMID 19734234 / DOI 10.4049/jimmunol.0901607. pIC50 = 9 − log10(IC50[nM]); larger is better. Assayed terminal/protonation forms are not independently certified by the structural conventions. No new wet-lab measurement was generated.

Each start uses an independent fixed seed, NumPy permutation of the same candidate order, eight initial labels and three batches of eight selected labels. Seeds 2026110400–2026110419 are the original 20, 2026110420–2026110599 are the added 180. Within each start all 151 arms share initial candidates. Oracle visibility is software-enforced, not cryptographic secrecy. All newly selected journals were sealed before the complete development endpoint evaluation.

The tie-aware top-8 metric contains seven strict winners and two tied boundary candidates sharing one remaining slot. Count the strict winners recovered plus at most one boundary hit. This tie convention means naive 8×32/160 is not the exact random expectation for this metric. Inclusion probability for each candidate remains 32/160.

## Sources

S: Facebook ESM2 `esm2_t6_8M_UR50D`, fixed revision `c731040fcd8d73dceaa04b0a8e6329b345b0f5df`; mean of final-layer nine residue tokens, excluding special/padding tokens; 320 dimensions, no fine-tuning.

C: 2048-dimensional chiral Morgan radius-2 fingerprint, averaged acid/amide terminal conventions; 10 RDKit descriptors: molecular weight, logP, TPSA, H-bond donors/acceptors, rotatable bonds, heavy atoms, rings, fractionCSP3 and formal charge. 2058 dimensions. These are chemical descriptors, not a separate learned chemistry oracle.

K: nine residues × nearest receptor-heavy distance and heavy-atom contact fractions at 0.35, 0.45 and 0.60 nm. 36 dimensions. The structures passed retained geometry/chemistry/constraint checks; they are prepared/template-informed, not validated blind docking.

E: direct peptide-chain C vs receptor-chains A/B Coulomb and Lennard-Jones cross terms using the actual Amber NonbondedForce parameters/mixing/exceptions on saved poses, all hydrogens, NoCutoff, kJ/mol. No GBSA/solvation, entropy, restraint energy or binding-free-energy interpretation. No new dynamics was generated in this study.

## Predictors and scaling

For each fit and selected source, fit the mean/std to current visible training rows only, std <1e-8 becomes one, divide the block by square root of its dimension. The GP averages RBF kernels over selected blocks. Length 0.5/1/2 chosen by four-fold visible-only inner CV MAE with first-grid ties. Standardized target mean/std fitted within each fit; noise variance .01, diagonal jitter 1e-8. Hyperparameter selection is repeated for each visible round.

XGB concatenates scaled blocks with additional square-root block-count normalization. XGBoost 3.4.1 CPU/hist, 100 trees, depths 1/2/3 via the same visible-only four-fold MAE, learning rate .05, full row/column subsampling, L2=1, min_child_weight=1, random_state=20261103, one thread. Sixteen row bootstrap fits at the selected depth provide mean/sample covariance. GP posterior and finite-bootstrap Gaussian approximation are both uncalibrated.

Acquisition/model seed = initialization seed +1000 + zero-based round. Shared identical visible fits are cached within a start; per-arm runtime is consequently not an independent full-workload cost estimate.

## Selectors

Greedy: eight largest predicted means, stable candidate-order ties. EI: analytic improvement over best visible response, xi=0, sequential covariance-only conditioning on mean fantasies, no within-batch labels. UCB: mean+2 SD with the same conditioning. Thompson: one joint Gaussian function draw then top eight. Top8utility: 256 joint Gaussian draws, greedy increment in the sum of the eight best among visible and fantasy-selected candidates. EI targets best-one improvement; it is not guaranteed to maximize top-8 recovery.

## Reporting

All 15 source subsets ×2 predictors ×5 selectors, plus random =151 arms. 200 starts ×151 =30,200 episodes; 20 original starts reused byte-for-byte, 180 added after seeing original results. 151 summaries, 280 matched one-source contrasts, four predeclared report contrasts. Endpoint evaluated once after complete new seals. SD is across initialization seeds; MCSE=SD/sqrt(200), paired analog for contrasts. No new p-value, superiority threshold or external confirmation. Full matrix selection makes the observed top configuration a development finding.
