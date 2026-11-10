# Add a model, selector or source

The three interfaces are independent. A new model does not require a new acquisition rule. A new source does not require a new model.

1. Predictor: receives only the visible response mapping, query IDs and selected feature blocks. Fit transformations and tune hyperparameters on visible rows only, including within CV. Return means in original pIC50 units and optionally joint covariance. Never read the assay table from inside the predictor.
2. Selector: receives prediction objects and visible responses, returns eight distinct unseen IDs. Stable candidate order resolves ties. Mask EI/UCB/TS/top8utility for mean-only predictions; do not invent uncertainty. Covariance must be symmetric, finite and positive semidefinite to numerical tolerance; the reference sampler rejects invalid factorization.
3. Source: provide a label-free matrix with the exact cohort order, feature names, units, generation code/version, input hashes and an availability mask. If not available, mask the action or predeclare an imputation rule. Do not use outcome information to select a descriptor or a pose.

`engine.run_episode` accepts predictor and selector callables. Their shapes are defined in `contracts.py`. The provided mock example demonstrates the interface without model fitting. `FrozenPredictor` and `FrozenSelector` adapt the original recipe.

The cached oracle counts one candidate assay reveal as one budget unit. It rejects duplicate, unknown, oversized or late reveals and refuses full-truth access before sealing. This is not a financial cost model. A future cost-aware experiment must predeclare units and compute prices, mask unavailable actions, select both candidate and layer, and compare under the same total budget. No such performance claim is made here.

Report all chosen configurations, failed runs and complete histories; do not select an apparent winner and then describe the same records as untouched validation. Independent researcher models may have trained on overlapping public IEDB measurements: disclose this provenance explicitly.
