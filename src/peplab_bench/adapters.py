"""The original numerical recipe behind portable predictor/selector contracts."""
from .contracts import Prediction, greedy

class FrozenPredictor:
    def __init__(self, ids, features, kind, deadline):
        from .frozen import replay
        self.ids, self.features, self.kind, self.core = tuple(ids), features, kind, replay
        self.lookup = {v: i for i, v in enumerate(ids)}
        replay.g.DEADLINE = deadline
        replay.CACHE.clear()

    def __call__(self, visible, query, sources, seed):
        import numpy as np
        train = np.array([self.lookup[i] for i in visible])
        q = np.array([self.lookup[i] for i in query])
        (mu, covariance, noise, settings), _ = self.core.model(
            self.features, train, np.array(list(visible.values())), q, list(sources), self.kind, seed)
        return Prediction(tuple(query), tuple(mu.tolist()), tuple(tuple(v) for v in covariance.tolist()),
                          float(noise), "UNCALIBRATED_GP_POSTERIOR" if self.kind == "GP" else "UNCALIBRATED_GAUSSIAN_BOOTSTRAP_APPROXIMATION")

class FrozenSelector:
    def __init__(self, kind): self.kind = kind
    def __call__(self, prediction, visible_values, batch_size, seed):
        if batch_size != 8: raise ValueError("The frozen acquisition recipe requires batch size eight")
        prediction.validate(prediction.ids)
        if self.kind == "greedy": return greedy(prediction, batch_size)
        if prediction.covariance is None: raise ValueError("This acquisition requires covariance; action unavailable")
        import numpy as np
        from .frozen.replay import select
        indices = select(np.array(prediction.means), np.array(prediction.covariance),
                         prediction.observation_noise, np.array(visible_values), self.kind, seed)
        return [prediction.ids[i] for i in indices]
