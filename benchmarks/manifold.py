from sklearn.manifold import TSNE

from .common import Benchmark, Estimator
from .datasets import _digits_dataset


class TSNE_bench(Benchmark, Estimator):
    """
    Benchmarks for t-SNE.
    """

    param_names = ['method', 'runtime']
    params = (['exact', 'barnes_hut'], ['skl', 'ort', 'pyrt'])

    def is_benchmark(self):
        return True

    def setup_cache(self):
        super().setup_cache()

    def setup_cache_(self, params):
        method, runtime = params

        n_samples = 500 if method == 'exact' else None
        data = _digits_dataset(n_samples=n_samples)
        estimator = TSNE(random_state=0, method=method)
        return data, estimator, runtime

    def make_scorers(self):
        self.train_scorer = lambda _, __: self.estimator.kl_divergence_
        self.test_scorer = lambda _, __: self.estimator.kl_divergence_
