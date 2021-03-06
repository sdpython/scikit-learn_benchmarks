from sklearn.svm import SVC

from .common import Benchmark, Estimator, Classifier
from .datasets import _synth_classification_dataset
from .utils import optimal_cache_size, make_gen_classif_scorers


class SVC_bench(Benchmark, Estimator, Classifier):
    """Benchmarks for SVC."""

    param_names = ['kernel', 'runtime']
    params = (['linear', 'poly', 'rbf', 'sigmoid'], ['skl', 'ort', 'pyrt'])

    def is_benchmark(self):
        return True

    def setup_cache(self):
        super().setup_cache()

    def setup_cache_(self, params):
        kernel, runtime = params

        data = _synth_classification_dataset()

        estimator = SVC(C=1,
                        cache_size=optimal_cache_size(data[0].shape[1]),
                        max_iter=100,
                        tol=1e-16,
                        kernel=kernel,
                        random_state=0,
                        shrinking=True,
                        gamma='scale',
                        probability=True)

        return data, estimator, runtime

    def make_scorers(self):
        make_gen_classif_scorers(self)
