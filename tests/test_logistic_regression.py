"""
Unit tests.
"""
import unittest
from benchmarks.linear_model import LogisticRegression_bench


class TestLogisticRegression_bench(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bench = LogisticRegression_bench()
        cls.bench.setup_cache()
        cls.bench.setup('dense', 'lbfgs', 1, 'skl')

    def test_setup(self):
        self.assertFalse(self.bench is None)
        bench = self.bench
        self.assertFalse(bench.estimator is None)
        self.assertFalse(bench.estimator_onnx is None)
        self.assertFalse(bench.estimator_onnx_ort is None)
        self.assertFalse(bench.estimator_onnx_pyrt is None)

    def test_track_test_score(self):
        bench = self.bench
        s1 = bench.track_test_score('dense', 'lbfgs', 1, 'skl')
        s2 = bench.track_test_score('dense', 'lbfgs', 1, 'skl')
        s3 = bench.track_test_score('dense', 'lbfgs', 1, 'skl')
        self.assertTrue(abs(s1 - s2) < 1e-7)
        self.assertTrue(abs(s1 - s3) < 1e-7)

    def test_time_predict(self):
        bench = self.bench
        bench.time_predict('dense', 'lbfgs', 1, 'skl')
        bench.time_predict('dense', 'lbfgs', 1, 'ort')
        bench.time_predict('dense', 'lbfgs', 1, 'pyrt')
        
    def test_time_predictproba(self):
        bench = self.bench
        bench.time_predictproba('dense', 'lbfgs', 1, 'skl')
        bench.time_predictproba('dense', 'lbfgs', 1, 'ort')
        bench.time_predictproba('dense', 'lbfgs', 1, 'pyrt')

    def test_peakmem_predict(self):
        bench = self.bench
        bench.time_predict('dense', 'lbfgs', 1, 'skl')
        bench.time_predict('dense', 'lbfgs', 1, 'ort')
        bench.time_predict('dense', 'lbfgs', 1, 'pyrt')

    def test_peakmem_predictproba(self):
        bench = self.bench
        bench.time_predictproba('dense', 'lbfgs', 1, 'skl')
        bench.time_predictproba('dense', 'lbfgs', 1, 'ort')
        bench.time_predictproba('dense', 'lbfgs', 1, 'pyrt')


if __name__ == "__main__":
    unittest.main()
