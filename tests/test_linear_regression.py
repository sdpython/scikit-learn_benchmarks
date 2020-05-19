"""
Unit tests.
"""
import unittest
from benchmarks.linear_model import LinearRegression_bench


class TestLinearRegression_bench(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bench = LinearRegression_bench()
        cls.bench.setup_cache()
        cls.bench.setup('dense', 'skl')

    def test_setup(self):
        self.assertFalse(self.bench is None)
        bench = self.bench
        self.assertFalse(bench.estimator is None)
        self.assertFalse(bench.estimator_onnx is None)
        self.assertFalse(bench.estimator_onnx_ort is None)
        self.assertFalse(bench.estimator_onnx_pyrt is None)

    def test_track_test_score(self):
        bench = self.bench
        s1 = bench.track_test_score('dense', 'skl')
        s2 = bench.track_test_score('dense', 'ort')
        s3 = bench.track_test_score('dense', 'pyrt')
        self.assertTrue(abs(s1 - s2) < 1e-7)
        self.assertTrue(abs(s1 - s3) < 1e-7)

    def test_time_predict(self):
        bench = self.bench
        bench.time_predict('dense', 'skl')
        bench.time_predict('dense', 'ort')
        bench.time_predict('dense', 'pyrt')

    def test_peakmem_predict(self):
        bench = self.bench
        bench.peakmem_predict('dense', 'skl')
        bench.peakmem_predict('dense', 'ort')
        bench.peakmem_predict('dense', 'pyrt')


if __name__ == "__main__":
    unittest.main()
