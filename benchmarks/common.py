import os
from multiprocessing import cpu_count
import json
import timeit
import pickle
import itertools
from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import is_classifier


ONNX_TARGET_OPSET = 12


def get_from_config():
    current_path = os.path.dirname(os.path.realpath(__file__))

    config_path = os.path.join(current_path, 'config.json')
    with open(config_path, 'r') as config_file:
        config_file = ''.join(line for line in config_file
                              if line and '//' not in line)
        config = json.loads(config_file)

    profile = config['profile']

    n_jobs_vals = config['n_jobs_vals']
    if not n_jobs_vals:
        n_jobs_vals = list(range(1, 1 + cpu_count()))

    save_estimators = config['save_estimators']
    save_folder = os.getenv('ASV_COMMIT', 'new')[:8]

    if save_estimators:
        save_path = os.path.join(current_path, 'cache',
                                 'estimators', save_folder)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    base_folder = config['base_folder']

    bench_predict = config['bench_predict']
    bench_predictproba = config['bench_predictproba']
    bench_transform = config['bench_transform']
    bench_onnx = config['bench_onnx']
    runtimes = config['runtimes']

    return (profile, n_jobs_vals, save_estimators, save_folder, base_folder,
            bench_predict, bench_predictproba, bench_transform, bench_onnx,
            runtimes)


def get_estimator_path(benchmark, folder, params, save=False):
    if params[-1] not in {'ort', 'pyrt', 'skl'}:
        raise ValueError("Unexpected parameters {}.".format(params))
    params = params[:-1]
    folder = os.path.join('estimators', folder) if save else 'tmp'
    f_name = (benchmark.__class__.__name__[:-6]
              + '_estimator_' + '_'.join(list(map(str, params))) + '.pkl')
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'cache', folder, f_name)
    return path


def get_data_path(benchmark, params):
    if params[-1] not in {'ort', 'pyrt', 'skl'}:
        raise ValueError("Unexpected parameters {}.".format(params))
    params = params[:-1]
    f_name = (benchmark.__class__.__name__[:-6]
              + '_data_' + '_'.join(list(map(str, params))) + '.pkl')
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'cache', 'tmp', f_name)
    return path


def clear_tmp():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'cache', 'tmp')
    list(map(os.remove, (os.path.join(path, f)
                         for f in os.listdir(path) if f != '.gitignore')))


class Benchmark:
    timer = timeit.default_timer  # wall time
    processes = 1
    timeout = 500

    (profile, n_jobs_vals, save_estimators, save_folder, base_folder,
     bench_predict, bench_predictproba, bench_transform,
     bench_onnx, runtimes) = get_from_config()

    if profile == 'fast':
        warmup_time = 0
        repeat = 1
        number = 1
        min_run_count = 1
        data_size = 'small'
    elif profile == 'regular':
        warmup_time = 1
        repeat = (3, 100, 30)
        data_size = 'small'
    elif profile == 'large_scale':
        warmup_time = 1
        repeat = 3
        number = 1
        data_size = 'large'


class Estimator(ABC):

    @abstractmethod
    def is_benchmark(self):
        return False
    
    def setup_cache(self):
        clear_tmp()

        param_grid = list(itertools.product(*self.params))
        if self.param_names[-1] != 'runtime':
            raise NameError(
                "Last parameter must be the runtime ({}).".format(self.param_names))

        for params in param_grid:
            if params[-1] != 'skl':
                continue
            data, estimator, _ = self.setup_cache_(params) or (None, None, None)
            if data is None:
                continue

            data_path = get_data_path(self, params)
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)

            X, _, y, _ = data
            estimator.fit(X, y)

            est_path = get_estimator_path(self, Benchmark.save_folder,
                                          params, Benchmark.save_estimators)
            with open(est_path, 'wb') as f:
                pickle.dump(estimator, f)

    def setup(self, *params):
        if hasattr(self, 'setup_'):
            self.setup_(params)

        data_path = get_data_path(self, params)
        with open(data_path, 'rb') as f:
            self.X, self.X_val, self.y, self.y_val = pickle.load(f)
        if Benchmark.bench_onnx:
            self.X32 = self.X.astype(np.float32)

        est_path = get_estimator_path(self, Benchmark.save_folder,
                                      params, Benchmark.save_estimators)
        with open(est_path, 'rb') as f:
            self.estimator = pickle.load(f)

        if Benchmark.bench_onnx:
            self._setup_onnx()
        else:
            raise NotImplementedError()

        self.make_scorers()

    def _setup_to_onnx(self):
        from skl2onnx import to_onnx
        if is_classifier(self.estimator):
            self.estimator_onnx = to_onnx(
                self.estimator, self.X[:1],
                options={id(self.estimator): {'zipmap': False}},
                target_opset=ONNX_TARGET_OPSET)
        else:
            self.estimator_onnx = to_onnx(
                self.estimator, self.X[:1],
                target_opset=ONNX_TARGET_OPSET)
        if self.estimator_onnx is None:
            raise RuntimeError("estimator_onnx cannot be None")

    def _setup_onnx(self):
        try:
            self._setup_to_onnx()
        except RuntimeError as e:
            self.estimator_onnx = None
        if self.estimator_onnx is not None:
            from onnxruntime import InferenceSession
            try:
                self.estimator_onnx_ort = InferenceSession(
                    self.estimator_onnx.SerializeToString())
            except RuntimeError as e:
                self.estimator_onnx_ort = None

            try:
                from mlprodict.onnxrt import OnnxInference
            except ImportError:
                self.estimator_onnx_pyrt = None
                return
            try:
                self.estimator_onnx_pyrt = OnnxInference(
                    self.estimator_onnx, runtime="python_compiled")
            except RuntimeError as e:
                self.estimator_onnx_pyrt = None
        else:
            self.estimator_onnx_ort = None
            self.estimator_onnx_pyrt = None

    def time_fit(self, *args):
        self.estimator.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        self.estimator.fit(self.X, self.y)

    def track_train_score(self, *args):
        if isinstance(self, Predictor):
            y_pred = self.estimator.predict(self.X)
        else:
            y_pred = None
        return float(self.train_scorer(self.y, y_pred))

    def track_test_score(self, *args):
        runtime = args[-1]
        if runtime == 'skl':
            if isinstance(self, Predictor):
                y_val_pred = self.estimator.predict(self.X_val)
            else:
                y_val_pred = None            
            return float(self.test_scorer(self.y_val, y_val_pred))

        if Benchmark.bench_onnx:

            if runtime == 'ort':
                if (isinstance(self, Predictor) and
                        self.estimator_onnx_ort is not None):
                    y_val_pred = self.estimator_onnx_ort.run(
                        None, {'X': self.X_val.astype(np.float32)})[0]
                else:
                    y_val_pred = None
                return float(self.test_scorer(self.y_val, y_val_pred))

            if runtime == 'pyrt':
                if (isinstance(self, Predictor) and
                        self.estimator_onnx_pyrt is not None):
                    res = self.estimator_onnx_pyrt.run(
                        {'X': self.X_val.astype(np.float32)})
                    for name in ['variable', 'output_label', 'label']:
                        if name in res:
                            y_val_pred = res[name]
                            break
                else:
                    y_val_pred = None
                return float(self.test_scorer(self.y_val, y_val_pred))


class Predictor(ABC):

    @abstractmethod
    def is_benchmark(self):
        return False

    def _common_predict_(self, *args):
        if len(args) == 0:
            raise ValueError("No arguments.")
        runtime = args[-1]
        if Benchmark.bench_predict:
            if runtime == 'skl':
                self.estimator.predict(self.X)
            elif runtime == 'ort':
                if self.estimator_onnx_ort is not None:
                    self.estimator_onnx_ort.run(
                        None, {'X': self.X32})[0]
            elif runtime == 'pyrt':
                if self.estimator_onnx_pyrt is not None:
                    self.estimator_onnx_pyrt.run(
                        {'X': self.X32})
            else:
                raise ValueError("Unknown runtime '{}'.".format(runtime))

    def time_predict(self, *args):
        self._common_predict_(*args)

    def peakmem_predict(self, *args):
        self._common_predict_(*args)


class Classifier(Predictor, ABC):

    @abstractmethod
    def is_benchmark(self):
        return False
    
    if Benchmark.bench_predictproba:

        def _common_predictproba_(self, *args):
            runtime = args[-1]            
            if runtime == 'skl':
                self.estimator.predict_proba(self.X)
            elif runtime == 'ort':
                if Benchmark.bench_onnx:
                    if self.estimator_onnx_ort is not None:
                        self.estimator_onnx_ort.run(
                            None, {'X': self.X.astype(np.float32)})[1]
                    else:
                        raise RuntimeError(
                            "estimator_onnx_ort could not be created.")
            elif runtime == 'pyrt':
                if Benchmark.bench_onnx:
                    if self.estimator_onnx_pyrt is not None:
                        self.estimator_onnx_pyrt.run(
                            {'X': self.X.astype(np.float32)})
                    else:
                        raise RuntimeError(
                            "estimator_onnx_pyrt could not be created.")
            else:
                raise ValueError("Unexpected runtime '{}'.".format(runtime))
                
        def peakmem_predictproba(self, *args):
            self._common_predictproba_(*args)

        def time_predictproba(self, *args):
            self._common_predictproba_(*args)


class Transformer(ABC):

    @abstractmethod
    def is_benchmark(self):
        return False
    
    if Benchmark.bench_transform:

        def _common_transform_(self, *args):
            runtime = args[-1]
            if runtime == 'skl':
                self.estimator.transform(self.X)
            elif runtime == 'ort':
                if Benchmark.bench_onnx:
                    if self.estimator_onnx_ort is not None:
                        self.estimator_onnx_ort.run(
                            None, {'X': self.X.astype(np.float32)})
                    else:
                        raise RuntimeError("estimator_onnx_ort could not be created.")
            elif runtime == 'pyrt':
                if Benchmark.bench_onnx:
                    if self.estimator_onnx_pyrt is not None:
                        self.estimator_onnx_pyrt.run(
                            None, {'X': self.X.astype(np.float32)})
                    else:
                        raise RuntimeError("estimator_onnx_ort could not be created.")
            else:
                raise ValueError("Unexpected runtime '{}'.".format(runtime))

        def peakmem_transform_ort(self, *args):
            self._common_transform_(*args)

        def time_transform_pyrt(self, *args):
            self._common_transform_(*args)
