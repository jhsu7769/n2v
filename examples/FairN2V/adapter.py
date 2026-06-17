"""
DatasetAdapter: a dataset-agnostic container for everything the fairness
verification loop needs to know about a dataset.

The adapter holds *nouns* (facts about the data): the samples, the trained
network, per-feature clamps, and a declaration of which feature is sensitive
and how the model picks its class. The verification loop and the fairness
*definitions* (the verbs) read these facts instead of hardcoding them.

This module provides the DatasetAdapter container, a per-dataset loader for
each supported dataset (load_adult, load_german, ...), and a LOADERS registry
that the verification driver selects from via config['dataset']. Adding a
dataset in the shared npz X/y format is a thin loader plus one registry entry.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from n2v.nn import NeuralNetwork
from n2v.utils.model_loader import load_onnx
from n2v.utils.model_preprocessing import strip_final_softmax


@dataclass
class DatasetAdapter:
    name: str                       # human-readable dataset name, e.g. 'adult'

    # --- data (loaded + normalized) ---
    X: np.ndarray                   # features, shape (n_features, n_samples)
    y: np.ndarray                   # int labels, shape (n_samples,)
    min_values: np.ndarray          # per-feature lower clamp, shape (n_features,)
    max_values: np.ndarray          # per-feature upper clamp, shape (n_features,)

    # --- trained model ---
    net: NeuralNetwork              # softmax already stripped, ready for reach()

    # --- fairness declaration ---
    sensitive_features: list        # column indices of the protected attribute
    perturbable_features: list      # numerical columns epsilon is allowed to move
    sensitive_encoding: str         # 'binary' | 'onehot'
    output_size: int                # number of output classes
    class_type: str                 # 'min' or 'max' -- how this net picks the class

    def counterfactuals(self, x):
        """Return the counterfactual versions of x w.r.t. the sensitive attribute.

        A counterfactual is x with ONLY the protected attribute changed to an
        alternative value; non-sensitive features are untouched.

        - 'binary'  -> exactly one counterfactual: the flipped value.
        - 'onehot'  -> one counterfactual per *other* category (k-1 of them),
                       each a valid one-hot vector over the sensitive columns.

        The number of counterfactuals therefore depends on the encoding; the
        caller treats a sample as fair only if the prediction is preserved
        across ALL returned counterfactuals.

        Args:
            x: 1-D feature vector, shape (n_features,)

        Returns:
            list[np.ndarray]: counterfactual copies of x (float64)
        """
        x = np.asarray(x, dtype=np.float64)
        cols = list(self.sensitive_features)

        if self.sensitive_encoding == 'binary':
            cf = x.copy()
            cf[cols] = 1.0 - cf[cols]
            return [cf]

        if self.sensitive_encoding == 'onehot':
            out = []
            for c in cols:
                if x[c] == 1.0:
                    continue  # this is the sample's actual category, not a counterfactual
                cf = x.copy()
                cf[cols] = 0.0  # clear the one-hot block...
                cf[c] = 1.0     # ...and activate the alternative category
                out.append(cf)
            return out

        raise ValueError(
            f"sensitive_encoding must be 'binary' or 'onehot', got {self.sensitive_encoding!r}"
        )


def _load_npz_adapter(name, data_dir, model_path, data_file, **declaration):
    """Shared loader for npz datasets in the FairNNV `X`/`y` format.

    Loads + min-max normalizes the data and wraps the ONNX model, then stamps
    the per-dataset fairness `declaration` onto a DatasetAdapter. All datasets
    that share this layout (Adult, German, ...) reuse this; each only differs
    in the declaration. Adding such a dataset is therefore one thin wrapper.

    Args:
        name:        dataset name
        data_dir:    directory containing the .npz data file
        model_path:  path to the .onnx model to verify
        data_file:   name of the .npz file inside data_dir
        declaration: the fairness fields (sensitive_features, perturbable_features,
                     sensitive_encoding, output_size, class_type)

    Returns:
        DatasetAdapter: populated, ready for the verification loop
    """
    data_dir = Path(data_dir)

    # --- load data ---
    data = np.load(data_dir / data_file)
    X = data['X']
    y = data['y']

    X_test_loaded = X.T
    y_test_loaded = y[:, 0].astype(int)

    # --- normalize ---
    min_values = X_test_loaded.min(axis=1)
    max_values = X_test_loaded.max(axis=1)

    # Ensure no division by zero for constant features
    variable_features = max_values - min_values > 0
    min_values[~variable_features] = 0.0  # avoids changing constant features
    max_values[~variable_features] = 1.0  # avoids division by zero

    X_test_loaded = (X_test_loaded - min_values[:, None]) / (max_values - min_values)[:, None]

    # --- load + wrap the model ---
    netONNX = load_onnx(model_path)
    netONNX = strip_final_softmax(netONNX)  # drop trailing softmax for Star reachability
    net = NeuralNetwork(netONNX)

    return DatasetAdapter(
        name=name,
        X=X_test_loaded,
        y=y_test_loaded,
        min_values=min_values,
        max_values=max_values,
        net=net,
        **declaration,
    )


def load_adult(data_dir, model_path, data_file='adult_data.npz'):
    """Adult-Income (UCI). Sensitive attribute: sex (binary, column 8)."""
    return _load_npz_adapter(
        'adult', data_dir, model_path, data_file,
        sensitive_features=[8],
        perturbable_features=[0, 9, 10, 11],
        sensitive_encoding='binary',
        output_size=2,
        class_type='min',
    )


# Registry: dataset key -> loader. The verification driver picks one via
# config['dataset'], so adding a dataset = write a loader + add one line here.
LOADERS = {
    'adult': load_adult,
}
