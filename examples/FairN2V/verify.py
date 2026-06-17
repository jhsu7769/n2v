"""
Exact Fairness Verification of a fairness classifier (NN), dataset-agnostic.
The dataset (and thus its model, data, and fairness declaration) is selected
via config['dataset'] / the DatasetAdapter loader registry in adapter.py.
Generates results for: (1) Counterfactual fairness table
                       (2) Individual fairness stacked bar charts
                       (3) Comprehensive timing table

This script can be run standalone or called from run_fairn2v.py
Standalone: uses default paths (./models, ./data, ./results/<ts>)
Runner-driven: paths come from the `config` dict passed by the runner
"""

import time
import datetime
from pathlib import Path

import numpy as np
import torch

from n2v.sets import Star, HalfSpace
from n2v.utils.verify_specification import verify_specification

from adapter import LOADERS


def perturbation_box(x, epsilon, perturbable_features):
    """Build the input Star: an epsilon-box around a (already counterfactual) sample.

    The sensitive attribute is expected to be ALREADY set by the caller via
    adapter.counterfactuals(); this function only widens the perturbable
    numerical features by +/- epsilon and clamps to the valid feature domain.

    The data is min-max normalized upstream, so every feature lives in [0, 1];
    that -- not the raw per-feature min/max -- is the correct clamp domain.
    (Clamping normalized values against raw bounds can invert lb/ub when a
    feature's raw range is not exactly [0, 1].)

    Args:
        x:          1-D feature vector, shape (n,) -- the counterfactual sample
        epsilon:    perturbation radius (0.0 -> counterfactual fairness, no widening;
                    >0.0 -> individual fairness, widen numerical features)
        perturbable_features: numerical columns epsilon is allowed to move

    Returns:
        Star: the input set (box) for reachability
    """

    x = np.asarray(x, dtype=np.float64)

    disturbance = np.zeros_like(x)
    if epsilon > 0:
        # Apply epsilon perturbation to the non-sensitive numerical features
        disturbance[perturbable_features] = epsilon

    # Clamp to the normalized feature domain [0, 1]
    lb = np.clip(x - disturbance, 0.0, 1.0)
    ub = np.clip(x + disturbance, 0.0, 1.0)

    # float32 at the boundary to match ONNX models' input dtype
    lb = lb.reshape(-1, 1).astype(np.float32)
    ub = ub.reshape(-1, 1).astype(np.float32)
    return Star.from_bounds(lb, ub)


def robustness_set(target, output_size, class_type):
    """
    Create unsafe/not robust region from a target label of a classification NN

    Args:
        target:      label idx of the given input set
        output_size: number of output classes of the NN
        class_type:  assume max, but could also be min like in ACAS Xu ('min', 'max')

    Returns:
        Hs: unsafe/not robust region as a list of HalfSpace objects
    """
    if target >= output_size:
        raise ValueError("Target idx must be less than the output size of the NN.")

    # Define HalfSpace Matrix and vector
    G = np.eye(output_size)
    G = np.delete(G, target, axis=0)
    if class_type == 'max':
        # predicted class is the largest logit: unsafe if a competitor is >= target
        G = -G
        G[:, target] = 1
    elif class_type == 'min':
        # predicted class is the smallest logit: unsafe if a competitor is <= target
        G[:, target] = -1
    else:
        raise ValueError(f"class_type must be 'min' or 'max', got {class_type!r}")

    # Create HalfSpace to define robustness specification
    return [HalfSpace(G[i, :], np.zeros((1, 1))) for i in range(G.shape[0])]


def main(config=None):
    ## Setup
    # Check if config exists (set by runner script), otherwise use defaults
    if config is None:
        # Default configuration for standalone execution
        # Paths are relative to this script's location
        script_dir = Path(__file__).resolve().parent
        ts = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        config = {
            'models_dir': script_dir / 'models',
            'data_dir': script_dir / 'data',
            'output_dir': script_dir / 'results' / ts,
            'data_file': 'adult_data.npz',
            'model_list': ['AC-1', 'AC-3'],
            'num_obs': 100,
            'random_seed': 500,
            'timeout': 600,
            'epsilon_counterfactual': [0.0],
            'epsilon_individual': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
        }

    # List all .onnx files in the models directory
    model_dir = config['models_dir']
    onnx_files = sorted(model_dir.glob('*.onnx'))

    # Create results directory if it doesn't exist
    config['output_dir'].mkdir(parents=True, exist_ok=True)

    # Initialize results storage
    results_counterfactual = []  # For counterfactual fairness (epsilon = 0)
    results_individual = []      # For individual fairness (epsilon > 0)
    results_timing = []          # For comprehensive timing table

    # List of models to process
    model_list = config['model_list']

    # Epsilon values
    # 0.0  -> counterfactual fairness (flips sensitive attribute)
    # >0.0 -> individual fairness (flips SA w/ perturbation of numerical features)
    epsilon_counterfactual = config['epsilon_counterfactual']
    epsilon_individual = config['epsilon_individual']
    epsilon = epsilon_counterfactual + epsilon_individual  # Combined for processing

    # Number of observations to test. If an explicit `sample_indices` list is
    # given (e.g. to match another tool's exact sample set), it overrides both
    # num_obs and the random selection below.
    sample_indices = config.get('sample_indices')
    num_obs = len(sample_indices) if sample_indices is not None else config['num_obs']

    # Pick the dataset adapter loader (default: adult, so existing runs are unchanged)
    loader = LOADERS[config.get('dataset', 'adult')]

    ## Loop through each model
    for onnx_path in onnx_files:
        model_name = onnx_path.stem
        if model_name not in model_list:
            continue

        # Build the dataset adapter: loads + normalizes data and wraps the model.
        # Everything dataset-specific now lives behind this single call.
        adapter = loader(config['data_dir'], onnx_path, config['data_file'])
        net = adapter.net
        X_test_loaded = adapter.X
        y_test_loaded = adapter.y

        # Count total observations
        total_obs = X_test_loaded.shape[1]

        # Cap requested sample count to what the dataset actually has (German has
        # 150 rows vs Adult's 9769); explicit sample_indices are respected as-is.
        if sample_indices is None and num_obs > total_obs:
            print(f"Requested num_obs={num_obs} exceeds dataset size {total_obs}; "
                  f"using {total_obs}.")
            num_obs = total_obs

        # Test accuracy --> verify matches with python
        total_corr = 0
        for i in range(total_obs):
            x_sample = X_test_loaded[:, i] # shape (n_features,)
            x_t = torch.tensor(x_sample, dtype=torch.float32).reshape(1, -1) # shape (1, n_features)
            predicted_labels = net.forward(x_t) # same as evaluate; returns output in (1, n_classes) tensor
            # class_type 'min' -> predicted class is the smaller logit (argmin)
            pred = int(predicted_labels.argmin()) if adapter.class_type == 'min' else int(predicted_labels.argmax())
            true_label = y_test_loaded[i]
            if pred == true_label:
                total_corr += 1
        print(f"Model: {model_name}")
        print(f"Accuracy of Model: {total_corr / total_obs}")

        ## Verification

        # First, we define the reachability options
        reach_method = 'exact'

        # Set up results
        nE = len(epsilon)
        res = np.zeros((num_obs, nE)) # robust result
        times = np.zeros((num_obs, nE)) # computation time
        # met (per-cell method tag) -- unused downstream, so omitted:
        # met = np.full((num_obs, nE), "exact", dtype=object) # method used to compute result

        # Select observations: explicit indices (to match an external tool) or random
        if sample_indices is not None:
            rand_indices = np.asarray(sample_indices, dtype=int)
        else:
            rng = np.random.default_rng(config['random_seed']) # set a seed for reproducibility
            rand_indices = rng.choice(total_obs, size=num_obs, replace=False)

        for e in range(nE):
            # Start the timer
            t_epsilon_start = time.time()

            for i in range(num_obs):
                idx = rand_indices[i]
                x_sample = X_test_loaded[:, idx]
                target = y_test_loaded[idx]

                t = time.time() # start timing the verification for each sample

                # The sample is fair only if the prediction is preserved across
                # EVERY counterfactual of the sensitive attribute (one for binary,
                # k-1 for a k-category one-hot attribute).
                is_robust = 1
                for cf in adapter.counterfactuals(x_sample):
                    IS = perturbation_box(cf, epsilon[e], adapter.perturbable_features)
                    R = net.reach(IS, method=reach_method) # generate output set

                    # Process fairness specification
                    spec = robustness_set(target, R[0].dim, adapter.class_type)

                    # one counterfactual violating the spec is enough to mark unfair
                    result = verify_specification(R, spec)
                    if result.verdict != 'UNSAT':
                        is_robust = 0
                        break

                # met[i,e] = 'exact' (met isn't used anywhere, so this is commented out for now)
                res[i, e] = is_robust
                times[i, e] = time.time() - t # store computation time

                # Check for timeout flag
                if (time.time() - t_epsilon_start > config['timeout']):
                    print(f"Timeout reached for epsilon = {epsilon[e]}: stopping verification for this epsilon.")
                    res[i+1:, e] = 2 # mark remaining as unknown
                    break # exit the inner loop after timeout

            # Get summary results
            rob = int(np.sum(res[:, e] == 1))
            not_rob = int(np.sum(res[:, e] == 0))
            unk = int(np.sum(res[:, e] == 2))
            total_time = float(np.sum(times[:, e]))
            avg_time = total_time / num_obs

            # Print results to screen
            print(f"Model: {model_name}")
            print(f"======= FAIRNESS RESULTS e: {epsilon[e]} ==========")
            print(" ")
            print(f"Number of fair samples = {rob}, equivalent to {100 * rob / num_obs}% of the samples.")
            print(f"Number of non-fair samples = {not_rob}, equivalent to {100 * not_rob / num_obs}% of the samples.")
            print(f"Number of unknown samples = {unk}, equivalent to {100 * unk / num_obs}% of the samples.")
            print(" ")
            print(f"It took a total of {total_time} seconds to compute the verification results, "
                  f"an average of {avg_time} seconds per sample")

            # Collect results based on epsilon type
            if epsilon[e] == 0.0:
                # Counterfactual fairness results
                results_counterfactual.append({
                    'model': model_name,
                    'fair_pct': 100 * rob / num_obs,
                    'unfair_pct': 100 * not_rob / num_obs,
                })
            else:
                # Individual fairness results (for stacked bar chart)
                results_individual.append({
                    'model': model_name,
                    'epsilon': epsilon[e],
                    'fair_pct': 100 * rob / num_obs,
                    'unfair_pct': 100 * not_rob / num_obs,
                    'unknown_pct': 100 * unk / num_obs,
                })

            # Timing results (all epsilon values)
            results_timing.append({
                'model': model_name,
                'epsilon': epsilon[e],
                'total_time': total_time,
                'avg_time': avg_time,
            })

    ## Save results to CSV files
    # Get the current timestamp using datetime
    timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M%S')

    # --- Save Counterfactual Fairness Results ---
    # For Table: Counterfactual Fairness (epsilon = 0)
    csv_counterfactual = config['output_dir'] / f"counterfactual_{timestamp}.csv"
    with open(csv_counterfactual, "w", encoding="utf-8") as file:
        file.write("Model,FairPercent,UnfairPercent\n")
        for row in results_counterfactual:
            file.write(f"{row['model']},{row['fair_pct']},{row['unfair_pct']}\n")
    print(f"Counterfactual results saved to {csv_counterfactual}")

    # --- Save Individual Fairness Results ---
    # For Stacked Bar Charts (epsilon > 0)
    csv_individual = config['output_dir'] / f"individual_{timestamp}.csv"
    with open(csv_individual, "w", encoding="utf-8") as file:
        file.write("Model,Epsilon,FairPercent,UnfairPercent,UnknownPercent\n")
        for row in results_individual:
            file.write(f"{row['model']},{row['epsilon']},{row['fair_pct']},{row['unfair_pct']},{row['unknown_pct']}\n")
    print(f"Individual results saved to {csv_individual}")

    # --- Save Comprehensive Timing Table ---
    csv_timing = config['output_dir'] / f"timing_{timestamp}.csv"
    with open(csv_timing, "w", encoding="utf-8") as file:
        file.write("Model,Epsilon,TotalTime,AvgTimePerSample\n")
        for row in results_timing:
            file.write(f"{row['model']},{row['epsilon']},{row['total_time']},{row['avg_time']}\n")
    print(f"Timing results saved to {csv_timing}")

    print(" ")
    print("======= FairNNV VERIFICATION COMPLETE ==========")
    print("Generated files:")
    print(f"  1. {csv_counterfactual} (for counterfactual fairness table)")
    print(f"  2. {csv_individual} (for individual fairness stacked bar charts)")
    print(f"  3. {csv_timing} (for comprehensive timing table)")

if __name__ == "__main__":
    main()