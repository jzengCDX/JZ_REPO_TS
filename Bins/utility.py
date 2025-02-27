import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from pykalman import KalmanFilter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import product
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score


def _prepare_training(data_path = '../../../Data_processing/Outputs/train.csv', features = ['aval_AlloMap',  'aval_AlloSure', 'ady_dna'], exclusion = True, split = 0.8, patient_level = True):

    random.seed(42)
    # Load the cleaned dataset
    data_cleaned = pd.read_csv(data_path)

    # Exclusion or inclusion only rejection samples
    if exclusion == True:
        data_cleaned = data_cleaned.loc[data_cleaned['usubjid'].isin(data_cleaned[data_cleaned['label'] == 1]["usubjid"].unique())]
    
    data_cleaned.dropna(inplace=True)

    # Preprocess the data 
    features = features
    target = 'label'

    # Group data by 'usubjid' to maintain sequence structure
    grouped_data = data_cleaned.groupby('usubjid')
    shuffled_df = grouped_data.sample(frac=1, random_state=42).sort_values(['usubjid', 'ady_dna'])
    grouped_data = shuffled_df.reset_index(drop=True).groupby('usubjid')

    X_grouped = [group[features].values for name, group in grouped_data]
    y_grouped = [group[target].values for name, group in grouped_data]

    # Calculate the split index for split %
    split_index = int(split * len(X_grouped))

    # Split the sequences into training and testing sets
    X_train_grouped = X_grouped[:split_index]
    X_test_grouped = X_grouped[split_index:]
    y_train_grouped = y_grouped[:split_index]
    y_test_grouped = y_grouped[split_index:]

    if patient_level == False:
        # Concatenate the sequences to form the training and testing sets
        X_train = np.concatenate(X_train_grouped)
        X_test = np.concatenate(X_test_grouped)
        y_train = np.concatenate(y_train_grouped)
        y_test = np.concatenate(y_test_grouped)
    else:
        return shuffled_df

    return X_train, X_test, y_train, y_test


def _find_init_parameters(X_train, X_test):
    # Define the grid of parameters
    # This parameter represents the initial guess for the state vector's mean values at the beginning of the filtering process.
    initial_state_means = [np.zeros(X_train.shape[1]), np.ones(X_train.shape[1])]
    # The transition matrix describes how the state vector evolves from one time step to the next in the absence of noise 
    transition_matrices = [np.eye(X_train.shape[1]), 0.95 * np.eye(X_train.shape[1])] 
    #The observation matrix maps the true state space (which might not be directly observable) to the observed data
    observation_matrices = [np.eye(X_train.shape[1]), 0.9 * np.eye(X_train.shape[1])]
    # This parameter defines the covariance of the process noise, which accounts for the uncertainty in the evolution of the state vector.
    process_noise_covariances = [0.01 * np.eye(X_train.shape[1]), 0.1 * np.eye(X_train.shape[1])]
    # This parameter defines the covariance of the observation noise, which reflects the uncertainty in your measurements.
    observation_noise_covariances = [0.1 * np.eye(X_train.shape[1]), 0.5 * np.eye(X_train.shape[1])]
    # The initial covariance matrix represents the uncertainty in the initial state estimate
    initial_covariances = [0.1 * np.eye(X_train.shape[1]), np.eye(X_train.shape[1])]

    # Lists to store all results
    results = []
    residual_means = []
    residual_stds = []
    covariance_diffs = []

    for param_combination in product(initial_state_means, transition_matrices, observation_matrices, process_noise_covariances, observation_noise_covariances, initial_covariances):
        initial_mean, A, C, Q, R, P_0 = param_combination
        
        # Initialize the Kalman Filter with current parameters
        kf = KalmanFilter(
            initial_state_mean=initial_mean,
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_covariance=P_0,
            n_dim_obs=X_train.shape[1],
            n_dim_state=X_train.shape[1]
        )
        
        # Train the filter using EM
        kf = kf.em(X_train, n_iter=50)
        
        # Filter the test data
        state_means, state_covariances = kf.filter(X_test)
        
        # Metric 1: Residual Analysis
        residuals = X_test - state_means
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        
        residual_means.append(residual_mean)
        residual_stds.append(residual_std)
        
        # Metric 2: Covariance Matrix Stability
        if len(covariance_diffs) > 0:
            covariance_diff = np.linalg.norm(state_covariances - prev_covariance_matrix)
            covariance_diffs.append(covariance_diff)
        else:
            covariance_diffs.append(0)  # No comparison for the first iteration
        
        prev_covariance_matrix = state_covariances
        
        # Store the current configuration and results
        results.append({
            'initial_state_mean': initial_mean,
            'transition_matrix': A,
            'observation_matrix': C,
            'process_noise_covariance': Q,
            'observation_noise_covariance': R,
            'initial_covariance': P_0,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'covariance_diff': covariance_diffs[-1],
        })

    return results

def _find_init_parameters_velocity(X_train, X_test, dt=1):
    # Define the dimension of the state space to include both 3 features and their velocities
    n_dim_obs = X_train.shape[1]  # Observed dimensions (3 features)
    n_dim_state = n_dim_obs * 2   # State dimensions (3 features + 3 velocities)

    # Define the grid of parameters
    initial_state_means = [
        np.zeros(n_dim_state),  # Initial state [feature_1, velocity_1, ..., feature_3, velocity_3] = 0
        np.ones(n_dim_state)    # Initial state [feature_1, velocity_1, ..., feature_3, velocity_3] = 1
    ]

    # Transition matrix including velocity (position updated by velocity)
    transition_matrices = [
        np.block([
            [np.eye(n_dim_obs), dt * np.eye(n_dim_obs)],  # Top-left: update positions based on velocities
            [np.zeros((n_dim_obs, n_dim_obs)), np.eye(n_dim_obs)]  # Bottom-right: velocities remain constant
        ]),
        0.95 * np.block([
            [np.eye(n_dim_obs), dt * np.eye(n_dim_obs)],  # Same structure with scaling factor
            [np.zeros((n_dim_obs, n_dim_obs)), np.eye(n_dim_obs)]
        ])
    ]

    # Observation matrix (we only observe the 3 features, not their velocities)
    observation_matrices = [
        np.hstack([np.eye(n_dim_obs), np.zeros((n_dim_obs, n_dim_obs))]),  # Map observed features, ignore velocities
        0.9 * np.hstack([np.eye(n_dim_obs), np.zeros((n_dim_obs, n_dim_obs))])
    ]

    # Process noise covariance (represents the uncertainty in the system's evolution)
    process_noise_covariances = [
        0.01 * np.eye(n_dim_state),
        0.1 * np.eye(n_dim_state)
    ]

    # Observation noise covariance (uncertainty in measurements)
    observation_noise_covariances = [
        0.1 * np.eye(n_dim_obs),  # Uncertainty in feature observations
        0.5 * np.eye(n_dim_obs)
    ]

    # Initial covariance matrix (uncertainty in the initial state estimate)
    initial_covariances = [
        0.1 * np.eye(n_dim_state),
        np.eye(n_dim_state)
    ]

    # Lists to store results
    results = []
    residual_means = []
    residual_stds = []
    covariance_diffs = []

    for param_combination in product(initial_state_means, transition_matrices, observation_matrices, process_noise_covariances, observation_noise_covariances, initial_covariances):
        initial_mean, A, C, Q, R, P_0 = param_combination
        
        # Initialize the Kalman Filter with current parameters
        kf = KalmanFilter(
            initial_state_mean=initial_mean,
            transition_matrices=A,
            observation_matrices=C,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_covariance=P_0,
            n_dim_obs=n_dim_obs,
            n_dim_state=n_dim_state
        )
        
        # Train the filter using EM
        kf = kf.em(X_train, n_iter=50)
        
        # Filter the test data
        state_means, state_covariances = kf.filter(X_test)
        
        # Metric 1: Residual Analysis (residual between observed and predicted 3 features)
        residuals = X_test - state_means[:, :n_dim_obs]  # Compare only features (exclude velocities)
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        
        residual_means.append(residual_mean)
        residual_stds.append(residual_std)
        
        # Metric 2: Covariance Matrix Stability
        if len(covariance_diffs) > 0:
            covariance_diff = np.linalg.norm(state_covariances - prev_covariance_matrix)
            covariance_diffs.append(covariance_diff)
        else:
            covariance_diffs.append(0)  # No comparison for the first iteration
        
        prev_covariance_matrix = state_covariances
        
        # Store the current configuration and results
        results.append({
            'initial_state_mean': initial_mean,
            'transition_matrix': A,
            'observation_matrix': C,
            'process_noise_covariance': Q,
            'observation_noise_covariance': R,
            'initial_covariance': P_0,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'covariance_diff': covariance_diffs[-1],
        })

    return results



def _save_parameters(train_split, output_dir, per_patient, full_data, features, results):
    # Change accordingly
    experiments_log = train_split + per_patient + full_data + features
    path = os.path.join(output_dir, experiments_log)

    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    # Define file paths
    results_csv_file = os.path.join(path, "all_results.csv")
    residual_plot_file = os.path.join(path, "residual_covariance_plots.png")
    combined_metric_plot_file = os.path.join(path, "combined_metric_plot.png")

        # Save all results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv_file, index=False)
    
    # Weights for the combined metric
    alpha = 0.5  # Weight for residual standard deviation
    beta = 0.5   # Weight for covariance matrix difference

    # Grid search
    best_combined_score = np.inf
    best_params = None

    # Lists to store all results
    residual_means = []
    residual_stds = []
    covariance_diffs = []
    combined_metrics = []

    for i in range(len(results)):
        residual_stds.append(results[i]["residual_std"])
        residual_means.append(results[i]["residual_mean"])
        covariance_diffs.append(results[i]["covariance_diff"])

    # Normalize after collecting all values
    scaler = MinMaxScaler()
    normalized_residual_stds = scaler.fit_transform(np.array(residual_stds).reshape(-1, 1)).flatten()
    normalized_covariance_diffs = scaler.fit_transform(np.array(covariance_diffs).reshape(-1, 1)).flatten()

    # Calculate combined metrics
    for i in range(len(results)):
        combined_metric = alpha * normalized_residual_stds[i] + beta * normalized_covariance_diffs[i]
        combined_metrics.append(combined_metric)
        
        # Update the best parameters based on the combined metric
        if combined_metric < best_combined_score:
            best_combined_score = combined_metric
            best_params = results[i]

        print(f"Residual Mean = {results[i]['residual_mean']:.4f}, Residual Std Dev = {results[i]['residual_std']:.4f}, Covariance Diff = {results[i]['covariance_diff']:.4f}, Combined Metric = {combined_metric:.4f}")

    # Store the best parameters
    print("\nBest Parameters Found Based on Combined Metric:")
    print(best_params)

    # # Save best parameters to a CSV file
    # best_params_file = os.path.join(path, "best_parameters.csv")
    # best_params_df = pd.DataFrame([best_params])
    # best_params_df.to_csv(best_params_file, index=False)

    # Convert numpy arrays to lists
    best_params_serializable = {k: v.tolist() for k, v in best_params.items()}
    best_params_file = os.path.join(path, "best_parameters.json")

    # Save the best parameters to a JSON file
    with open(best_params_file, 'w') as json_file:
        json.dump(best_params_serializable, json_file, indent=4)

    print(f"Best parameters saved to {best_params_file}")


    # Plot Residual Metrics
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(residual_means, 'b-', label='Residual Mean')
    plt.plot(residual_stds, 'g-', label='Residual Std Dev')
    plt.xlabel('Parameter Combination Index')
    plt.ylabel('Residual Value')
    plt.title('Residual Analysis Over Parameter Combinations')
    plt.legend()

    # Plot Covariance Matrix Differences
    plt.subplot(1, 2, 2)
    plt.plot(covariance_diffs, 'r-', label='Covariance Matrix Difference')
    plt.xlabel('Parameter Combination Index')
    plt.ylabel('Difference')
    plt.title('Covariance Matrix Stability Over Parameter Combinations')
    plt.legend()

    # Save the residual and covariance plots
    plt.tight_layout()
    plt.savefig(residual_plot_file)

    plt.show()

    # Plot Combined Metric
    plt.figure(figsize=(6, 4))
    plt.plot(combined_metrics, 'm-', label='Combined Metric')
    plt.xlabel('Parameter Combination Index')
    plt.ylabel('Combined Metric Value')
    plt.title('Combined Metric Over Parameter Combinations')
    plt.legend()

    # Save the combined metric plot
    plt.tight_layout()
    plt.savefig(combined_metric_plot_file)

    plt.show()

    print(f"Best parameters saved to {best_params_file}")
    print(f"All results saved to {results_csv_file}")
    print(f"Plots saved to {residual_plot_file} and {combined_metric_plot_file}")

def fine_tuning(X_train, X_test, path, best_params, velocity = 1, iteration = 125):

    combined_metric_plot_file = os.path.join(path, "fine_tuning_residual_plot.png")

    n_dim_obs = X_train.shape[1]

    initial_mean = np.array(best_params["initial_state_mean"])
    A = np.array(best_params["transition_matrix"])
    C = np.array(best_params["observation_matrix"])
    Q = np.array(best_params["process_noise_covariance"])
    R = np.array(best_params["observation_noise_covariance"])
    P_0 = np.array(best_params["initial_covariance"])

    kf = KalmanFilter(
        initial_state_mean=initial_mean,
        transition_matrices=A,
        observation_matrices=C,
        transition_covariance=Q,
        observation_covariance=R,
        initial_state_covariance=P_0,
        n_dim_obs=X_train.shape[1],
        n_dim_state=X_train.shape[1] * velocity
    )

    # Estimate the Kalman Filter parameters using EM algorithm and track convergence metrics
    residual_means = []
    residual_stds = []
    covariance_diffs = []
    prev_covariance_matrix = None
    for i in range(iteration):
        kf = kf.em(X_train, n_iter=1)
        
        # Use the filter to estimate the hidden states
        state_means, state_covariances = kf.filter(X_test)
        
        # Calculate residuals
        residuals = X_test - state_means[:, :n_dim_obs]
        
        # Metric 1: Residual Analysis
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        
        residual_means.append(residual_mean)
        residual_stds.append(residual_std)
        
        print(f"Iteration {i+1}: Residual Mean = {residual_mean:.4f}, Residual Std Dev = {residual_std:.4f}")
        
        # Metric 2: Covariance Matrix Stability
        if prev_covariance_matrix is not None:
            covariance_diff = np.linalg.norm(state_covariances - prev_covariance_matrix)
            covariance_diffs.append(covariance_diff)
            print(f"Iteration {i+1}: Covariance Matrix Difference = {covariance_diff:.4f}")
        prev_covariance_matrix = state_covariances

    # Final Filter Application
    state_means, state_covariances = kf.filter(X_test)

    # Plot the metrics
    plt.figure(figsize=(12, 6))

    # Plot Residual Std Dev with Residual Mean on a secondary Y-axis
    plt.subplot(1, 2, 1)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(residual_stds, 'g-', label='Residual Std Dev')
    ax2.plot(residual_means, 'b-', label='Residual Mean')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Residual Std Dev', color='g')
    ax2.set_ylabel('Residual Mean', color='b')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Residual Metrics Over Iterations')

    # Plot Covariance Matrix Differences
    plt.subplot(1, 2, 2)
    plt.plot(covariance_diffs, label='Covariance Matrix Difference')
    plt.ylim(0, 10000)
    plt.title('Covariance Matrix Stability Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Difference')
    plt.legend()

    # Save the combined metric plot
    plt.tight_layout()
    plt.savefig(combined_metric_plot_file)

    return kf


def _inference_with_future_steps(kf, X_test, X_train, y_test, y_train, features="AS", threshold=0.5, n_future_steps=1, df=None, single = False):
    # Initialize a DataFrame if not provided
    if df is None:
        df = pd.DataFrame()

    # Perform filtering to get the state means and covariances
    state_means, state_covariances = kf.filter(X_test)

    if features == "AS":
        # Use the state means for predictions
        if single == False:
            prediction = state_means[:, 1]
            predicted_labels = (prediction >= threshold).astype(int)
            true_predictor = X_test[:, 1]
        else:
            prediction = state_means[:, 0]
            predicted_labels = (prediction >= threshold).astype(int)
            true_predictor = X_test[:, 0]
    elif features == "AM":
        prediction = state_means[:, 0]
        predicted_labels = (prediction >= threshold).astype(int)
        true_predictor = X_test[:, 0]
    elif features == "AS&AM":
        AM = state_means[:, 0]
        AS = state_means[:, 1]
        prediction = state_means[:, 1]
        predicted_labels = ((AS >= 0.2) & (AM >= 34)).astype(int)
        true_predictor = X_test[:, 1]

    # Calculate current state metrics
    model_auc = roc_auc_score(y_test, prediction)
    base_auc = roc_auc_score(y_test, true_predictor)
    model_acc = accuracy_score(y_test, predicted_labels)
    baseline_predicted_labels = (true_predictor >= threshold).astype(int)
    base_acc = accuracy_score(y_test, baseline_predicted_labels)

    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels).ravel()
    ppv = precision_score(y_test, predicted_labels)
    sensitivity = recall_score(y_test, predicted_labels)
    specificity = tn / (tn + fp)

    tn_base, fp_base, fn_base, tp_base = confusion_matrix(y_test, baseline_predicted_labels).ravel()
    base_ppv = precision_score(y_test, baseline_predicted_labels)
    base_sensitivity = recall_score(y_test, baseline_predicted_labels)
    base_specificity = tn_base / (tn_base + fp_base)

    # Predict future states for all samples
    future_state_means_list = []
    future_state_covariances_list = []

    for sample_index in range(X_test.shape[0]):
        # Get the last state mean and covariance for the current sample
        last_state_mean = state_means[sample_index]
        last_state_covariance = state_covariances[sample_index]
        
        # Predict future state for this sample
        for step in range(n_future_steps):
            predicted_state_mean = kf.transition_matrices @ last_state_mean
            predicted_state_covariance = (
                kf.transition_matrices @ last_state_covariance @ kf.transition_matrices.T
                + kf.transition_covariance
            )
            
            # Update the last state for the next iteration (if predicting more than 1 step)
            last_state_mean = predicted_state_mean
            last_state_covariance = predicted_state_covariance
        
        # Store the prediction results for this sample
        future_state_means_list.append(predicted_state_mean.flatten())
        future_state_covariances_list.append(predicted_state_covariance)

    # Convert to arrays
    future_state_means = np.array(future_state_means_list)
    
    # Calculate metrics for future predictions
    if features == "AS":
        if single == True:
            future_prediction = future_state_means[:, 0]
            future_predicted_labels = (future_prediction >= threshold).astype(int)
        else:
            future_prediction = future_state_means[:, 1]
            future_predicted_labels = (future_prediction >= threshold).astype(int)
    elif features == "AM":
        future_prediction = future_state_means[:, 0]
        future_predicted_labels = (future_prediction >= threshold).astype(int)
    elif features == "AS&AM":
        AM = future_state_means[:, 0]
        AS = future_state_means[:, 1]
        future_prediction = AS
        future_predicted_labels = ((AS >= 0.2) & (AM >= 34)).astype(int)

    future_auc = roc_auc_score(y_test, future_prediction)
    future_acc = accuracy_score(y_test, future_predicted_labels)
    
    tn_future, fp_future, fn_future, tp_future = confusion_matrix(y_test, future_predicted_labels).ravel()
    future_ppv = precision_score(y_test, future_predicted_labels)
    future_sensitivity = recall_score(y_test, future_predicted_labels)
    future_specificity = tn_future / (tn_future + fp_future)

    # Create a dictionary with the new results for both current and future states
    results = {
        f'Threshold_{threshold}': [
            model_auc, model_acc, ppv, sensitivity, specificity,base_auc, base_acc, base_ppv, base_sensitivity, base_specificity,
            future_auc, future_acc, future_ppv, future_sensitivity, future_specificity
        ]
    }

    # List of metric names
    metric_names = [
        'Model_AUC','Model_Accuracy',
        'Model_PPV', 'Model_Sensitivity', 'Model_Specificity','Baseline_AUC','Baseline_Accuracy',
        'Baseline_PPV', 'Baseline_Sensitivity', 'Baseline_Specificity',
        'Future_AUC', 'Future_Accuracy', 'Future_PPV', 'Future_Sensitivity', 'Future_Specificity'
    ]

    # Create a new DataFrame with these results and metric names
    new_row = pd.DataFrame(results, index=metric_names)

    # Concatenate the new row to the existing DataFrame
    df = pd.concat([df, new_row], axis=1)

    return df