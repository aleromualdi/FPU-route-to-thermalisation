import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class FPUTData:
    data_matrix: np.array
    energy_matrix: np.array
    dst: np.array
    times: np.array


# util function to load FPUT data
def load_data(data_dir: str, beta: float) -> FPUTData:

    data_path = os.path.join(data_dir, f"fermi_{beta}")
    data_matrix = np.load(os.path.join(data_path, "dataMatrix.npy"))
    energy_matrix = np.load(os.path.join(data_path, "energyMatrix.npy"))
    dst = np.load(os.path.join(data_path, "dst.npy"))
    time = np.load(os.path.join(data_path, "time.npy"))

    return FPUTData(
        data_matrix=data_matrix, energy_matrix=energy_matrix, dst=dst, times=time
    )


def temporalize_data(q: np.array, timesteps: np.array) -> np.array:
    """Temporalizes a given dataset by organizing it into sequences of past records.

    Parameters:
    - q (np.array): The input dataset, assumed to be a 2D NumPy array where each row represents a record.
    - timesteps (int): The number of past timesteps to include in each sequence.

    Returns:
    - np.array: A 3D NumPy array where each element is a sequence of past records, organized by timesteps.

    Example:
    >>> input_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> temporalize_data(input_data, timesteps=2)
    array([[[1, 2, 3],
            [4, 5, 6]],
           [[4, 5, 6],
            [7, 8, 9]]])
    """
    temporalized_q = []
    for i in range(len(q) - timesteps - 1):
        t = []
        for j in range(1, timesteps + 1):
            # Gather past records upto the timesteps period
            t.append(q[(i + j + 1), :])
        temporalized_q.append(t)
    return np.array(temporalized_q)


def compute_spectral_entropy(
    energies: np.array, axis=1, modes: Tuple[float, float] = None
) -> np.array:
    """Compute spectral entropy over time based on energy distribution.

    When a tuple `modes` is provided, its values are utilized to specify a range of energy
    modes for which the spectral entropy will be computed.

    Parameters:
    - energies (np.array): 2D NumPy array representing energy values over time and modes.
    - axis (int): The axis along which to compute spectral entropy. Default is axis 1 (time axis).
    - modes (tuple[float, float]): Optional tuple specifying the range of modes to consider.
      If provided, only energies within the specified mode range will be used.

    Returns:
    - np.array: 1D NumPy array representing the spectral entropy over time.

    Notes:
    - Spectral entropy is a measure of the disorder or unpredictability in the energy distribution.
    - Entropy values are normalized between 0 and log2(number of modes).

    Example:
    >>> energy_data = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    >>> compute_spectral_entropy(energy_data, axis=1)
    array([1.04644047, 1.07389879])
    """
    if modes:
        energies = energies[:, modes[0] : modes[1]]

    # Normalize energy values along the specified axis to get probabilities
    probs = np.apply_along_axis(lambda x: x / np.sum(x), axis, energies)

    # Compute spectral entropy over time
    entropy_per_time = -np.sum(probs * np.log2(probs + 1e-10), axis)

    return entropy_per_time


def find_time_ranges_above_threshold(
    mae_loss: np.array, feature_thresholds: np.array, min_range_length: int = 10
) -> List[Tuple[int, int]]:
    """Calculate time ranges where the mean absolute error (MAE) is above the specified thresholds.

    Parameters:
    - mae_loss (numpy.ndarray): The matrix of mean absolute errors with shape (num_samples, num_dimensions).
    - feature_thresholds (list): A list of threshold values for each dimension.
    - min_range_length (int, optional): Minimum length of the identified ranges. Defaults to 10.

    Returns:
    - list: A list of tuples representing time ranges where the MAE is above the threshold.
      Each tuple contains start and end indices of the range.
    """
    time_ranges_per_dimension = []

    for i in range(mae_loss.shape[1]):
        threshold = feature_thresholds[i]
        above_threshold = mae_loss[:, i] > threshold

        # Find non-empty ranges
        if np.any(above_threshold):
            ranges = np.where(
                np.diff(np.concatenate(([0], above_threshold, [0]))) != 0
            )[0]
            ranges = ranges.reshape(-1, 2)
            ranges = [(start, end - 1) for start, end in ranges]

            # Convert indices to time ranges
            time_ranges_per_dimension.extend(ranges)

    # Sort and merge overlapping ranges
    time_ranges_per_dimension.sort(key=lambda x: x[0])
    merged_ranges = []
    for start, end in time_ranges_per_dimension:
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1] = (merged_ranges[-1][0], max(end, merged_ranges[-1][1]))
        else:
            if end - start < min_range_length:
                merged_ranges.append((start, end))

    return merged_ranges


def calculate_cumulative_time_above_threshold(
    mae_loss: np.array, feature_thresholds: np.array
) -> np.array:
    """Calculate the cumulative time spent above the specified thresholds for each feature.

    Parameters:
    - mae_loss (numpy.ndarray): The matrix of mean absolute errors with shape (num_samples, num_dimensions).
    - feature_thresholds (list): A list of threshold values for each dimension.

    Returns:
    - numpy.ndarray: An array containing the cumulative time spent above the threshold for each feature.
    """
    cumulative_time_above_threshold = np.zeros(mae_loss.shape[1])

    for i in range(mae_loss.shape[1]):
        threshold = feature_thresholds[i]
        above_threshold = mae_loss[:, i] > threshold

        # Calculate cumulative time above threshold
        cumulative_time_above_threshold[i] = np.sum(above_threshold)

    return cumulative_time_above_threshold


def smeared_logistic_function(
    times: np.array,
    transition_time,
    amplitude_before: float,
    amplitude_after: float,
    smear_factor: float,
) -> float:
    return amplitude_before + (amplitude_after - amplitude_before) / (
        1 + np.exp(-smear_factor * (times - transition_time))
    )


def fit_smeared_logistic(
    series: np.array, initial_guess: List[float]
) -> Tuple[np.array, np.array]:
    t = np.arange(0, series.shape[0])
    params, _ = curve_fit(
        smeared_logistic_function, t, series, p0=initial_guess, maxfev=10000
    )
    return t, params
