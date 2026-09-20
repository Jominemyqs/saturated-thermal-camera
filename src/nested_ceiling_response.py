from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SINGLE_THRESHOLD_FEATURES = (
    "posterior_exceedance_K",
    "posterior_sd_K",
    "local_censored_fraction",
    "posterior_standardized_gap",
    "prior_standardized_gap",
    "posterior_exceedance_probability",
    "log_posterior_sd",
    "normalized_heat_flux",
)

RESPONSE_FEATURES = (
    "dmu_dc_near",
    "dsd_dc_near",
    "dmu_dc_far",
    "dsd_dc_far",
    "dmu_dc_change",
    "dsd_dc_change",
)


@dataclass(frozen=True)
class GroupedPrediction:
    prediction: np.ndarray
    selected_penalty: dict[str, float]


@dataclass(frozen=True)
class RidgeModel:
    feature_mean: np.ndarray
    feature_sd: np.ndarray
    coefficients: np.ndarray
    penalty: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = (np.asarray(features, dtype=float) - self.feature_mean) / self.feature_sd
        design = np.column_stack([np.ones(len(values)), values])
        return design.dot(self.coefficients)


def build_nested_response_table(
    examples: pd.DataFrame,
    *,
    levels: tuple[float, float, float] = (0.05, 0.075, 0.10),
) -> pd.DataFrame:
    """Match pixels censored at three nested pseudo-camera ceilings."""
    near, middle, far = levels
    index = ["trajectory", "pixel_index"]
    pivot_columns = [
        "ceiling_K",
        "known_reference_measurement_K",
        "latent_truth_K_evaluation_only",
        "posterior_mean_K",
        "posterior_sd_K",
        "local_censored_fraction",
        "posterior_standardized_gap",
        "prior_standardized_gap",
        "posterior_exceedance_probability",
        "log_posterior_sd",
        "normalized_heat_flux",
    ]
    pivot = examples.pivot(
        index=index,
        columns="target_censored_fraction",
        values=pivot_columns,
    )
    required = [(column, level) for column in pivot_columns for level in levels]
    matched = pivot[required].dropna()

    def values(column: str, level: float) -> np.ndarray:
        return matched[(column, level)].to_numpy(dtype=float)

    ceiling_near = values("ceiling_K", near)
    ceiling_middle = values("ceiling_K", middle)
    ceiling_far = values("ceiling_K", far)
    if not np.all((ceiling_near > ceiling_middle) & (ceiling_middle > ceiling_far)):
        raise ValueError("Nested ceilings must decrease as censoring increases")
    mean_near = values("posterior_mean_K", near)
    mean_middle = values("posterior_mean_K", middle)
    mean_far = values("posterior_mean_K", far)
    sd_near = values("posterior_sd_K", near)
    sd_middle = values("posterior_sd_K", middle)
    sd_far = values("posterior_sd_K", far)
    dmu_near = (mean_near - mean_middle) / (ceiling_near - ceiling_middle)
    dmu_far = (mean_middle - mean_far) / (ceiling_middle - ceiling_far)
    dsd_near = (sd_near - sd_middle) / (ceiling_near - ceiling_middle)
    dsd_far = (sd_middle - sd_far) / (ceiling_middle - ceiling_far)
    output = pd.DataFrame(
        {
            "trajectory": matched.index.get_level_values("trajectory"),
            "pixel_index": matched.index.get_level_values("pixel_index"),
            "target_censored_fraction": near,
            "target_ceiling_K": ceiling_near,
            "known_reference_measurement_K": values(
                "known_reference_measurement_K", near
            ),
            "latent_truth_K_evaluation_only": values(
                "latent_truth_K_evaluation_only", near
            ),
            "measurement_exceedance_K": values(
                "known_reference_measurement_K", near
            )
            - ceiling_near,
            "truth_exceedance_K_evaluation_only": values(
                "latent_truth_K_evaluation_only", near
            )
            - ceiling_near,
            "posterior_exceedance_K": mean_near - ceiling_near,
            "posterior_sd_K": sd_near,
            "dmu_dc_near": dmu_near,
            "dsd_dc_near": dsd_near,
            "dmu_dc_far": dmu_far,
            "dsd_dc_far": dsd_far,
            "dmu_dc_change": dmu_near - dmu_far,
            "dsd_dc_change": dsd_near - dsd_far,
        }
    )
    for column in SINGLE_THRESHOLD_FEATURES[2:]:
        output[column] = values(column, near)
    return output


def _ridge_fit(
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    penalty: float,
) -> np.ndarray:
    feature_mean = np.mean(train_features, axis=0)
    feature_sd = np.std(train_features, axis=0)
    feature_sd = np.where(feature_sd > 1e-8, feature_sd, 1.0)
    train = (train_features - feature_mean) / feature_sd
    test = (test_features - feature_mean) / feature_sd
    design = np.column_stack([np.ones(len(train)), train])
    regularizer = np.diag([0.0] + [float(penalty)] * train_features.shape[1])
    coefficients = np.linalg.solve(
        design.T.dot(design) + regularizer,
        design.T.dot(train_target),
    )
    return np.column_stack([np.ones(len(test)), test]).dot(coefficients)


def fit_grouped_ridge_model(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    penalties: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> RidgeModel:
    """Select a ridge penalty by leave-one-group-out CV, then fit all rows."""
    values = np.asarray(features, dtype=float)
    response = np.asarray(target, dtype=float).reshape(-1)
    labels = np.asarray(groups)
    unique_groups = np.unique(labels)
    if len(unique_groups) < 3:
        raise ValueError("Grouped tuning requires at least three groups")
    scores = []
    for penalty in penalties:
        squared_errors = []
        for held_out in unique_groups:
            validation = labels == held_out
            prediction = _ridge_fit(
                values[~validation],
                response[~validation],
                values[validation],
                penalty,
            )
            squared_errors.extend((prediction - response[validation]) ** 2)
        scores.append(float(np.sqrt(np.mean(squared_errors))))
    penalty = float(penalties[int(np.argmin(scores))])
    feature_mean = np.mean(values, axis=0)
    feature_sd = np.std(values, axis=0)
    feature_sd = np.where(feature_sd > 1e-8, feature_sd, 1.0)
    standardized = (values - feature_mean) / feature_sd
    design = np.column_stack([np.ones(len(standardized)), standardized])
    regularizer = np.diag([0.0] + [penalty] * values.shape[1])
    coefficients = np.linalg.solve(
        design.T.dot(design) + regularizer,
        design.T.dot(response),
    )
    return RidgeModel(feature_mean, feature_sd, coefficients, penalty)


def grouped_ridge_predictions(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    penalties: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> GroupedPrediction:
    """Outer leave-one-group-out predictions with inner grouped tuning."""
    values = np.asarray(features, dtype=float)
    response = np.asarray(target, dtype=float).reshape(-1)
    labels = np.asarray(groups)
    unique_groups = np.unique(labels)
    if len(unique_groups) < 3:
        raise ValueError("Nested grouped validation requires at least three groups")
    predictions = np.empty(len(response))
    selected: dict[str, float] = {}
    for held_out in unique_groups:
        outer_train = labels != held_out
        candidate_scores = []
        training_groups = np.unique(labels[outer_train])
        for penalty in penalties:
            squared_errors = []
            for validation_group in training_groups:
                inner_validation = labels == validation_group
                inner_train = outer_train & ~inner_validation
                prediction = _ridge_fit(
                    values[inner_train],
                    response[inner_train],
                    values[inner_validation],
                    penalty,
                )
                squared_errors.extend(
                    (prediction - response[inner_validation]) ** 2
                )
            candidate_scores.append(float(np.sqrt(np.mean(squared_errors))))
        penalty = float(penalties[int(np.argmin(candidate_scores))])
        selected[str(held_out)] = penalty
        predictions[~outer_train] = _ridge_fit(
            values[outer_train],
            response[outer_train],
            values[~outer_train],
            penalty,
        )
    return GroupedPrediction(predictions, selected)


def grouped_ridge_transfer_predictions(
    train_features: np.ndarray,
    train_target: np.ndarray,
    train_groups: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_groups: np.ndarray,
    *,
    penalties: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> GroupedPrediction:
    """Train on one regime and predict another under outer grouped validation."""
    train_values = np.asarray(train_features, dtype=float)
    response = np.asarray(train_target, dtype=float).reshape(-1)
    train_labels = np.asarray(train_groups)
    evaluation_values = np.asarray(evaluation_features, dtype=float)
    evaluation_labels = np.asarray(evaluation_groups)
    unique_groups = np.unique(evaluation_labels)
    if len(unique_groups) < 3:
        raise ValueError("Grouped transfer requires at least three groups")
    if not set(unique_groups).issubset(set(np.unique(train_labels))):
        raise ValueError("Every evaluation group must occur in training data")
    predictions = np.empty(len(evaluation_values))
    selected: dict[str, float] = {}
    for held_out in unique_groups:
        outer_train = train_labels != held_out
        candidate_scores = []
        training_groups = np.unique(train_labels[outer_train])
        for penalty in penalties:
            squared_errors = []
            for validation_group in training_groups:
                validation = train_labels == validation_group
                inner_train = outer_train & ~validation
                prediction = _ridge_fit(
                    train_values[inner_train],
                    response[inner_train],
                    train_values[validation],
                    penalty,
                )
                squared_errors.extend((prediction - response[validation]) ** 2)
            candidate_scores.append(float(np.sqrt(np.mean(squared_errors))))
        penalty = float(penalties[int(np.argmin(candidate_scores))])
        selected[str(held_out)] = penalty
        evaluation = evaluation_labels == held_out
        predictions[evaluation] = _ridge_fit(
            train_values[outer_train],
            response[outer_train],
            evaluation_values[evaluation],
            penalty,
        )
    return GroupedPrediction(predictions, selected)


def prediction_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    estimate = np.asarray(prediction, dtype=float)
    truth = np.asarray(target, dtype=float)
    error = estimate - truth
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    return {
        "rmse_K": float(np.sqrt(np.mean(error**2))),
        "mae_K": float(np.mean(np.abs(error))),
        "bias_K": float(np.mean(error)),
        "spearman_rho": float(spearmanr(estimate, truth).statistic),
        "r_squared": float(1.0 - np.sum(error**2) / denominator),
    }
