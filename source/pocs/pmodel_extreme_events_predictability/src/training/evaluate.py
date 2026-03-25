# evaluate.py

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Imports from project modules
from src.models.model import PeakPredictorLSTM
from src.data.pmodel_generation import generate_and_prepare_series


def create_ground_truth_for_prediction(peak_series, lookahead):
    """
    Creates the "imminent peak" ground-truth signal used for visualization.
    """
    rolled_max = (
        peak_series
        .rolling(window=lookahead, min_periods=1)
        .max()
        .shift(-lookahead + 1)
        .bfill()
    )
    ground_truth = (rolled_max > 0).astype(int)
    return ground_truth


def prepare_evaluation_dataframe(config):
    """
    Generates a p-model time series and prepares a DataFrame for evaluation.
    """
    print("\nGenerating new evaluation data...")

    df_eval = generate_and_prepare_series(
        length=config["p_model"]["series_length_eval"],
        p_value=config["p_model"]["p_value"],
        peak_percentile=config["data"]["peak_percentile"],
    )

    df_eval["ground_truth_for_prediction"] = create_ground_truth_for_prediction(
        df_eval["peak"],
        config["data"]["lookahead"]
    )

    print(
        f"Generated a series of length {len(df_eval)} "
        f"with {df_eval['peak'].sum()} peak events."
    )

    return df_eval


def plot_evaluation_matplotlib(
    df,
    p_value,
    lookahead=10,
    filename="data/evaluation_plot_pmodel.png"
):
    """
    Generates and saves a styled PNG plot for qualitative evaluation.

    The plot includes:
    - Normalized signal
    - Ground-truth peak prediction window
    - Shifted predicted probabilities (anticipating peaks)
    - True peak markers
    """
    print(f"\nGenerating evaluation plot for p-model (p = {p_value})...")

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # -------- X-axis setup --------
    x_axis_data = df.index
    x_label = "Time Step"
    signal_label = f"P-model Signal (Normalized, p = {p_value})"

    # -------- Scale signal for visualization --------
    plot_df = df.copy()
    max_val = plot_df["normalized"].max()
    scaling_factor = 0.9 / max_val if max_val > 0 else 1.0
    plot_df["normalized_scaled"] = plot_df["normalized"] * scaling_factor

    # 1. Plot ground-truth peak prediction window
    ax.fill_between(
        x_axis_data,
        plot_df["ground_truth_for_prediction"],
        0,
        color="orange",
        alpha=0.2,
        label="Peak Prediction Window",
        step="post",
    )

    # 2. Plot normalized signal
    ax.plot(
        x_axis_data,
        plot_df["normalized_scaled"],
        label=signal_label,
        color="blue",
        lw=0.6,
    )

    # -------- Compute step size for temporal shift --------
    if len(x_axis_data) > 1:
        step_size = np.mean(np.diff(x_axis_data))
    else:
        step_size = 1.0

    # 3. Shift predictions to the left (anticipating peaks)
    x_pred = x_axis_data - lookahead * step_size
    mask = x_pred >= x_axis_data.min()

    ax.plot(
        x_pred[mask],
        plot_df["prediction"].values[mask],
        label="Predicted Peak Probability",
        color="limegreen",
        linestyle="--",
        lw=1.5,
    )

    # 4. Plot peak markers
    peak_indices = plot_df[plot_df["peak"] == 1].index
    peak_y_values = plot_df.loc[peak_indices, "normalized_scaled"]

    ax.scatter(
        peak_indices,
        peak_y_values,
        color="red",
        s=50,
        edgecolor="black",
        label="Peak Event",
        zorder=5,
    )

    # -------- Formatting --------
    ax.set_title(
        f"LSTM Peak Prediction on P-model Time Series (p = {p_value})"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Normalized Value / Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend(loc="best", fontsize="small", framealpha=0.7)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Evaluation plot saved to '{filename}'")
    plt.show()


def evaluate_and_plot():
    """
    Runs stateful step-by-step inference on a p-model time series
    and generates a qualitative evaluation plot.
    """
    with open("config/config.json", "r") as f:
        config = json.load(f)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    model = PeakPredictorLSTM(config["model"]).to(device)

    try:
        model.load_state_dict(
            torch.load(
                config["files"]["model_save_path"],
                map_location=device
            )
        )
        model.eval()
    except FileNotFoundError:
        print(
            f"Error: Model file not found at "
            f"'{config['files']['model_save_path']}'."
        )
        return

    df_eval = prepare_evaluation_dataframe(config)
    signal = df_eval["normalized"].values
    predictions = []

    print("Running stateful, step-by-step inference...")
    hidden_states = model.init_hidden()

    with torch.no_grad():
        for t in tqdm(range(len(signal)), desc="Predicting"):
            x_t = signal[t]
            input_tensor = torch.tensor(
                [[[x_t]]],
                dtype=torch.float32,
                device=device
            )

            prediction, hidden_states = model.step(
                input_tensor,
                hidden_states
            )
            predictions.append(prediction.item())

    df_eval["prediction"] = predictions

    plot_evaluation_matplotlib(
        df_eval,
        config["p_model"]["p_value"],
        lookahead=config["data"]["lookahead"],
    )


if __name__ == "__main__":
    evaluate_and_plot()
