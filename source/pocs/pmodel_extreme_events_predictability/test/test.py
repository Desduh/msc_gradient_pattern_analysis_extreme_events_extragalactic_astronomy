import os
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
import numpy as np
import json
import pandas as pd

from src.models.model import PeakPredictorLSTM
from src.data.pmodel_generation import generate_multiple_series

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


def recall_vs_anticipation(
    predictions,
    true_peaks,
    lookahead,
    threshold,
    burn_in=100,
):
    """
    Computes recall as a function of minimum anticipation time,
    discarding early points without valid prediction context.

    Returns:
        ks: minimum anticipation (in time steps)
        recalls: fraction of peaks anticipated at least k steps ahead
    """

    # Positions of real peaks
    peak_positions = np.where(true_peaks == 1)[0]

    # Discard peaks too close to the beginning (no valid anticipation window)
    peak_positions = peak_positions[
        peak_positions >= burn_in + lookahead
    ]

    ks = np.arange(1, lookahead + 1)
    recalls = []

    if len(peak_positions) == 0:
        return ks, np.zeros_like(ks, dtype=float)

    for k in ks:
        detected = 0

        for t_peak in peak_positions:
            t_start = t_peak - lookahead

            window = predictions[t_start:t_peak]

            # Ignore windows with NaNs (extra safety)
            if np.isnan(window).any():
                continue

            alarm_idx = np.where(window >= threshold)[0]

            if len(alarm_idx) > 0:
                t_alarm = t_start + alarm_idx[0]
                if (t_peak - t_alarm) >= k:
                    detected += 1

        recalls.append(detected / len(peak_positions))

    return ks, np.array(recalls)

def build_anticipatory_labels(peaks, lookahead):
    """
    peaks: array binário (0/1) indicando pico instantâneo
    lookahead: horizonte de previsão

    Retorna:
    y_future[t] = 1 se existir pico em [t+1, t+lookahead]
    """
    y_future = np.zeros_like(peaks)

    for t in range(len(peaks)):
        end = min(t + lookahead + 1, len(peaks))
        if np.any(peaks[t + 1 : end]):
            y_future[t] = 1

    return y_future



def evaluate_and_plot_streaming(
    model,
    df,
    device,
    lookahead,
    threshold=0.6,
    title_extra="",
):
    """
    Evaluates a single time series in a streaming (causal) regime.
    """

    # Select appropriate signal column
    signal_col = (
        "normalized_amplitude"
        if "normalized_amplitude" in df.columns
        else "normalized"
    )

    series = df[signal_col].values
    true_peaks = df["peak"].values.astype(int)
    instant_peaks = df["peak"].values.astype(int)
    true_peaks_classic_metrics = build_anticipatory_labels(instant_peaks, lookahead)


    model.eval()
    hidden = model.init_hidden()
    predictions = []

    with torch.no_grad():
        for value in series:
            x_t = torch.tensor([[[value]]], dtype=torch.float32).to(device)
            y_hat, hidden = model.step(x_t, hidden)
            predictions.append(y_hat.item())

    predictions = np.array(predictions)

    # Apply sigmoid if the model outputs logits
    predictions = 1 / (1 + np.exp(-predictions))

    # Temporal adjustment (anticipatory prediction)
    predictions[:lookahead] = np.nan
    valid = ~np.isnan(predictions)

    y_true = true_peaks_classic_metrics[valid]
    y_score = predictions[valid]
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "AUC-ROC": roc_auc_score(y_true, y_score)
        if len(np.unique(y_true)) > 1
        else np.nan,
        "AUC-PR": average_precision_score(y_true, y_score)
        if len(np.unique(y_true)) > 1
        else np.nan,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }

    print(f"\nEvaluation metrics {title_extra}")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")

    # Recall vs Anticipation Plot
    thresholds = [threshold, 0.7, 0.8, 0.9]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))] 
    markers = ["o", "x", "^", "D"]

    plt.figure(figsize=(6, 4))

    for th, c, ls, m in zip(thresholds, colors, linestyles, markers):
        ks, recalls = recall_vs_anticipation(
            predictions,
            true_peaks,
            lookahead,
            threshold=th,
        )

        plt.plot(
            ks,
            recalls,
            color=c,
            linestyle=ls,
            marker=m,
            markersize=6,
            linewidth=1.2,
            label=f"threshold = {th}"
        )

    plt.xlabel("Minimum anticipation (time steps before the peak)")
    plt.ylabel("Recall (fraction of anticipated peaks)")
    plt.title("Extreme event anticipation capability")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.show()



    # Full series  plots
    fig, ax = plt.subplots(figsize=(7, 2.5))

    peak_idx = np.where(true_peaks == 1)[0]

    # Full series
    ax.plot(
        series, 
        lw=0.9, 
        label="P-model time series"
    )

    ax.plot(
        peak_idx,
        series[peak_idx],
        "x",
        markersize=6,
        label="Extreme events",
        color="red"
    )

    ax.plot(
        predictions,
        "--",
        lw=0.9,
        label=f"Predicted P(extreme event within {lookahead} steps)",
    )
    ax.set_title(
        f"P-model time series with extreme event labels and LSTM prediction"
    )

    ax.legend()
    ax.grid(alpha=0.3)
    ax.legend(frameon=True, fontsize=10)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Normalized amplitude / probability")

    plt.tight_layout()
    plt.show()

    return metrics, ks, recalls



def load_and_prepare_csv(file_path, peak_percentile):
    """
    Loads a real CSV time series and generates peak labels using a percentile.
    """
    print(f"\nLoading file: {os.path.basename(file_path)}")

    df = pd.read_csv(file_path)
    df.columns = ["time", "amplitude"]

    amp = df["amplitude"].values
    df["normalized_amplitude"] = (
        (amp - amp.min()) / (amp.max() - amp.min() + 1e-6)
        if amp.max() > amp.min()
        else np.zeros_like(amp)
    )

    threshold = np.percentile(
        df["normalized_amplitude"], peak_percentile
    )
    df["peak"] = (df["normalized_amplitude"] > threshold).astype(int)
    df["peak_threshold"] = threshold

    return df


def main():
    # ---------- Configuration ----------
    np.random.seed(30)

    with open("./config/config_exo_endo.json", "r") as f:
        config = json.load(f)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # ---------- Model ----------
    model = PeakPredictorLSTM(config["model"]).to(device)
    model.load_state_dict(
        torch.load("./results/best_model_exo_endo.pth", map_location=device)
    )
    print("Model loaded successfully")

    test_series = generate_multiple_series(
        length=config["p_model"]["series_length"],
        p_value=config["p_model"]["p_value"],
        num_series=config["data"]["num_series_test"],
        peak_percentile=config["data"]["peak_percentile"],
    )

    all_metrics = []
    all_recalls = []
    ks_ref = None

    indices = np.linspace(0, len(test_series) - 1, 3, dtype=int)

    # for count, i in enumerate([0]):
    #     df = test_series[i]

    #     print(
    #         f"\n Synthetic series {count + 1}/{len(indices)} "
    #         f"(index {i})"
    #     )

    #     metrics, ks, recalls = evaluate_and_plot_streaming(
    #         model,
    #         df,
    #         device,
    #         lookahead=config["data"]["lookahead"],
    #         title_extra=f"(P-model, series {i})",
    #     )

    #     all_metrics.append(metrics)
    #     all_recalls.append(recalls)

    #     if ks_ref is None:
    #         ks_ref = ks

    
    # all_recalls = np.array(all_recalls)  # shape = (num_series, lookahead)

    # mean_recall = np.nanmean(all_recalls, axis=0)
    # std_recall = np.nanstd(all_recalls, axis=0)

    # plt.figure(figsize=(7, 5))

    # plt.plot(
    #     ks_ref,
    #     mean_recall,
    #     marker="o",
    #     label="Mean Recall (synthetic test set)"
    # )

    # plt.fill_between(
    #     ks_ref,
    #     mean_recall - std_recall,
    #     mean_recall + std_recall,
    #     alpha=0.3,
    #     label="±1 std"
    # )

    # plt.xlabel("Minimum anticipation (time steps before peak)")
    # plt.ylabel("Fraction of anticipated peaks")
    # plt.title("Mean Recall vs Anticipation (P-model test set)")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()



    # Evaluation on real Tokamak data
    file_path = "test/data/jorek/heat_flux_iter.csv"
    # file_path = "test/data/jorek/heat_flux_iter_resampled.csv"
    # file_path = "test/data/geomagnetic/ae_index_normalizado.csv"
    # file_path = "test/data/sdo/sdo_time_series.csv"
    # file_path = "test/data/sdo/sdo_resampled.csv"

    df_real = load_and_prepare_csv(
        file_path,
        peak_percentile=config["data"]["peak_percentile"],
    )
    
    # Use percentile of lower values as reference and subtract it
    percentile_val = np.percentile(df_real['normalized_amplitude'], 10)
    df_real['normalized_amplitude'] = df_real['normalized_amplitude'] - percentile_val
    max_val = df_real['normalized_amplitude'].max()
    df_real['normalized_amplitude'] = df_real['normalized_amplitude'] / max_val
    print(df_real)


    evaluate_and_plot_streaming(
        model,
        df_real,
        device,
        lookahead=config["data"]["lookahead"],
        title_extra="(Tokamak)",
    )


if __name__ == "__main__":
    main()
