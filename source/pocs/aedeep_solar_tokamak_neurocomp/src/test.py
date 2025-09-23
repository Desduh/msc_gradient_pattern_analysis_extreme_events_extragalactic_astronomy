import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd

from model import PeakPredictorLSTM
from pmodel_generation import generate_multiple_series

from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score

def evaluate_and_plot_streaming(model, df, device, lookahead, threshold=0.5, zoom_range=(1000, 1500), title_extra=""):
    """
    Avalia série (sintética ou real), calcula métricas e plota série completa + zoom.
    """
    series = df['normalized_amplitude' if 'normalized_amplitude' in df else 'normalized'].values
    true_peaks = df['peak'].values.astype(int)
    x_axis = np.arange(len(series))

    hidden_states = model.init_hidden()
    predictions = []
    model.eval()
    with torch.no_grad():
        for value in series:
            x_t = torch.tensor([[[value]]], dtype=torch.float32).to(device)
            pred, hidden_states = model.step(x_t, hidden_states)
            predictions.append(pred.item())
    predictions = np.array(predictions)

    x_pred = x_axis - lookahead
    predictions[:lookahead] = np.nan
    valid_mask = ~np.isnan(predictions)

    y_true = true_peaks[valid_mask]
    y_score = predictions[valid_mask]
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "AUC-ROC": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
        "AUC-PR": average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }

    print(f"\n🔎 Métricas de desempenho {title_extra}:")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)

    axes[0].plot(x_axis, series, label="Sinal Normalizado", color="blue", lw=0.7)
    peak_indices = np.where(true_peaks == 1)[0]
    axes[0].scatter(peak_indices, series[peak_indices], color="red", label="Pico Real", s=40, zorder=5)
    axes[0].plot(x_pred, predictions, label=f"Probabilidade Prevista (lookahead={lookahead})", color="green", linestyle="--", lw=1.5)
    axes[0].set_title(f"Série Completa {title_extra}")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(x_axis, series, color="blue", lw=0.7)
    axes[1].scatter(peak_indices, series[peak_indices], color="red", s=40, zorder=5)
    axes[1].plot(x_pred, predictions, color="green", linestyle="--", lw=1.5)
    axes[1].set_xlim(*zoom_range)
    axes[1].set_title(f"Zoom {zoom_range[0]}–{zoom_range[1]} {title_extra}")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    return metrics


def load_and_prepare_csv(file_path, peak_threshold, lookahead):
    """
    Carrega CSV, normaliza e cria labels de pico.
    """
    print(f"\n--- Processando arquivo: {os.path.basename(file_path)} ---")
    df = pd.read_csv(file_path)
    df.columns = ['time', 'amplitude']

    # Normalização
    amp = df['amplitude'].values
    if amp.min() >= 0 and amp.max() <= 1:
        df['normalized_amplitude'] = df['amplitude']
    else:
        min_val, max_val = amp.min(), amp.max()
        df['normalized_amplitude'] = (amp - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(amp)

    # Labels
    df['peak'] = (df['normalized_amplitude'] > peak_threshold).astype(int)

    # Ground truth antecipado
    rolled_max = df['peak'].rolling(window=lookahead, min_periods=1).max().shift(-lookahead+1).bfill()
    df['ground_truth_for_prediction'] = (rolled_max > 0).astype(int)

    return df


def main():
    # Carregar config
    with open('src/config.json', 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carregar modelo salvo
    model = PeakPredictorLSTM(config['model']).to(device)
    model.load_state_dict(torch.load(config['files']['model_save_path'], map_location=device))
    print("✅ Modelo carregado")

    # ---------- Avaliação com dados sintéticos ----------
    df_test = generate_multiple_series(
        length=config['p_model']['series_length_train'],
        p_value=config['p_model']['p_value'],
        num_series=config['data']['num_series_test'],
        peak_threshold=config['data']['peak_threshold'],
        norm_percentile=config['data']['normalization_percentile']
    )

    evaluate_and_plot_streaming(
        model,
        df_test,
        device,
        lookahead=config['data']['lookahead'],
        title_extra="(P-Model)"
    )

    # ---------- Avaliação com dados reais ----------
    file_path = "test/heat_flux_iter.csv"  
    df_real = load_and_prepare_csv(file_path, config['data']['peak_threshold'], config['data']['lookahead'])

    evaluate_and_plot_streaming(
        model,
        df_real,
        device,
        lookahead=config['data']['lookahead'],
        title_extra="(Tokamak)"
    )


if __name__ == "__main__":
    main()
