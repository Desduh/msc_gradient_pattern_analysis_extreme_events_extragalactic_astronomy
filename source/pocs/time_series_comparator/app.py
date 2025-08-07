import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from models.time_series_comparator import TimeSeriesComparator
from models.pmodel_generation import generate_and_prepare_series

# ===============================
# Dataset configuration
# ===============================
DATASETS = {
    "tokamak": {
        "path": r"C:\Users\Carlos\Documents\INPE\master\aedeep_deeplearning_plasma_astrophysics_masters_research\source\data\jorek\heat_flux_iter.csv",
        "column": "Amplitude",
        "label": "Tokamak"
    },
    "sdo": {
        "path": r"C:\Users\Carlos\Documents\INPE\master\aedeep_deeplearning_plasma_astrophysics_masters_research\source\data\sdo\sdo_time_series.csv",
        "column": "Amplitude",
        "label": "SDO"
    },
    "rho": {
        "path": r"C:\Users\Carlos\Documents\INPE\master\aedeep_deeplearning_plasma_astrophysics_masters_research\source\data\rho\ae_index_normalizado.csv",
        "column": "norm",
        "label": "RHO"
    }
}

# ===============================
# Helper functions
# ===============================
def load_and_normalize_series(series_key):
    """
    Load a time series from CSV and normalize it to [0, 1].
    """
    if series_key not in DATASETS:
        raise ValueError(f"Dataset '{series_key}' is not defined.")

    config = DATASETS[series_key]
    df = pd.read_csv(config["path"])
    raw_series = df[config["column"]].values
    scaler = MinMaxScaler()
    normalized_series = scaler.fit_transform(raw_series.reshape(-1, 1)).flatten()
    return normalized_series, config["label"]

def run_comparisons(series_data, series_key, series_label, p_values, plot=True):
    """
    Compare a time series with multiple P-models for a range of p values.
    """
    results = []
    for p in p_values:
        print(f"\n=============================")
        print(f"🔍 Evaluating P-model with p = {p:.2f}")

        # Generate P-model
        pmodel_df = generate_and_prepare_series(length=len(series_data), p_value=p, peak_threshold=0.5, seed=42)
        pmodel_series = pmodel_df['normalized'].values

        # Center both series
        s1 = series_data - np.mean(series_data)
        s2 = pmodel_series - np.mean(pmodel_series)

        # Create comparator
        ts_comp = TimeSeriesComparator(s1, s2, series1_name=series_label, series2_name=f"P-model p={p:.2f}", plot=plot)

        # Extract metrics
        stats1, stats2 = ts_comp.compare_basic_stats()
        ent1, ent2 = ts_comp.compare_entropy()
        kl_pq, kl_qp = ts_comp.kullback_leibler()
        dfa_alpha1, dfa_alpha2, beta1, beta2 = ts_comp.compare_dfa_psd()

        # Optional: plots
        ts_comp.display_similarity_report()
        ts_comp.plot_distributions()
        ts_comp.plot_fft()
        ts_comp.plot_autocorrelation()

        # Store result
        results.append({
            'p_value': p,
            f'{series_key}_mean': stats1['mean'],
            'pmodel_mean': stats2['mean'],
            f'{series_key}_std': stats1['std'],
            'pmodel_std': stats2['std'],
            f'{series_key}_skewness': stats1['skewness'],
            'pmodel_skewness': stats2['skewness'],
            f'{series_key}_kurtosis': stats1['kurtosis'],
            'pmodel_kurtosis': stats2['kurtosis'],
            f'{series_key}_min': stats1['min'],
            'pmodel_min': stats2['min'],
            f'{series_key}_max': stats1['max'],
            'pmodel_max': stats2['max'],
            f'{series_key}_entropy': ent1,
            'pmodel_entropy': ent2,
            'kl_series_pmodel': kl_pq,
            'kl_pmodel_series': kl_qp,
            f'dfa_{series_key}': dfa_alpha1,
            'dfa_pmodel': dfa_alpha2,
            f'psd_{series_key}': beta1,
            'psd_pmodel': beta2
        })

    return pd.DataFrame(results)

def plot_all_metrics_vs_p(results_df, series_key):
    """
    Plot comparison metrics between the time series and P-models.
    """
    basic_metrics = ['skewness', 'kurtosis']

    # Total de métricas: skewness, kurtosis, entropy, KL, DFA, PSD
    n_metrics = len(basic_metrics) + 1 + 1 + 1 + 1  # +1 entropy, +1 KL, +1 DFA, +1 PSD
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    # Plot basic stats
    for i, metric in enumerate(basic_metrics):
        ax = plt.subplot(n_rows, n_cols, i + 1)

        p_vals = results_df['p_value']
        pmodel_vals = results_df[f'pmodel_{metric}']
        series_val = results_df[f'{series_key}_{metric}'].iloc[0]
        abs_diff = np.abs(pmodel_vals - series_val)

        ax.plot(p_vals, pmodel_vals, marker='o', label='P-model')
        ax.axhline(y=series_val, color='r', linestyle='--', label=series_key.capitalize())
        ax.plot(p_vals, abs_diff, marker='s', linestyle='--', color='black', label='|Diff|')

        ax.set_title(f'{metric.capitalize()} Comparison')
        if i >= n_metrics - n_cols:
            ax.set_xlabel('p value')
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)

    # Plot entropy
    i = len(basic_metrics)
    ax = plt.subplot(n_rows, n_cols, i + 1)
    pmodel_vals = results_df['pmodel_entropy']
    series_val = results_df[f'{series_key}_entropy'].iloc[0]
    abs_diff = np.abs(pmodel_vals - series_val)

    ax.plot(results_df['p_value'], pmodel_vals, marker='o', label='P-model')
    ax.axhline(y=series_val, color='r', linestyle='--', label=series_key.capitalize())
    ax.plot(results_df['p_value'], abs_diff, marker='s', linestyle='--', color='black', label='|Diff|')

    ax.set_title('Sample Entropy Comparison')
    if i >= n_metrics - n_cols:
        ax.set_xlabel('p value')
    else:
        ax.set_xticklabels([])
    ax.set_ylabel('Sample Entropy')
    ax.legend()
    ax.grid(True)

    # Plot KL divergence
    i += 1
    ax = plt.subplot(n_rows, n_cols, i + 1)
    ax.plot(results_df['p_value'], results_df['kl_series_pmodel'], marker='o', label=f'KL {series_key.capitalize()} → P-model')
    # ax.plot(results_df['p_value'], results_df['kl_pmodel_series'], marker='o', label=f'KL P-model → {series_key.capitalize()}')

    ax.set_title('KL Divergence')
    if i >= n_metrics - n_cols:
        ax.set_xlabel('p value')
    else:
        ax.set_xticklabels([])
    ax.set_ylabel('KL Divergence')
    ax.legend()
    ax.grid(True)

    # Plot DFA exponent
    i += 1
    ax = plt.subplot(n_rows, n_cols, i + 1)
    pmodel_vals = results_df['dfa_pmodel']
    series_val = results_df[f'dfa_{series_key}'].iloc[0]
    abs_diff = np.abs(pmodel_vals - series_val)

    ax.plot(results_df['p_value'], pmodel_vals, marker='o', label='P-model')
    ax.axhline(y=series_val, color='r', linestyle='--', label=series_key.capitalize())
    ax.plot(results_df['p_value'], abs_diff, marker='s', linestyle='--', color='black', label='|Diff|')

    ax.set_title('DFA Exponent (α) Comparison')
    if i >= n_metrics - n_cols:
        ax.set_xlabel('p value')
    else:
        ax.set_xticklabels([])
    ax.set_ylabel('DFA α')
    ax.legend()
    ax.grid(True)

    # Plot PSD exponent
    i += 1
    ax = plt.subplot(n_rows, n_cols, i + 1)
    pmodel_vals = results_df['psd_pmodel']
    series_val = results_df[f'psd_{series_key}'].iloc[0]
    abs_diff = np.abs(pmodel_vals - series_val)

    ax.plot(results_df['p_value'], pmodel_vals, marker='o', label='P-model')
    ax.axhline(y=series_val, color='r', linestyle='--', label=series_key.capitalize())
    ax.plot(results_df['p_value'], abs_diff, marker='s', linestyle='--', color='black', label='|Diff|')

    ax.set_title('PSD Exponent (β) Comparison')
    if i >= n_metrics - n_cols:
        ax.set_xlabel('p value')
    else:
        ax.set_xticklabels([])
    ax.set_ylabel('PSD β')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

# ===============================
# Main execution
# ===============================
if __name__ == "__main__":
    # Select series to analyze
    series_key = "tokamak"  # Change to 'tokamak' 'sdo' or 'rho'

    # Load and normalize the series
    series_data, series_label = load_and_normalize_series(series_key)

    # Define p values to test
    p_values = np.linspace(0.1, 0.49, 20)

    # Run the comparisons
    results_df = run_comparisons(series_data, series_key, series_label, p_values, plot=False)

    # Save results
    results_df.to_csv(f"{series_key}_pmodel_comparison_results.csv", index=False)
    print(f"\n✅ Comparison completed. Results saved to '{series_key}_pmodel_comparison_results.csv'.")

    # Plot results
    plot_all_metrics_vs_p(results_df, series_key)
