import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy
from statsmodels.tsa.stattools import acf
import antropy as ant
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import welch

class TimeSeriesComparator:
    def __init__(self, series1, series2, series1_name="Series1", series2_name="Series2", plot=True):
        self.s1 = np.array(series1)
        self.s2 = np.array(series2)
        self.name1 = series1_name
        self.name2 = series2_name
        self.plot = plot

    def basic_statistics(self, series):
        return {
            'mean': np.mean(series),
            'std': np.std(series),
            'skewness': skew(series),
            'kurtosis': kurtosis(series),
            'min': np.min(series),
            'max': np.max(series)
        }

    def compare_basic_stats(self):
        stats1 = self.basic_statistics(self.s1)
        stats2 = self.basic_statistics(self.s2)
        return stats1, stats2

    def plot_distributions(self):
        if not self.plot:
            return
        sns.kdeplot(self.s1, label=self.name1)
        sns.kdeplot(self.s2, label=self.name2)
        plt.title("Distribution Comparison")
        plt.legend()
        plt.show()

    def fft_magnitude(self, series):
        fft_vals = np.fft.fft(series)
        mag = np.abs(fft_vals)
        return mag[:len(mag)//2]

    def plot_fft(self):
        mag1 = self.fft_magnitude(self.s1)
        mag2 = self.fft_magnitude(self.s2)
        if self.plot:
            plt.plot(mag1, label=f'{self.name1} FFT Magnitude')
            plt.plot(mag2, label=f'{self.name2} FFT Magnitude', alpha=0.7)
            plt.title("FFT Magnitude Comparison")
            plt.legend()
            plt.show()

    def autocorrelation(self, series, nlags=100):
        return acf(series, nlags=nlags, fft=True)

    def plot_autocorrelation(self, nlags=100):
        acf1 = self.autocorrelation(self.s1, nlags=nlags)
        acf2 = self.autocorrelation(self.s2, nlags=nlags)
        if self.plot:
            plt.plot(acf1, label=f'{self.name1} Autocorrelation')
            plt.plot(acf2, label=f'{self.name2} Autocorrelation')
            plt.title("Autocorrelation Comparison")
            plt.legend()
            plt.show()

    def sample_entropy(self, series):
        return ant.sample_entropy(series)

    def compare_entropy(self):
        ent1 = self.sample_entropy(self.s1)
        ent2 = self.sample_entropy(self.s2)
        return ent1, ent2

    def dfa(self, signal, scale_min=4, scale_max=100, scale_density=20):
        N = len(signal)
        Y = np.cumsum(signal - np.mean(signal))
        scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num=scale_density).astype(int)
        scales = np.unique(scales)
        fluct = []

        for s in scales:
            n_segments = N // s
            if n_segments == 0:
                # pula escala que não comporta segmentos
                continue
            RMS = []
            for i in range(n_segments):
                idx_start = i * s
                idx_end = idx_start + s
                segment = Y[idx_start:idx_end]
                C = np.polyfit(range(s), segment, 1)
                fit = np.polyval(C, range(s))
                RMS.append(np.sqrt(np.mean((segment - fit) ** 2)))
            fluct.append(np.sqrt(np.mean(np.array(RMS) ** 2)))
        return np.array(scales[:len(fluct)]), np.array(fluct)

    def compute_psd(self, signal):
        freqs, psd = welch(signal, nperseg=len(signal) // 4)
        return freqs[1:], psd[1:]  # remove DC (freq = 0)

    def compare_dfa_psd(self):
        # --- DFA ---
        s1_scales, s1_fluct = self.dfa(self.s1)
        s2_scales, s2_fluct = self.dfa(self.s2)

        def filter_positive_pairs(x, y):
            x = np.array(x)
            y = np.array(y)
            mask = (x > 0) & (y > 0)
            return x[mask], y[mask]

        log_s1_scales, log_s1_fluct = filter_positive_pairs(s1_scales, s1_fluct)
        log_s2_scales, log_s2_fluct = filter_positive_pairs(s2_scales, s2_fluct)

        log_s1_scales = np.log10(log_s1_scales)
        log_s1_fluct = np.log10(log_s1_fluct)
        log_s2_scales = np.log10(log_s2_scales)
        log_s2_fluct = np.log10(log_s2_fluct)

        if len(log_s1_scales) < 2 or len(log_s1_fluct) < 2:
            slope_dfa1 = np.nan
        else:
            slope_dfa1, *_ = stats.linregress(log_s1_scales, log_s1_fluct)

        if len(log_s2_scales) < 2 or len(log_s2_fluct) < 2:
            slope_dfa2 = np.nan
        else:
            slope_dfa2, *_ = stats.linregress(log_s2_scales, log_s2_fluct)

        # --- PSD ---
        freqs1, psd1 = self.compute_psd(self.s1)
        freqs2, psd2 = self.compute_psd(self.s2)

        log_freqs1 = np.log10(freqs1)
        log_psd1 = np.log10(psd1)
        log_freqs2 = np.log10(freqs2)
        log_psd2 = np.log10(psd2)

        slope_psd1, *_ = stats.linregress(log_freqs1, log_psd1)
        slope_psd2, *_ = stats.linregress(log_freqs2, log_psd2)

        beta1 = -slope_psd1
        beta2 = -slope_psd2

        # --- Plot ---
        if self.plot:
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(2, 2)

            ax0 = fig.add_subplot(gs[0, :])
            ax0.plot(self.s1, label=self.name1)
            ax0.plot(self.s2, label=self.name2, alpha=0.7)
            ax0.set_title("Time Series Comparison")
            ax0.set_xlabel("Time Step")
            ax0.set_ylabel("Value")
            ax0.legend()
            ax0.grid(True)

            ax1 = fig.add_subplot(gs[1, 0])
            ax1.loglog(freqs1, psd1, 'o-', label=f'{self.name1} β={beta1:.3f}')
            ax1.loglog(freqs2, psd2, 'o-', label=f'{self.name2} β={beta2:.3f}', alpha=0.7)
            ax1.set_title("PSD (Welch)")
            ax1.set_xlabel("Frequency")
            ax1.set_ylabel("PSD")
            ax1.legend()
            ax1.grid(True, which='both', ls='--')

            ax2 = fig.add_subplot(gs[1, 1])
            ax2.loglog(s1_scales, s1_fluct, 'o-', label=f'{self.name1} α={slope_dfa1:.3f}')
            ax2.loglog(s2_scales, s2_fluct, 'o-', label=f'{self.name2} α={slope_dfa2:.3f}', alpha=0.7)
            ax2.set_title("DFA")
            ax2.set_xlabel("Window size (n)")
            ax2.set_ylabel("Fluctuation")
            ax2.legend()
            ax2.grid(True, which='both', ls='--')

            fig.tight_layout()
            plt.show()

        return slope_dfa1, slope_dfa2, beta1, beta2

    # --- Kullback-Leibler divergence ---

    @staticmethod
    def kullback_leibler_divergence(p, q):
        from scipy.stats import entropy
        return entropy(p, q)

    @staticmethod
    def estimate_distribution(data, bins=50, range_=None):
        hist, bin_edges = np.histogram(data, bins=bins, range=range_, density=True)
        pdf = hist / np.sum(hist)
        return pdf, bin_edges

    @staticmethod
    def compute_kl_divergence(series1, series2, bins=50):
        data_min = min(np.min(series1), np.min(series2))
        data_max = max(np.max(series1), np.max(series2))
        range_ = (data_min, data_max)

        p, _ = TimeSeriesComparator.estimate_distribution(series1, bins=bins, range_=range_)
        q, _ = TimeSeriesComparator.estimate_distribution(series2, bins=bins, range_=range_)

        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon

        p /= p.sum()
        q /= q.sum()

        return TimeSeriesComparator.kullback_leibler_divergence(p, q)

    def kullback_leibler(self, bins=50):
        kl_12 = self.compute_kl_divergence(self.s1, self.s2, bins)
        kl_21 = self.compute_kl_divergence(self.s2, self.s1, bins)
        return kl_12, kl_21

    def similarity_report(self):
        stats1, stats2 = self.compare_basic_stats()
        ent1, ent2 = self.compare_entropy()
        kl_12, kl_21 = self.kullback_leibler()

        report = {
            "Basic Statistics": {
                self.name1: stats1,
                self.name2: stats2,
            },
            "Sample Entropy": {
                self.name1: ent1,
                self.name2: ent2
            },
            "Kullback-Leibler Divergence": {
                f"{self.name1} vs {self.name2}": kl_12,
                f"{self.name2} vs {self.name1}": kl_21
            }
        }
        return report

    def display_similarity_report(self):
        stats1, stats2 = self.compare_basic_stats()
        ent1, ent2 = self.compare_entropy()
        kl_12, kl_21 = self.kullback_leibler()

        stats_df = pd.DataFrame({
            self.name1: stats1,
            self.name2: stats2
        })

        print("\n📊 Basic Statistics Comparison:")
        print(stats_df.round(4))

        entropy_df = pd.DataFrame({
            'Sample Entropy': [ent1, ent2]
        }, index=[self.name1, self.name2])

        print("\n🧠 Sample Entropy Comparison:")
        print(entropy_df.round(4))

        print("\n🔍 Kullback-Leibler Divergence:")
        print(f"{self.name1} vs {self.name2}: {kl_12:.4f}")
        print(f"{self.name2} vs {self.name1}: {kl_21:.4f}")
