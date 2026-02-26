import numpy as np
import pandas as pd
from services.adaptive_fusion import AdaptiveFusionEngine


def _filter_company_data_benchmark(df: pd.DataFrame, esg_categories, start, end) -> pd.DataFrame:
    """Replicate production filter_company_data for benchmark (no Streamlit dependency)."""
    comps = []
    for cat in esg_categories:
        if cat in df.columns:
            comps.append(df[df[cat] == True])
    if not comps:
        filtered = df.copy()
    else:
        filtered = pd.concat(comps).drop_duplicates()
    if hasattr(start, "date"):
        start = start.date()
    if hasattr(end, "date"):
        end = end.date()
    if "DATE" in filtered.columns:
        filtered = filtered[filtered["DATE"].between(start, end)]
    return filtered


def run_fusion_mode_benchmark(
    avg_tone: float,
    structured_score: float,
    momentum: float,
    tone_std: float,
    article_count: int,
    daily_sentiment_series=None,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
) -> dict:
    """
    Controlled benchmark comparison between three ESG fusion modes.
    Isolated evaluation - does NOT modify production logic.

    Quantifies whether temporal reliability reduces volatility compared to
    static and old-reliability approaches.

    Args:
        avg_tone: Base aggregated sentiment score (used when daily_sentiment_series is sparse)
        structured_score: Mean of pillar-based (E, S, G) structured data
        momentum: Sentiment/Score delta over time
        tone_std: Base standard deviation of tone (used for rolling when series available)
        article_count: Number of evidence signals
        daily_sentiment_series: Optional array-like of daily mean sentiment values (one per day)
        alpha: Sentiment weight base (default 0.4)
        beta: Structured score weight base (default 0.4)
        gamma: Momentum weight base (default 0.2)

    Returns:
        dict with keys:
            static_variance, old_reliability_variance, new_reliability_variance,
            volatility_reduction_vs_static, volatility_reduction_vs_old
    """
    # ---- Normalize helper (mirrors engine logic) ----
    def _normalize(w_sent, w_struct, w_momentum):
        total = w_sent + w_struct + w_momentum
        if total <= 0:
            return 1.0 / 3, 1.0 / 3, 1.0 / 3
        return w_sent / total, w_struct / total, w_momentum / total

    # ---- Safe scalar conversion ----
    safe_struct = float(structured_score) if not pd.isna(structured_score) else 0.0
    safe_momentum = float(momentum) if not pd.isna(momentum) else 0.0

    # ---- Build time series for simulation ----
    if daily_sentiment_series is not None:
        try:
            arr = np.asarray(daily_sentiment_series, dtype=float)
            arr = arr[~np.isnan(arr)]
        except (TypeError, ValueError):
            arr = np.array([avg_tone])
    else:
        arr = np.array([avg_tone])

    if len(arr) < 2:
        arr = np.array([avg_tone, avg_tone])

    n_steps = len(arr)
    static_scores = []
    old_scores = []
    new_scores = []

    engine = AdaptiveFusionEngine(alpha=alpha, beta=beta, gamma=gamma)

    for i in range(n_steps):
        avg_tone_i = float(arr[i]) if not pd.isna(arr[i]) else 0.0
        window = arr[max(0, i - 6) : i + 1]
        tone_std_i = (
            float(np.std(window))
            if len(window) >= 2
            else max(0.0, float(tone_std) if not pd.isna(tone_std) else 0.0)
        )
        article_count_i = int(article_count) if not pd.isna(article_count) else 0
        history_for_temporal = arr[: i + 1] if i >= 0 else None

        # ---- A. Static Mode ----
        w_s, w_st, w_m = _normalize(alpha, beta, gamma)
        score_static = w_s * avg_tone_i + w_st * safe_struct + w_m * safe_momentum
        static_scores.append(score_static)

        # ---- B. Old Reliability Mode (no temporal factor) ----
        if article_count_i == 0 or pd.isna(tone_std_i):
            reliability_old = 0.0
        else:
            safe_std = max(0.0, tone_std_i)
            stability = 1.0 / (1.0 + safe_std)
            volume_factor = np.log1p(float(article_count_i))
            reliability_old = stability * volume_factor
        w_sent_old = alpha * reliability_old
        w_s_old, w_st_old, w_m_old = _normalize(w_sent_old, beta, gamma)
        score_old = w_s_old * avg_tone_i + w_st_old * safe_struct + w_m_old * safe_momentum
        old_scores.append(score_old)

        # ---- C. New Reliability Mode (with temporal factor) ----
        reliability_new = engine.compute_reliability(
            tone_std_i, article_count_i, daily_sentiment_series=history_for_temporal
        )
        w_sent_new = alpha * reliability_new
        w_s_new, w_st_new, w_m_new = _normalize(w_sent_new, beta, gamma)
        score_new = w_s_new * avg_tone_i + w_st_new * safe_struct + w_m_new * safe_momentum
        new_scores.append(score_new)

    static_series = np.array(static_scores, dtype=float)
    old_series = np.array(old_scores, dtype=float)
    new_series = np.array(new_scores, dtype=float)

    var_static = float(np.var(static_series))
    var_old = float(np.var(old_series))
    var_new = float(np.var(new_series))

    volatility_reduction_vs_static = (
        (1.0 - (var_new / var_static)) * 100.0 if var_static > 0 else 0.0
    )
    volatility_reduction_vs_old = (
        (1.0 - (var_new / var_old)) * 100.0 if var_old > 0 else 0.0
    )

    return {
        "static_variance": var_static,
        "old_reliability_variance": var_old,
        "new_reliability_variance": var_new,
        "volatility_reduction_vs_static": volatility_reduction_vs_static,
        "volatility_reduction_vs_old": volatility_reduction_vs_old,
    }


def run_real_data_benchmark(
    start_label: str = "dec30",
    end_label: str = "jan12",
    esg_categories=None,
    min_companies: int = 10,
) -> list:
    """
    Generate real benchmark results using actual company data from the dataset.
    Builds context exactly as production does and calls run_fusion_mode_benchmark.
    Isolated - does NOT modify production fusion logic or UI.

    Args:
        start_label: Dataset start (e.g. dec30 for dec30_to_jan12)
        end_label: Dataset end (e.g. jan12)
        esg_categories: E/S/G filter, default ["E", "S", "G"]
        min_companies: Minimum companies to process (default 10)

    Returns:
        List of per-company result dicts.
    """
    if esg_categories is None:
        esg_categories = ["E", "S", "G"]

    from download_data import Data
    from utils.analytics import build_company_context, filter_on_date

    data = Data().read(start_label, end_label)
    df_data = data["data"]
    companies = sorted(df_data["Organization"].dropna().unique().tolist())
    companies = [c for c in companies if c and str(c).strip() != "Select a Company"]

    start = df_data["DATE"].min()
    end = df_data["DATE"].max()
    market_scope = filter_on_date(df_data, start, end)
    df_market = _filter_company_data_benchmark(market_scope, esg_categories, start, end)

    results = []
    for company in companies:
        df_company_full = df_data[df_data["Organization"] == company]
        df_company = _filter_company_data_benchmark(
            df_company_full, esg_categories, start, end
        )
        if df_company.empty or len(df_company) < 2:
            continue

        context = build_company_context(
            company, df_company, df_market, data, start, end
        )

        daily_series = context.get("daily_sentiment_series")
        if daily_series is None or (hasattr(daily_series, "__len__") and len(daily_series) < 2):
            continue

        esg_scores = context.get("esg_scores") or {}
        structured_score = (
            float(np.nanmean(list(esg_scores.values())))
            if esg_scores else 0.0
        )

        benchmark_result = run_fusion_mode_benchmark(
            avg_tone=context.get("avg_tone") or 0.0,
            structured_score=structured_score,
            momentum=context.get("tone_change") or 0.0,
            tone_std=context.get("tone_std") or 0.0,
            article_count=context.get("article_count") or 0,
            daily_sentiment_series=daily_series,
        )

        results.append({
            "company": company,
            **benchmark_result,
        })

    return results


def print_benchmark_report(results: list) -> None:
    """
    Print benchmark results with precise aggregate metrics.
    Uses correct formulae for percentage reductions.
    Suitable for patent documentation.
    """
    if not results:
        print("No benchmark results to display.")
        return

    df = pd.DataFrame(results)

    # Aggregate means (pandas)
    mean_static_variance = float(df["static_variance"].mean())
    mean_old_reliability_variance = float(df["old_reliability_variance"].mean())
    mean_new_reliability_variance = float(df["new_reliability_variance"].mean())

    # Percentage reductions using correct formula:
    # reduction = ((base_variance - new_variance) / base_variance) * 100
    mean_volatility_reduction_vs_static = (
        ((mean_static_variance - mean_new_reliability_variance) / mean_static_variance)
        * 100.0
        if mean_static_variance > 0
        else 0.0
    )
    mean_volatility_reduction_vs_old = (
        ((mean_old_reliability_variance - mean_new_reliability_variance) / mean_old_reliability_variance)
        * 100.0
        if mean_old_reliability_variance > 0
        else 0.0
    )

    # Per-company reduction vs old: for min, max, std
    reduction_vs_old = df["volatility_reduction_vs_old"]
    min_reduction_vs_old = float(reduction_vs_old.min()) if len(reduction_vs_old) else 0.0
    max_reduction_vs_old = float(reduction_vs_old.max()) if len(reduction_vs_old) else 0.0
    std_reduction_vs_old = float(reduction_vs_old.std()) if len(reduction_vs_old) > 1 else 0.0

    total_companies = len(df)

    # Print aggregate block
    print("\n" + "=" * 60)
    print("Benchmark Results (Averaged Across Companies)")
    print("=" * 60)
    print(f"Static Variance: {mean_static_variance:.6f}")
    print(f"Old Reliability Variance: {mean_old_reliability_variance:.6f}")
    print(f"New Reliability Variance: {mean_new_reliability_variance:.6f}")
    print(f"Volatility Reduction vs Static: {mean_volatility_reduction_vs_static:.2f} %")
    print(f"Volatility Reduction vs Old: {mean_volatility_reduction_vs_old:.2f} %")
    print("=" * 60)

    # Additional metrics
    print("\n--- Evaluation Summary ---")
    print(f"Total companies evaluated: {total_companies}")
    print(f"Minimum reduction vs old reliability: {min_reduction_vs_old:.2f} %")
    print(f"Maximum reduction vs old reliability: {max_reduction_vs_old:.2f} %")
    print(f"Standard deviation of reduction vs old: {std_reduction_vs_old:.2f} %")

    display_cols = [
        "company",
        "static_variance",
        "old_reliability_variance",
        "new_reliability_variance",
        "volatility_reduction_vs_static",
        "volatility_reduction_vs_old",
    ]
    table_df = df[display_cols].copy()
    table_df["volatility_reduction_vs_static"] = (
        table_df["volatility_reduction_vs_static"].round(2).astype(str) + " %"
    )
    table_df["volatility_reduction_vs_old"] = (
        table_df["volatility_reduction_vs_old"].round(2).astype(str) + " %"
    )
    table_df.columns = [
        "Company",
        "Static Var",
        "Old Rel Var",
        "New Rel Var",
        "Reduction vs Static",
        "Reduction vs Old",
    ]
    print("\nPer-company results:")
    print(table_df.to_string(index=False))
    print()


class StabilityBenchmarker:
    """
    Compares the Adaptive Fusion Engine against a baseline static model.
    Used to demonstrate technical improvement and non-obviousness.
    """
    def __init__(self, engine: AdaptiveFusionEngine = None):
        self.engine = engine or AdaptiveFusionEngine()

    def run_benchmark(self, sample_data_list: list) -> pd.DataFrame:
        """
        Runs comparison between Adaptive and Static models.
        
        Args:
            sample_data_list: List of dicts with (avg_tone, structured_score, momentum, tone_std, article_count)
            
        Returns:
            DataFrame with comparison results.
        """
        results = []
        for i, data in enumerate(sample_data_list):
            # 1. Static Baseline (Equal 50/50 weighting of sentiment and structured)
            # Ensure safe numerical conversion
            val_tone = float(data.get('avg_tone', 0)) if not pd.isna(data.get('avg_tone')) else 0.0
            val_struct = float(data.get('structured_score', 0)) if not pd.isna(data.get('structured_score')) else 0.0
            
            static_score = (val_tone * 0.5) + (val_struct * 0.5)
            
            # 2. Adaptive Fusion (The Invention)
            fusion_result = self.engine.fuse(
                avg_tone=data.get('avg_tone', 0.0),
                structured_score=data.get('structured_score', 0.0),
                momentum=data.get('momentum', 0.0),
                tone_std=data.get('tone_std', 0.0),
                article_count=data.get('article_count', 0),
                daily_sentiment_series=data.get('daily_sentiment_series', None)
            )
            adaptive_score = fusion_result['final_esg_score']
            
            results.append({
                "Sample": i + 1,
                "Noise_Lev": data['tone_std'],
                "Volume": data['article_count'],
                "Static_Score": static_score,
                "Adaptive_Score": adaptive_score,
                "Variance_Delta": abs(adaptive_score - static_score),
                "Reliability": fusion_result['reliability_metric']
            })
            
        return pd.DataFrame(results)

    def measure_noise_resistance(self):
        """
        Simulates noise injection to prove adaptive stability.
        """
        # Scenario: Corporate claim is constant (0.8), but media sentiment is erratic
        samples = []
        for i in range(10):
            # Media tone swings wildly
            noisy_tone = np.random.uniform(-1, 1)
            # High noise (std), low volume (article_count)
            samples.append({
                "avg_tone": noisy_tone, 
                "structured_score": 0.8, 
                "momentum": 0.0, 
                "tone_std": 2.0, 
                "article_count": 5
            })
        
        df = self.run_benchmark(samples)
        
        # Calculate volatility metric
        static_volatility = df['Static_Score'].std()
        adaptive_volatility = df['Adaptive_Score'].std()
        
        # In this scenario, we want adaptive volatility to be LOWER than static
        reduction = (1 - (adaptive_volatility / static_volatility)) * 100 if static_volatility > 0 else 0
        
        return {
            "static_volatility": static_volatility,
            "adaptive_volatility": adaptive_volatility,
            "volatility_reduction_pct": reduction
        }

if __name__ == "__main__":
    print("--- SYNTHETIC NOISE RESISTANCE ---")
    benchmarker = StabilityBenchmarker()
    report = benchmarker.measure_noise_resistance()
    print(f"Static Model Volatility: {report['static_volatility']:.4f}")
    print(f"Adaptive Fusion Volatility: {report['adaptive_volatility']:.4f}")
    print(f"Invention reduces noise sensitivity by: {report['volatility_reduction_pct']:.2f}%")

    print("\n--- REAL DATA BENCHMARK ---")
    try:
        results = run_real_data_benchmark(min_companies=10)
        if len(results) >= 10:
            print_benchmark_report(results)
        else:
            print(
                f"Processed {len(results)} companies (need 10+ with sufficient daily data)."
            )
            if results:
                print_benchmark_report(results)
    except Exception as e:
        print(f"Real data benchmark failed: {e}")
        print("Ensure Data/dec30_to_jan12/ exists with data_as_csv.csv")
