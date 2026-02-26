import numpy as np
import pandas as pd

class AdaptiveFusionEngine:
    """
    Computes an adaptive ESG index by dynamically weighting sentiment, 
    structured scores, and temporal momentum based on data reliability.
    
    PATENTABLE ASPECT: Reliability-based dynamic weight modulation.
    """
    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2, decay_lambda=0.05):
        self.alpha = alpha   # sentiment weight base
        self.beta = beta     # structured ESG weight base
        self.gamma = gamma   # momentum weight base
        self.decay_lambda = decay_lambda

    def compute_reliability(self, tone_std, article_count):
        """
        Calculates the reliability of the sentiment signal.
        Reliability decreases as variance (std) increases and increases with volume.
        """
        if article_count == 0 or pd.isna(tone_std) or pd.isna(article_count):
            return 0.0
        # Stability is inverse of variance (clip std to avoid division by zero or negative)
        safe_std = max(0, float(tone_std))
        stability = 1 / (1 + safe_std)
        # Volume factor is logarithmic to prevent over-weighting massive spikes
        volume_factor = np.log1p(float(article_count))
        return float(stability * volume_factor)

    def time_decay(self, days_since_event):
        """Temporal decay function for event relevance."""
        return np.exp(-self.decay_lambda * days_since_event)

    def normalize(self, values):
        """Normalizes weights to ensure they sum to 1."""
        total = sum(values)
        if total <= 0:
            # Fallback to equal weights if total is zero
            return [1.0/len(values)] * len(values)
        return [v / total for v in values]

    def fuse(self, avg_tone, structured_score, momentum, tone_std, article_count):
        """
        Perform the adaptive fusion of ESG factors.
        
        Args:
            avg_tone: Aggregated sentiment score from unstructured data (GDELT/News)
            structured_score: Mean of pillar-based (E, S, G) structured data
            momentum: Sentiment/Score delta over time
            tone_std: Standard deviation of tone (signal noise)
            article_count: Number of evidence signals
            
        Returns:
            Dictionary with final score and dynamic weights used.
        """
        # 1. Compute Reliability Context
        reliability = self.compute_reliability(tone_std, article_count)

        # 2. Modulate Weights based on Reliability
        # If sentiment is low reliability (high std or low volume), its weight alpha decreases
        w_sent = self.alpha * reliability
        w_struct = self.beta
        w_momentum = self.gamma

        # 3. Normalize to preserve scale
        w_sent, w_struct, w_momentum = self.normalize(
            [w_sent, w_struct, w_momentum]
        )

        # 4. Compute Composite Index
        # Ensure values are float and not NaN
        safe_avg_tone = float(avg_tone) if not pd.isna(avg_tone) else 0.0
        safe_struct = float(structured_score) if not pd.isna(structured_score) else 0.0
        safe_momentum = float(momentum) if not pd.isna(momentum) else 0.0

        final_score = (
            w_sent * safe_avg_tone +
            w_struct * safe_struct +
            w_momentum * safe_momentum
        )

        return {
            "final_esg_score": float(final_score),
            "weights": {
                "sentiment": float(w_sent),
                "structured": float(w_struct),
                "momentum": float(w_momentum),
            },
            "reliability_metric": float(reliability)
        }
