import numpy as np
import pandas as pd
from services.adaptive_fusion import AdaptiveFusionEngine

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
                article_count=data.get('article_count', 0)
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
    benchmarker = StabilityBenchmarker()
    report = benchmarker.measure_noise_resistance()
    print("--- INVENTION STABILITY REPORT ---")
    print(f"Static Model Volatility: {report['static_volatility']:.4f}")
    print(f"Adaptive Fusion Volatility: {report['adaptive_volatility']:.4f}")
    print(f"Invention reduces noise sensitivity by: {report['volatility_reduction_pct']:.2f}%")
