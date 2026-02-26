"""
Explainability Service for ESG Classification Models
Provides SHAP and LIME explanations for model predictions
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
SHAP_AVAILABLE = False
LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP not available. Install with: pip install shap")

try:
    from lime import lime_text
    LIME_AVAILABLE = True
except ImportError:
    logger.warning("LIME not available. Install with: pip install lime")


class ESGExplainer:
    """
    Explain ESG classification model predictions using SHAP and LIME.
    """
    
    def __init__(self, classifier=None):
        """
        Initialize explainer with optional classifier.
        
        Args:
            classifier: The ESG classifier model to explain
        """
        self.classifier = classifier
        self.shap_explainer = None
        self.lime_explainer = None
        
        if LIME_AVAILABLE:
            self.lime_explainer = lime_text.LimeTextExplainer(class_names=['E', 'S', 'G', 'None'])
    
    def explain_with_lime(self, text: str, num_features: int = 10) -> Dict:
        """
        Generate LIME explanation for a prediction.
        
        Args:
            text: Text to explain
            num_features: Number of top features to show
            
        Returns:
            Dictionary with explanation data
        """
        if not LIME_AVAILABLE:
            return {'error': 'LIME not available'}
        
        if not self.lime_explainer or not self.classifier:
            return {'error': 'Explainer or classifier not initialized'}
        
        try:
            # LIME expects a function that returns prediction probabilities
            def predict_proba_wrapper(texts):
                results = []
                for txt in texts:
                    # Handle text list or text strings
                    text_content = txt if isinstance(txt, str) else txt
                    result = self.classifier.classify(text_content)
                    # Convert to probability vector [E, S, G, None]
                    probas = [0.0, 0.0, 0.0, 0.0]
                    cat_map = {'E': 0, 'S': 1, 'G': 2}
                    if result['category'] in cat_map:
                        idx = cat_map[result['category']]
                        probas[idx] = result['confidence']
                        # Distribute remaining confidence
                        remaining = 1.0 - result['confidence']
                        probas[3] = remaining
                    else:
                        probas[3] = 0.8
                        # Small probability to other categories
                        probas[0] = 0.07
                        probas[1] = 0.07
                        probas[2] = 0.06
                    results.append(probas)
                return np.array(results)
            
            # Generate explanation with faster settings
            explanation = self.lime_explainer.explain_instance(
                text,
                predict_proba_wrapper,
                num_features=num_features,
                top_labels=1,
                num_samples=100  # Reduce from default 5000 to 100 for faster execution
            )
            
            # Extract important words
            try:
                exp_list = explanation.as_list(label=explanation.top_labels[0])
            except Exception as e:
                logger.error(f"Error getting explanation list: {e}")
                # Fallback: return empty list
                exp_list = []
            
            return {
                'predicted_category': explanation.top_labels[0],
                'top_features': exp_list,
                'num_features': num_features,
                'method': 'lime'
            }
            
        except Exception as e:
            logger.error(f"LIME explanation error: {str(e)}")
            return {'error': str(e)}
    
    def explain_with_shap(self, text: str, background_texts: List[str] = None, max_evals: int = 50) -> Dict:
        """
        Generate SHAP explanation for a prediction.
        
        Args:
            text: Text to explain
            background_texts: Background texts for SHAP (optional)
            max_evals: Maximum evaluations
            
        Returns:
            Dictionary with SHAP values
        """
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not available'}
        
        if not self.classifier:
            return {'error': 'Classifier not initialized'}
        
        try:
            # Create a wrapper function for SHAP
            def model_wrapper(texts):
                results = []
                for txt in texts:
                    result = self.classifier.classify(txt)
                    # Return probability vector
                    probas = [0.0, 0.0, 0.0, 0.0]
                    cat_map = {'E': 0, 'S': 1, 'G': 2}
                    if result['category'] in cat_map:
                        idx = cat_map[result['category']]
                        probas[idx] = result['confidence']
                    else:
                        probas[3] = 1.0
                    results.append(probas)
                return np.array(results)
            
            # Initialize SHAP explainer
            if background_texts:
                masker = shap.maskers.Text(text_tokenizer=tokenizer)
                explainer = shap.Explainer(model_wrapper, masker)
            else:
                explainer = shap.Explainer(model_wrapper)
            
            # Get SHAP values
            shap_values = explainer([text], max_evals=max_evals)
            
            # Extract important features
            feature_names = []
            feature_values = []
            
            # Note: SHAP for text models is complex - this is simplified
            for val_array in shap_values.values:
                for i, val in enumerate(val_array[0]):  # Assuming first label
                    if abs(val) > 0.001:  # Threshold
                        feature_names.append(f"feature_{i}")
                        feature_values.append(float(val))
            
            return {
                'shap_values': shap_values.values.tolist(),
                'feature_names': feature_names,
                'feature_importances': dict(zip(feature_names, feature_values)),
                'method': 'shap'
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation error: {str(e)}")
            return {'error': str(e), 'fallback': True}
    
    def explain(self, text: str, method: str = "lime") -> Dict:
        """
        Main explanation function.
        
        Args:
            text: Text to explain
            method: 'lime' or 'shap'
            
        Returns:
            Explanation results
        """
        if method.lower() == "lime":
            return self.explain_with_lime(text)
        elif method.lower() == "shap":
            return self.explain_with_shap(text)
        else:
            return {'error': f'Unknown method: {method}'}
    
    def get_word_highlights(self, text: str, explanation: Dict) -> List[Dict]:
        """
        Extract word-level highlights from explanation.
        
        Args:
            text: Original text
            explanation: LIME or SHAP explanation
            
        Returns:
            List of word highlights with importance scores
        """
        if 'error' in explanation:
            return []
        
        words = text.split()
        highlights = []
        
        if explanation.get('method') == 'lime' and 'top_features' in explanation:
            # LIME gives us tuples like ('climate', 0.15) or ('word1 word2', 0.2)
            feature_map = {}
            for item in explanation['top_features']:
                text_part, score = item
                feature_map[text_part] = score
            
            # Find matching words
            for i, word in enumerate(words):
                # Check if this word or its stem matches any feature
                importance = 0.0
                for feature_text, score in feature_map.items():
                    if word.lower() in feature_text.lower() or feature_text.lower() in word.lower():
                        importance = score
                        break
                
                highlights.append({
                    'word': word,
                    'position': i,
                    'importance': float(importance)
                })
        
        else:
            # Default: equal importance
            for i, word in enumerate(words):
                highlights.append({
                    'word': word,
                    'position': i,
                    'importance': 0.0
                })
        
        return highlights


def create_explainer(classifier, method: str = "lime") -> ESGExplainer:
    """
    Factory function to create an explainer.
    
    Args:
        classifier: The model to explain
        method: Preferred explanation method
        
    Returns:
        ESGExplainer instance
    """
    return ESGExplainer(classifier=classifier)

