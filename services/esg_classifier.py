"""
ESG Classification Service using Pretrained Transformer Models
Supports ESGBERT and FinBERT for automated ESG category classification
"""

import os
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. ESG classification will use rule-based fallback.")


class ESGBertClassifier:
    """
    Classify text into ESG categories using pretrained ESGBERT model.
    
    Model: nbroad/ESG-BERT from HuggingFace
    Can classify text into: Environmental, Social, Governance, or None
    """
    
    def __init__(self, model_name: str = "nbroad/ESG-BERT", use_gpu: bool = False):
        """
        Initialize the ESGBERT classifier.
        
        Args:
            model_name: HuggingFace model identifier
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = None
        self.tokenizer = None
        self.classifier = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.error("Transformers not available. Cannot load ESGBERT.")
    
    def _load_model(self):
        """Load the pretrained model and tokenizer."""
        try:
            logger.info(f"Loading ESGBERT model: {self.model_name}")
            logger.info(f"Using GPU: {self.use_gpu}")
            
            device = 0 if self.use_gpu else -1
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=device,
                tokenizer=self.model_name
            )
            
            logger.info("ESGBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ESGBERT model: {str(e)}")
            self.classifier = None
    
    def classify(self, text: str) -> Dict[str, any]:
        """
        Classify a single text into an ESG category.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary with 'category', 'confidence', and 'all_scores'
        """
        if not self.classifier:
            return self._fallback_classification(text)
        
        try:
            # ESGBERT returns a list of predictions
            result = self.classifier(text, top_k=4)
            
            # Result format: [{'label': 'LABEL', 'score': 0.99}, ...]
            best = result[0] if result else None
            
            if best:
                # Try to map the top label
                category = self._map_label_to_category(best['label'])
                
                # If top label doesn't map, try other predictions
                if category is None and len(result) > 1:
                    for pred in result[1:]:
                        category = self._map_label_to_category(pred['label'])
                        if category is not None:
                            # Use the first mappable prediction and its confidence
                            return {
                                'category': category,
                                'confidence': pred['score'],
                                'raw_label': pred['label'],
                                'all_scores': {r['label']: r['score'] for r in result},
                                'all_predictions': result,
                                'note': f'Mapped from prediction #{result.index(pred)+1}'
                            }
                
                # Debug: print what labels we're getting
                logger.debug(f"ESGBERT output: {best['label']} -> {category}")
                logger.debug(f"All scores: {result}")
                
                return {
                    'category': category,
                    'confidence': best['score'],
                    'raw_label': best['label'],
                    'all_scores': {r['label']: r['score'] for r in result},
                    'all_predictions': result
                }
            
            return {'category': None, 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Error classifying text: {str(e)}")
            return self._fallback_classification(text)
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Classify multiple texts at once for efficiency.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of classification results
        """
        return [self.classify(text) for text in texts]
    
    def _map_label_to_category(self, label: str) -> Optional[str]:
        """
        Map ESGBERT output labels to our ESG categories.
        
        ESGBERT returns detailed labels like:
        - GHG_Emissions, Energy_Management, Water_Management -> E
        - Employee_Relations, Data_Privacy, Human_Rights -> S
        - Board, Executive_Compensation, Audit_Related -> G
        
        Args:
            label: Model output label
            
        Returns:
            'E', 'S', 'G', or None
        """
        label_lower = label.lower()
        
        # Environmental (E) keywords
        if any(kw in label_lower for kw in [
            'ghg', 'emission', 'carbon', 'energy', 'water', 'waste',
            'biodiversity', 'environmental', 'climate', 'pollution',
            'resource', 'renewable', 'sustainability', 'ecological'
        ]):
            return 'E'
        
        # Social (S) keywords
        elif any(kw in label_lower for kw in [
            'employee', 'human_rights', 'community', 'data_privacy',
            'customer', 'social', 'labor', 'workplace', 'diversity',
            'inclusion', 'safety', 'health', 'accessibility'
        ]):
            return 'S'
        
        # Governance (G) keywords
        elif any(kw in label_lower for kw in [
            'governance', 'board', 'executive', 'audit', 'compliance',
            'legal', 'regulatory', 'management_of_legal', 'ethics',
            'corporate', 'oversight', 'transparency', 'accountability',
            'systemic_risk', 'risk_management', 'data_security',
            'business_ethics', 'product_quality', 'quality_and_safety',
            'bribery', 'corruption', 'conflicts_of_interest'
        ]):
            return 'G'
        
        else:
            # Unmapped label
            logger.debug(f"Unmapped label: {label}")
            return None
    
    def _fallback_classification(self, text: str) -> Dict[str, any]:
        """
        Fallback rule-based classification if model unavailable.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary with basic classification
        """
        text_lower = text.lower()
        
        # Simple keyword-based classification
        env_keywords = ['environment', 'climate', 'carbon', 'emission', 'renewable', 'sustainability', 'green']
        social_keywords = ['diversity', 'workplace', 'employee', 'labor', 'social', 'community', 'human rights']
        gov_keywords = ['board', 'executive', 'governance', 'ethics', 'compliance', 'audit', 'regulation']
        
        env_score = sum(1 for kw in env_keywords if kw in text_lower)
        social_score = sum(1 for kw in social_keywords if kw in text_lower)
        gov_score = sum(1 for kw in gov_keywords if kw in text_lower)
        
        scores = {'E': env_score, 'S': social_score, 'G': gov_score}
        max_category = max(scores.items(), key=lambda x: x[1])
        
        if max_category[1] > 0:
            total = sum(scores.values())
            confidence = max_category[1] / total if total > 0 else 0
            return {
                'category': max_category[0],
                'confidence': confidence,
                'raw_label': f'rule_based_{max_category[0]}',
                'method': 'rule_based'
            }
        
        return {'category': None, 'confidence': 0.0, 'method': 'rule_based'}


class FinBERTClassifier:
    """
    Alternative classifier using FinBERT for financial sentiment and ESG classification.
    
    Model: ProsusAI/finbert (general financial sentiment)
    Can be fine-tuned for ESG-specific tasks
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize FinBERT classifier.
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.classifier = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.error("Transformers not available. Cannot load FinBERT.")
    
    def _load_model(self):
        """Load FinBERT model."""
        try:
            logger.info("Loading FinBERT model")
            device = 0 if self.use_gpu else -1
            self.classifier = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=device
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FinBERT: {str(e)}")
            self.classifier = None
    
    def get_sentiment(self, text: str) -> Dict[str, any]:
        """
        Get financial sentiment from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment label and score
        """
        if not self.classifier:
            return {'label': 'neutral', 'score': 0.0}
        
        try:
            result = self.classifier(text)
            return {
                'label': result['label'],
                'score': result['score']
            }
        except Exception as e:
            logger.error(f"Error in FinBERT sentiment: {str(e)}")
            return {'label': 'neutral', 'score': 0.0}


def get_classifier(model_type: str = "esgbert", use_gpu: bool = False):
    """
    Factory function to get appropriate classifier.
    
    Args:
        model_type: 'esgbert' or 'finbert'
        use_gpu: Whether to use GPU
        
    Returns:
        Classifier instance
    """
    if model_type.lower() == "esgbert":
        return ESGBertClassifier(use_gpu=use_gpu)
    elif model_type.lower() == "finbert":
        return FinBERTClassifier(use_gpu=use_gpu)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

