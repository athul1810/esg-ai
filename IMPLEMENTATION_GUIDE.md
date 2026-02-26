# ESG AI Enhancement Implementation Guide 🛠️

## Overview
This guide shows you how to implement and test the new ML features in your ESG AI dashboard.

---

## 🎯 What's Been Added

### 1. New Services Created

✅ **`services/esg_classifier.py`** - ESGBERT & FinBERT integration
✅ **`services/explainability.py`** - SHAP & LIME explanations
✅ **`services/data_ingestion.py`** - Real-time data fetching (NewsAPI, SEC)
✅ **`ENHANCEMENT_ROADMAP.md`** - Full roadmap for all features
✅ **Updated `requirements.txt`** - All necessary dependencies

---

## 🚀 How to Get Started

### Step 1: Install Dependencies

```bash
# Activate your virtual environment
source venv/bin/activate

# Install new requirements
pip install -r requirements.txt
```

**Note:** Some packages are large and may take time:
- `torch` (~2GB) - for transformer models
- `transformers` - for ESGBERT/FinBERT
- `shap`, `lime` - for explainability

### Step 2: Get API Keys (Optional but Recommended)

Create a `.env` file in your project root:

```bash
# .env
NEWSAPI_KEY=your_newsapi_key_here
OPENAI_API_KEY=your_existing_openai_key
```

Get NewsAPI key: https://newsapi.org/register

### Step 3: Test the Models

Create a test script:

```python
# test_new_services.py
from services.esg_classifier import ESGBertClassifier
from services.explainability import ESGExplainer
from services.data_ingestion import fetch_company_news

# Test ESGBERT
print("Testing ESGBERT...")
classifier = ESGBertClassifier()
text = "Company announces new renewable energy initiative and carbon neutrality goals"
result = classifier.classify(text)
print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.2f}")

# Test Explainability
print("\nTesting LIME explanations...")
explainer = ESGExplainer(classifier=classifier)
explanation = explainer.explain_with_lime(text)
print(f"Top features: {explanation['top_features'][:5]}")

# Test Data Ingestion
print("\nTesting NewsAPI...")
articles = fetch_company_news("Apple", days=1)
print(f"Found {len(articles)} articles")
```

Run: `python test_new_services.py`

---

## 🔧 Integration with Existing App

### Option 1: Add New Tab to Streamlit App

Add to `app.py`:

```python
# Add new imports
from services.esg_classifier import ESGBertClassifier
from services.explainability import ESGExplainer

# In your main() function, add a new tab
esg_ml_tab = st.tabs(["Overview", "Advisory", "ML Analysis"])[2]

with esg_ml_tab:
    st.markdown("### 🤖 ML-Powered ESG Classification")
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        with st.spinner("Loading ESGBERT..."):
            st.session_state.classifier = ESGBertClassifier()
    
    # Text input
    text_to_analyze = st.text_area(
        "Enter text to classify",
        placeholder="Company announces new diversity initiatives..."
    )
    
    if st.button("Classify"):
        result = st.session_state.classifier.classify(text_to_analyze)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Category", result['category'] or "None")
        with col2:
            st.metric("Confidence", f"{result['confidence']:.0%}")
        
        # Explainability
        if st.checkbox("Show explanation"):
            explainer = ESGExplainer(classifier=st.session_state.classifier)
            explanation = explainer.explain_with_lime(text_to_analyze)
            
            st.write("**Why this classification?**")
            for word, score in explanation['top_features'][:5]:
                st.write(f"- '{word}': {score:.3f}")
```

### Option 2: Enhance Existing Features

In your advisory tab, you can now:

1. **Classify existing articles automatically**
2. **Show explainability for GPT recommendations**
3. **Fetch real-time news for companies**

---

## 📊 Testing Strategy

### Unit Tests

Create `tests/test_services.py`:

```python
import pytest
from services.esg_classifier import ESGBertClassifier, FinBERTClassifier
from services.explainability import ESGExplainer

class TestESGClassifier:
    def test_esgbert_initialization(self):
        classifier = ESGBertClassifier()
        assert classifier.classifier is not None
    
    def test_classification(self):
        classifier = ESGBertClassifier()
        result = classifier.classify("Company invests in renewable energy")
        assert result['category'] in ['E', 'S', 'G', None]
        assert 0 <= result['confidence'] <= 1

class TestExplainability:
    def test_lime_explanation(self):
        classifier = ESGBertClassifier()
        explainer = ESGExplainer(classifier=classifier)
        result = explainer.explain_with_lime("test text")
        assert 'top_features' in result or 'error' in result
```

Run: `pytest tests/`

### Integration Tests

Test the full pipeline:

```python
from services.data_ingestion import ingest_and_classify
from services.esg_classifier import ESGBertClassifier

# Test end-to-end
classifier = ESGBertClassifier()
articles = ingest_and_classify("Apple", classifier, days=1)

# Should return classified articles
for article in articles:
    assert 'esg_category' in article
    assert 'esg_confidence' in article
```

---

## 🎨 UI Enhancements

### Real-Time Data Section

Add to dashboard:

```python
# New section for real-time data
with st.expander("🔄 Real-Time News Updates"):
    if st.button("Fetch Latest News"):
        with st.spinner("Fetching articles..."):
            articles = fetch_company_news(company, days=1)
            
            if articles:
                st.success(f"Found {len(articles)} articles")
                for article in articles[:5]:
                    st.markdown(f"**{article['title']}**")
                    st.caption(f"Source: {article['source']} | {article['published_at']}")
                    if article.get('esg_category'):
                        st.badge(article['esg_category'])
            else:
                st.info("No recent articles found")
```

### Explainability Widget

```python
# Show why a prediction was made
if st.checkbox("🔬 Show Model Explanation"):
    explainer = ESGExplainer(classifier=classifier)
    explanation = explainer.explain_with_lime(text)
    
    # Word cloud or bar chart
    import plotly.graph_objects as go
    
    features = explanation.get('top_features', [])[:10]
    if features:
        words = [f[0] for f in features]
        scores = [f[1] for f in features]
        
        fig = go.Figure(go.Bar(
            x=scores,
            y=words,
            orientation='h'
        ))
        fig.update_layout(title="Word Importance")
        st.plotly_chart(fig)
```

---

## 🐛 Troubleshooting

### Issue: "Transformers not available"
**Fix:** `pip install transformers torch`

### Issue: "CUDA out of memory"
**Fix:** Use CPU mode: `ESGBertClassifier(use_gpu=False)`

### Issue: "NewsAPI rate limit exceeded"
**Fix:** Add delays between requests or upgrade plan

### Issue: Model downloads slowly
**Fix:** First run downloads pretrained models. Subsequent runs use cache.

---

## 📈 Next Steps

After testing these services:

1. ✅ **Fine-tune models** on your specific data
2. ✅ **Add real-time scheduler** (APScheduler)
3. ✅ **Build predictive analytics** (LSTM/Prophet)
4. ✅ **Deploy to cloud** (Dockerize + deploy)

See `ENHANCEMENT_ROADMAP.md` for full plan.

---

## 💡 Quick Reference

### Classify Text
```python
classifier = ESGBertClassifier()
result = classifier.classify("text here")
```

### Get Explanations
```python
explainer = ESGExplainer(classifier)
explanation = explainer.explain_with_lime(text)
```

### Fetch News
```python
articles = fetch_company_news("Company Name", days=7)
```

### Classify + Explain
```python
articles = ingest_and_classify("Company", classifier, days=7)
```

---

## 🔗 Resources

- ESGBERT: https://huggingface.co/nbroad/ESG-BERT
- FinBERT: https://huggingface.co/ProsusAI/finbert
- SHAP: https://github.com/slundberg/shap
- LIME: https://github.com/marcotcr/lime
- NewsAPI: https://newsapi.org/docs

---

Good luck! 🚀

