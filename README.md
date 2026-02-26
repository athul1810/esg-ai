# 🌍 ESG AI Intelligence Platform

<div align="center">

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![ML](https://img.shields.io/badge/ML-Transformer-FFA500?style=for-the-badge&logo=tensorflow&logoColor=white)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**🚀 Executive-grade ESG Intelligence powered by ML & AI**

*Automated ESG analysis, sentiment tracking, and predictive insights for 239+ companies*

[📊 Live Demo](#) • [📖 Documentation](docs/) • [🚀 Quick Start](#-quick-start)

</div>

---

## ✨ Overview

**ESG AI** is a comprehensive intelligence platform that transforms how financial analysts assess Environmental, Social, and Governance (ESG) performance. Leveraging **machine learning**, **natural language processing**, and **real-time data ingestion**, it provides actionable insights that automate what traditionally takes weeks of manual analysis.

---

## 🌟 Key Features

### 🤖 **ML-Powered Analysis**
- **ESGBERT Classification**: Transformer-based NLP model for automated E/S/G categorization
- **Explainable AI**: LIME/SHAP explanations showing "why" predictions were made
- **Real-time Classification**: Instant ESG category detection from news articles

### 📊 **Comprehensive Dashboard**
- **Multi-Company Comparison**: Side-by-side analysis of up to 4 companies with table format
- **Interactive Visualizations**: Trend analysis, sentiment tracking, and peer comparisons
- **Real-time Data**: NewsAPI integration for live ESG news ingestion
- **Predictive Analytics**: Forecast ESG trends with time-series analysis

### 🎯 **Professional Intelligence**
- **GPT-4 Advisory**: AI-powered investment recommendations with company-specific insights
- **Risk Alerts**: Automated monitoring of ESG risks and opportunities
- **Catalyst Detection**: Identify high-impact ESG events and their investment implications
- **Portfolio Management**: Track multiple companies across custom time windows

### 🔒 **Enterprise-Ready**
- **Role-Based Access**: Admin, Analyst, and Viewer permissions
- **Audit Logs**: Complete tracking of user actions and data changes
- **PDF Export**: Professional advisory reports with executive summaries
- **Secure Authentication**: Token-based access control

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/athul1810/esg-ai.git
cd esg-ai
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
NEWSAPI_KEY=your_newsapi_key_here  # Optional
```

5. **Run the application**
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
esg-ai/
├── app.py                          # Main Streamlit application
├── services/
│   ├── esg_classifier.py          # ML-powered ESG classification
│   ├── explainability.py          # LIME/SHAP explanations
│   ├── data_ingestion.py          # Real-time news ingestion
│   ├── synthetic_data_generator.py # Historical data generation
│   └── portfolio.py               # Portfolio management
├── pages/
│   ├── 01_Portfolio_Command.py    # Portfolio analytics
│   └── 02_Compare_Companies.py    # Multi-company comparison
├── utils/
│   ├── analytics.py               # Data analysis utilities
│   ├── auth.py                    # Authentication system
│   └── logging.py                 # Logging configuration
├── Data/
│   ├── synthetic_2021_to_now/    # Historical ESG data (239 companies)
│   └── dec30_to_jan12/           # Legacy dataset
├── backend/
│   ├── main.py                    # FastAPI backend
│   ├── schemas.py                 # API schemas
│   └── dependencies.py            # Dependency injection
├── docs/
│   ├── architecture_overview.md   # System architecture
│   ├── implementation_roadmap.md  # Development roadmap
│   └── working_features.md        # Feature documentation
└── requirements.txt               # Python dependencies
```

---

## 🎨 Features Deep Dive

### 🧠 ML-Powered Classification

Our system uses **ESGBERT**, a fine-tuned transformer model, to automatically classify text into ESG categories:

```python
from services.esg_classifier import ESGBertClassifier

classifier = ESGBertClassifier()
result = classifier.classify("Company commits to 100% renewable energy by 2030")
# Returns: {'category': 'E', 'confidence': 0.92, 'method': 'esgbert'}
```

### 🔬 Explainable AI

Understand *why* predictions were made using LIME explanations:

```python
from services.explainability import ESGExplainer

explainer = ESGExplainer(classifier=classifier)
explanation = explainer.explain_with_lime(text_input)
# Shows word-level importance scores
```

### 📰 Real-Time Data Ingestion

Fetch and classify live news articles:

```python
from services.data_ingestion import fetch_company_news

articles = fetch_company_news("Apple", days=7)
# Returns classified ESG articles from last 7 days
```

---

## 🎯 Use Cases

### 1. **ESG Research Analysts**
- Quickly assess ESG performance of potential investments
- Compare multiple companies side-by-side
- Identify ESG risks and opportunities
- Generate professional advisory reports

### 2. **Portfolio Managers**
- Track ESG scores across portfolio companies
- Set automated risk alerts
- Monitor sentiment trends over time
- Export comprehensive portfolio reports

### 3. **Compliance Officers**
- Verify ESG disclosure accuracy
- Detect potential greenwashing
- Track regulatory compliance
- Generate audit reports

### 4. **Academics & Researchers**
- Study ESG impact on financial performance
- Analyze ESG trends across industries
- Access explainable ML predictions
- Export data for further analysis

---

## 📊 Data Coverage

### Companies
- **239 companies** across multiple industries
- **S&P 500** representation
- **Custom portfolio** support

### Time Windows
- **2021-2025**: Comprehensive historical data
- **Real-time**: Live news ingestion
- **Custom ranges**: Analyze any time period

### Data Sources
- **NewsAPI**: Real-time news articles
- **GDELT**: Historical news archive
- **SEC EDGAR**: Company filings
- **Synthetic Data**: Generated realistic profiles

---

## 🛠️ Technology Stack

### Frontend
- **Streamlit**: Interactive web interface
- **Altair/Plotly**: Visualizations
- **Custom CSS**: Professional styling

### Backend
- **FastAPI**: RESTful API
- **SQLite**: Data storage
- **JWT**: Authentication

### Machine Learning
- **Transformers**: HuggingFace library
- **ESGBERT**: Fine-tuned classification model
- **LIME**: Explainable AI
- **Node2Vec**: Graph embeddings

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **NetworkX**: Graph analysis

---

## 📈 Performance Metrics

- ⚡ **Classification Speed**: ~50ms per article
- 🎯 **Accuracy**: >85% ESG categorization
- 📊 **Explainability**: Word-level importance scores
- 🔄 **Real-time**: <2s news ingestion
- 📱 **Responsive**: Works on desktop & tablet

---

## 🔐 Security

- **Token-based authentication**
- **Role-based access control** (Admin, Analyst, Viewer)
- **Audit logging** of all actions
- **Secure credential storage**
- **Environment variable** API key management

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Contribution Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Finastra** for the Hack to the Future competition
- **Streamlit** for the amazing framework
- **HuggingFace** for transformer models
- **GDELT** for news data
- **NewsAPI** for real-time news access

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/athul1810/esg-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/athul1810/esg-ai/discussions)
- **Documentation**: See [docs/](docs/) directory

---

<div align="center">

**Made with ❤️ for sustainable investing**

⭐ **Star us on GitHub** if you find this project useful!

</div>
