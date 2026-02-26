# GPT-Enhanced Advisory Intelligence Setup Guide

## Overview
The ESG AI application now includes GPT-4 powered advisory capabilities that analyze all ESG data to provide intelligent, context-aware investment insights.

## What's New

### 🤖 AI Advisory Features
- **Comprehensive Data Analysis**: The GPT system analyzes all ESG data including:
  - Article coverage and sentiment metrics
  - Tone trends and momentum
  - ESG pillar scores (E, S, G) vs industry
  - Source influence and media landscape
  - Catalyst timelines and key events
  - Risk indicators
  - Representative coverage samples

- **Intelligent Context Building**: Automatically creates rich data summaries from all available metrics

- **Adaptive Advisory**: Provides tailored recommendations based on:
  - Client mandates and portfolio focus
  - Specific advisory prompts
  - Risk profiles and preferences

- **Multiple Fallback Layers**:
  1. GPT-4 powered analysis (primary)
  2. Backend API service (fallback)
  3. Rule-based logic (final fallback)

## Setup Instructions

### 1. Install OpenAI Library
```bash
source venv/bin/activate
pip install openai langchain langchain-openai
```

### 2. Configure OpenAI API Key

You have two options:

#### Option A: Environment Variable (Recommended for Production)
```bash
export OPENAI_API_KEY="your-api-key-here"
```

#### Option B: Add to .streamlit/config.toml (Development)
```toml
[api]
OPENAI_API_KEY = "your-api-key-here"
```

### 3. Restart the Application
After setting the API key, restart the Streamlit app:
```bash
streamlit run app.py
```

## Usage

### Accessing the AI Advisory
1. Navigate to the **Advisory** tab in the ESG AI application
2. You'll see a status indicator showing whether GPT mode is enabled
3. Enter your advisory request in the text area
4. Optionally specify client focus or mandate
5. Click **🤖 Generate AI Advisory**

### Example Prompts
- "Draft a briefing for the sustainability committee"
- "Analyze environmental risks for a low-carbon transition portfolio"
- "Provide investment outlook for EU pension fund clients"
- "Summarize key governance concerns"

### Output Structure
The AI advisory provides:
- **Executive Summary**: 2-3 sentence overview
- **Key Talking Points**: 3-5 strategic highlights
- **Risk Radar**: Potential concerns to monitor
- **Recommended Actions**: Specific next steps
- **Evidence Summary**: Supporting data points

## Features

### Intelligent Data Synthesis
The GPT system processes:
- Hundreds of articles and their sentiment scores
- ESG pillar breakdowns
- Industry benchmark comparisons
- Temporal trends and momentum
- Source credibility and influence
- Catalyst events and their impact

### Client-Specific Customization
- Adapts tone and focus to mandate
- Considers risk tolerance
- Aligns with portfolio objectives
- Incorporates regional constraints

### Professional Output
- Investment-grade language
- Data-driven insights
- Actionable recommendations
- Evidence-backed analysis

## Cost Considerations

The system uses **GPT-4o-mini** for:
- Cost efficiency
- Fast response times
- High-quality outputs
- Lower latency

Typical costs: ~$0.001-0.002 per advisory generation

## Troubleshooting

### GPT Mode Not Available
- Verify OPENAI_API_KEY is set correctly
- Check that OpenAI library is installed
- Restart the application after configuration

### Fallback Mode Active
If GPT fails or API key is missing, the system automatically falls back to:
1. Backend API service (if running)
2. Rule-based advisory logic

All modes provide useful advisory content, but GPT mode offers the richest analysis.

## Next Steps

For production deployment:
1. Set OPENAI_API_KEY as environment variable
2. Consider rate limiting for cost control
3. Add caching for frequent queries
4. Monitor API usage and costs
5. Implement user feedback loop

## Support

For issues or questions:
- Check the application logs
- Review GPT API response in console
- Verify data availability for selected company
- Ensure analysis_context is populated correctly

