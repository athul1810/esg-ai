#!/usr/bin/env python3
"""Test GPT Advisory with sample data"""

import pandas as pd
import sys
import os

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import the function
from app import generate_gpt_advisory, build_comprehensive_data_summary

# Sample context data
sample_context = {
    "company": "Apple",
    "company_label": "Apple",
    "start": "2024-01-01",
    "end": "2024-01-31",
    "range_label": "Jan 01, 2024 — Jan 31, 2024",
    "article_count": 150,
    "avg_tone": 0.85,
    "positive_share": 0.68,
    "tone_change": 0.15,
    "tone_vs_industry": 0.32,
    "negative_share": 0.32,
    "publisher_counts": pd.Series({
        "TechCrunch": 45,
        "Bloomberg": 32,
        "Reuters": 28,
        "CNBC": 25,
        "WSJ": 20
    }),
    "catalysts": [
        {
            "date": "2024-01-15",
            "volume": 25,
            "avg_tone": 1.2,
            "positive_share": 0.8
        },
        {
            "date": "2024-01-22",
            "volume": 18,
            "avg_tone": -0.5,
            "positive_share": 0.3
        }
    ],
    "top_positive_articles": pd.DataFrame([{
        "DATE": "2024-01-15",
        "SourceCommonName": "TechCrunch",
        "Tone": 2.5
    }]),
    "esg_scores": {"E": 0.8, "S": 0.75, "G": 0.85, "T": 0.8},
    "esg_industry": {"E": 0.6, "S": 0.65, "G": 0.7, "T": 0.65}
}

sample_df = pd.DataFrame([{
    "DATE": "2024-01-15",
    "SourceCommonName": "TechCrunch",
    "Tone": 2.5,
    "PositiveTone": 8.0,
    "NegativeTone": 1.5,
    "Polarity": 9.5,
    "ActivityDensity": 22.3,
    "WordCount": 450,
    "E": True,
    "S": False,
    "G": True
}])

print("Testing GPT Advisory System")
print("=" * 60)
print(f"OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")

# Test data summary
print("\n1. Testing data summary building...")
try:
    summary = build_comprehensive_data_summary(sample_context, sample_df)
    print(f"✓ Data summary created: {len(summary)} characters")
    print(f"\nSummary preview:\n{summary[:500]}...\n")
except Exception as e:
    print(f"✗ Error building summary: {e}")
    sys.exit(1)

# Test GPT advisory
print("\n2. Testing GPT advisory generation...")
try:
    result = generate_gpt_advisory(
        "Apple",
        sample_context,
        sample_df,
        prompt="Provide investment recommendations",
        client_profile={"mandate": "Tech sector focus"}
    )
    if result:
        print("✓ GPT advisory generated successfully!")
        print(f"\nExecutive Summary:\n{result.get('executive_summary', 'N/A')}")
        print(f"\nTalking Points: {len(result.get('talking_points', []))} items")
        print(f"Risk Radar: {len(result.get('risk_radar', []))} items")
        print(f"Actions: {len(result.get('recommended_actions', []))} items")
    else:
        print("✗ No result returned (check API key and connection)")
except Exception as e:
    print(f"✗ Error generating advisory: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")

