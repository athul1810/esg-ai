"""
Synthetic ESG Data Generator
Creates realistic historical ESG data from 2021 to now based on companies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def load_real_companies():
    """Load actual companies from the real dataset."""
    try:
        import os
        data_path = "Data/dec30_to_jan12/data_as_csv.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            return sorted(df['Organization'].unique().tolist())
        return []
    except Exception as e:
        logger.warning(f"Could not load real companies: {e}")
        return []

# Get real companies from dataset
REAL_COMPANIES = load_real_companies()

# Fallback list
MAJOR_COMPANIES = [
    "apple", "microsoft", "alphabet", "amazon com", "meta", "tesla", "nvidia", 
    "intel corporation", "netflix", "pfizer", "johnson and johnson", "visa",
    "jpmorgan chase", "bank of america", "walmart", "costco", "target",
    "starbucks", "coca cola", "pepsico", "nike", "adidas", "boeing", 
    "general electric", "international business machines", "oracle", "adobe", 
    "twitter", "facebook", "uber technologies", "delta air lines", 
    "exxon mobil", "chevron", "abbvie"
]

# Realistic ESG profiles based on known company characteristics
# Format: {company_name: {industry, E_score, S_score, G_score, E_focus, S_focus, G_focus}}
COMPANY_PROFILES = {
    # Tech/SaaS - Strong E, Good S, Excellent G
    "apple": {"industry": "tech", "E": 0.75, "S": 0.70, "G": 0.80, "E_focus": 0.8, "S_focus": 0.3, "G_focus": 0.2},
    "microsoft": {"industry": "tech", "E": 0.85, "S": 0.75, "G": 0.85, "E_focus": 0.9, "S_focus": 0.4, "G_focus": 0.3},
    "alphabet": {"industry": "tech", "E": 0.80, "S": 0.75, "G": 0.70, "E_focus": 0.9, "S_focus": 0.3, "G_focus": 0.4},
    "facebook": {"industry": "tech", "E": 0.75, "S": 0.60, "G": 0.55, "E_focus": 0.7, "S_focus": 0.5, "G_focus": 0.6},
    "meta": {"industry": "tech", "E": 0.75, "S": 0.60, "G": 0.55, "E_focus": 0.7, "S_focus": 0.5, "G_focus": 0.6},
    "amazon com": {"industry": "tech_retail", "E": 0.60, "S": 0.55, "G": 0.50, "E_focus": 0.6, "S_focus": 0.7, "G_focus": 0.5},
    "netflix": {"industry": "tech_media", "E": 0.70, "S": 0.65, "G": 0.75, "E_focus": 0.6, "S_focus": 0.4, "G_focus": 0.4},
    "adobe": {"industry": "tech", "E": 0.80, "S": 0.75, "G": 0.80, "E_focus": 0.8, "S_focus": 0.3, "G_focus": 0.3},
    "oracle": {"industry": "tech", "E": 0.75, "S": 0.70, "G": 0.75, "E_focus": 0.7, "S_focus": 0.3, "G_focus": 0.4},
    "nvidia": {"industry": "tech_semiconductor", "E": 0.70, "S": 0.75, "G": 0.80, "E_focus": 0.7, "S_focus": 0.4, "G_focus": 0.3},
    "intel corporation": {"industry": "tech_semiconductor", "E": 0.75, "S": 0.70, "G": 0.75, "E_focus": 0.8, "S_focus": 0.3, "G_focus": 0.3},
    "advanced micro devices": {"industry": "tech_semiconductor", "E": 0.70, "S": 0.70, "G": 0.75, "E_focus": 0.7, "S_focus": 0.3, "G_focus": 0.4},
    "cisco systems": {"industry": "tech", "E": 0.80, "S": 0.75, "G": 0.80, "E_focus": 0.8, "S_focus": 0.3, "G_focus": 0.3},
    "international business machines": {"industry": "tech", "E": 0.85, "S": 0.75, "G": 0.85, "E_focus": 0.9, "S_focus": 0.4, "G_focus": 0.3},
    
    # Healthcare/Pharma - Moderate E, Strong S, Good G
    "pfizer": {"industry": "healthcare", "E": 0.65, "S": 0.80, "G": 0.75, "E_focus": 0.3, "S_focus": 0.9, "G_focus": 0.3},
    "johnson and johnson": {"industry": "healthcare", "E": 0.70, "S": 0.80, "G": 0.75, "E_focus": 0.3, "S_focus": 0.9, "G_focus": 0.2},
    "abbvie": {"industry": "healthcare", "E": 0.65, "S": 0.75, "G": 0.70, "E_focus": 0.3, "S_focus": 0.8, "G_focus": 0.3},
    "merck": {"industry": "healthcare", "E": 0.65, "S": 0.80, "G": 0.75, "E_focus": 0.3, "S_focus": 0.9, "G_focus": 0.3},
    "bristol myers squibb": {"industry": "healthcare", "E": 0.65, "S": 0.78, "G": 0.73, "E_focus": 0.3, "S_focus": 0.85, "G_focus": 0.3},
    "amgen": {"industry": "healthcare", "E": 0.68, "S": 0.75, "G": 0.72, "E_focus": 0.3, "S_focus": 0.85, "G_focus": 0.3},
    "gilead sciences": {"industry": "healthcare", "E": 0.66, "S": 0.80, "G": 0.74, "E_focus": 0.3, "S_focus": 0.9, "G_focus": 0.3},
    "eli lilly": {"industry": "healthcare", "E": 0.67, "S": 0.78, "G": 0.73, "E_focus": 0.3, "S_focus": 0.88, "G_focus": 0.3},
    
    # Electric Vehicles/Auto - Excellent E, Good S, Moderate G  
    "tesla": {"industry": "ev_automotive", "E": 0.95, "S": 0.55, "G": 0.50, "E_focus": 1.0, "S_focus": 0.3, "G_focus": 0.5},
    "ford motor": {"industry": "automotive", "E": 0.60, "S": 0.65, "G": 0.70, "E_focus": 0.7, "S_focus": 0.4, "G_focus": 0.3},
    "general motors": {"industry": "automotive", "E": 0.62, "S": 0.68, "G": 0.72, "E_focus": 0.7, "S_focus": 0.4, "G_focus": 0.3},
    
    # Energy/Oil - Weak E, Moderate S, Good G
    "exxon mobil": {"industry": "energy", "E": 0.30, "S": 0.55, "G": 0.70, "E_focus": 0.5, "S_focus": 0.2, "G_focus": 0.4},
    "chevron": {"industry": "energy", "E": 0.32, "S": 0.57, "G": 0.72, "E_focus": 0.5, "S_focus": 0.2, "G_focus": 0.4},
    "conocophillips": {"industry": "energy", "E": 0.31, "S": 0.56, "G": 0.71, "E_focus": 0.5, "S_focus": 0.2, "G_focus": 0.4},
    "marathon petroleum": {"industry": "energy", "E": 0.33, "S": 0.58, "G": 0.73, "E_focus": 0.5, "S_focus": 0.2, "G_focus": 0.4},
    "schlumberger": {"industry": "energy", "E": 0.34, "S": 0.59, "G": 0.74, "E_focus": 0.5, "S_focus": 0.2, "G_focus": 0.4},
    
    # Finance - Weak E, Good S, Strong G
    "jpmorgan chase": {"industry": "finance", "E": 0.45, "S": 0.65, "G": 0.80, "E_focus": 0.2, "S_focus": 0.4, "G_focus": 0.9},
    "bank of america": {"industry": "finance", "E": 0.47, "S": 0.67, "G": 0.82, "E_focus": 0.2, "S_focus": 0.4, "G_focus": 0.9},
    "goldman sachs": {"industry": "finance", "E": 0.50, "S": 0.70, "G": 0.85, "E_focus": 0.2, "S_focus": 0.4, "G_focus": 0.95},
    "morgan stanley": {"industry": "finance", "E": 0.48, "S": 0.68, "G": 0.83, "E_focus": 0.2, "S_focus": 0.4, "G_focus": 0.9},
    "wells fargo": {"industry": "finance", "E": 0.46, "S": 0.60, "G": 0.70, "E_focus": 0.2, "S_focus": 0.4, "G_focus": 0.8},
    "visa": {"industry": "finance_tech", "E": 0.75, "S": 0.70, "G": 0.80, "E_focus": 0.7, "S_focus": 0.3, "G_focus": 0.5},
    "mastercard": {"industry": "finance_tech", "E": 0.75, "S": 0.70, "G": 0.80, "E_focus": 0.7, "S_focus": 0.3, "G_focus": 0.5},
    
    # Retail - Moderate E, Strong S, Good G
    "walmart": {"industry": "retail", "E": 0.55, "S": 0.70, "G": 0.75, "E_focus": 0.4, "S_focus": 0.8, "G_focus": 0.4},
    "costco": {"industry": "retail", "E": 0.60, "S": 0.75, "G": 0.80, "E_focus": 0.5, "S_focus": 0.9, "G_focus": 0.4},
    "target": {"industry": "retail", "E": 0.62, "S": 0.73, "G": 0.77, "E_focus": 0.5, "S_focus": 0.85, "G_focus": 0.4},
    
    # Airlines - Weak E, Good S, Moderate G
    "delta air lines": {"industry": "airlines", "E": 0.45, "S": 0.70, "G": 0.68, "E_focus": 0.4, "S_focus": 0.5, "G_focus": 0.3},
    "southwest airlines": {"industry": "airlines", "E": 0.48, "S": 0.75, "G": 0.72, "E_focus": 0.4, "S_focus": 0.6, "G_focus": 0.3},
    "united airlines": {"industry": "airlines", "E": 0.44, "S": 0.68, "G": 0.67, "E_focus": 0.4, "S_focus": 0.5, "G_focus": 0.3},
    
    # Food & Beverage - Moderate E, Strong S, Good G
    "starbucks": {"industry": "food_bev", "E": 0.68, "S": 0.85, "G": 0.80, "E_focus": 0.5, "S_focus": 0.9, "G_focus": 0.3},
    "coca cola": {"industry": "food_bev", "E": 0.58, "S": 0.75, "G": 0.78, "E_focus": 0.5, "S_focus": 0.85, "G_focus": 0.3},
    "pepsico": {"industry": "food_bev", "E": 0.60, "S": 0.77, "G": 0.78, "E_focus": 0.5, "S_focus": 0.85, "G_focus": 0.3},
    "mcdonalds": {"industry": "food_bev", "E": 0.52, "S": 0.70, "G": 0.75, "E_focus": 0.4, "S_focus": 0.8, "G_focus": 0.3},
    
    # Travel/Hospitality - Moderate E, Strong S, Good G
    "marriott": {"industry": "hospitality", "E": 0.58, "S": 0.78, "G": 0.75, "E_focus": 0.5, "S_focus": 0.9, "G_focus": 0.3},
    "hilton worldwide holdings": {"industry": "hospitality", "E": 0.60, "S": 0.75, "G": 0.73, "E_focus": 0.5, "S_focus": 0.85, "G_focus": 0.3},
    
    # Consumer Goods - Moderate E, Good S, Moderate G
    "nike": {"industry": "consumer", "E": 0.65, "S": 0.75, "G": 0.70, "E_focus": 0.6, "S_focus": 0.7, "G_focus": 0.4},
    "p&g": {"industry": "consumer", "E": 0.60, "S": 0.70, "G": 0.75, "E_focus": 0.5, "S_focus": 0.7, "G_focus": 0.4},
    
    # Aerospace/Defense - Weak E, Good S, Strong G
    "boeing": {"industry": "aerospace", "E": 0.45, "S": 0.70, "G": 0.75, "E_focus": 0.4, "S_focus": 0.5, "G_focus": 0.8},
    "lockheed martin": {"industry": "aerospace", "E": 0.40, "S": 0.68, "G": 0.77, "E_focus": 0.3, "S_focus": 0.5, "G_focus": 0.85},
    "northrop grumman": {"industry": "aerospace", "E": 0.38, "S": 0.65, "G": 0.78, "E_focus": 0.3, "S_focus": 0.5, "G_focus": 0.85},
    "raytheon technologies": {"industry": "aerospace", "E": 0.42, "S": 0.67, "G": 0.79, "E_focus": 0.3, "S_focus": 0.5, "G_focus": 0.85},
    
    # Industrial - Moderate E, Good S, Good G
    "general electric": {"industry": "industrial", "E": 0.65, "S": 0.70, "G": 0.75, "E_focus": 0.7, "S_focus": 0.4, "G_focus": 0.4},
    "caterpillar": {"industry": "industrial", "E": 0.58, "S": 0.68, "G": 0.72, "E_focus": 0.6, "S_focus": 0.4, "G_focus": 0.4},
    "3m": {"industry": "industrial", "E": 0.62, "S": 0.73, "G": 0.76, "E_focus": 0.6, "S_focus": 0.5, "G_focus": 0.4},
}


def generate_synthetic_esg_data(
    companies: List[str] = None,
    start_date: str = "2021-01-01",
    end_date: str = None,
    articles_per_company_per_month: int = 30
) -> pd.DataFrame:
    """
    Generate synthetic historical ESG data for companies.
    
    Args:
        companies: List of company names (default: MAJOR_COMPANIES)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date (default: today)
        articles_per_company_per_month: Average articles per company per month
        
    Returns:
        DataFrame with synthetic ESG data similar to real data format
    """
    if companies is None:
        # Use real companies from dataset if available, otherwise fallback
        companies = REAL_COMPANIES if REAL_COMPANIES else MAJOR_COMPANIES
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Generating synthetic ESG data from {start_date} to {end_date} for {len(companies)} companies")
    
    # Date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate daily data
    all_records = []
    date_range = pd.date_range(start=start, end=end, freq='D')
    
    # Sources for realistic variety
    sources = [
        "Reuters", "Bloomberg", "Financial Times", "Wall Street Journal",
        "Forbes", "CNBC", "MarketWatch", "Business Insider", "Yahoo Finance",
        "Seeking Alpha", "TechCrunch", "The Verge", "CNET", "Engadget",
        "Harvard Business Review", "Harvard Law School"
    ]
    
    # ESG topics by category
    e_topics = ["climate", "carbon", "emission", "renewable", "sustainability", "energy", "water", "waste"]
    s_topics = ["employee", "diversity", "workplace", "safety", "community", "labor", "rights", "health"]
    g_topics = ["board", "governance", "compliance", "ethics", "executive", "regulation", "audit", "transparency"]
    
    np.random.seed(42)  # For reproducibility
    
    for company in companies:
        # Get company profile or use defaults
        if company.lower() in COMPANY_PROFILES:
            profile = COMPANY_PROFILES[company.lower()]
            company_e_focus = profile['E_focus']
            company_s_focus = profile['S_focus']
            company_g_focus = profile['G_focus']
            base_e_score = profile['E']
            base_s_score = profile['S']
            base_g_score = profile['G']
            industry = profile['industry']
        else:
            # Default profiles for companies without specific data
            company_e_focus = np.random.uniform(0.2, 0.7)
            company_s_focus = np.random.uniform(0.1, 0.6)
            company_g_focus = np.random.uniform(0.2, 0.6)
            base_e_score = 0.6
            base_s_score = 0.6
            base_g_score = 0.6
            industry = "general"
        
        # Monthly article volume with some variability
        total_days = len(date_range)
        total_articles = int((total_days / 30) * articles_per_company_per_month * np.random.uniform(0.8, 1.2))
        
        # Select random dates
        article_dates = sorted(np.random.choice(date_range, size=min(total_articles, len(date_range)), replace=False))
        
        for date in article_dates:
            # Determine ESG category based on company focus
            rand = np.random.random()
            if rand < company_e_focus:
                category = 'E'
                topic = np.random.choice(e_topics)
            elif rand < company_e_focus + company_s_focus:
                category = 'S'
                topic = np.random.choice(s_topics)
            elif rand < company_e_focus + company_s_focus + company_g_focus:
                category = 'G'
                topic = np.random.choice(g_topics)
            else:
                # Neutral/non-ESG article
                category = None
            
            # Generate tone based on company's ESG performance and trend
            time_factor = (date - start).days / 365  # Years since start
            
            # Base tone from company's ESG score for this category
            if category == 'E':
                base_tone = (base_e_score - 0.5) * 10 + np.random.normal(0, 2)
            elif category == 'S':
                base_tone = (base_s_score - 0.5) * 10 + np.random.normal(0, 2)
            elif category == 'G':
                base_tone = (base_g_score - 0.5) * 10 + np.random.normal(0, 2)
            else:
                base_tone = np.random.normal(0, 1.5)
            
            # Trend based on known companies
            if company.lower() in ["apple", "microsoft", "nvidia", "tesla", "meta", "alphabet"]:
                trend = 0.02  # Slight improvement
            elif company.lower() in ["facebook", "twitter", "boeing", "exxon mobil", "chevron"]:
                trend = -0.01  # Slight decline
            else:
                trend = np.random.uniform(-0.01, 0.01)
            
            tone = base_tone + trend * time_factor
            tone = np.clip(tone, -10, 10)
            
            # Generate related scores
            positive_tone = max(0, tone) if tone > 0 else 0
            negative_tone = -min(0, tone) if tone < 0 else 0
            polarity = positive_tone + negative_tone
            activity_density = np.random.uniform(15, 30)
            self_density = np.random.uniform(0, 2)
            word_count = np.random.randint(200, 1500)
            
            # URL (fake but realistic)
            date_str = pd.to_datetime(date).strftime('%Y/%m')
            url = f"https://example-news.com/companies/{company.lower().replace(' ', '-')}/{date_str}/{topic}"
            
            # Create record
            record = {
                'DATE': date,
                'SourceCommonName': np.random.choice(sources),
                'URL': url,
                'E': category == 'E',
                'S': category == 'S',
                'G': category == 'G',
                'Organization': company,
                'Tone': tone,
                'PositiveTone': positive_tone,
                'NegativeTone': negative_tone,
                'Polarity': polarity,
                'ActivityDensity': activity_density,
                'SelfDensity': self_density,
                'WordCount': word_count
            }
            
            all_records.append(record)
    
    df = pd.DataFrame(all_records)
    df = df.sort_values(['Organization', 'DATE']).reset_index(drop=True)
    
    logger.info(f"Generated {len(df):,} synthetic ESG records")
    return df


def generate_daily_esg_scores(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Generate daily ESG scores from article data.
    Optimized version using pandas groupby.
    
    Args:
        df: DataFrame with article data
        
    Returns:
        Dictionary of DataFrames: E_score, S_score, G_score, overall_score
    """
    logger.info("Computing daily ESG scores (optimized)...")
    
    # Convert date column
    df['date'] = pd.to_datetime(df['DATE']).dt.date
    
    # Group by Organization and date for efficient processing
    grouped = df.groupby(['Organization', 'date'])
    
    # Calculate scores efficiently using aggregations
    agg_dict = {
        'Tone': ['mean', 'count']
    }
    
    daily_scores = []
    
    for (org, date), group in grouped:
        # Overall scores
        overall_tone = group['Tone'].mean()
        overall_count = len(group)
        
        # ESG-specific scores
        e_filter = group['E'] == True
        s_filter = group['S'] == True
        g_filter = group['G'] == True
        
        e_tone = group[e_filter]['Tone'].mean() if e_filter.any() else 0
        s_tone = group[s_filter]['Tone'].mean() if s_filter.any() else 0
        g_tone = group[g_filter]['Tone'].mean() if g_filter.any() else 0
        
        # Normalize to 0-1 scale
        e_score = (e_tone + 10) / 20 if e_filter.any() else 0
        s_score = (s_tone + 10) / 20 if s_filter.any() else 0
        g_score = (g_tone + 10) / 20 if g_filter.any() else 0
        overall_score = (overall_tone + 10) / 20 if overall_count > 0 else 0
        
        daily_scores.append({
            'Organization': org,
            'date': date,
            'score': overall_score,
            'E_score': e_score,
            'S_score': s_score,
            'G_score': g_score,
            'article_count': overall_count
        })
    
    daily_df = pd.DataFrame(daily_scores)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values(['Organization', 'date'])
    
    # Get all unique dates
    all_dates = sorted(daily_df['date'].unique())
    
    # Create separate DataFrames for each score type with company columns
    results = {}
    for name in ['E_score', 'S_score', 'G_score', 'overall_score']:
        # Pivot to have dates as rows and companies as columns
        if name == 'overall_score':
            pivot_df = daily_df.pivot_table(
                index='date', 
                columns='Organization', 
                values='score', 
                aggfunc='mean'
            )
        else:
            pivot_df = daily_df.pivot_table(
                index='date', 
                columns='Organization', 
                values=f'{name}', 
                aggfunc='mean'
            )
        
        # Rename columns to add _diff suffix and replace spaces with underscores
        pivot_df.columns = [f"{col.replace(' ', '_')}_diff" for col in pivot_df.columns]
        
        # Add industry_tone column (average across all companies per date)
        industry_tone = pivot_df.mean(axis=1)
        pivot_df['industry_tone'] = industry_tone
        
        # Reorder columns to have industry_tone first
        cols = ['industry_tone'] + [c for c in pivot_df.columns if c != 'industry_tone']
        pivot_df = pivot_df[cols]
        
        # Set date as index
        pivot_df.index.name = 'date'
        
        results[name] = pivot_df
    
    return results


def create_synthetic_dataset(
    output_dir: str = "Data/synthetic_2021_to_now",
    start_date: str = "2021-01-01",
    end_date: str = None
):
    """
    Create a complete synthetic ESG dataset and save to files.
    
    Args:
        output_dir: Directory to save files
        start_date: Start date
        end_date: End date
    """
    import os
    
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ESG"), exist_ok=True)
    
    logger.info(f"Creating synthetic dataset in {output_dir}")
    
    # Generate main data
    df = generate_synthetic_esg_data(start_date=start_date, end_date=end_date)
    
    # Save main data
    main_file = os.path.join(output_dir, "data_as_csv.csv")
    df.to_csv(main_file, index=False)
    logger.info(f"Saved {main_file}")
    
    # Generate and save daily scores
    scores = generate_daily_esg_scores(df)
    
    for name, df_score in scores.items():
        filename = f"daily_{name}.csv" if name != 'overall_score' else f"overall_daily_esg_scores.csv"
        filepath = os.path.join(output_dir, "ESG", filename)
        df_score.to_csv(filepath)
        logger.info(f"Saved {filepath}")
    
    # Create average ESG scores (transposed: rows=E/S/G/Total, columns=companies)
    logger.info("Creating average ESG scores...")
    avg_scores_dict = {'E': {}, 'S': {}, 'G': {}, 'Total': {}}
    
    for company in df['Organization'].unique():
        company_data = df[df['Organization'] == company]
        e_avg = company_data[company_data['E'] == True]['Tone'].mean() if len(company_data[company_data['E'] == True]) > 0 else 0
        s_avg = company_data[company_data['S'] == True]['Tone'].mean() if len(company_data[company_data['S'] == True]) > 0 else 0
        g_avg = company_data[company_data['G'] == True]['Tone'].mean() if len(company_data[company_data['G'] == True]) > 0 else 0
        total_avg = company_data['Tone'].mean()
        
        # Store normalized scores (0-1 scale)
        avg_scores_dict['E'][company] = (e_avg + 10) / 20 if not pd.isna(e_avg) else 0
        avg_scores_dict['S'][company] = (s_avg + 10) / 20 if not pd.isna(s_avg) else 0
        avg_scores_dict['G'][company] = (g_avg + 10) / 20 if not pd.isna(g_avg) else 0
        avg_scores_dict['Total'][company] = (total_avg + 10) / 20 if not pd.isna(total_avg) else 0
    
    # Create DataFrame with companies as columns, then transpose
    avg_df = pd.DataFrame(avg_scores_dict)
    avg_df = avg_df.T  # Transpose: now rows are E/S/G/Total, columns are companies
    # Save without index name (will create empty first column)
    avg_file = os.path.join(output_dir, "ESG", "average_esg_scores.csv")
    avg_df.to_csv(avg_file)
    logger.info(f"Saved {avg_file}")
    
    # Create connections.csv (company relationships)
    # Format: company, n0_rec, n0_conf, n1_rec, n1_conf, ...
    logger.info("Creating connections data...")
    connections = []
    orgs = df['Organization'].unique()
    
    # Pre-compute metrics for all companies (vectorized)
    company_metrics = {}
    for org in orgs:
        org_data = df[df['Organization'] == org]
        # Get profile if available
        profile = COMPANY_PROFILES.get(org.lower(), {})
        company_metrics[org] = {
            'E': org_data['E'].any(),
            'S': org_data['S'].any(),
            'G': org_data['G'].any(),
            'tone': org_data['Tone'].mean(),
            'industry': profile.get('industry', 'general'),
            'e_score': profile.get('E', 0.6),
            's_score': profile.get('S', 0.6),
            'g_score': profile.get('G', 0.6)
        }
    
    # For each company, find its most connected companies
    for org in orgs:
        org_metrics = company_metrics[org]
        
        # Score other companies
        scores = []
        for other_org in orgs:
            if other_org == org:
                continue
            other_metrics = company_metrics[other_org]
            
            # Industry similarity (strongest factor)
            if org_metrics['industry'] == other_metrics['industry']:
                industry_score = 0.8
            elif org_metrics['industry'].split('_')[0] == other_metrics['industry'].split('_')[0]:
                industry_score = 0.4  # Same broad category
            else:
                industry_score = 0.1
            
            # ESG score similarity
            e_sim = 1 - abs(org_metrics['e_score'] - other_metrics['e_score'])
            s_sim = 1 - abs(org_metrics['s_score'] - other_metrics['s_score'])
            g_sim = 1 - abs(org_metrics['g_score'] - other_metrics['g_score'])
            avg_esg_sim = (e_sim + s_sim + g_sim) / 3
            
            # Tone similarity
            tone_sim = 1 - abs(org_metrics['tone'] - other_metrics['tone']) / 20
            
            # Combined score (industry weighted more heavily)
            score = (industry_score * 0.5 + avg_esg_sim * 0.3 + tone_sim * 0.2) * np.random.uniform(0.7, 1.0)
            scores.append((other_org, score))
        
        # Get top connections
        scores.sort(key=lambda x: x[1], reverse=True)
        top_connections = scores[:25]  # Keep top 25
        
        # Create row with company and its neighbors
        row_data = {'company': org}
        for i, (neighbor, conf) in enumerate(top_connections):
            row_data[f'n{i}_rec'] = neighbor
            row_data[f'n{i}_conf'] = conf
        
        # Pad with empty values if needed
        for i in range(len(top_connections), 25):
            row_data[f'n{i}_rec'] = ''
            row_data[f'n{i}_conf'] = 0.0
        
        connections.append(row_data)
    
    connections_df = pd.DataFrame(connections)
    conn_file = os.path.join(output_dir, "connections.csv")
    connections_df.to_csv(conn_file, index=False)
    logger.info(f"Saved {conn_file}")
    
    # Create pca_embeddings.csv (reduced embeddings for visualization)
    # Format: unnamed index column, then 0,1,2,company columns
    logger.info("Creating PCA embeddings...")
    orgs_list = list(orgs)
    # Create random 3D embeddings for each company
    embeddings = []
    for org in orgs_list:
        embeddings.append({
            '0': np.random.uniform(-5, 5),
            '1': np.random.uniform(-5, 5),
            '2': np.random.uniform(-5, 5),
            'company': org
        })
    
    embeddings_df = pd.DataFrame(embeddings)
    # Reorder columns to match expected format
    embeddings_df = embeddings_df[['0', '1', '2', 'company']]
    embed_file = os.path.join(output_dir, "pca_embeddings.csv")
    embeddings_df.to_csv(embed_file, index=False)
    logger.info(f"Saved {embed_file}")
    
    logger.info("Synthetic dataset creation complete!")
    return df, scores, avg_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df, scores, avg = create_synthetic_dataset(
        output_dir="Data/synthetic_2021_to_now",
        start_date="2021-01-01"
    )
    print(f"\n✅ Generated {len(df):,} records for {df['Organization'].nunique()} companies")

