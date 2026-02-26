"""
Real-Time Data Ingestion Service for ESG AI
Supports GDELT, NewsAPI, and SEC EDGAR filings
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Optional imports
NEWSAPI_AVAILABLE = False
SEC_EDGAR_AVAILABLE = False

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    logger.warning("NewsAPI not available. Install with: pip install newsapi-python")

try:
    from sec_edgar_downloader import Downloader
    SEC_EDGAR_AVAILABLE = True
except ImportError:
    logger.warning("SEC Edgar not available. Install with: pip install sec-edgar-downloader")


class DataIngestionService:
    """
    Service for ingesting real-time ESG data from various sources.
    """
    
    def __init__(self):
        """Initialize data ingestion services."""
        self.newsapi_client = None
        self.edgar_downloader = None
        self.setup_sources()
    
    def setup_sources(self):
        """Setup API clients for various data sources."""
        # NewsAPI
        if NEWSAPI_AVAILABLE:
            api_key = os.getenv('NEWSAPI_KEY')
            if api_key:
                try:
                    self.newsapi_client = NewsApiClient(api_key=api_key)
                    logger.info("NewsAPI client initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize NewsAPI: {str(e)}")
            else:
                logger.warning("NEWSAPI_KEY not found in environment")
        
        # SEC EDGAR
        if SEC_EDGAR_AVAILABLE:
            try:
                self.edgar_downloader = Downloader("ESG-AI", "athul1810@example.com")
                logger.info("SEC EDGAR downloader initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SEC EDGAR: {str(e)}")
    
    def fetch_news_articles(self, company_name: str, days: int = 7, language: str = 'en') -> List[Dict]:
        """
        Fetch recent news articles about a company from NewsAPI.
        
        Args:
            company_name: Company to search for
            days: Number of days to look back
            language: Language code
            
        Returns:
            List of article dictionaries
        """
        if not self.newsapi_client:
            logger.warning("NewsAPI client not available")
            return []
        
        try:
            # Calculate date range
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Search for company in headlines and content
            all_articles = self.newsapi_client.get_everything(
                q=company_name,
                from_param=from_date,
                language=language,
                sort_by='publishedAt',
                page_size=100
            )
            
            # Debug: log response
            logger.info(f"NewsAPI response - Total: {all_articles.get('totalResults', 0)}, Status: {all_articles.get('status', 'unknown')}")
            
            articles = []
            for article in all_articles.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'published_at': article.get('publishedAt', ''),
                    'company': company_name
                })
            
            logger.info(f"Fetched {len(articles)} articles for {company_name}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news articles: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def fetch_esg_filings(self, ticker: str, filing_type: str = "10-K", limit: int = 1) -> List[Dict]:
        """
        Fetch SEC filings for a company.
        
        Args:
            ticker: Stock ticker symbol
            filing_type: Type of filing (10-K, 8-K, DEF 14A, etc.)
            limit: Maximum number of filings to fetch
            
        Returns:
            List of filing information
        """
        if not self.edgar_downloader:
            logger.warning("SEC EDGAR downloader not available")
            return []
        
        try:
            # Download filings
            self.edgar_downloader.get(filing_type, ticker, limit=limit)
            
            # Note: Returns file paths, would need to parse the actual content
            # This is a simplified version
            filings = [{
                'ticker': ticker,
                'filing_type': filing_type,
                'status': 'downloaded',
                'note': 'Manual parsing required'
            }]
            
            logger.info(f"Downloaded {filing_type} for {ticker}")
            return filings
            
        except Exception as e:
            logger.error(f"Error fetching SEC filings: {str(e)}")
            return []
    
    def fetch_gdelt_articles(self, company_name: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Fetch articles from GDELT (Global Database of Events, Language, and Tone).
        
        Note: GDELT integration is complex. This is a placeholder.
        Would need GDELT API or GKG (Global Knowledge Graph) data.
        
        Args:
            company_name: Company to search for
            start_date: Start date (YYYYMMDDHHMMSS)
            end_date: End date (YYYYMMDDHHMMSS)
            
        Returns:
            List of articles
        """
        # Placeholder - would need actual GDELT API implementation
        logger.info("GDELT integration requires API setup")
        return []
    
    def classify_and_enrich_articles(self, articles: List[Dict], classifier) -> List[Dict]:
        """
        Classify articles into ESG categories and enrich with scores.
        
        Args:
            articles: List of article dictionaries
            classifier: ESG classifier model
            
        Returns:
            Enriched articles with ESG classifications
        """
        if not classifier:
            logger.warning("No classifier provided")
            return articles
        
        enriched = []
        for article in articles:
            # Combine title and content for classification
            text = f"{article.get('title', '')} {article.get('description', '')}"
            
            # Classify
            classification = classifier.classify(text)
            
            # Add ESG metadata
            article['esg_category'] = classification.get('category')
            article['esg_confidence'] = classification.get('confidence', 0.0)
            article['classification_method'] = classification.get('method', 'unknown')
            
            enriched.append(article)
        
        return enriched


# Convenience functions
def setup_ingestion() -> DataIngestionService:
    """Create and configure data ingestion service."""
    return DataIngestionService()


def fetch_company_news(company_name: str, days: int = 7) -> List[Dict]:
    """Quick function to fetch company news."""
    service = setup_ingestion()
    return service.fetch_news_articles(company_name, days)


def ingest_and_classify(company_name: str, classifier, days: int = 7) -> List[Dict]:
    """
    Fetch news for a company and classify into ESG categories.
    
    Args:
        company_name: Company name
        classifier: ESG classifier
        days: Days to look back
        
    Returns:
        Classified articles
    """
    service = setup_ingestion()
    articles = service.fetch_news_articles(company_name, days)
    return service.classify_and_enrich_articles(articles, classifier)

