import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import pickle
import itertools
import re
from collections import Counter
from typing import Optional, Dict, List
import plotly.express as px
from plot_setup import finastra_theme
from download_data import Data
import sys
import requests
from requests import RequestException
from fpdf import FPDF  # fpdf2 package
import time

import metadata_parser

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# GPT Integration imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("OpenAI library not available. Enhanced advisory features will use rule-based fallback.")

from services.storage import initialise_storage
from utils.logging import configure_logging, get_logger
from utils.auth import ensure_authenticated, render_user_controls, _load_token_map, _verify_credentials, _safe_rerun
from utils.analytics import (
    filter_on_date,
    format_number,
    format_percentage,
    build_pillar_breakdown,
    build_company_context,
    identify_catalyst_timeline,
)


configure_logging()
LOGGER = get_logger(__name__)
initialise_storage()

# Check authentication first
mapping = _load_token_map()
USER_SESSION = st.session_state.get("auth_user")

# If not authenticated, show landing page with login
if not USER_SESSION:
    # Hide Streamlit UI completely for landing page
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');
        
        /* Hide EVERYTHING Streamlit */
        header {visibility: hidden !important; height: 0 !important; max-height: 0 !important;}
        footer {visibility: hidden !important; height: 0 !important; max-height: 0 !important;}
        #MainMenu {visibility: hidden !important; height: 0 !important; max-height: 0 !important;}
        .stDeployButton {visibility: hidden !important; height: 0 !important; max-height: 0 !important;}
        div[data-testid="stToolbar"] {visibility: hidden !important; height: 0 !important; max-height: 0 !important;}
        div[data-testid="collapsedControl"] {visibility: hidden !important; height: 0 !important; max-height: 0 !important;}
        div[data-testid="stDecoration"] {visibility: hidden !important; height: 0 !important;}
        
        .stApp {
            font-family: 'Inter', sans-serif !important;
            background-color: #0b0e11 !important;
            background-image: radial-gradient(circle at 50% 50%, #1a1a1a 0%, #0b0e11 100%) !important;
            min-height: 100vh !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        .main {
            background: transparent !important;
        }
        
        .block-container {
            padding: 4rem 2rem !important;
            max-width: 1000px !important;
            margin: 0 auto !important;
        }
        
        section[data-testid="stSidebar"] {
            display: none !important;
        }

        /* Terminal-style inputs */
        div[data-baseweb="input"] {
            background-color: #1a1a1a !important;
            border-radius: 2px !important;
            border: 1px solid #333 !important;
            transition: border-color 0.2s ease !important;
        }
        
        div[data-baseweb="input"]:focus-within {
            border-color: #0076ff !important;
            box-shadow: none !important;
        }
        
        div[data-baseweb="input"] input {
            color: #ffffff !important;
            font-family: 'Roboto Mono', monospace !important;
            font-size: 0.9rem !important;
            padding: 0.75rem 1rem !important;
        }
        
        /* Bloomberg-style primary button */
        button[kind="primary"] {
            background: #0076ff !important;
            border: none !important;
            color: white !important;
            font-weight: 500 !important;
            border-radius: 2px !important;
            padding: 0.75rem 1.5rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.8rem !important;
            transition: opacity 0.2s ease !important;
        }
        
        button[kind="primary"]:hover {
            opacity: 0.9 !important;
            background: #0076ff !important;
        }

        .login-card-container {
            background: #151515;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 3.5rem;
            max-width: 450px;
            margin: 0 auto;
        }

        h1, h2, h3, h4, h5, h6, p, label, span, div, input, button {
            color: #ffffff !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* Global Font Enforcement */
        * {
            color: #ffffff !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown("""
        <div class="login-card-container">
            <div style="text-align: center; margin-bottom: 3rem;">
                <h1 style="margin: 0; font-size: 2.25rem; font-weight: 800; letter-spacing: -0.025em; color: #ffffff;">ESG Intelligence</h1>
                <p style="margin-top: 0.75rem; color: #cbd5e1; font-size: 1rem; font-weight: 500;">Professional Investment Analytics Terminal</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Center the inputs and button
        st.markdown('<div style="max-width:320px; margin: 0 auto;">', unsafe_allow_html=True)
        
        st.markdown('<label style="color: #ffffff; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; display: block;">Authorized Identifier</label>', unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="admin", key="auth_username", label_visibility="collapsed")
        
        st.markdown('<div style="margin-top: 1.25rem;"></div>', unsafe_allow_html=True)
        st.markdown('<label style="color: #ffffff; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; display: block;">Secure Access Key</label>', unsafe_allow_html=True)
        token = st.text_input("Access Code", type="password", placeholder="••••••••", key="auth_token", label_visibility="collapsed")
        
        st.markdown('<div style="margin-top: 2.5rem;"></div>', unsafe_allow_html=True)
        submitted = st.button("ESTABLISH SECURE SESSION", key="auth_submit", use_container_width=True, type="primary")
        
        if submitted:
            if username and token:
                verified = _verify_credentials(username, token, mapping)
                if verified:
                    st.session_state["auth_user"] = verified
                    st.success(f"Session established: {verified['username']}")
                    _safe_rerun()
                else:
                    st.error("Authentication failed: Invalid credentials")
            else:
                st.warning("Action required: Please provide credentials")
        
        # Help text / Demo credentials
        st.markdown("""
        <div style="text-align: center; margin-top: 3rem; padding: 1.25rem; background: rgba(255,255,255,0.03); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
            <p style="color: #ffffff; font-size: 0.8rem; font-weight: 500; margin: 0; font-family: 'Inter', sans-serif;">
                DEMO ACCESS: <b>admin</b> / <b>admin123</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Add some bottom padding
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.stop()

# User is authenticated, show controls
render_user_controls(USER_SESSION)
st.sidebar.markdown("---")
LOGGER.info(
    "page_access",
    extra={"page": "issuer_overview", "user": USER_SESSION.get("username", "unknown"), "role": USER_SESSION.get("role", "unknown")},
)

API_BASE_URL = os.getenv("ESG_AI_API_BASE", "http://127.0.0.1:8000/v1")

####### CACHED FUNCTIONS ######
@st.cache_data(show_spinner=False)
def filter_company_data(df_company, esg_categories, start, end):
    #Filter E,S,G Categories
    comps = []
    for i in esg_categories:
        X = df_company[df_company[i] == True]
        comps.append(X)
    df_company = pd.concat(comps)
    # Convert date inputs to the same type as the DATE column
    # The data loader converts DATE to datetime.date, so we need to match that
    if hasattr(start, 'date'):
        start = start.date() if hasattr(start, 'date') else start
    if hasattr(end, 'date'):
        end = end.date() if hasattr(end, 'date') else end
    
    df_company = df_company[df_company.DATE.between(start, end)]
    return df_company


@st.cache_resource(show_spinner=False)
def load_data(start_data, end_data):
    data = Data().read(start_data, end_data)
    companies = data["data"].Organization.sort_values().unique().tolist()
    companies.insert(0,"Select a Company")
    return data, companies


@st.cache_data(show_spinner=False)
def filter_publisher(df_company,publisher):
    if publisher != 'all':
        df_company = df_company[df_company['SourceCommonName'] == publisher]
    return df_company


def get_melted_frame(data_dict, frame_names, keepcol=None, dropcol=None):
    if keepcol:
        reduced = {k: df[keepcol].rename(k) for k, df in data_dict.items()
                   if k in frame_names}
    else:
        reduced = {k: df.drop(columns=dropcol).mean(axis=1).rename(k)
                   for k, df in data_dict.items() if k in frame_names}
    df = (pd.concat(list(reduced.values()), axis=1).reset_index().melt("date")
            .sort_values("date").ffill())
    df.columns = ["DATE", "ESG", "Score"]
    return df.reset_index(drop=True)


def get_clickable_name(url):
    try:
        T = metadata_parser.MetadataParser(url=url, search_head_only=True)
        title = T.metadata["og"]["title"].replace("|", " - ")
        return f"[{title}]({url})"
    except:
        return f"[{url}]({url})"


def inject_global_styles():
	st.markdown(
		"""
		<style>
			@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
			
			:root {
				--brand-primary: #10b981; /* Emerald Green */
				--brand-secondary: #3b82f6; /* Modern Blue */
				--bg-deep: #020617; /* Slate 950 */
				--card-bg: rgba(30, 41, 59, 0.92); /* Higher opacity for contrast */
				--card-border: rgba(255, 255, 255, 0.28);
				--text-strong: #ffffff;
				--text-muted: #cbd5e1; /* Brighter for readability */
				--success: #10b981;
				--error: #ef4444;
				--warning: #f59e0b;
			}

			/* Global Typography - Minimum 14px base for readability */
			* {
				font-family: 'Inter', sans-serif !important;
			}
			
			span, p, label, div, h1, h2, h3, h4, section, input, button {
				color: #ffffff !important;
			}
			
			/* Base font size for body text */
			.stMarkdown, .stMarkdown p {
				font-size: 0.95rem !important;
				line-height: 1.5 !important;
			}
			
			/* Layout & Background */
			.stApp {
				background-color: var(--bg-deep) !important;
				background-image: 
					radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
					radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 50%) !important;
			}
			
			.main {
				background: transparent !important;
			}
			
			.block-container { 
				padding: 3rem 5rem !important;
				max-width: 1400px !important;
			}

			/* Sidebar - Stronger contrast */
			section[data-testid="stSidebar"] {
				background-color: #0f172a !important;
				border-right: 1px solid var(--card-border) !important;
			}
			section[data-testid="stSidebar"] > div {
				background: transparent !important;
			}
			section[data-testid="stSidebar"] .stMarkdown,
			section[data-testid="stSidebar"] p,
			section[data-testid="stSidebar"] label {
				color: #e2e8f0 !important;
				font-size: 0.95rem !important;
				font-weight: 500 !important;
			}
			section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
				color: #cbd5e1 !important;
				font-size: 0.9rem !important;
				font-weight: 500 !important;
			}
			section[data-testid="stSidebar"] h1, 
			section[data-testid="stSidebar"] h2, 
			section[data-testid="stSidebar"] h3 {
				color: #ffffff !important;
				font-weight: 700 !important;
			}

			/* Premium Headers */
			h1, h2, h3 {
				color: var(--text-strong) !important;
				font-weight: 700 !important;
				letter-spacing: -0.02em !important;
			}

			/* Metrics & Cards - Higher contrast, larger text */
			.metric-card {
				background: var(--card-bg) !important;
				backdrop-filter: blur(12px) !important;
				border: 1px solid var(--card-border) !important;
				border-radius: 16px !important;
				padding: 1.75rem !important;
				box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
			}
			
			.metric-card h4 {
				color: #e2e8f0 !important;
				font-size: 0.95rem !important;
				font-weight: 700 !important;
				text-transform: uppercase !important;
				letter-spacing: 0.06em !important;
				margin-bottom: 0.75rem !important;
			}
			
			.metric-value { 
				font-size: 2.75rem !important;
				font-weight: 800 !important;
				color: #ffffff !important;
				letter-spacing: -0.03em !important;
				text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
			}

			/* Sidebar Nav Links - Readable */
			[data-testid="stSidebarNav"] ul {
				padding-top: 1rem !important;
			}
			[data-testid="stSidebarNav"] a {
				color: #e2e8f0 !important;
				font-size: 0.95rem !important;
				font-weight: 500 !important;
			}
			[data-testid="stSidebarNav"] a:hover {
				color: #ffffff !important;
			}

			/* Modern Tabs - Larger text */
			.stTabs [role="tablist"] {
				gap: 2rem !important;
				border-bottom: 1px solid var(--card-border) !important;
			}
			.stTabs [role="tab"] {
				height: auto !important;
				padding: 1rem 0 !important;
				background: transparent !important;
				border: none !important;
				color: #e2e8f0 !important;
				font-weight: 700 !important;
				font-size: 1rem !important;
			}
			.stTabs [role="tab"][aria-selected="true"] {
				color: var(--brand-secondary) !important;
				border-bottom: 2px solid var(--brand-secondary) !important;
			}

			/* Inputs & Dropdowns - Larger, readable */
			div[data-baseweb="select"] > div, div[data-baseweb="input"] {
				background-color: rgba(15, 23, 42, 0.9) !important;
				border: 1px solid var(--card-border) !important;
				border-radius: 8px !important;
			}
			div[data-baseweb="select"] span,
			div[data-baseweb="input"] input {
				color: #ffffff !important;
				font-size: 0.95rem !important;
			}
			input {
				color: #ffffff !important;
				font-size: 0.95rem !important;
			}
			/* Selectbox label */
			div[data-testid="stSelectbox"] label {
				color: #e2e8f0 !important;
				font-size: 0.95rem !important;
			}

			/* Buttons - Higher visibility */
			button[kind="primary"] {
				background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
				border: none !important;
				color: #ffffff !important;
				border-radius: 8px !important;
				padding: 0.65rem 1.5rem !important;
				font-weight: 600 !important;
				font-size: 0.9rem !important;
				box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39) !important;
			}
			div[data-testid="stSidebar"] button {
				color: #e2e8f0 !important;
				font-size: 0.9rem !important;
			}
			div[data-testid="stSidebar"] button:hover {
				color: #ffffff !important;
			}

			/* Multiselect & Slider labels */
			div[data-testid="stMultiSelect"] label,
			div[data-testid="stSlider"] label {
				color: #e2e8f0 !important;
				font-size: 0.95rem !important;
				font-weight: 500 !important;
			}

			/* Dataframes */
			.stDataFrame {
				border: 1px solid var(--card-border) !important;
				border-radius: 12px !important;
				overflow: hidden !important;
			}

			/* Custom Footer - Larger */
			.app-footer {
				text-align: center;
				padding: 4rem 0 2rem;
				color: #cbd5e1 !important;
				font-size: 0.9rem !important;
				font-weight: 600 !important;
				letter-spacing: 0.05em !important;
			}

			/* Expanders, Radio, Tab content - readable */
			.streamlit-expanderHeader p, .streamlit-expanderHeader span {
				color: #e2e8f0 !important;
				font-size: 0.95rem !important;
			}
			div[data-testid="stRadio"] label {
				color: #e2e8f0 !important;
				font-size: 0.95rem !important;
			}
			/* Alerts/Info boxes */
			div[data-baseweb="notification"] {
				font-size: 0.95rem !important;
			}
		</style>
		""",
		unsafe_allow_html=True,
	)
	st.markdown('<div class="app-footer">INTELLIGENCE TERMINAL · SECURE ACCESS</div>', unsafe_allow_html=True)


def format_metric(value, precision=1, suffix=""):
    if value is None or pd.isna(value):
        return "–"
    if isinstance(value, (int, np.integer)):
        return f"{value:,d}{suffix}"
    return f"{value:,.{precision}f}{suffix}"


def build_company_summary(df_company):
    if df_company.empty:
        return {
            "article_count": 0,
            "avg_tone": None,
            "positive_ratio": None,
            "avg_polarity": None,
        }

    total = len(df_company)
    avg_tone = df_company["Tone"].mean()
    avg_polarity = df_company["Polarity"].mean()
    positive_ratio = (
        (df_company["PositiveTone"] > df_company["NegativeTone"]).mean()
        if total else None
    )

    return {
        "article_count": total,
        "avg_tone": avg_tone,
        "positive_ratio": positive_ratio,
        "avg_polarity": avg_polarity,
    }


def render_metrics(summary, context=None):
    if context and "final_esg_score" in context:
        cols = st.columns(4)
        art_col, tone_col, sentiment_col, fusion_col = cols
    else:
        art_col, tone_col, sentiment_col = st.columns(3)
        fusion_col = None
        
    with art_col:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Analysed Articles</h4>
                <div class="metric-value">{format_metric(summary['article_count'], precision=0)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tone_col:
        tone_value = format_metric(summary["avg_tone"], precision=2)
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Average Tone</h4>
                <div class="metric-value">{tone_value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sentiment_col:
        ratio = (
            f"{summary['positive_ratio']*100:,.1f}%"
            if summary["positive_ratio"] is not None
            else "–"
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Positive Sentiment Share</h4>
                <div class="metric-value">{ratio}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    if fusion_col and context:
        fusion_score = context.get("final_esg_score", 0)
        st.markdown(
            f"""
			<div class="metric-card" style="border-top: 4px solid var(--brand-primary) !important; box-shadow: 0 4px 20px rgba(0,0,0,0.3), 0 0 0 1px rgba(16, 185, 129, 0.15) !important;">
                <h4 style="color: #10b981 !important; font-size: 1rem !important;">Fusion ESG Index</h4>
                <div class="metric-value" style="color: #ffffff !important;">{fusion_score:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


STOPWORDS = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"}


def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def build_summary_package(raw_text, question=None, focus=None, sentence_target=3):
    sentences = split_into_sentences(raw_text)
    tokenised = [tokenize(sentence) for sentence in sentences]

    flat_tokens = list(itertools.chain.from_iterable(tokenised))
    if not flat_tokens:
        return {"summary": "", "highlights": [], "keywords": []}

    frequencies = Counter(flat_tokens)
    question_tokens = set(tokenize(question)) if question else set()
    focus_tokens = set()
    if focus:
        focus_tokens = set(tokenize(focus))
        focus_tokens.update({segment.strip().lower() for segment in focus.split(',') if segment.strip()})

    scored_sentences = []
    for idx, (sentence, tokens) in enumerate(zip(sentences, tokenised)):
        if not tokens:
            continue
        base_score = sum(frequencies[token] for token in tokens) / len(tokens)
        question_overlap = len(question_tokens.intersection(tokens))
        focus_overlap = len(focus_tokens.intersection(tokens)) if focus_tokens else 0
        positional_boost = 1.05 if idx == 0 else 1.0
        score = base_score * positional_boost
        if question_overlap:
            score += question_overlap * 2.5
        if focus_overlap:
            score += focus_overlap * 1.8
        scored_sentences.append((score, idx, sentence, tokens))

    if not scored_sentences:
        return {"summary": "", "highlights": [], "keywords": []}

    sentence_target = max(1, min(sentence_target, len(scored_sentences)))
    top_sentences = sorted(
        sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:sentence_target],
        key=lambda x: x[1],
    )
    summary_chunks = [sentence for _, _, sentence, _ in top_sentences]

    highlights = []
    for _, _, sentence, _ in top_sentences:
        cleaned = sentence.strip()
        if len(cleaned) > 220:
            cleaned = cleaned[:217].rsplit(' ', 1)[0] + '…'
        highlights.append(cleaned)

    dominant_terms = [word for word, _ in frequencies.most_common(12)
                      if word not in question_tokens and word not in focus_tokens]
    keywords = dominant_terms[:8]

    return {
        "summary": " ".join(summary_chunks),
        "highlights": highlights,
        "keywords": keywords,
    }

def format_article_reference(row):
    url = row.get("URL", "")
    date = row.get("DATE")
    source = row.get("SourceCommonName", "")
    tone = row.get("Tone")
    date_str = pd.to_datetime(date).strftime("%d %b %Y") if pd.notna(date) else "—"
    tone_str = format_number(tone)
    label = f"{date_str} · {source} (Tone {tone_str})"
    if url:
        return f"[{label}]({url})"
    return label


def answer_company_question(question, context):
    if not question or not question.strip():
        return {"status": "error", "message": "Enter a question to analyse."}

    if context.get("article_count", 0) == 0:
        return {"status": "info", "message": "No coverage is available for this company with the current filters."}

    text = question.lower()
    simplified = re.sub(r"[^a-z0-9\s]", " ", text)

    def contains_trigger(body, trigger):
        trigger = trigger.strip()
        if not trigger:
            return False
        if " " in trigger:
            return trigger in body
        return re.search(rf"\b{re.escape(trigger)}\b", body) is not None

    def contains_any(body, triggers):
        return any(contains_trigger(body, trig) for trig in triggers)

    unsupported_topics = {
        "gender diversity": ["female", "male", "gender", "ratio", "women", "men"],
        "financial performance": ["revenue", "earnings", "eps", "profit", "guidance", "margin"],
        "valuation": ["valuation", "price target", "share price", "stock price"],
    }
    for label, triggers in unsupported_topics.items():
        if contains_any(simplified, triggers):
            return {
                "status": "info",
                "message": f"The current ESG media dataset does not contain {label} metrics, so I can't answer that precisely. Try focusing on sentiment, coverage, or ESG narratives.",
            }

    answer_lines = []
    insights = []
    evidence = []
    handled = False

    def add_evidence(rows, note):
        if rows is None or rows.empty:
            return
        evidence.append(note)
        for _, row in rows.iterrows():
            evidence.append(f"- {format_article_reference(row)}")

    sentiment_triggers = ["sentiment", "tone", "mood", "overall score"]
    positive_triggers = ["positive", "favourable", "favorable", "bullish"]
    negative_triggers = ["negative", "bearish", "critical"]
    coverage_triggers = ["how many", "number of", "article count", "coverage", "volume", "articles", "sources", "publisher", "outlet"]
    activity_triggers = ["activity", "momentum", "density", "buzz"]
    wordcount_triggers = ["word count", "length", "depth"]
    polarity_triggers = ["polarity"]
    esg_triggers = ["esg", "environment", "environmental", "social", "governance", "sustainability"]
    invest_triggers = ["invest", "investment", "recommend", "should i", "bullish", "bearish", "buy", "sell"]
    risk_triggers = ["risk", "concern", "issue", "challenge", "headwind"]

    if contains_any(simplified, sentiment_triggers):
        handled = True
        avg_tone = context.get("avg_tone")
        industry_tone = context.get("industry_tone_avg")
        answer_lines.append(
            f"Average media tone on {context['company_label']} over {context['range_label']} was {format_number(avg_tone)}"
            + (f", versus {format_number(industry_tone)} across the peer set." if industry_tone is not None else ".")
        )
        if context.get("tone_change") is not None:
            delta = context["tone_change"]
            direction = "improved" if delta > 0 else "softened" if delta < 0 else "held steady"
            insights.append(
                f"Tone {direction} by {format_number(abs(delta))} points from the start to the end of the window."
            )
        if context.get("positive_share") is not None:
            industry_positive = context.get("industry_positive_share")
            insights.append(
                f"Positive sentiment share: {format_percentage(context['positive_share'])}"
                + (f" (industry {format_percentage(industry_positive)})" if industry_positive is not None else "")
            )
        add_evidence(context.get("top_positive_articles"), "Top positive coverage")
        add_evidence(context.get("top_negative_articles"), "Top negative coverage")

    if contains_any(simplified, positive_triggers) and context.get("positive_share") is not None:
        handled = True
        insights.append(
            f"Positive narratives made up {format_percentage(context['positive_share'])} of recent coverage."
        )

    if contains_any(simplified, negative_triggers) and context.get("negative_share") is not None:
        handled = True
        insights.append(
            f"Negative-to-neutral coverage accounted for {format_percentage(context['negative_share'])}."
        )

    if contains_any(simplified, polarity_triggers) and context.get("avg_polarity") is not None:
        handled = True
        insights.append(
            f"Average polarity (tone intensity) sat at {format_number(context['avg_polarity'])}."
        )

    if contains_any(simplified, activity_triggers) and context.get("avg_activity") is not None:
        handled = True
        insights.append(
            f"Activity density averaged {format_number(context['avg_activity'])}, signalling how concentrated the coverage was."
        )

    if contains_any(simplified, wordcount_triggers) and context.get("avg_wordcount") is not None:
        handled = True
        insights.append(
            f"Typical article length was {format_number(context['avg_wordcount'])} words."
        )

    coverage_triggered = False
    for trigger in coverage_triggers:
        if trigger in simplified:
            coverage_triggered = True
            break
    if coverage_triggered:
        handled = True
        answer_lines.append(
            f"The filtered dataset contains {context['article_count']:,d} articles mentioning {context['company_label']}."
        )
        publishers = context.get("publisher_counts")
        if publishers is not None and not publishers.empty:
            top_publishers = ", ".join(
                f"{publisher} ({count})" for publisher, count in publishers.items()
            )
            insights.append(f"Top sources by volume: {top_publishers}.")
        add_evidence(context.get("recent_articles"), "Most recent coverage")

    if contains_any(simplified, esg_triggers):
        handled = True
        esg_scores = context.get("esg_scores", {})
        if esg_scores:
            answer_lines.append(
                f"ESG benchmark snapshot for {context['company_label']} compared with the peer average:"
            )
            esg_lines = []
            labels = {"E": "Environment", "S": "Social", "G": "Governance", "T": "Total"}
            industry_scores = context.get("esg_industry", {})
            for bucket, label in labels.items():
                if bucket in esg_scores and pd.notna(esg_scores[bucket]):
                    comparison = ""
                    if bucket in industry_scores and pd.notna(industry_scores[bucket]):
                        comparison = f" (industry {format_number(industry_scores[bucket])})"
                    esg_lines.append(f"{label}: {format_number(esg_scores[bucket])}{comparison}")
            if esg_lines:
                insights.append("; ".join(esg_lines))
        else:
            insights.append("ESG benchmark data is unavailable for this issuer in the loaded dataset.")

    if contains_any(simplified, invest_triggers):
        handled = True
        answer_lines.append(
            "Media-driven investment sentiment should complement, not replace, fundamental diligence."
        )
        if context.get("avg_tone") is not None and context.get("positive_share") is not None:
            insights.append(
                f"Sentiment snapshot: tone {format_number(context['avg_tone'])}, positive share {format_percentage(context['positive_share'])}."
            )
        insights.append("No valuation or financial metrics are embedded in this ESG dataset.")

    if contains_any(simplified, risk_triggers):
        handled = True
        if context.get("negative_share") is not None:
            insights.append(
                f"Watchpoints: {format_percentage(context['negative_share'])} of coverage skewed negative or neutral."
            )
        add_evidence(context.get("top_negative_articles"), "Most critical coverage")

    if not handled:
        answer_lines.append(
            f"Here's what the ESG media dataset shows for {context['company_label']} over {context['range_label']}."
        )
        answer_lines.append(
            f"{context['article_count']:,d} articles with average tone {format_number(context.get('avg_tone'))}"
            + (f" and positive share {format_percentage(context.get('positive_share'))}." if context.get('positive_share') is not None else ".")
        )
        publishers = context.get("publisher_counts")
        if publishers is not None and not publishers.empty:
            top_pub = ", ".join(f"{pub} ({count})" for pub, count in publishers.items())
            insights.append(f"Top sources: {top_pub}.")
        add_evidence(context.get("recent_articles"), "Representative coverage")

    final_answer = " ".join(answer_lines).strip()
    return {
        "status": "ok",
        "answer": final_answer,
        "insights": insights,
        "evidence": evidence,
    }


METRIC_DEFINITIONS = {
    "Tone": (
        "Average sentiment score per article on a -10 to +10 scale. Scores above zero indicate favourable coverage; "
        "scores below zero suggest critical or negative narratives."
    ),
    "NegativeTone": (
        "Weighted share of negative sentiment expressed within articles. Higher values flag coverage that emphasises "
        "risks, controversies, or critical viewpoints."
    ),
    "PositiveTone": (
        "Weighted share of positive language in the articles. Elevated readings point to supportive commentary or "
        "favourable stakeholder reactions."
    ),
    "Polarity": (
        "Magnitude of sentiment, regardless of direction. A high polarity means narratives are strongly worded—" 
        "whether positive or negative—while low polarity implies neutral tonality."
    ),
    "ActivityDensity": (
        "Measure of article concentration around the topic, capturing how much attention the company receives relative "
        "to peers in the timeframe."
    ),
    "WordCount": (
        "Average article length. Longer pieces may indicate deeper analysis or investigative coverage, while shorter "
        "articles typically signal news briefs."
    ),
    "Overall Score": (
        "Composite roll-up of tone metrics produced by ESG<sup>AI</sup> to benchmark overall media sentiment against the "
        "sector baseline."
    ),
    "ESG Scores": (
        "Derived scores for Environment (E), Social (S), Governance (G), and Total (T) pillars, benchmarked against the "
        "industry universe. Positive deltas imply leadership; negative deltas highlight lagging narratives."
    ),
}


def generate_executive_summary(context):
    company = context["company_label"]
    summary_parts = [
        f"Between {context['range_label']} the platform captured {context['article_count']:,d} ESG-relevant articles relating to {company}."
    ]
    if context.get("avg_tone") is not None:
        tone_sentence = f"Average media tone sat at {format_number(context['avg_tone'])}"
        if context.get("tone_vs_industry") is not None:
            delta = context["tone_vs_industry"]
            qualifier = "above" if delta > 0 else "below" if delta < 0 else "in line with"
            if qualifier == "in line with":
                tone_sentence += "—in line with the industry benchmark."
            else:
                tone_sentence += f", {abs(delta):.2f} points {qualifier} the industry average."
        else:
            tone_sentence += "."
        summary_parts.append(tone_sentence)
    if context.get("positive_share") is not None:
        pos_sentence = f"Positive sentiment represented {format_percentage(context['positive_share'])} of coverage"
        if context.get("positive_vs_industry") is not None:
            delta = context["positive_vs_industry"]
            direction = "ahead of" if delta > 0 else "behind" if delta < 0 else "matching"
            if direction == "matching":
                pos_sentence += ", in line with peer momentum."
            else:
                pos_sentence += f", {abs(delta)*100:.1f} percentage points {direction} the industry baseline."
        else:
            pos_sentence += "."
        summary_parts.append(pos_sentence)
    if context.get("tone_change") is not None:
        delta = context["tone_change"]
        if abs(delta) > 0.2:
            direction = "improved" if delta > 0 else "softened"
            summary_parts.append(
                f"Tone {direction} by {format_number(abs(delta))} points between the start and end of the review period."
            )
    if context.get("busiest_day") is not None:
        summary_parts.append(
            f"Media concentration peaked on {pd.to_datetime(context['busiest_day']).strftime('%d %b %Y')} with {context['busiest_day_count']} pieces published."
        )
    return " ".join(summary_parts)


def generate_trend_narrative(context, df_company):
    narrative = []
    if context.get("tone_change") is not None:
        delta = context["tone_change"]
        if abs(delta) > 0.2:
            direction = "upward" if delta > 0 else "downward"
            narrative.append(
                f"Tone trend: the fitted curve shows a {direction} move of {format_number(abs(delta))} points across the period,"
                f" indicating {'strengthening market confidence' if delta > 0 else 'heightened scrutiny'}."
            )
        else:
            narrative.append("Tone trend: sentiment held broadly steady with only marginal movement week over week.")
    if context.get("tone_best_date") is not None:
        narrative.append(
            f"Peak tone of {format_number(context['tone_best_value'])} was observed on {pd.to_datetime(context['tone_best_date']).strftime('%d %b %Y')},"
            " signalling a favourable media moment."
        )
    if context.get("tone_worst_date") is not None:
        narrative.append(
            f"Lowest tone of {format_number(context['tone_worst_value'])} occurred on {pd.to_datetime(context['tone_worst_date']).strftime('%d %b %Y')},"
            " warranting review of the underlying narrative drivers."
        )
    if context.get("daily_volume") is not None and not context["daily_volume"].empty:
        avg_vol = context["daily_volume"].mean()
        max_vol = context["daily_volume"].max()
        if max_vol > avg_vol * 1.8:
            narrative.append(
                "Coverage volume shows a pronounced spike relative to the daily average, suggesting a catalyst or news event worth examining."
            )
    tone_quantiles = df_company["Tone"].quantile([0.1, 0.5, 0.9]) if not df_company.empty else None
    if tone_quantiles is not None:
        narrative.append(
            f"Distribution: 80% of articles fall between tone {tone_quantiles.loc[0.1]:.2f} and {tone_quantiles.loc[0.9]:.2f},"
            f" with the median at {tone_quantiles.loc[0.5]:.2f}, implying {'balanced coverage' if abs(tone_quantiles.loc[0.5]) < 0.5 else 'a skew towards critical commentary' if tone_quantiles.loc[0.5] < 0 else 'a positive skew'}."
        )
    return narrative


def build_tone_trend_chart(context, industry_df):
    company_series = context.get("tone_daily")
    if company_series is None or company_series.empty:
        return None
    comp_df = company_series.reset_index().rename(columns={"DATE": "Date", "Tone": "Tone"})
    comp_df["Entity"] = context["company_label"]
    industry_series = industry_df.groupby("DATE")["Tone"].mean().reset_index()
    industry_series = industry_series.rename(columns={"DATE": "Date", "Tone": "Tone"})
    industry_series["Entity"] = "Industry Average"
    plot_df = pd.concat([comp_df, industry_series]).reset_index(drop=True)
    chart = (
        alt.Chart(plot_df, title="Tone trend vs industry")
        .mark_line(point=True)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Tone:Q", title="Tone"),
            color=alt.Color("Entity:N", title=""),
            tooltip=["Date:T", "Entity:N", alt.Tooltip("Tone:Q", format=".2f")],
        )
        .properties(height=320)
        .interactive()
    )
    return chart


def build_tone_distribution_chart(df_company):
    if df_company.empty:
        return None
    chart = (
        alt.Chart(df_company, title="Document tone distribution")
        .transform_density(density="Tone", as_=["Tone", "density"])
        .mark_area(opacity=0.55, color="#694ED6")
        .encode(
            x=alt.X("Tone:Q", scale=alt.Scale(domain=(-10, 10))),
            y="density:Q",
            tooltip=[alt.Tooltip("Tone", format=".2f"), alt.Tooltip("density:Q", format=".4f")],
        )
        .properties(height=260)
        .interactive()
    )
    return chart


def build_industry_comparison_table(context):
    records = []
    def append_metric(label, company_value, industry_value, formatter):
        if company_value is None and industry_value is None:
            return
        delta, delta_str = None, "n/a"
        if company_value is not None and industry_value is not None:
            delta = company_value - industry_value
            delta_str = formatter(delta)
        records.append({
            "Metric": label,
            "Company": formatter(company_value) if company_value is not None else "n/a",
            "Industry": formatter(industry_value) if industry_value is not None else "n/a",
            "Delta": delta_str,
        })

    append_metric("Average tone", context.get("avg_tone"), context.get("industry_tone_avg"), lambda v: format_number(v))
    append_metric("Positive sentiment share", context.get("positive_share"), context.get("industry_positive_share"), format_percentage)
    append_metric("Polarity", context.get("avg_polarity"), context.get("industry_polarity"), lambda v: format_number(v))

    esg_scores = context.get("esg_scores", {})
    industry_scores = context.get("esg_industry", {})
    for pillar, label in {"E": "Environment", "S": "Social", "G": "Governance", "T": "Total ESG"}.items():
        company_val = esg_scores.get(pillar)
        industry_val = industry_scores.get(pillar)
        append_metric(f"{label} score", company_val, industry_val, lambda v: format_number(v))

    return pd.DataFrame(records)


def build_source_sentiment_table(df_company, top_n=8):
    columns = ["Source", "Articles", "Average Tone", "Positive Share", "Last Mention"]
    if df_company.empty:
        return pd.DataFrame(columns=columns)
    positive_mask = df_company["PositiveTone"] > df_company["NegativeTone"]
    agg_df = (
        df_company.assign(_positive=positive_mask)
        .groupby("SourceCommonName")
        .agg(
            Articles=("SourceCommonName", "size"),
            Average_Tone=("Tone", "mean"),
            Positive_Share=("_positive", "mean"),
            Last_Mention=("DATE", "max"),
        )
        .sort_values("Articles", ascending=False)
        .head(top_n)
        .reset_index()
    )
    agg_df.rename(
        columns={
            "SourceCommonName": "Source",
            "Average_Tone": "Average Tone",
            "Positive_Share": "Positive Share",
            "Last_Mention": "Last Mention",
        },
        inplace=True,
    )
    return agg_df


def collect_article_highlights(context, limit=3):
    highlights = {"positive": [], "negative": [], "recent": []}
    def prepare_rows(df):
        entries = []
        if df is None or df.empty:
            return entries
        for _, row in df.head(limit).iterrows():
            entries.append(
                {
                    "date": pd.to_datetime(row.get("DATE")).strftime("%d %b %Y"),
                    "source": row.get("SourceCommonName", ""),
                    "tone": row.get("Tone"),
                    "url": row.get("URL", ""),
                }
            )
        return entries

    highlights["positive"] = prepare_rows(context.get("top_positive_articles"))
    highlights["negative"] = prepare_rows(context.get("top_negative_articles"))
    highlights["recent"] = prepare_rows(context.get("recent_articles"))
    return highlights








def clean_text_for_pdf(text):
    """Remove or replace Unicode characters that aren't supported by PDF fonts."""
    if not text:
        return ""
    # Replace common Unicode characters with ASCII equivalents
    replacements = {
        '⚠️': '[IMPORTANT]',
        '✓': '[OK]',
        '→': '->',
        '—': '-',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '…': '...',
        '°': 'deg',
    }
    cleaned = str(text)
    for unicode_char, ascii_replacement in replacements.items():
        cleaned = cleaned.replace(unicode_char, ascii_replacement)
    # Remove any other non-ASCII characters that might cause issues
    cleaned = cleaned.encode('ascii', 'ignore').decode('ascii')
    return cleaned

def build_advisory_pdf(payload: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    # Get page width with margins (A4 width 210mm - left margin 10mm - right margin 10mm)
    page_width = 190

    def add_heading(title):
        pdf.set_font("Arial", "B", 14)
        title_text = clean_text_for_pdf(str(title)) if title else ""
        if title_text:
            pdf.cell(page_width, 10, title_text, ln=True)
        pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(page_width, 10, "Advisory Brief", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", size=12)

    if payload.get("executive_summary"):
        add_heading("Executive Summary")
        summary_text = clean_text_for_pdf(str(payload["executive_summary"]))
        if summary_text:
            pdf.multi_cell(page_width, 8, summary_text)
        pdf.ln(2)

    def add_list(title, items):
        if not items:
            return
        add_heading(title)
        for item in items:
            if item:
                item_text = clean_text_for_pdf(str(item))
                if item_text:
                    # Ensure text isn't too long and wraps properly
                    bullet_text = f"- {item_text}"
                    pdf.multi_cell(page_width, 8, bullet_text)
        pdf.ln(2)

    add_list("Key Talking Points", payload.get("talking_points"))
    add_list("Risk Radar", payload.get("risk_radar"))
    add_list("Recommended Actions", payload.get("recommended_actions"))
    add_list("Evidence", payload.get("evidence"))

    if payload.get("disclaimer"):
        pdf.set_font("Arial", "I", 11)
        disclaimer_text = clean_text_for_pdf(str(payload['disclaimer']))
        if disclaimer_text:
            pdf.multi_cell(page_width, 8, f"Disclaimer: {disclaimer_text}")

    pdf_bytes = pdf.output(dest="S")
    # Check if already bytes/bytearray, else encode
    if isinstance(pdf_bytes, str):
        return pdf_bytes.encode("latin-1")
    elif isinstance(pdf_bytes, bytearray):
        return bytes(pdf_bytes)
    else:
        return pdf_bytes

def build_advisory_markdown(payload: dict) -> str:
    sections = ["# Advisory Brief"]
    if payload.get("executive_summary"):
        sections.append("## Executive Summary")
        sections.append(payload["executive_summary"])
    if payload.get("talking_points"):
        sections.append("## Key Talking Points")
        sections.extend(f"- {item}" for item in payload["talking_points"])
    if payload.get("risk_radar"):
        sections.append("## Risk Radar")
        sections.extend(f"- {item}" for item in payload["risk_radar"])
    if payload.get("recommended_actions"):
        sections.append("## Recommended Actions")
        sections.extend(f"- {item}" for item in payload["recommended_actions"])
    if payload.get("evidence"):
        sections.append("## Evidence")
        sections.extend(f"- {item}" for item in payload["evidence"])
    if payload.get("disclaimer"):
        sections.append("")
        sections.append(f"> {payload['disclaimer']}")
    return "\n\n".join(sections)

def call_event_log_api(portfolio_name=None, limit=100):
    params = {"limit": limit}
    if portfolio_name:
        params["portfolio"] = portfolio_name
    try:
        response = requests.get(
            f"{API_BASE_URL}/events",
            params=params,
            headers={"X-User": USER_SESSION["username"], "X-Role": USER_SESSION["role"]},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except RequestException:
        LOGGER.warning("events_api_unavailable")
        return []


def _record_event(action: str, portfolio_name: Optional[str] = None, payload: Optional[dict] = None):
    try:
        requests.post(
            f"{API_BASE_URL}/events",
            json={
                "action": action,
                "portfolio": portfolio_name,
                "payload": payload or {},
            },
            headers={"X-User": USER_SESSION["username"], "X-Role": USER_SESSION["role"]},
            timeout=10,
        )
    except RequestException:
        LOGGER.warning("event_record_failed", extra={"action": action, "portfolio": portfolio_name})


def call_advisory_service(company, start, end, prompt=None, profile=None):
    payload = {
        "company": company,
        "prompt": prompt,
        "profile": profile,
        "start_date": pd.to_datetime(start).date().isoformat() if start else None,
        "end_date": pd.to_datetime(end).date().isoformat() if end else None,
    }
    try:
        response = requests.post(
            f"{API_BASE_URL}/advisory/generate",
            json=payload,
            headers={"X-User": USER_SESSION["username"], "X-Role": USER_SESSION["role"]},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except RequestException:
        LOGGER.warning(
            "advisory_api_fallback",
            extra={"company": company, "prompt": prompt},
        )
        return None


def compute_sentiment_momentum(context, df_company):
    bullets = []
    metrics = {}
    tone_change = context.get("tone_change")
    avg_tone = context.get("avg_tone")
    positive_share = context.get("positive_share")
    industry_tone = context.get("industry_tone_avg")

    if tone_change is not None:
        metrics["tone_change"] = tone_change
        if abs(tone_change) > 0.15:
            direction = "strengthened" if tone_change > 0 else "softened"
            bullets.append(
                f"Tone {direction} by {format_number(abs(tone_change))} points from the start of the window."
            )
        else:
            bullets.append("Tone held broadly stable across the review period.")

    tone_std = context.get("tone_std")
    if tone_std is not None:
        metrics["tone_volatility"] = tone_std
        if tone_std > 1.5:
            bullets.append("Narratives were volatile with wide tone dispersion across articles.")
        elif tone_std < 0.7:
            bullets.append("Coverage stayed tightly clustered around the average tone.")

    if avg_tone is not None:
        metrics["avg_tone"] = avg_tone
        if industry_tone is not None:
            delta = avg_tone - industry_tone
            metrics["tone_vs_industry"] = delta
            if abs(delta) >= 0.3:
                orientation = "ahead of" if delta > 0 else "below"
                bullets.append(
                    f"Average tone sits {format_number(abs(delta))} points {orientation} the industry baseline."
                )

    if positive_share is not None:
        metrics["positive_share"] = positive_share
        if positive_share >= 0.6:
            bullets.append("Positive narratives dominate coverage (≥60% of mentions).")
        elif positive_share <= 0.4:
            bullets.append("Less than 40% of coverage is positive, flagging sentiment headwinds.")

    volume_series = context.get("daily_volume")
    if volume_series is not None and not volume_series.empty:
        avg_vol = volume_series.mean()
        peak_vol = volume_series.max()
        metrics["avg_daily_volume"] = avg_vol
        metrics["peak_volume"] = peak_vol
        if peak_vol >= avg_vol * 2:
            peak_date = volume_series.idxmax()
            bullets.append(
                f"Coverage spiked on {pd.to_datetime(peak_date).strftime('%d %b %Y')} reaching {int(peak_vol)} articles."
            )

    return {"bullets": bullets, "metrics": metrics}


def derive_pillar_commentary(pillar_table, context):
    commentary = []
    if pillar_table.empty:
        return commentary
    avg_tone = context.get("avg_tone")
    for row in pillar_table.to_dict("records"):
        share = row.get("Share")
        tone = row.get("Average Tone")
        positive_share = row.get("Positive Share")
        label = row.get("Pillar")
        if share and share < 0.1:
            commentary.append(
                f"{label}: Limited coverage (<10% of mentions) — monitor for disclosure gaps."
            )
            continue
        if tone is not None and not pd.isna(tone):
            if avg_tone is not None and tone > avg_tone + 0.3:
                commentary.append(
                    f"{label}: Tone {tone:.2f} outperforms the overall narrative; leverage as a proof-point."
                )
            elif avg_tone is not None and tone < avg_tone - 0.3:
                commentary.append(
                    f"{label}: Tone {tone:.2f} lags overall sentiment — address stakeholder concerns."
                )
            elif tone < 0:
                commentary.append(
                    f"{label}: Negative leaning tone ({tone:.2f}) indicates pressure points."
                )
        if positive_share is not None and not pd.isna(positive_share):
            if positive_share >= 0.65:
                commentary.append(
                    f"{label}: {positive_share*100:.1f}% of articles are supportive — amplify related initiatives."
                )
            elif positive_share <= 0.4:
                commentary.append(
                    f"{label}: Only {positive_share*100:.1f}% positive coverage; prioritise remediation."
                )
    return commentary


def derive_source_influence(df_company, limit=5):
    if df_company.empty:
        return {"supportive": [], "critical": [], "neutral": []}
    df = df_company.assign(_positive=df_company["PositiveTone"] > df_company["NegativeTone"])
    summary = (
        df.groupby("SourceCommonName")
        .agg(
            articles=("SourceCommonName", "size"),
            avg_tone=("Tone", "mean"),
            positive_share=("_positive", "mean"),
            last_mention=("DATE", "max"),
        )
        .reset_index()
    )
    supportive = (
        summary[summary["avg_tone"] > 0]
        .sort_values(["avg_tone", "articles"], ascending=[False, False])
        .head(limit)
        .to_dict("records")
    )
    critical = (
        summary[summary["avg_tone"] < 0]
        .sort_values(["avg_tone", "articles"], ascending=[True, False])
        .head(limit)
        .to_dict("records")
    )
    neutral = (
        summary[(summary["avg_tone"] >= 0) & (summary["avg_tone"] <= 0.3)]
        .sort_values("articles", ascending=False)
        .head(limit)
        .to_dict("records")
    )
    return {"supportive": supportive, "critical": critical, "neutral": neutral}


def build_risk_opportunity(context, pillar_table, source_influence, catalysts):
    risks, opportunities = [], []
    negative_share = context.get("negative_share")
    if negative_share is not None and negative_share > 0.5:
        risks.append(
            f"{negative_share*100:.1f}% of coverage skews negative or neutral — investor perception is at risk."
        )
    avg_tone = context.get("avg_tone")
    if avg_tone is not None and avg_tone < 0:
        risks.append("Average tone remains below zero, indicating net negative sentiment.")
    if pillar_table is not None and not pillar_table.empty:
        worst_pillar = pillar_table.sort_values("Average Tone").iloc[0]
        if worst_pillar["Average Tone"] < 0:
            risks.append(
                f"{worst_pillar['Pillar']} pillar is underwater (tone {worst_pillar['Average Tone']:.2f})."
            )
        low_share = pillar_table.sort_values("Share").iloc[0]
        if low_share["Share"] < 0.1:
            risks.append(
                f"Coverage on {low_share['Pillar']} is sparse (<10%), creating an information vacuum."
            )

    critical_sources = source_influence.get("critical", []) if source_influence else []
    if critical_sources:
        top = critical_sources[0]
        risks.append(
            f"{top['SourceCommonName'] if 'SourceCommonName' in top else top['Source']} frequently covers the company with negative tone ({top['avg_tone']:.2f})."
        )

    for event in catalysts:
        if event.get("avg_tone") is not None and event["avg_tone"] < -0.5:
            risks.append(
                f"Catalyst {pd.to_datetime(event['date']).strftime('%d %b %Y')} drove negative sentiment (tone {event['avg_tone']:.2f})."
            )

    positive_share = context.get("positive_share")
    if positive_share is not None and positive_share >= 0.55:
        opportunities.append(
            f"Positive coverage accounts for {positive_share*100:.1f}% — momentum to reinforce."
        )
    if avg_tone is not None and avg_tone > 0.5:
        opportunities.append("Media tone is distinctly favourable; lean into supportive narratives.")

    supportive_sources = source_influence.get("supportive", []) if source_influence else []
    if supportive_sources:
        lead = supportive_sources[0]
        opportunities.append(
            f"{lead['SourceCommonName'] if 'SourceCommonName' in lead else lead['Source']} is a supportive amplifier (tone {lead['avg_tone']:.2f})."
        )

    for event in catalysts:
        if event.get("avg_tone") is not None and event["avg_tone"] > 0.7:
            opportunities.append(
                f"Catalyst {pd.to_datetime(event['date']).strftime('%d %b %Y')} landed strongly positive (tone {event['avg_tone']:.2f})."
            )

    return risks, opportunities


def build_watchlist(catalysts, limit=3):
    watch_items = []
    for event in catalysts[:limit]:
        date_label = pd.to_datetime(event["date"]).strftime("%d %b %Y")
        source = event.get("top_source") or event.get("highlight_source")
        tone = event.get("avg_tone")
        tone_str = "n/a" if tone is None else f"{tone:.2f}"
        watch_items.append(
            {
                "date": date_label,
                "source": source,
                "tone": tone_str,
                "url": event.get("highlight_url"),
            }
        )
    return watch_items


def generate_actionable_insights(context):
    actions = []
    if context.get("positive_share") is not None and context["positive_share"] < 0.4:
        actions.append(
            "Positive sentiment trails the desired threshold. Consider amplifying communications around sustainability achievements, governance milestones, or social impact programmes to rebalance perception."
        )
    if context.get("tone_vs_industry") is not None and context["tone_vs_industry"] < -0.3:
        actions.append(
            "Overall tone underperforms the peer set, signalling a need to address the root causes of negative narratives—potentially through targeted stakeholder engagement or proactive media outreach."
        )
    if context.get("tone_change") is not None and context["tone_change"] < -0.5:
        actions.append(
            "Sentiment erosion over the period indicates emerging concerns. Commission a rapid review of the underlying coverage to isolate themes and craft responses."
        )
    esg_scores = context.get("esg_scores", {})
    industry_scores = context.get("esg_industry", {})
    for pillar, label in {"E": "environmental", "S": "social", "G": "governance"}.items():
        pillar_score = esg_scores.get(pillar)
        industry_score = industry_scores.get(pillar)
        if pillar_score is not None and industry_score is not None and pillar_score + 5 < industry_score:
            actions.append(
                f"Media narratives highlight a relative gap on {label} factors. Reinforce disclosure and programme delivery in this pillar to lift perception."
            )
    if not actions:
        actions.append(
            "Maintain current ESG communication cadence while monitoring upcoming news cycles for potential risks or opportunities."
        )
    return actions


def generate_conclusion(context):
    tone_phrase = "balanced" if context.get("avg_tone") is None else (
        "favourable" if context["avg_tone"] > 0.5 else "challenging" if context["avg_tone"] < -0.5 else "mixed"
    )
    closing = [
        f"Overall, {context['company_label']} experienced a {tone_phrase} ESG media profile during {context['range_label']}."
    ]
    if context.get("positive_vs_industry") is not None:
        delta = context["positive_vs_industry"]
        if abs(delta) > 0.02:
            closing.append(
                f"Positive share sits {abs(delta)*100:.1f} percentage points {'ahead of' if delta > 0 else 'below'} the peer average, underscoring {'momentum to preserve' if delta > 0 else 'areas for remedial action'}."
            )
    closing.append(
        "Continued surveillance of sentiment drivers, coupled with transparent ESG disclosures, will be critical to sustaining investor and stakeholder confidence."
    )
    return " ".join(closing)


def generate_report_markdown(
    context,
    executive_summary,
    comparison_table,
    pillar_table,
    pillar_commentary,
    momentum,
    catalysts,
    risks,
    opportunities,
    source_influence,
    actions,
    conclusion,
    highlights,
    watchlist,
):
    lines = []
    lines.append(f"# ESG AI Insight Report — {context['company_label']}")
    lines.append(f"**Review window:** {context['range_label']}")
    lines.append("")

    lines.append("## Executive Snapshot")
    lines.append(executive_summary)
    lines.append("")

    lines.append("## Key Metrics vs Industry")
    if comparison_table is not None and not comparison_table.empty:
        cmp = comparison_table.copy()
        lines.append(cmp.to_markdown(index=False))
    else:
        lines.append("No benchmark data available.")
    lines.append("")

    lines.append("## Sentiment Momentum")
    momentum_points = momentum.get("bullets", []) if momentum else []
    if momentum_points:
        for item in momentum_points:
            lines.append(f"- {item}")
    else:
        lines.append("- Insufficient data to determine sentiment momentum.")
    metrics = momentum.get("metrics", {}) if momentum else {}
    if metrics:
        metric_table = pd.DataFrame([
            {
                "Metric": "Average tone",
                "Value": format_number(metrics.get("avg_tone")) if metrics.get("avg_tone") is not None else "n/a",
            },
            {
                "Metric": "Tone vs industry",
                "Value": format_number(metrics.get("tone_vs_industry")) if metrics.get("tone_vs_industry") is not None else "n/a",
            },
            {
                "Metric": "Tone change",
                "Value": format_number(metrics.get("tone_change")) if metrics.get("tone_change") is not None else "n/a",
            },
            {
                "Metric": "Tone volatility",
                "Value": format_number(metrics.get("tone_volatility")) if metrics.get("tone_volatility") is not None else "n/a",
            },
            {
                "Metric": "Positive share",
                "Value": format_percentage(metrics.get("positive_share")) if metrics.get("positive_share") is not None else "n/a",
            },
        ])
        lines.append("")
        lines.append(metric_table.to_markdown(index=False))
    lines.append("")

    lines.append("## Catalyst Timeline")
    if catalysts:
        catalyst_rows = []
        for event in catalysts:
            catalyst_rows.append(
                {
                    "Date": pd.to_datetime(event["date"]).strftime("%d %b %Y"),
                    "Articles": event.get("volume"),
                    "Tone": format_number(event.get("avg_tone")),
                    "Positive share": format_percentage(event.get("positive_share")),
                    "Lead source": event.get("top_source") or event.get("highlight_source") or "—",
                    "Pillars": ", ".join(event.get("pillars", [])) or "—",
                }
            )
        lines.append(pd.DataFrame(catalyst_rows).to_markdown(index=False))
        lines.append("")
        lines.append("Top catalyst articles:")
        for event in catalysts:
            url = event.get("highlight_url")
            date_label = pd.to_datetime(event["date"]).strftime("%d %b %Y")
            tone = event.get("highlight_tone")
            tone_str = "n/a" if tone is None or pd.isna(tone) else f"Tone {tone:.2f}"
            base = f"- {date_label} · {event.get('highlight_source', 'Unknown source')} · {tone_str}"
            lines.append(f"{base} — {url}" if url else base)
    else:
        lines.append("No catalysts identified in this period.")
    lines.append("")

    lines.append("## ESG Pillar Commentary")
    if pillar_commentary:
        for item in pillar_commentary:
            lines.append(f"- {item}")
    else:
        lines.append("- No ESG pillar signals available.")
    if pillar_table is not None and not pillar_table.empty:
        p_df = pillar_table.copy()
        p_df["Share"] = p_df["Share"].apply(lambda v: f"{v*100:.1f}%")
        p_df["Average Tone"] = p_df["Average Tone"].apply(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
        p_df["Positive Share"] = p_df["Positive Share"].apply(
            lambda v: "n/a" if pd.isna(v) else f"{v*100:.1f}%"
        )
        lines.append("")
        lines.append(p_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Source Influence")
    if source_influence:
        for label, key in [("Supportive amplifiers", "supportive"), ("Critical voices", "critical"), ("Neutral/monitor", "neutral")]:
            entries = source_influence.get(key, []) if source_influence else []
            lines.append(f"### {label}")
            if entries:
                source_names = [e.get("SourceCommonName") or e.get("Source") for e in entries]
                article_counts = [e.get("articles") for e in entries]
                tones = [format_number(e.get("avg_tone")) for e in entries]
                positive = [format_percentage(e.get("positive_share")) for e in entries]
                last_mentions = [
                    pd.to_datetime(e.get("last_mention")).strftime("%d %b %Y")
                    if e.get("last_mention") is not None and not pd.isna(e.get("last_mention"))
                    else "—"
                    for e in entries
                ]
                table = pd.DataFrame(
                    {
                        "Source": source_names,
                        "Articles": article_counts,
                        "Tone": tones,
                        "Positive share": positive,
                        "Last mention": last_mentions,
                    }
                )
                lines.append(table.to_markdown(index=False))
            else:
                lines.append("No sources in this category.")
            lines.append("")
    else:
        lines.append("No source analytics available.")

    lines.append("## Risk Radar")
    if risks:
        for risk in risks:
            lines.append(f"- {risk}")
    else:
        lines.append("- No significant risks detected in this window.")
    lines.append("")

    lines.append("## Opportunity Drivers")
    if opportunities:
        for item in opportunities:
            lines.append(f"- {item}")
    else:
        lines.append("- No clear opportunity signals surfaced.")
    lines.append("")

    lines.append("## Action Checklist")
    for action in actions:
        lines.append(f"- {action}")
    lines.append("")

    lines.append("## Watchlist")
    if watchlist:
        for item in watchlist:
            base = f"- {item['date']} · {item['source'] or 'Unknown source'} · Tone {item['tone']}"
            lines.append(f"{base} — {item['url']}" if item.get("url") else base)
    else:
        lines.append("- No immediate follow-ups recorded.")
    lines.append("")

    lines.append("## Evidence Appendix")
    for label, key in [
        ("Positive momentum", "positive"),
        ("Risks to monitor", "negative"),
        ("Most recent coverage", "recent"),
    ]:
        entries = highlights.get(key, []) if highlights else []
        lines.append(f"### {label}")
        if not entries:
            lines.append("- No articles available within this filter.")
        else:
            for entry in entries:
                tone = entry["tone"]
                tone_str = "n/a" if pd.isna(tone) else f"Tone {tone:.2f}"
                url = entry.get("url")
                base = f"- {entry['date']} · {entry['source']} · {tone_str}"
                lines.append(f"{base} — {url}" if url else base)
        lines.append("")

    lines.append("## Conclusion")
    lines.append(conclusion)

    return "\n".join(lines)


def interpret_esg_score(context):
    esg_scores = context.get("esg_scores", {})
    total_score = esg_scores.get("T")
    if total_score is None or pd.isna(total_score):
        return None
    # Convert to a 100 scale if data is in basis points
    if total_score > 1 and total_score <= 100:
        scaled = total_score
    else:
        scaled = total_score / 100 if total_score > 100 else total_score * 100
    return max(0, min(100, scaled))


def build_comprehensive_data_summary(context, df_company) -> str:
    """Build a comprehensive text summary of all ESG data for GPT context."""
    
    summary_lines = []
    
    # Company and timeframe - EMPHASIZE COMPANY NAME
    company_name = context.get('company_label', 'the company')
    summary_lines.append(f"╔════════════════════════════════════════════════════════════╗")
    summary_lines.append(f"║  COMPANY ANALYSIS: {company_name.upper():<40} ║")
    summary_lines.append(f"╚════════════════════════════════════════════════════════════╝")
    summary_lines.append(f"Analysis Period: {context.get('range_label', 'N/A')}")
    summary_lines.append(f"")
    summary_lines.append(f"⚠️ CRITICAL: This entire dataset is EXCLUSIVELY for {company_name}. All metrics, scores, and trends below are specific to {company_name}.\n")
    
    # Article volume
    article_count = context.get("article_count", 0)
    summary_lines.append(f"## Media Coverage for {company_name}:")
    summary_lines.append(f"- Total Articles Analyzed: {article_count:,d}")
    summary_lines.append(f"- All data points below are specific to {company_name}\n")
    
    # Sentiment metrics - Make company-specific
    if context.get("avg_tone") is not None:
        tone = context['avg_tone']
        sentiment_label = "positive" if tone > 0 else "negative" if tone < 0 else "neutral"
        summary_lines.append(f"## Sentiment Analysis for {company_name}:")
        summary_lines.append(f"- {company_name}'s Average Tone: {tone:.2f} ({sentiment_label})")
    if context.get("positive_share") is not None:
        pos_share = context['positive_share']*100
        summary_lines.append(f"- {company_name}'s Positive Coverage: {pos_share:.1f}% of articles")
    if context.get("negative_share") is not None:
        neg_share = context.get("negative_share", 0)*100
        summary_lines.append(f"- {company_name}'s Negative Coverage: {neg_share:.1f}% of articles")
    if context.get("tone_change") is not None:
        direction = "improving" if context['tone_change'] > 0 else "declining"
        summary_lines.append(f"- {company_name}'s Trend: {direction} by {abs(context['tone_change']):.2f} points")
    if context.get("tone_vs_industry") is not None:
        vs_ind = context['tone_vs_industry']
        comparison = "above" if vs_ind > 0 else "below"
        summary_lines.append(f"- {company_name} vs Industry: {abs(vs_ind):.2f} points {comparison} industry average")
    
    # ESG scores - Make company-specific
    esg_scores = context.get("esg_scores", {})
    esg_industry = context.get("esg_industry", {})
    if esg_scores:
        summary_lines.append(f"\n## ESG Performance for {company_name} vs Industry:")
        for pillar, label in {"E": "Environmental", "S": "Social", "G": "Governance", "T": "Total"}.items():
            company_score = esg_scores.get(pillar)
            industry_score = esg_industry.get(pillar)
            if company_score is not None:
                delta = company_score - industry_score if industry_score is not None else None
                if delta is not None:
                    perf_label = "outperforming" if delta > 0 else "underperforming"
                    summary_lines.append(f"- {company_name}'s {label} Score: {company_score:.2f} ({delta:+.2f} vs industry, {perf_label})")
                else:
                    summary_lines.append(f"- {company_name}'s {label} Score: {company_score:.2f}")
    
    # Source analysis
    publisher_counts = context.get("publisher_counts")
    if publisher_counts is not None and not publisher_counts.empty:
        summary_lines.append("\n## Top Media Sources:")
        for pub, count in publisher_counts.head(5).items():
            summary_lines.append(f"- {pub}: {count} articles ({count/article_count*100:.1f}% of coverage)")
    
    # Catalyst timeline - Make more detailed and company-specific
    catalysts = context.get("catalysts", [])
    if catalysts:
        summary_lines.append(f"\n## Key Catalysts for {company_name} (High Media Activity Periods):")
        summary_lines.append(f"These are specific events/periods when {company_name} received significant ESG-related media attention:")
        for i, event in enumerate(catalysts[:5], 1):
            date_str = pd.to_datetime(event.get('date', '')).strftime('%d %b %Y') if event.get('date') else 'N/A'
            tone = event.get('avg_tone', 'N/A')
            volume = event.get('volume', 'N/A')
            tone_desc = "positive" if isinstance(tone, (int, float)) and tone > 0 else "negative" if isinstance(tone, (int, float)) and tone < 0 else "neutral"
            summary_lines.append(f"  Catalyst {i} - {date_str}: {volume} articles about {company_name} with {tone_desc} sentiment (avg tone: {tone:.2f})")
        summary_lines.append(f"  INVESTMENT IMPLICATION: These catalysts indicate periods when {company_name} had significant ESG news that could impact stock performance.")
    
    # Risk indicators
    risk_items = []
    if context.get("negative_share", 0) > 0.5:
        risk_items.append(f"High negative coverage ({context['negative_share']*100:.1f}% of articles)")
    if context.get("tone_vs_industry", 0) < -0.5:
        risk_items.append("Significantly underperforming industry sentiment average")
    if context.get("tone_change", 0) < -0.3:
        risk_items.append("Declining sentiment trend (-{:.2f} points)".format(abs(context['tone_change'])))
    
    if risk_items:
        summary_lines.append("\n## ⚠️ Risk Indicators:")
        for risk in risk_items:
            summary_lines.append(f"- {risk}")
    
    # Positive indicators
    positive_items = []
    if context.get("positive_share", 0) > 0.6:
        positive_items.append(f"Strong positive coverage ({context['positive_share']*100:.1f}% of articles)")
    if context.get("tone_vs_industry", 0) > 0.5:
        positive_items.append("Outperforming industry sentiment")
    if context.get("tone_change", 0) > 0.3:
        positive_items.append("Improving sentiment trend (+{:.2f} points)".format(context['tone_change']))
    
    if positive_items:
        summary_lines.append("\n## ✓ Positive Indicators:")
        for pos in positive_items:
            summary_lines.append(f"- {pos}")
    
    # Sample headlines
    top_articles = context.get("top_positive_articles")
    if top_articles is not None and hasattr(top_articles, 'empty') and not top_articles.empty:
        summary_lines.append("\n## Representative Articles:")
        for i, (_, row) in enumerate(top_articles.head(3).iterrows(), 1):
            date_str = pd.to_datetime(row.get('DATE', '')).strftime('%d %b') if row.get('DATE') else 'N/A'
            source = row.get('SourceCommonName', 'Unknown')
            tone = row.get('Tone', 0)
            summary_lines.append(f"- {i}. {date_str} - {source} (tone: {tone:.2f})")
    
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append(f"⚠️ CRITICAL REMINDER: This entire dataset is EXCLUSIVELY for {company_name}.")
    summary_lines.append(f"Every metric, score, percentage, and trend mentioned above is specific to {company_name}.")
    summary_lines.append(f"When generating the advisory, ALWAYS mention {company_name} by name and cite exact metrics.")
    summary_lines.append(f"{'='*60}")
    
    return "\n".join(summary_lines)


def generate_gpt_advisory(company, context, df_company, prompt=None, client_profile=None) -> Dict[str, any]:
    """Generate AI-powered advisory using OpenAI GPT with full ESG data context."""
    
    print(f"DEBUG: Starting GPT advisory for {company}")
    print(f"DEBUG: OPENAI_AVAILABLE = {OPENAI_AVAILABLE}")
    
    if not OPENAI_AVAILABLE:
        print("DEBUG: OpenAI library not available, falling back")
        return None
    
    # Check if API key is configured
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"DEBUG: API key exists: {bool(api_key)}")
    if not api_key:
        print("DEBUG: No API key found")
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        print("DEBUG: OpenAI client created successfully")
        
        # Build comprehensive data summary
        data_summary = build_comprehensive_data_summary(context, df_company)
        print(f"DEBUG: Data summary length: {len(data_summary)} characters")
        print(f"DEBUG: Data summary preview: {data_summary[:200]}...")
        
        # Build GPT prompt
        system_prompt = """You are an expert ESG analyst and investment advisor providing strategic guidance to institutional investors. 
You analyze ESG sentiment data, media coverage, and company performance to deliver actionable investment insights.

CRITICAL RULES - VIOLATION WILL RESULT IN REJECTION:
1. The data provided is for ONE SPECIFIC COMPANY ONLY. Mention the company by name in EVERY sentence or bullet point.
2. EVERY bullet point MUST cite EXACT numbers from the data (e.g., "Apple's Environmental score of 0.82 outperforms industry by +0.15 points", NOT "Monitor ESG performance")
3. FORBIDDEN GENERIC PHRASES - DO NOT USE THESE:
   - "Review ESG data and trends"
   - "Maintain regular ESG monitoring cadence"
   - "Continue monitoring ESG performance metrics"
   - "Monitor ESG performance"
   - "Track ESG trends"
   - Any variation of "monitor", "track", "review" without specific metrics
4. If asked about catalysts, you MUST list the SPECIFIC dates and volumes from the catalyst timeline in the data
5. If asked about investment implications, you MUST connect specific ESG scores to concrete investment outcomes
6. Every sentence must reference a SPECIFIC metric, date, percentage, or score from the provided data
7. Answer the EXACT question asked - if they ask for catalysts, provide catalysts; if they ask for risks, provide specific risks
8. NO generic statements allowed - every point must have a number, date, or specific metric"""
        
        user_prompt = f"""**Company ESG Analysis Request for {company}**

⚠️ CRITICAL INSTRUCTION: The data below is SPECIFIC to {company}. Your response MUST be tailored exclusively to {company}'s performance. 
DO NOT use generic statements. Every insight must reference specific metrics from this company's data.

**Data Summary for {company}:**
{data_summary}

**COMPANY NAME: {company}**
Remember: Every section of your response must be about {company} specifically.

"""
        
        if prompt:
            user_prompt += f"**Analyst's Specific Request:**\n{prompt}\n\n"
        
        if client_profile:
            mandate = client_profile.get("mandate")
            portfolio = client_profile.get("portfolio")
            user_prompt += "**Client Context:**\n"
            if mandate:
                user_prompt += f"- Investment Mandate: {mandate}\n"
            if portfolio:
                user_prompt += f"- Portfolio: {portfolio}\n"
            user_prompt += "\n"
        
        # Check if user asked about catalysts specifically
        is_catalyst_query = prompt and any(word in prompt.lower() for word in ["catalyst", "catalysts", "event", "events", "trigger", "milestone"])
        
        if is_catalyst_query:
            user_prompt += f"""**REQUIRED OUTPUT - CATALYST ANALYSIS for {company}:**

⚠️ CRITICAL: The user specifically asked about ESG CATALYSTS. You MUST analyze and list the catalysts from the data above.

**Executive Summary**
Start with "{company}'s catalyst analysis shows..." Cite {company}'s ESG metrics AND reference specific catalyst dates from the timeline. If catalysts are listed in the data, mention them by date. Example: "{company} had a major catalyst on [DATE] with [X] articles and tone [Y], which correlates with [ESG score]."

**Key Talking Points - MUST INCLUDE CATALYSTS**
List 4-6 bullet points. FORBIDDEN generic phrases. Every point MUST:
- Start with "{company}'s catalyst on [DATE]" or "{company}'s [DATE] event..." (use ACTUAL dates from catalyst timeline)
- Cite EXACT data: "{company}'s catalyst on 15 Jan 2021: 25 articles, tone 1.2, positive sentiment"
- Explain investment implications: "This {company} catalyst suggests [specific impact]"
- If no catalysts in data, state: "{company} shows no major catalyst events in the analysis period, but {company}'s baseline ESG score is [X.XX]"
- NEVER say "Review ESG data" or "Monitor performance" - only specific catalyst dates/metrics

**Risk Radar - CATALYST-RELATED RISKS**
Identify risks based on {company}'s catalyst patterns:
- Start with "{company}'s negative catalyst on [DATE]" or "{company}'s [DATE] event..."
- Reference ACTUAL dates from negative catalysts: "{company}'s catalyst on [DATE] showed [X] articles with negative tone [Y]"
- Cite specific metrics: "{company}'s negative sentiment spike to [X]% during [DATE] catalyst"
- Connect to investment risk: "{company}'s [DATE] catalyst pattern suggests [specific risk]"

**Recommended Actions - CATALYST-FOCUSED**
List actions based on {company}'s catalyst data:
- Start with "Monitor {company}'s catalyst pattern around [DATE]" (use actual dates)
- Reference specific catalyst dates: "Watch for {company} catalysts similar to [DATE BBLE] pattern"
- Be specific: "{company}'s next potential catalyst window based on historical pattern: [month/year]"
- NO generic advice - must reference specific dates/metrics from catalyst timeline

**Evidence Summary - CATALYST EVIDENCE REQUIRED**
Cite MINIMUM 3-5 EXACT catalyst data points for {company}:
- Format: "{company}'s catalyst on [DATE]: [X] articles, tone [Y], sentiment [positive/negative]"
- Include dates, volumes, and sentiment for each catalyst
- If no catalysts: List {company}'s other specific metrics instead"""
        else:
            user_prompt += f"""**Required Output Structure for {company}:**

⚠️ CRITICAL: Every section MUST reference SPECIFIC metrics from {company}'s data. FORBIDDEN generic phrases.

**Executive Summary**
Start with "{company} shows..." Cite specific ESG scores with numbers, sentiment trends with numbers, or performance metrics. Example: "{company} demonstrates Environmental score of [X.XX], which is [above/below] industry by [X.XX] points." Must have numbers.

**Key Talking Points**
List 3-5 bullet points. FORBIDDEN: "Review ESG data", "Monitor performance", "Track trends". Each MUST:
- Start with "{company}'s [specific metric]" - include the number
- Example: "{company}'s Environmental score of 0.82 outperforms industry by +0.15 points, suggesting strong sustainability positioning"
- Explain implications using the exact numbers
- NEVER generic advice without numbers

**Risk Radar**
Identify 2-4 concerns with SPECIFIC numbers:
- Start with "{company}'s [metric] shows [number]" or "{company} faces [specific issue] with [number]"
- Example: "{company}'s negative sentiment reached 35% in [month], 15 points above industry average"
- Cite exact underperformance: "{company}'s Social score of 0.55 lags industry by 0.20 points"
- Reference catalyst dates if in data: "{company}'s negative catalyst on [DATE]"

**Recommended Actions**
List 2-4 actions with SPECIFIC metrics:
- Start with "For {company}, monitor [specific metric] at [specific value/threshold]"
- Example: "Monitor {company}'s Social ESG score which currently at 0.55 needs to reach 0.65 to match industry"
- Reference dates: "Track {company}'s next catalyst window around [month/year]"
- NO generic "monitor ESG" - must have specific metric + number

**Evidence Summary**
Cite 3-5 EXACT data points. Start each "{company}..." with numbers:
- "{company}'s tone: [X.XX], declined [Y.YY] points"
- "{company}'s Environmental score: [X.XX], [above/below] industry by [Y.YY]"
- "{company}'s negative coverage: [X]% in [month]"
- "{company}'s catalyst on [DATE]: [X] articles"

Every sentence must have {company} name + specific number/metric. NO GENERIC STATEMENTS."""
        
        # Call OpenAI API with fallback to cheaper model if quota exceeded
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for better instruction following and company-specific responses
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.9,  # Higher temperature for more varied, company-specific responses
                max_tokens=2500,  # Increased for more comprehensive responses
                top_p=0.95  # Nucleus sampling for more diverse outputs
            )
        except Exception as api_error:
            error_str = str(api_error)
            # If quota exceeded, try with cheaper model
            if "429" in error_str or "quota" in error_str.lower():
                print(f"DEBUG: Quota exceeded, falling back to gpt-4o-mini")
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",  # Fallback to cheaper model
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.9,
                        max_tokens=2000,
                        top_p=0.95
                    )
                except Exception as fallback_error:
                    raise Exception(f"API quota exceeded. Please check your OpenAI billing: {error_str}")
            else:
                raise api_error
        
        gpt_response = response.choices[0].message.content
        print(f"DEBUG: Received GPT response, length: {len(gpt_response) if gpt_response else 0}")
        print(f"DEBUG: GPT response preview: {gpt_response[:300] if gpt_response else 'None'}...")
        
        # Parse GPT response into structured format
        sections = gpt_response.split("\n\n")
        
        advisory_data = {
            "executive_summary": "",
            "talking_points": [],
            "risk_radar": [],
            "recommended_actions": [],
            "evidence": [],
            "disclaimer": "AI-generated advisory draft powered by GPT-4. Review and customize before distribution."
        }
        
        current_section = None
        for section in sections:
            section_lower = section.lower().strip()
            section_clean = section.strip()
            
            # Detect section headers (with or without **bold**)
            if "**executive summary**" in section_lower or "executive summary" in section_lower[:20]:
                current_section = "executive_summary"
                # Extract content after header
                content = section_clean.replace("**Executive Summary**", "").replace("Executive Summary", "").strip().lstrip("-").strip().lstrip(":").strip()
                if content and not any(char in content for char in ["**", "##"]):
                    advisory_data[current_section] = content
            elif "**talking points**" in section_lower or "talking points" in section_lower[:20] or "key points" in section_lower[:20]:
                current_section = "talking_points"
            elif "**risk radar**" in section_lower or "risk radar" in section_lower[:20]:
                current_section = "risk_radar"
            elif "**recommended actions**" in section_lower or "recommended actions" in section_lower[:20]:
                current_section = "recommended_actions"
            elif "**evidence" in section_lower or "evidence" in section_lower[:20]:
                current_section = "evidence"
            else:
                # Process content based on current section
                if current_section:
                    # For list items (starting with - or *)
                    if section_clean.startswith("-") or section_clean.startswith("*"):
                        item = section_clean.lstrip("-* ").strip()
                        if item and current_section in advisory_data:
                            if isinstance(advisory_data[current_section], list):
                                advisory_data[current_section].append(item)
                            else:
                                advisory_data[current_section] = [item]
                    # For executive summary continuation (non-list content)
                    elif current_section == "executive_summary":
                        content = section_clean
                        if content and not any(char in content for char in ["**", "##", "-"]):
                            if advisory_data["executive_summary"]:
                                advisory_data["executive_summary"] += " " + content
                            else:
                                advisory_data["executive_summary"] = content
        
        # Check for generic responses and replace them
        generic_phrases = [
            "review esg data", "monitor esg performance", "track esg trends",
            "maintain regular", "continue monitoring", "monitoring cadence",
            "detailed insights", "performance metrics"
        ]
        
        def is_generic(text):
            if not text:
                return True
            text_lower = text.lower()
            # Check if it contains generic phrases AND doesn't have company name or specific numbers
            has_generic = any(phrase in text_lower for phrase in generic_phrases)
            has_company = company.lower() in text_lower
            has_numbers = any(char.isdigit() for char in text)
            return has_generic and not (has_company or has_numbers)
        
        # Ensure executive summary exists and is not generic
        if not advisory_data["executive_summary"] or is_generic(advisory_data["executive_summary"]):
            if gpt_response:
                # Try to extract first non-generic sentence
                first_line = gpt_response.split("\n")[0]
                if not is_generic(first_line):
                    advisory_data["executive_summary"] = first_line
                else:
                    # Create company-specific fallback
                    tone = context.get("avg_tone", 0)
                    article_count = context.get("article_count", 0)
                    advisory_data["executive_summary"] = f"{company} shows an average tone of {tone:.2f} across {article_count} articles analyzed in this period."
            else:
                advisory_data["executive_summary"] = f"{company} ESG analysis: Review the data above for specific metrics."
        
        # Filter out generic talking points and replace with company-specific ones
        if not advisory_data["talking_points"] or all(is_generic(pt) for pt in advisory_data["talking_points"]):
            # Create company-specific talking points from context
            esg_scores = context.get("esg_scores", {})
            tone = context.get("avg_tone", 0)
            advisory_data["talking_points"] = [
                f"{company}'s average tone: {tone:.2f}",
                f"{company} analyzed across {context.get('article_count', 0)} articles",
                f"{company}'s ESG scores: {', '.join([f'{k}={v:.2f}' for k, v in esg_scores.items() if v])}" if esg_scores else f"{company} ESG data available in detail above"
            ]
        else:
            # Filter out any generic ones that slipped through
            advisory_data["talking_points"] = [pt for pt in advisory_data["talking_points"] if not is_generic(pt)]
        
        # Filter out generic risk radar items
        if not advisory_data["risk_radar"] or all(is_generic(r) for r in advisory_data["risk_radar"]):
            tone_vs_ind = context.get("tone_vs_industry", 0)
            negative_share = context.get("negative_share", 0)
            advisory_data["risk_radar"] = [
                f"{company}'s tone vs industry: {tone_vs_ind:+.2f} points",
                f"{company}'s negative coverage: {negative_share*100:.1f}%"
            ]
        else:
            advisory_data["risk_radar"] = [r for r in advisory_data["risk_radar"] if not is_generic(r)]
        
        # Filter out generic recommended actions
        if not advisory_data["recommended_actions"] or all(is_generic(ra) for ra in advisory_data["recommended_actions"]):
            esg_scores = context.get("esg_scores", {})
            advisory_data["recommended_actions"] = [
                f"Monitor {company}'s ESG scores: {', '.join([f'{k}={v:.2f}' for k, v in esg_scores.items() if v])}" if esg_scores else f"Review {company}'s specific ESG metrics above"
            ]
        else:
            advisory_data["recommended_actions"] = [ra for ra in advisory_data["recommended_actions"] if not is_generic(ra)]
        
        return advisory_data
        
    except Exception as e:
        error_str = str(e)
        error_msg = f"GPT advisory generation failed: {error_str}"
        LOGGER.error(error_msg)
        
        # User-friendly error messages
        if "429" in error_str or "quota" in error_str.lower():
            st.error("""
            **API Quota Exceeded**
            
            Your OpenAI API quota has been exceeded. Please:
            1. Check your OpenAI billing at https://platform.openai.com/account/billing
            2. Add credits or upgrade your plan
            3. Try again later
            
            The system will automatically use a cheaper model (gpt-4o-mini) when available.
            """)
        else:
            st.error(f"[ERROR] GPT Advisory Error: {error_str}")
        
        print(f"DEBUG: GPT Error details: {type(e).__name__}: {error_str}")
        return None


def generate_chatbot_response(context, user_message, preferences=None):
    if not user_message:
        return "Let me know what kind of investment guidance you’re looking for.", []

    response_lines = []
    evidence = []
    esg_score = interpret_esg_score(context)
    positive_share = context.get("positive_share")
    tone_change = context.get("tone_change")
    industry_positive = context.get("industry_positive_share")
    governance_score = context.get("esg_scores", {}).get("G")
    social_score = context.get("esg_scores", {}).get("S")
    environment_score = context.get("esg_scores", {}).get("E")

    def score_band(score):
        if score is None:
            return "unknown"
        if score >= 70:
            return "strong"
        if score >= 50:
            return "moderate"
        if score >= 40:
            return "subdued"
        return "weak"

    if esg_score is not None:
        response_lines.append(
            f"ESG<sup>AI</sup> benchmarks show an overall ESG score of {esg_score:.0f}/100 ({score_band(esg_score)})."
        )
    if positive_share is not None:
        pos_pct = positive_share * 100
        response_lines.append(
            f"Positive sentiment share stands at {pos_pct:.1f}%,"
            + (f" compared with {industry_positive*100:.1f}% for the industry" if industry_positive is not None else "")
            + "."
        )

    if tone_change is not None and abs(tone_change) > 0.2:
        direction = "strengthening" if tone_change > 0 else "weakening"
        response_lines.append(
            f"Tone trajectory is {direction} by {abs(tone_change):.2f} points over the review period."
        )

    recommendation = None
    if esg_score is not None and positive_share is not None:
        if esg_score >= 70 and positive_share >= 0.6:
            recommendation = (
                "The company combines a robust ESG profile with supportive sentiment. This aligns with a favourable investment stance,"
                " especially for investors seeking sustainability leaders."
            )
        elif esg_score < 40 or positive_share <= 0.4:
            recommendation = (
                "ESG<sup>AI</sup> flags elevated risk: either the ESG benchmark lags peers or sentiment remains subdued. Exercising caution before allocating capital is advisable."
            )
        else:
            recommendation = (
                "Signals are mixed. Monitoring upcoming disclosures and sentiment catalysts is prudent before taking a definitive position."
            )

    if recommendation:
        response_lines.append(recommendation)

    if governance_score is not None and governance_score < context.get("esg_industry", {}).get("G", float("inf")) - 5:
        response_lines.append(
            "Governance narratives trail the peer baseline, indicating board or oversight topics may attract scrutiny."
        )
    if environment_score is not None and environment_score < context.get("esg_industry", {}).get("E", float("inf")) - 5:
        response_lines.append(
            "Environmental reporting underperforms, suggesting a need for clearer sustainability roadmaps if the company seeks green-focused capital."
        )
    if social_score is not None and social_score < context.get("esg_industry", {}).get("S", float("inf")) - 5:
        response_lines.append(
            "Social metrics sit below sector norms—stakeholder engagement or workforce initiatives may require reinforcement."
        )

    if tone_change is not None and tone_change > 0.5:
        response_lines.append(
            "Momentum check: sentiment is improving meaningfully, hinting that remediation efforts are resonating with the market."
        )
    elif tone_change is not None and tone_change < -0.5:
        response_lines.append(
            "Momentum check: sentiment is deteriorating, highlighting unresolved ESG concerns that could weigh on valuation."
        )

    if preferences:
        resp_lower = preferences.lower()
        if "high-risk" in resp_lower or "high risk" in resp_lower:
            response_lines.append(
                "Given a higher risk appetite, tactical exposure could be justified if you anticipate near-term catalysts, but position sizing should remain disciplined."
            )
        elif "low-risk" in resp_lower or "stable" in resp_lower:
            response_lines.append(
                "For a low-risk mandate, prioritise issuers with consistently strong ESG scores and stable positive sentiment trajectories." 
                " Consider diversifying until sentiment stabilises."
            )

    if not response_lines:
        response_lines.append(
            "I could not map your query to the available ESG signals. Please ask about sentiment, ESG scores, or investment outlooks."
        )

    if context.get("tone_daily") is not None and not context["tone_daily"].empty:
        latest_date = context["tone_daily"].index.max()
        latest_tone = context["tone_daily"].iloc[-1]
        evidence.append(
            f"Latest tone reading ({pd.to_datetime(latest_date).strftime('%d %b %Y')}): {latest_tone:.2f}"
        )
    if esg_score is not None:
        evidence.append(f"ESG score (Total): {esg_score:.0f}/100")
    if positive_share is not None:
        evidence.append(f"Positive sentiment share: {positive_share*100:.1f}%")
    if industry_positive is not None:
        evidence.append(f"Industry sentiment benchmark: {industry_positive*100:.1f}%")

    return " ".join(response_lines), evidence

def inject_dark_theme():
	st.markdown(
		"""
		<style>
			:root {
				--brand-primary: #00ff00;
				--brand-secondary: #0076ff;
				--text-strong: #ffffff;
				--text-muted: #cbd5e1;
				--bg-soft: #0b0e11;
				--card-bg: rgba(30, 41, 59, 0.95);
				--card-border: rgba(255, 255, 255, 0.25);
			}
			.main { background-color: var(--bg-soft) !important; }
			section[data-testid="stSidebar"] > div { background: #0f172a !important; }
			.dataframe thead tr th { background: #1e293b !important; color: #e2e8f0 !important; font-weight: 600 !important; }
			.dataframe tbody tr { color: #e2e8f0 !important; }
			.dataframe tbody tr:hover { background: rgba(59, 130, 246, 0.1) !important; }
		</style>
		""",
		unsafe_allow_html=True,
	)


def main(start_data, end_data):
	###### CUSTOMIZE COLOR THEME ######
	# Configure Altair theme
	try:
		# Try Altair 5+ API first
		alt.theme.register("finastra", finastra_theme)
		alt.theme.enable("finastra")
	except (AttributeError, TypeError):
		# Fallback to old Altair 4.x API
		try:
			alt.themes.register("finastra", finastra_theme)
			alt.themes.enable("finastra")
		except Exception:
			pass
	violet, fuchsia = ["#694ED6", "#C137A2"]


    ###### SET UP PAGE ######
	icon_path = os.path.join(".", "raw", "esg_ai_logo.png")
	if not os.path.exists(icon_path):
		icon_path = None
	st.set_page_config(page_title="ESG AI", page_icon=icon_path if icon_path else None,
					   layout='wide', initial_sidebar_state="expanded")
	# Sidebar theme toggle
	with st.sidebar:
		st.markdown("### APPEARANCE")
		dark_mode = st.toggle("DARK TERMINAL MODE", value=True, help="Switch between performance modes")
	inject_global_styles()
	if 'dark_mode' not in st.session_state:
		st.session_state.dark_mode = dark_mode
	else:
		st.session_state.dark_mode = dark_mode
	if st.session_state.dark_mode:
		inject_dark_theme()

	hero = st.container()
	with hero:
		col_logo, col_copy = st.columns([1, 4])
		with col_logo:
			if icon_path:
				try:
					st.image(icon_path, width=150, use_container_width=False)
				except Exception as e:
					st.markdown("<div style='height:135px;display:flex;align-items:center;justify-content:center;'><em>ESG AI Dashboard</em></div>", unsafe_allow_html=True)
					LOGGER.error(f"Error loading dashboard logo: {e}")
			else:
				st.markdown("<div style='height:135px;display:flex;align-items:center;justify-content:center;'><em>No logo available</em></div>", unsafe_allow_html=True)
		with col_copy:
			st.markdown(
				"""
				<div style="background: linear-gradient(90deg, rgba(59, 130, 246, 0.15) 0%, transparent 100%); padding: 2rem; border-radius: 16px; border-left: 4px solid var(--brand-secondary); margin-bottom: 3rem; border: 1px solid rgba(255,255,255,0.12);">
					<h1 style="margin: 0; font-size: 2.5rem; font-weight: 800; letter-spacing: -0.025em; color: #ffffff;">Intelligence Console</h1>
					<p style="margin: 0.75rem 0 0; color: #cbd5e1; font-size: 1.2rem; font-weight: 500; line-height: 1.4;">Advanced ESG Signal Decomposition & Narrative Modeling</p>
				</div>
				""",
				unsafe_allow_html=True,
			)


	###### LOAD DATA ######
	with st.spinner(text="Fetching Data..."):
		data, companies = load_data(start_data, end_data)
	df_conn = data["conn"]
	df_data = data["data"]
	embeddings = data["embed"]


	###### CREATE SIDEBAR CATEGORY FILTER######
	with st.sidebar:
		st.markdown("### ANALYST CONTROLS")
		st.caption("Configure the ESG signals for this review.")
		esg_categories = st.multiselect("News Categories", ["E", "S", "G"],
										default=["E", "S", "G"],
										help="Filter narratives by Environmental, Social and Governance tags.")
		st.markdown("---")
		num_neighbors = st.slider(
			"Relationship Depth",
			min_value=1,
			max_value=20,
			value=8,
			help="Number of peer organisations highlighted in the network map.",
		)





	###### RUN COMPUTATIONS WHEN A COMPANY IS SELECTED ######
	company = st.selectbox(
		"Select a company to analyze",
		companies,
		help="Start typing to search across covered organisations.",
	)

	if company and company != "Select a Company":
		df_company = df_data[df_data.Organization == company]
		if df_company.empty:
			st.warning("No coverage available for the selected company.")
			return

		diff_col = f"{company.replace(' ', '_')}_diff"
		esg_keys = ["E_score", "S_score", "G_score"]
		esg_df = get_melted_frame(data, esg_keys, keepcol=diff_col)
		ind_esg_df = get_melted_frame(data, esg_keys, dropcol="industry_tone")
		tone_df = get_melted_frame(data, ["overall_score"], keepcol=diff_col)
		ind_tone_df = get_melted_frame(data, ["overall_score"],
									   dropcol="industry_tone")

		start = pd.to_datetime(df_company.DATE.min()).date()
		end = pd.to_datetime(df_company.DATE.max()).date()
		selected_dates = st.sidebar.date_input(
			"Date range",
			value=(start, end),
			min_value=start,
			max_value=end,
			help="Focus the analysis on a specific reporting period.",
		)
		if isinstance(selected_dates, tuple):
			start, end = selected_dates
		else:
			start, end = selected_dates, selected_dates

		df_company = filter_company_data(df_company, esg_categories, start, end)
		esg_df = filter_on_date(esg_df, start, end)
		ind_esg_df = filter_on_date(ind_esg_df, start, end)
		tone_df = filter_on_date(tone_df, start, end)
		ind_tone_df = filter_on_date(ind_tone_df, start, end)
		market_scope = filter_on_date(df_data, start, end)
		date_filtered = filter_company_data(market_scope, esg_categories, start, end)

		publishers = df_company.SourceCommonName.sort_values().unique().tolist()
		publishers.insert(0, "all")
		publisher = st.sidebar.selectbox(
			"Publisher",
			publishers,
			help="Drill into narratives from a single outlet.",
		)
		df_company = filter_publisher(df_company, publisher)

		if df_company.empty:
			st.warning("No articles match the selected filters.")
			return

		summary = build_company_summary(df_company)
		st.markdown(
			f"<div style='margin-bottom: 2rem;'><h3 style='margin: 0; font-size: 1.5rem; font-weight: 700; color: #ffffff;'>Narrative Analysis: {company}</h3>"
			f"<p style='margin: 0.5rem 0 0; color: #cbd5e1; font-size: 1rem; font-weight: 500;'>{pd.to_datetime(start).strftime('%B %d, %Y')} — {pd.to_datetime(end).strftime('%B %d, %Y')}</p></div>",
			unsafe_allow_html=True,
		)
		analysis_context = build_company_context(
			company,
			df_company,
			date_filtered,
			data,
			start,
			end,
		)

		render_metrics(summary, analysis_context)

		overview_tab, insight_tab, fusion_tab, library_tab, network_tab, report_tab, advisory_tab, ml_tab = st.tabs(
			["OVERVIEW", "INSIGHTS", "SIGNAL FUSION (PATENT)", "SOURCE LIBRARY", "CONNECTIONS", "INSIGHT REPORT", "ADVISORY", "ML ANALYSIS"]
		)

		with overview_tab:
			st.markdown("### Trend intelligence")
			selector_col, chart_col = st.columns((1, 3))
			metric_options = [
				"Tone",
				"NegativeTone",
				"PositiveTone",
				"Polarity",
				"ActivityDensity",
				"WordCount",
				"Overall Score",
				"ESG Scores",
			]
			line_metric = selector_col.radio("Choose metric", options=metric_options)

			if line_metric == "ESG Scores":
				esg_df["WHO"] = company.title()
				ind_esg_df["WHO"] = "Industry Average"
				esg_plot_df = pd.concat([esg_df, ind_esg_df]).reset_index(drop=True)
				esg_plot_df.replace({
					"E_score": "Environment",
					"S_score": "Social",
					"G_score": "Governance",
				}, inplace=True)

				metric_chart = alt.Chart(esg_plot_df, title="Trends Over Time").mark_line().encode(
					x=alt.X("yearmonthdate(DATE):O", title="DATE"),
					y=alt.Y("Score:Q"),
					color=alt.Color("ESG", sort=None, legend=alt.Legend(title=None, orient="top")),
					strokeDash=alt.StrokeDash(
						"WHO",
						sort=None,
						legend=alt.Legend(
							title=None,
							symbolType="stroke",
							symbolFillColor="gray",
							symbolStrokeWidth=4,
							orient="top",
						),
					),
					tooltip=["DATE", "ESG", alt.Tooltip("Score", format=".5f")],
				)
			else:
				if line_metric == "Overall Score":
					metric_col = "Score"
					tone_df["WHO"] = company.title()
					ind_tone_df["WHO"] = "Industry Average"
					plot_df = pd.concat([tone_df, ind_tone_df]).reset_index(drop=True)
				else:
					metric_col = line_metric
					df1 = df_company.groupby("DATE")[metric_col].mean().reset_index()
					df2 = filter_on_date(
						df_data.groupby("DATE")[metric_col].mean().reset_index(), start, end
					)
					df1["WHO"] = company.title()
					df2["WHO"] = "Industry Average"
					plot_df = pd.concat([df1, df2]).reset_index(drop=True)

				metric_chart = alt.Chart(plot_df, title="Trends Over Time").mark_line().encode(
					x=alt.X("yearmonthdate(DATE):O", title="DATE"),
					y=alt.Y(f"{metric_col}:Q", scale=alt.Scale(type="linear")),
					color=alt.Color("WHO", legend=None),
					strokeDash=alt.StrokeDash(
						"WHO",
						sort=None,
						legend=alt.Legend(
							title=None,
							symbolType="stroke",
							symbolFillColor="gray",
							symbolStrokeWidth=4,
							orient="top",
						),
					),
					tooltip=["DATE", alt.Tooltip(metric_col, format=".3f")],
				)

			metric_chart = metric_chart.properties(height=340, width=200).interactive()
			chart_col.altair_chart(metric_chart, use_container_width=True)

			radar_col, dist_col = st.columns(2)
			with radar_col:
				avg_esg = data["ESG"].copy()
				avg_esg.rename(columns={"Unnamed: 0": "Type"}, inplace=True)
				avg_esg.replace({"T": "Overall", "E": "Environment", "S": "Social", "G": "Governance"}, inplace=True)
				numeric_cols = avg_esg.select_dtypes(include=[np.number]).columns
				avg_esg["Industry Average"] = avg_esg[numeric_cols].mean(axis=1)
				radar_df = avg_esg[["Type", company, "Industry Average"]].melt(
					"Type", value_name="score", var_name="entity"
				)
				radar = px.line_polar(
					radar_df,
					r="score",
					theta="Type",
					color="entity",
					line_close=True,
					hover_name="Type",
					hover_data={"Type": True, "entity": True, "score": ":.2f"},
					color_discrete_map={"Industry Average": "#94a3b8", company: "#10b981"},
				)
				radar.update_layout(
					template=None,
					polar={
						"radialaxis": {"showticklabels": False, "ticks": ""},
						"angularaxis": {"showticklabels": False, "ticks": ""},
					},
					legend={"title": None, "orientation": "h", "yanchor": "bottom"},
					margin={"l": 5, "r": 5, "t": 35, "b": 5},
				)
				st.plotly_chart(radar, use_container_width=True)

			with dist_col:
				dist_chart = (
					alt.Chart(df_company, title="Document Tone Distribution")
					.transform_density(density="Tone", as_=["Tone", "density"])
					.mark_area(opacity=0.4, color="#3b82f6")
					.encode(
						x=alt.X("Tone:Q", scale=alt.Scale(domain=(-10, 10))),
						y="density:Q",
						tooltip=[
							alt.Tooltip("Tone", format=".3f"),
							alt.Tooltip("density:Q", format=".4f"),
						],
					)
					.properties(height=300)
					.interactive()
				)
				st.altair_chart(dist_chart, use_container_width=True)

		with fusion_tab:
			st.markdown("### SIGNAL FUSION ANALYSIS")
			st.caption("Patent-Pending Adaptive Weighting & Reliability Modeling")
			
			f_col1, f_col2 = st.columns((1, 2))
			
			with f_col1:
				st.markdown("#### DYNAMIC WEIGHTS")
				weights = analysis_context.get("fusion_weights", {})
				if weights:
					weight_data = pd.DataFrame([
						{"Source": "Sentiment (Unstructured)", "Weight": weights.get("sentiment", 0)},
						{"Source": "Pillar Scores (Structured)", "Weight": weights.get("structured", 0)},
						{"Source": "Momentum (Temporal)", "Weight": weights.get("momentum", 0)},
					])
					
					weight_chart = alt.Chart(weight_data).mark_arc(innerRadius=50).encode(
						theta=alt.Theta(field="Weight", type="quantitative"),
						color=alt.Color(field="Source", type="nominal", scale=alt.Scale(range=["#3b82f6", "#10b981", "#f59e0b"])),
						tooltip=["Source", alt.Tooltip("Weight", format=".1%")]
					).properties(height=300)
					st.altair_chart(weight_chart, use_container_width=True)
				
				st.metric("SIGNAL RELIABILITY", f"{analysis_context.get('reliability_score', 0):.2f}", 
						  help="Determined by article volume vs. tone variance.")

			with f_col2:
				st.markdown("#### INVENTION STABILITY REPORT")
				from services.evaluation import StabilityBenchmarker
				benchmarker = StabilityBenchmarker()
				
				# Create a synthetic sequence based on current context for demonstration
				tone_val = analysis_context.get("avg_tone", 0.0)
				current_tone = float(tone_val) if not pd.isna(tone_val) else 0.0
				
				# NaN-safe pillar mean
				scores = analysis_context.get("esg_scores", {})
				if scores:
					current_struct = float(np.nanmean(list(scores.values())))
				else:
					current_struct = 0.0
				
				if pd.isna(current_struct):
					current_struct = 0.0
				
				demo_samples = []
				for i in range(5):
					# Simulate noise/fluctuation
					noise = np.random.uniform(-0.5, 0.5)
					demo_samples.append({
						"avg_tone": current_tone + noise,
						"structured_score": current_struct,
						"momentum": 0.0,
						"tone_std": 1.0 + (i * 0.5), # Increasing noise
						"article_count": 10 - i # Decreasing volume
					})
				
				benchmark_df = benchmarker.run_benchmark(demo_samples)
				
				# Plot stability comparison
				plot_data = benchmark_df.melt(id_vars=["Sample"], value_vars=["Static_Score", "Adaptive_Score"], 
											  var_name="Model", value_name="ESG_Index")
				
				stability_chart = alt.Chart(plot_data, title="Adaptive vs Static Baseline").mark_line(point=True).encode(
					x="Sample:O",
					y=alt.Y("ESG_Index:Q", scale=alt.Scale(zero=False)),
					color=alt.Color("Model:N", scale=alt.Scale(range=["#ef4444", "#10b981"])),
					tooltip=["Sample", "Model", "ESG_Index"]
				).properties(height=300)
				
				st.altair_chart(stability_chart, use_container_width=True)
				
				noise_res = benchmarker.measure_noise_resistance()
				st.info(f"The Adaptive Fusion Engine reduces noise sensitivity by **{noise_res['volatility_reduction_pct']:.1f}%** compared to equal-weighted models.")

		with insight_tab:
			st.markdown("### ARTICLE SIGNALS")
			scatter = (
				alt.Chart(df_company, title="Article Tone")
				.mark_circle()
				.encode(
					x="NegativeTone:Q",
					y="PositiveTone:Q",
					size="WordCount:Q",
					color=alt.Color("Polarity:Q", scale=alt.Scale()),
					tooltip=[
						alt.Tooltip("Polarity", format=".3f"),
						alt.Tooltip("NegativeTone", format=".3f"),
						alt.Tooltip("PositiveTone", format=".3f"),
						alt.Tooltip("DATE"),
						alt.Tooltip("WordCount", format=",d"),
						alt.Tooltip("SourceCommonName", title="Site"),
					],
				)
				.properties(height=450)
				.interactive()
			)
			st.altair_chart(scatter, use_container_width=True)

		with library_tab:
			st.markdown("### COVERAGE DETAIL")
			display_cols = [
				"DATE",
				"SourceCommonName",
				"Tone",
				"Polarity",
				"NegativeTone",
				"PositiveTone",
			]
			st.dataframe(
				df_company[display_cols].sort_values("DATE", ascending=False),
				use_container_width=True,
			)
			st.markdown("#### FEATURED ARTICLES")
			link_df = df_company[["DATE", "URL"]].head(3).copy()
			link_df["ARTICLE"] = link_df.URL.apply(get_clickable_name)
			st.markdown(link_df[["DATE", "ARTICLE"]].to_markdown(index=False))

		with network_tab:
			neighbor_cols = [f"n{i}_rec" for i in range(num_neighbors)]
			company_df = df_conn[df_conn.company == company]
			if company_df.empty:
				st.warning("No connection data available for this company.")
			else:
				neighbors = company_df[neighbor_cols].iloc[0]
				overlays = embeddings.copy()
				color_f = lambda f: (
					f"Company: {company.title()}"
					if f == company
					else ("Connected Company" if f in neighbors.values else "Other Company")
				)
				overlays["colorCode"] = overlays.company.apply(color_f)
				point_colors = {
					company: violet,
					"Connected Company": fuchsia,
					"Other Company": "lightgrey",
				}
				fig_3d = px.scatter_3d(
					overlays,
					x="0",
					y="1",
					z="2",
					color="colorCode",
					color_discrete_map=point_colors,
					opacity=0.4,
					hover_name="company",
					hover_data={c: False for c in overlays.columns},
				)
				fig_3d.update_layout(
					legend={"orientation": "h", "yanchor": "bottom", "title": None},
					margin={"l": 0, "r": 0, "t": 0, "b": 0},
				)
				st.plotly_chart(fig_3d, use_container_width=True)

				conf_cols = [f"n{i}_conf" for i in range(num_neighbors)]
				neighbor_conf = pd.DataFrame(
					{
						"Neighbor": neighbors,
						"Confidence": company_df[conf_cols].values[0],
					}
				)
				conf_plot = (
					alt.Chart(neighbor_conf, title="Connected companies")
					.mark_bar()
					.encode(
						x="Confidence:Q",
						y=alt.Y("Neighbor:N", sort="-x"),
						tooltip=[
							"Neighbor",
							alt.Tooltip("Confidence", format=".3f"),
						],
						color=alt.Color("Confidence:Q", scale=alt.Scale(), legend=None),
					)
					.properties(height=25 * num_neighbors + 100)
					.configure_axis(grid=False)
				)
				st.altair_chart(conf_plot, use_container_width=True)

		with report_tab:
			st.markdown("### DECISION SUPPORT MATRIX")

			exec_summary = generate_executive_summary(analysis_context)
			comparison_table = build_industry_comparison_table(analysis_context)
			pillar_table = build_pillar_breakdown(df_company)
			pillar_commentary = derive_pillar_commentary(pillar_table, analysis_context)
			momentum_summary = compute_sentiment_momentum(analysis_context, df_company)
			trend_points = generate_trend_narrative(analysis_context, df_company)
			if trend_points:
				momentum_summary.setdefault("bullets", []).extend(trend_points)
			catalysts = identify_catalyst_timeline(df_company)
			source_influence = derive_source_influence(df_company)
			risks, opportunities = build_risk_opportunity(
				analysis_context, pillar_table, source_influence, catalysts
			)
			watchlist = build_watchlist(catalysts)
			highlight_data = collect_article_highlights(analysis_context)
			actions = generate_actionable_insights(analysis_context)
			conclusion_text = generate_conclusion(analysis_context)
			report_markdown = generate_report_markdown(
				analysis_context,
				exec_summary,
				comparison_table,
				pillar_table,
				pillar_commentary,
				momentum_summary,
				catalysts,
				risks,
				opportunities,
				source_influence,
				actions,
				conclusion_text,
				highlight_data,
				watchlist,
			)
			company_slug = re.sub(
				r"[^a-z0-9]+",
				"-",
				analysis_context["company"].lower(),
			).strip("-")
			start_label = pd.to_datetime(analysis_context["start"]).strftime("%Y%m%d")
			end_label = pd.to_datetime(analysis_context["end"]).strftime("%Y%m%d")
			st.download_button(
				label="EXPORT DECISION BRIEF (.MD)",
				data=report_markdown,
				file_name=f"esg-ai-report_{company_slug}_{start_label}_{end_label}.md",
				mime="text/markdown",
			)

			st.markdown("#### EXECUTIVE SUMMARY")
			metric_cols = st.columns(4)
			with metric_cols[0]:
				st.metric("Articles analysed", format_metric(analysis_context.get("article_count"), precision=0))
			with metric_cols[1]:
				st.metric("Average tone", format_metric(analysis_context.get("avg_tone"), precision=2))
			with metric_cols[2]:
				st.metric(
					"Positive share",
					format_percentage(analysis_context.get("positive_share")),
				)
			with metric_cols[3]:
				st.metric(
					"Tone vs industry",
					format_number(analysis_context.get("tone_vs_industry")),
				)
			st.markdown(exec_summary)

			st.divider()
			st.markdown("#### KEY METRICS VS INDUSTRY")
			if comparison_table is not None and not comparison_table.empty:
				st.table(comparison_table)
			else:
				st.info("No benchmark data available for the selected view.")

			st.divider()
			st.markdown("#### SENTIMENT MOMENTUM")
			for point in momentum_summary.get("bullets", []) or ["Sentiment momentum not available."]:
				st.markdown(f"- {point}")
			momentum_metrics = momentum_summary.get("metrics", {})
			if momentum_metrics:
				metrics_df = pd.DataFrame(
					{
						"Metric": [
							"Average tone",
							"Tone vs industry",
							"Tone change",
							"Tone volatility",
							"Positive share",
						],
						"Value": [
							format_number(momentum_metrics.get("avg_tone")),
							format_number(momentum_metrics.get("tone_vs_industry")),
							format_number(momentum_metrics.get("tone_change")),
							format_number(momentum_metrics.get("tone_volatility")),
							format_percentage(momentum_metrics.get("positive_share")),
						],
					}
				)
				st.table(metrics_df)

			st.divider()
			st.markdown("#### CATALYST TIMELINE")
			if catalysts:
				catalyst_table = pd.DataFrame(
					{
						"Date": [pd.to_datetime(event["date"]).strftime("%d %b %Y") for event in catalysts],
						"Articles": [event.get("volume") for event in catalysts],
						"Tone": [format_number(event.get("avg_tone")) for event in catalysts],
						"Positive share": [format_percentage(event.get("positive_share")) for event in catalysts],
						"Lead source": [
							event.get("top_source") or event.get("highlight_source") or "—"
							for event in catalysts
						],
						"Pillars": [", ".join(event.get("pillars", [])) or "—" for event in catalysts],
					}
				)
				st.table(catalyst_table)
				st.markdown("**Representative articles**")
				for event in catalysts:
					url = event.get("highlight_url")
					label = get_clickable_name(url) if url else "No URL"
					date_label = pd.to_datetime(event["date"]).strftime("%d %b %Y")
					st.markdown(f"- {date_label} · {event.get('highlight_source', 'Unknown source')} · {format_number(event.get('highlight_tone'))} — {label}")
			else:
				st.info("No catalysts detected for this selection.")

			st.divider()
			st.markdown("#### ESG PILLAR COMMENTARY")
			if pillar_commentary:
				for item in pillar_commentary:
					st.markdown(f"- {item}")
			else:
				st.caption("No ESG pillar commentary available.")
			if pillar_table is not None and not pillar_table.empty:
				display_pillars = pillar_table.copy()
				display_pillars["Share"] = display_pillars["Share"].apply(lambda v: f"{v*100:.1f}%")
				display_pillars["Average Tone"] = display_pillars["Average Tone"].apply(
					lambda v: "n/a" if pd.isna(v) else f"{v:.2f}"
				)
				display_pillars["Positive Share"] = display_pillars["Positive Share"].apply(
					lambda v: "n/a" if pd.isna(v) else f"{v*100:.1f}%"
				)
				st.table(display_pillars)
			else:
				st.info("No ESG pillar tags recorded for this cohort.")

			st.divider()
			st.markdown("#### SOURCE INFLUENCE")
			source_cols = st.columns(3)
			categories = [
				("Supportive amplifiers", "supportive"),
				("Critical voices", "critical"),
				("Neutral/monitor", "neutral"),
			]
			for col, (label, key) in zip(source_cols, categories):
				with col:
					entries = source_influence.get(key, []) if source_influence else []
					st.markdown(f"**{label}**")
					if entries:
						records = []
						for e in entries:
							records.append(
								{
									"Source": e.get("SourceCommonName") or e.get("Source"),
									"Tone": format_number(e.get("avg_tone")),
									"Articles": e.get("articles"),
								}
							)
						display_df = pd.DataFrame(records)
						st.table(display_df)
					else:
						st.caption("None identified.")

			st.divider()
			st.markdown("#### RISK & OPPORTUNITY OUTLOOK")
			ro_cols = st.columns(2)
			with ro_cols[0]:
				st.markdown("**Risk radar**")
				if risks:
					for risk in risks:
						st.markdown(f"- {risk}")
				else:
					st.caption("No acute risks flagged.")
			with ro_cols[1]:
				st.markdown("**Opportunity drivers**")
				if opportunities:
					for item in opportunities:
						st.markdown(f"- {item}")
				else:
					st.caption("No clear opportunities surfaced.")

			st.divider()
			st.markdown("#### ACTION CHECKLIST")
			for action in actions:
				st.markdown(f"- {action}")

			st.divider()
			st.markdown("#### WATCHLIST")
			if watchlist:
				for item in watchlist:
					url = item.get("url")
					line = f"- {item['date']} · {item['source'] or 'Unknown source'} · Tone {item['tone']}"
					if url:
						st.markdown(f"{line} [Read more]({url})")
					else:
						st.markdown(line)
			else:
				st.caption("Watchlist is clear for this period.")

			st.divider()
			st.markdown("#### EVIDENCE APPENDIX")
			highlight_cols = st.columns(3)
			labels = [
				("Positive momentum", "positive"),
				("Risks to monitor", "negative"),
				("Latest mentions", "recent"),
			]
			for column, (label, key) in zip(highlight_cols, labels):
				with column:
					st.markdown(f"**{label}**")
					entries = highlight_data.get(key, []) if highlight_data else []
					if not entries:
						st.caption("No articles available.")
					else:
						for entry in entries:
							tone_val = entry.get("tone")
							tone_str = "Tone n/a" if pd.isna(tone_val) else f"Tone {tone_val:.2f}"
							date_label = entry.get("date", "—")
							source_label = entry.get("source", "")
							url = entry.get("url")
							text = f"- {date_label} · {source_label} · {tone_str}"
							if url:
								st.markdown(f"{text} [Read more]({url})")
							else:
								st.markdown(text)

			st.divider()
			st.markdown("#### TRENDS & VISUALS")
			tone_chart = build_tone_trend_chart(analysis_context, date_filtered)
			distribution_chart = build_tone_distribution_chart(df_company)
			chart_col1, chart_col2 = st.columns(2)
			with chart_col1:
				if tone_chart is not None:
					st.altair_chart(tone_chart, use_container_width=True)
					st.caption("Tone trend benchmarked against industry average.")
				else:
					st.info("Insufficient data to plot tone trend.")
			with chart_col2:
				if distribution_chart is not None:
					st.altair_chart(distribution_chart, use_container_width=True)
					st.caption("Distribution of article tone scores for the selected period.")
				else:
					st.info("Insufficient data to plot tone distribution.")

			st.divider()
			st.markdown("#### CONCLUSION")
			st.markdown(conclusion_text)

		with advisory_tab:
			st.markdown("### AI ADVISORY INTELLIGENCE WORKSPACE")
			st.caption(
				"Powered by GPT-4: Craft tailored client guidance using comprehensive ESG data analysis."
			)

			# Check GPT availability
			gpt_available = OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY")
			if gpt_available:
				st.success("[OK] GPT-powered advisory enabled")
			else:
				st.info("[INFO] GPT mode unavailable. Configure OPENAI_API_KEY in environment for enhanced AI advisory.")

			# Add catalysts to analysis context for GPT
			if "catalysts" not in analysis_context:
				analysis_context["catalysts"] = identify_catalyst_timeline(df_company)

			st.markdown("#### ADVISORY DESK")
			
			# Quick prompt examples
			with st.expander("[INFO] Example Prompts", expanded=False):
				st.markdown("""
				**Try these prompts:**
				- "Provide investment recommendations based on ESG trends"
				- "Analyze environmental risks and opportunities"
				- "Draft a briefing for the sustainability committee"
				- "Summarize key governance concerns and improvement areas"
				- "Assess social impact performance vs industry peers"
				- "Identify top ESG catalysts and their investment implications"
				- "Evaluate media sentiment trends and risk indicators"
				""")
			
			pref_col1, pref_col2 = st.columns((2, 1))
			with pref_col1:
				advisory_prompt = st.text_area(
					"ADVISORY REQUEST",
					height=140,
					placeholder="Describe what analysis or briefing you need...",
					help="Be specific about what insights you want from the ESG data",
					key="advisory_prompt",
				)
			with pref_col2:
				client_focus = st.text_input(
					"Client Context (Optional)",
					placeholder="e.g. EU pension fund, ESG-focused",
					help="Client mandate, portfolio focus, or risk preferences",
					key="advisory_profile",
				)
				advisory_button = st.button(
					"GENERATE AI ADVISORY",
					key="advisory_button",
					type="primary",
					use_container_width=True,
				)

			if advisory_button:
				client_profile = {
					"mandate": client_focus or None,
					"portfolio": st.session_state.get("portfolio_active_name"),
				}
				
				# Try GPT-powered advisory first
				advisory_payload = None
				if gpt_available:
					with st.spinner("[AI] Analyzing ESG data with GPT-4..."):
						try:
							advisory_payload = generate_gpt_advisory(
								company,
								analysis_context,
								df_company,
								prompt=advisory_prompt,
								client_profile=client_profile,
							)
							if advisory_payload:
								st.success("[OK] GPT-4 analysis complete")
						except Exception as e:
							st.error(f"GPT analysis error: {str(e)}")
							print(f"DEBUG: Full error: {repr(e)}")
							import traceback
							traceback.print_exc()
							advisory_payload = None
				
				# Fallback to API service
				if not advisory_payload:
					advisory_payload = call_advisory_service(
						company,
						start,
						end,
						prompt=advisory_prompt,
						profile=client_profile,
					)

				# Final fallback to rule-based
				if not advisory_payload:
					fallback_text, evidence = generate_chatbot_response(
						analysis_context,
						advisory_prompt,
						preferences=client_focus,
					)
					advisory_payload = {
						"executive_summary": fallback_text,
						"talking_points": [fallback_text],
						"risk_radar": [],
						"recommended_actions": [],
						"evidence": evidence or [],
						"disclaimer": "Rule-based advisory draft. Review before circulation.",
					}

				_record_event(
					"generate_advisory",
					st.session_state.get("portfolio_active_name"),
					{"company": company, "prompt": advisory_prompt},
				)

				st.markdown("#### EXECUTIVE SUMMARY")
				st.write(advisory_payload.get("executive_summary", "No summary available."))

				col_tp, col_rr = st.columns(2)
				with col_tp:
					st.markdown("**Key talking points**")
					for item in advisory_payload.get("talking_points", []) or ["No key points generated."]:
						st.markdown(f"- {item}")
				with col_rr:
					st.markdown("**Risk radar**")
					for item in advisory_payload.get("risk_radar", []) or ["No emergent risks flagged."]:
						st.markdown(f"- {item}")

				st.markdown("**Recommended actions**")
				for action in advisory_payload.get("recommended_actions", []) or ["Maintain monitoring cadence."]:
					st.markdown(f"- {action}")

				st.markdown("**Evidence citations**")
				for item in advisory_payload.get("evidence", []) or ["No supporting evidence captured."]:
					st.markdown(f"- {item}")

				st.caption(advisory_payload.get("disclaimer", ""))

				pdf_bytes = build_advisory_pdf(advisory_payload)
				if st.download_button(
					label="DOWNLOAD ADVISORY BRIEF (PDF)",
					data=pdf_bytes,
					file_name=f"advisory_{company}_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M')}.pdf",
					mime="application/pdf",
				):
					_record_event(
						"export_advisory",
						st.session_state.get("portfolio_active_name"),
						{"format": "pdf", "company": company},
					)

		with ml_tab:
			st.markdown("### ML-POWERED ESG CLASSIFICATION & EXPLAINABILITY")
			st.caption(
				"Automated ESG category classification using ESGBERT with explainable AI insights"
			)
			
			# Check ML availability
			try:
				from services.esg_classifier import ESGBertClassifier
				from services.explainability import ESGExplainer
				ML_AVAILABLE = True
			except ImportError:
				ML_AVAILABLE = False
				st.warning("[WARN] ML services not available. Install: pip install transformers shap lime")
			
			if ML_AVAILABLE:
				# Initialize classifier
				if 'ml_classifier' not in st.session_state:
					with st.spinner("Loading ESGBERT model..."):
						st.session_state.ml_classifier = ESGBertClassifier(use_gpu=False)
					st.success("[OK] ESGBERT loaded successfully")
				
				st.divider()
				
				# Text classification section
				st.markdown("#### TEXT CLASSIFICATION")
				
				col1, col2 = st.columns([2, 1])
				with col1:
					text_input = st.text_area(
						"Enter text to classify",
						height=150,
						placeholder="Enter company news, press release, or article text...",
						help="The model will classify into Environmental, Social, or Governance categories",
						key="ml_text_input",
					)
				with col2:
					st.markdown("<br>", unsafe_allow_html=True)
					classify_btn = st.button(
						"CLASSIFY",
						use_container_width=True,
						type="primary",
						key="ml_classify_btn",
					)
					show_explanation = st.checkbox("Show explanation", value=True, key="ml_show_explanation")
				
				if classify_btn and text_input:
					with st.spinner("Classifying..."):
						result = st.session_state.ml_classifier.classify(text_input)
					
					# Display results
					st.divider()
					st.markdown("#### CLASSIFICATION RESULTS")
					
					metric_cols = st.columns(3)
					with metric_cols[0]:
						category = result.get('category', 'None')
						category_label = category if category else "None"
						st.metric("ESG Category", category_label)
					with metric_cols[1]:
						confidence = result.get('confidence', 0)
						st.metric("Confidence", f"{confidence:.1%}")
					with metric_cols[2]:
						method = result.get('method', 'unknown')
						st.metric("Method", method.title())
					
					# Show all predictions if available
					if result.get('all_predictions'):
						with st.expander("📋 All Predictions", expanded=False):
							for pred in result['all_predictions'][:5]:
								st.write(f"**{pred['label']}**: {pred['score']:.2%}")
					
					# Explainability
					if show_explanation:
						st.divider()
						st.markdown("#### 🔬 Explainability")
						st.caption("Understanding why the model made this prediction")
						
						try:
							with st.spinner("Generating explanation..."):
								explainer = ESGExplainer(classifier=st.session_state.ml_classifier)
								# Use fewer features and faster settings
								explanation = explainer.explain_with_lime(text_input, num_features=5)
							
							if 'error' not in explanation:
								# Show top features
								st.markdown("**Most Important Words:**")
								top_features = explanation.get('top_features', [])[:10]
								if top_features:
									for word, score in top_features:
										# Convert numpy types to Python native types
										word_str = str(word)
										score_float = float(score)
										
										# Create visual bar
										bar_html = f"""
										<div style="display: flex; align-items: center; margin: 5px 0;">
											<div style="flex: 0 0 150px;">{word_str}</div>
											<div style="flex: 1; background: #e0e0e0; height: 20px; border-radius: 10px; margin: 0 10px;">
												<div style="background: {'green' if score_float > 0 else 'red'}; height: 100%; width: {abs(score_float)*100:.1f}%; border-radius: 10px;"></div>
											</div>
											<div style="flex: 0 0 60px; text-align: right;">{score_float:.3f}</div>
										</div>
										"""
										st.markdown(bar_html, unsafe_allow_html=True)
								else:
									st.info("No explainability features available for this text.")
							else:
								st.warning(f"⚠️ Explanation not available: {explanation.get('error', 'Unknown error')}")
						except Exception as e:
							st.error(f"Error generating explanation: {str(e)}")
				
				# Real-time data section
				try:
					from services.data_ingestion import fetch_company_news
					DATA_FETCH_AVAILABLE = True
				except ImportError:
					DATA_FETCH_AVAILABLE = False
				
					if DATA_FETCH_AVAILABLE:
						st.divider()
						st.markdown("#### 🔄 Real-Time Data Ingestion")
						st.caption("⚠️ Note: This fetches current/live articles from today. The chart data above is historical (Dec 2020 - Jan 2021).")
						
						# Check if we have newsapi key
						has_api_key = os.getenv('NEWSAPI_KEY') is not None
					
					if has_api_key:
						st.info("📡 NewsAPI configured - Real-time news fetch available")
						
						# Fetch button for current company
						if company:
							# Date range selector
							col_date, col_fetch = st.columns([2, 1])
							with col_date:
								days_back = st.selectbox(
									"📅 Articles from last",
									options=[7, 14, 21, 30],
									index=0,
									key="news_days_selector"
								)
							with col_fetch:
								st.markdown("<br>", unsafe_allow_html=True)
								fetch_btn = st.button("📰 Fetch News", key="fetch_news_btn", use_container_width=True)
							
							if fetch_btn:
								with st.spinner(f"Fetching latest news for {company} (last {days_back} days)..."):
									articles = fetch_company_news(company, days=days_back)
									
									if articles:
										st.success(f"✅ Found {len(articles)} recent articles")
										
										# Classify articles (first 20 for performance)
										articles_to_show = articles[:20]
										classified = st.session_state.ml_classifier.classify_batch(
											[a.get('title', '') + ' ' + a.get('description', '') for a in articles_to_show]
										)
										
										# Category breakdown
										category_counts = {}
										for result in classified:
											cat = result.get('category', 'None')
											category_counts[cat] = category_counts.get(cat, 0) + 1
										
										# Show summary
										if category_counts:
											st.markdown("**Category Breakdown:**")
											cat_cols = st.columns(len(category_counts))
											for i, (cat, count) in enumerate(sorted(category_counts.items(), key=lambda x: -x[1])):
												with cat_cols[i]:
													st.metric(f"{cat if cat else 'None'}", count)
										
										st.divider()
										
										# Display
										for article, class_result in zip(articles_to_show, classified):
											with st.expander(f"📄 {article.get('title', 'No title')[:60]}..."):
												st.caption(f"Source: {article.get('source')} | {article.get('published_at')}")
												st.write(article.get('description', '')[:200] + "...")
												
												# Show classification
												cat = class_result.get('category', 'Unknown')
												conf = class_result.get('confidence', 0)
												
												col_cat, col_conf = st.columns(2)
												with col_cat:
													st.badge(cat if cat else "None", delta=cat if cat else None)
												with col_conf:
													st.metric("Confidence", f"{conf:.1%}")
										
										if len(articles) > 20:
											st.caption(f"... and {len(articles) - 20} more articles")
									else:
										st.info(f"No recent articles found for {company}")
					else:
						st.info("🔑 NewsAPI key not configured. Add NEWSAPI_KEY to .env for real-time news.")
				
				# Quick examples
				st.divider()
				with st.expander("💡 Quick Examples", expanded=False):
					examples = {
						"Environmental": "Company announces new renewable energy initiative and carbon neutrality goals by 2030",
						"Social": "Board approves diversity and inclusion program expanding employee benefits",
						"Governance": "Corporate governance reforms address executive compensation transparency and board oversight"
					}
					
					# Show example texts as copyable code blocks
					for cat, example in examples.items():
						st.markdown(f"**{cat} Example:**")
						st.code(example, language=None)
			
			else:
				st.info("""
				**ML Features Not Available**
				
				To enable ML-powered classification:
				1. Install dependencies: `pip install -r requirements.txt`
				2. Restart the app
				
				See `IMPLEMENTATION_GUIDE.md` for more details.
				""")
	else:
		st.info("Select a company from the drop-down to launch the ESG briefing.")


if __name__ == "__main__":
	args = sys.argv
	if len(args) != 3:
		# Default: try to use synthetic dataset, fall back to original
		if "synthetic2021_to_synthetic2025" in os.listdir("Data"):
			start_data = "synthetic2021"
			end_data = "synthetic2025"
		else:
			start_data = "dec30"
			end_data = "jan12"
	else:
		start_data = args[1]
		end_data = args[2]

	if f"{start_data}_to_{end_data}" not in os.listdir("Data"):
		print(f"There isn't data for {start_data}_to_{end_data}")
		raise NameError(f"Please pick from {os.listdir('Data')}")
		sys.exit()
		st.stop()
	else:
		main(start_data, end_data)
	alt.themes.enable("default")


# one_month, ten_days
