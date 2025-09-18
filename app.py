import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import pickle
import itertools
import re
from collections import Counter
import plotly.express as px
from plot_setup import finastra_theme
from download_data import Data
import sys

import metadata_parser

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


def filter_on_date(df, start, end, date_col="DATE"):
    # Handle different date column types (some are date objects, some are Timestamps)
    if df[date_col].dtype == 'object':
        # Date column contains datetime.date objects - convert comparison values to date
        if hasattr(start, 'date'):
            start = start.date() if hasattr(start, 'date') else start
        if hasattr(end, 'date'):
            end = end.date() if hasattr(end, 'date') else end
        df = df[(df[date_col] >= start) & (df[date_col] <= end)]
    else:
        # Date column contains Timestamps - convert comparison values to Timestamp
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        df = df[(df[date_col] >= start_ts) & (df[date_col] <= end_ts)]
    return df


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
			:root {
				--brand-primary: #533fd7;
				--brand-secondary: #C137A2;
				--text-strong: #1f2333;
				--text-muted: #4f5567;
				--bg-soft: #f5f6fb;
				--card-bg: #ffffff;
				--card-border: rgba(105, 78, 214, 0.12);
				--shadow-1: 0 6px 20px rgba(15, 31, 64, 0.06);
				--shadow-2: 0 10px 35px rgba(15, 31, 64, 0.08);
			}

			/* App background */
			.main {
				background: linear-gradient(135deg, var(--bg-soft) 0%, #ffffff 45%);
				padding: 0 1.25rem;
			}
			.block-container { padding-top: 1.25rem; }

			/* Sidebar */
			section[data-testid="stSidebar"] > div {
				background: #ffffff;
				border-right: 1px solid var(--card-border);
			}
			section[data-testid="stSidebar"] .stSelectbox, 
			section[data-testid="stSidebar"] .stMultiSelect,
			section[data-testid="stSidebar"] .stDateInput { margin-bottom: .5rem; }
			section[data-testid="stSidebar"] * { font-size: 0.96rem; }

			/* Hero/Header */
			.app-hero {
				background: var(--card-bg);
				border-radius: 18px;
				padding: 1.6rem 2.0rem;
				box-shadow: var(--shadow-2);
				border: 1px solid var(--card-border);
				display: flex; align-items: center; justify-content: space-between; gap: 1.25rem;
			}
			.app-hero h1 {
				font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
				font-size: 2.2rem; font-weight: 700; color: var(--text-strong); margin: 0;
			}
			.app-hero p { font-size: 1rem; color: var(--text-muted); margin: 0; }

			/* Tabs */
			.stTabs [role="tab"] {
				border-radius: 12px 12px 0 0;
				padding: 0.65rem 1.25rem; font-weight: 600; color: #5e6482;
			}
			.stTabs [role="tab"][aria-selected="true"] {
				background: #ffffff; color: var(--brand-primary);
				box-shadow: 0 -1px 12px rgba(83, 63, 215, 0.18);
			}

			/* Cards/Metrics */
			.metric-card {
				background: var(--card-bg); padding: 1rem 1.15rem; border-radius: 14px;
				border: 1px solid var(--card-border); box-shadow: var(--shadow-1);
			}
			.metric-card h4 {
				font-size: 0.82rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
				color: #5a5f73; margin-bottom: .3rem;
			}
			.metric-card .metric-value { font-size: 1.7rem; font-weight: 700; color: #181c2f; }
			.metric-card .delta-positive { color: #0d7a60; font-weight: 600; font-size: 0.85rem; }
			.metric-card .delta-negative { color: #a52b3d; font-weight: 600; font-size: 0.85rem; }

			/* Tables */
			.dataframe { border-radius: 12px; overflow: hidden; border: 1px solid var(--card-border); }
			.dataframe thead tr th { background: #fafbff; color: #555b74; font-weight: 700; }
			.dataframe tbody tr:hover { background: #fafbff; }
			.dataframe tbody tr th { display: none; }

			/* Buttons */
			.stButton > button {
				background: var(--brand-primary); color: #fff; font-weight: 600;
				border-radius: 10px; border: none; padding: .5rem .9rem; box-shadow: var(--shadow-1);
			}
			.stButton > button:hover { filter: brightness(1.05); }

			/* Inputs */
			.stTextInput input, .stSelectbox > div, .stMultiSelect > div, .stDateInput > div {
				border-radius: 10px; border-color: var(--card-border);
			}

			/* Scrollbar (WebKit) */
			::-webkit-scrollbar { height: 10px; width: 10px; }
			::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 8px; }
			::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.25); }
		</style>
		""",
		unsafe_allow_html=True,
	)
	# Optional footer
	st.markdown(
		"""
		<div style="text-align:center; color:#8a8faa; font-size:.85rem; margin-top:1rem;">
			ESG AI · Enhanced UI — built with Streamlit
		</div>
		""",
		unsafe_allow_html=True,
	)


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


def render_metrics(summary):
    art_col, tone_col, sentiment_col = st.columns(3)
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


def format_number(value, decimals=2, suffix=""):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{decimals}f}{suffix}"


def format_percentage(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


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


def build_company_context(company, df_company, df_market, data, start, end):
    context = {
        "company": company,
        "company_label": company.title(),
        "start": start,
        "end": end,
        "range_label": f"{pd.to_datetime(start).strftime('%b %d, %Y')} — {pd.to_datetime(end).strftime('%b %d, %Y')}"
    }

    context["article_count"] = len(df_company)
    if context["article_count"] == 0:
        return context

    context["avg_tone"] = df_company["Tone"].mean()
    context["median_tone"] = df_company["Tone"].median()
    context["tone_std"] = df_company["Tone"].std()
    tone_daily = df_company.groupby("DATE")["Tone"].mean().sort_index()
    context["tone_daily"] = tone_daily
    if not tone_daily.empty:
        window = min(3, len(tone_daily))
        context["tone_early"] = tone_daily.head(window).mean()
        context["tone_recent"] = tone_daily.tail(window).mean()
        context["tone_change"] = tone_daily.iloc[-1] - tone_daily.iloc[0] if len(tone_daily) > 1 else 0.0
        context["tone_best_date"] = tone_daily.idxmax()
        context["tone_best_value"] = tone_daily.max()
        context["tone_worst_date"] = tone_daily.idxmin()
        context["tone_worst_value"] = tone_daily.min()
    else:
        context["tone_early"] = context["tone_recent"] = context["tone_change"] = None
        context["tone_best_date"] = context["tone_worst_date"] = None
        context["tone_best_value"] = context["tone_worst_value"] = None

    positive_mask = df_company["PositiveTone"] > df_company["NegativeTone"]
    context["positive_share"] = positive_mask.mean() if len(df_company) else None
    context["negative_share"] = (1 - context["positive_share"]) if context["positive_share"] is not None else None
    context["avg_polarity"] = df_company["Polarity"].mean()
    context["avg_activity"] = df_company["ActivityDensity"].mean()
    context["avg_wordcount"] = df_company["WordCount"].mean()

    context["publisher_counts"] = df_company["SourceCommonName"].value_counts().head(5)
    context["recent_articles"] = df_company.sort_values("DATE", ascending=False).head(5)
    context["top_positive_articles"] = df_company.sort_values("Tone", ascending=False).head(3)
    context["top_negative_articles"] = df_company.sort_values("Tone", ascending=True).head(3)

    context["daily_volume"] = df_company.groupby("DATE").size().sort_index()
    if not context["daily_volume"].empty:
        context["busiest_day"] = context["daily_volume"].idxmax()
        context["busiest_day_count"] = int(context["daily_volume"].max())
    else:
        context["busiest_day"] = None
        context["busiest_day_count"] = None

    context["industry_tone_avg"] = df_market["Tone"].mean() if not df_market.empty else None
    industry_pos_mask = df_market["PositiveTone"] > df_market["NegativeTone"] if not df_market.empty else pd.Series(dtype=float)
    context["industry_positive_share"] = industry_pos_mask.mean() if len(industry_pos_mask) else None
    context["industry_polarity"] = df_market["Polarity"].mean() if not df_market.empty else None
    if context["industry_tone_avg"] is not None:
        context["tone_vs_industry"] = context["avg_tone"] - context["industry_tone_avg"]
    else:
        context["tone_vs_industry"] = None
    if context.get("positive_share") is not None and context.get("industry_positive_share") is not None:
        context["positive_vs_industry"] = context["positive_share"] - context["industry_positive_share"]
    else:
        context["positive_vs_industry"] = None

    esg_matrix = data.get("ESG")
    context["esg_scores"] = {}
    context["esg_industry"] = {}
    if esg_matrix is not None:
        esg_df = esg_matrix.set_index("Unnamed: 0")
        if company in esg_df.columns:
            comp_scores = esg_df[company]
            industry_scores = esg_df.select_dtypes(include=[np.number]).mean(axis=1)
            context["esg_scores"] = comp_scores.to_dict()
            context["esg_industry"] = industry_scores.to_dict()

    return context


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
				--brand-primary: #a89cff;
				--brand-secondary: #ff7bd7;
				--text-strong: #e9eaf5;
				--text-muted: #b5b9d3;
				--bg-soft: #0f1320;
				--card-bg: #151a2c;
				--card-border: rgba(168, 156, 255, 0.15);
			}
			.main { background: linear-gradient(135deg, var(--bg-soft) 0%, #0d111c 45%); }
			section[data-testid="stSidebar"] > div { background: #0f1320; }
			.dataframe thead tr th { background: #11162a; }
			.dataframe tbody tr:hover { background: #11162a; }
			.stButton > button { box-shadow: none; }
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
	st.set_page_config(page_title="ESG AI", page_icon=icon_path,
					   layout='wide', initial_sidebar_state="expanded")
	# Sidebar theme toggle
	with st.sidebar:
		st.markdown("**Appearance**")
		dark_mode = st.toggle("Dark mode", value=False, help="Switch between light and dark themes")
	inject_global_styles()
	if 'dark_mode' not in st.session_state:
		st.session_state.dark_mode = dark_mode
	else:
		st.session_state.dark_mode = dark_mode
	if st.session_state.dark_mode:
		inject_dark_theme()

	hero = st.container()
	with hero:
		col_logo, col_copy = st.columns([1, 3])
		with col_logo:
			st.image(icon_path, width=135)
		with col_copy:
			st.markdown(
				"""
				<div class="app-hero">
					<div>
						<h1>ESG<sup>AI</sup> Intelligence Console</h1>
						<p>Executive-grade visibility into ESG narratives, sentiment, and peer linkages.</p>
					</div>
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
		st.markdown("### Analyst Controls")
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
			f"<p class='esg-section-title'>Narrative summary for <strong>{company}</strong> · "
			f"{pd.to_datetime(start).strftime('%b %d, %Y')} — {pd.to_datetime(end).strftime('%b %d, %Y')}</p>",
			unsafe_allow_html=True,
		)
		render_metrics(summary)

		analysis_context = build_company_context(
			company,
			df_company,
			date_filtered,
			data,
			start,
			end,
		)

		overview_tab, insight_tab, library_tab, network_tab, report_tab, ai_tab = st.tabs(
			["Overview", "Insights", "Source Library", "Connections", "Insight Report", "AI Summarizer"]
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
					color_discrete_map={"Industry Average": fuchsia, company: violet},
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
					.mark_area(opacity=0.55, color="#694ED6")
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

		with insight_tab:
			st.markdown("### Article-level signals")
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
			st.markdown("### Coverage detail")
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
			st.markdown("#### Featured articles")
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
			st.markdown("### ESG Insight Report")

			exec_summary = generate_executive_summary(analysis_context)
			st.markdown("#### Executive Summary")
			st.markdown(exec_summary)

			st.divider()
			st.markdown("#### Key Metrics Breakdown")
			for metric, description in METRIC_DEFINITIONS.items():
				st.markdown(f"- **{metric}**: {description}", unsafe_allow_html=True)

			st.divider()
			st.markdown("#### Trends and Insights")
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

			trend_points = generate_trend_narrative(analysis_context, df_company)
			for point in trend_points:
				st.markdown(f"- {point}")

			st.divider()
			st.markdown("#### Industry Comparison")
			comparison_table = build_industry_comparison_table(analysis_context)
			if not comparison_table.empty:
				st.table(comparison_table)
			else:
				st.info("No benchmark data available for the selected view.")

			st.divider()
			st.markdown("#### Actionable Insights and Recommendations")
			for action in generate_actionable_insights(analysis_context):
				st.markdown(f"- {action}")

			st.divider()
			st.markdown("#### Conclusion")
			st.markdown(generate_conclusion(analysis_context))

		with ai_tab:
			st.markdown("### Analyst Intelligence Workspace")
			st.caption(
				"Ask questions of the filtered ESG coverage or paste external passages for summarisation."
			)

			st.markdown("#### Ask this dataset")
			dataset_question = st.text_area(
				"Question to ESG<sup>AI</sup>",
				height=135,
				placeholder="e.g. How has sentiment shifted? Which outlets are most critical?",
				key="dataset_question",
				help="Questions draw on the articles and ESG benchmarks currently in view.",
			)
			ask_button = st.button(
				"Answer from current coverage",
				key="qa_button",
				use_container_width=True,
			)

			if ask_button:
				qa_result = answer_company_question(dataset_question, analysis_context)
				status = qa_result.get("status")
				if status == "ok":
					st.success(qa_result.get("answer", ""))
					if qa_result.get("insights"):
						st.markdown("**Key datapoints**")
						for bullet in qa_result["insights"]:
							st.markdown(f"- {bullet}")
					if qa_result.get("evidence"):
						st.markdown("**Supporting coverage**")
						for line in qa_result["evidence"]:
							if line.startswith("-"):
								st.markdown(line)
							else:
								st.markdown(f"*{line}*")
				elif status == "info":
					st.info(qa_result.get("message", ""))
				else:
					st.warning(qa_result.get("message", ""))

			st.markdown("#### Investment advisor")
			pref_col1, pref_col2 = st.columns((2, 1))
			with pref_col1:
				investment_prompt = st.text_area(
					"Ask for investment guidance",
					height=120,
					placeholder="e.g. Should I consider investing now?",
					key="investment_prompt",
				)
			with pref_col2:
				investor_profile = st.text_input(
					"Risk appetite or preferences (optional)",
					placeholder="e.g. low-risk income focus",
					key="investment_preferences",
				)
				get_advice = st.button(
					"Generate advice",
					key="advice_button",
					use_container_width=True,
				)

			if get_advice:
				advice, evidence = generate_chatbot_response(
					analysis_context,
					investment_prompt,
					preferences=investor_profile,
				)
				st.markdown(advice, unsafe_allow_html=True)
				if evidence:
					st.markdown("**Key data points**")
					for item in evidence:
						st.markdown(f"- {item}")
				else:
					st.caption("No supporting metrics available for this request.")
	else:
		st.info("Select a company from the drop-down to launch the ESG briefing.")


if __name__ == "__main__":
	args = sys.argv
	if len(args) != 3:
		start_data = "dec30"
		end_data = "jan12"
	else:
		start_data = args[1]
		end_data = args[2]

	if f"{start_data}_to_{end_data}" not in os.listdir("Data"):
		print(f"There isn't data for {dir_name}")
		raise NameError(f"Please pick from {os.listdir('Data')}")
		sys.exit()
		st.stop()
	else:
		main(start_data, end_data)
	alt.themes.enable("default")


# one_month, ten_days
