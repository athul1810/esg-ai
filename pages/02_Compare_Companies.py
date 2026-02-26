import streamlit as st
import pandas as pd
import numpy as np
import os
from download_data import Data
from app import filter_company_data, render_metrics, build_company_summary, filter_on_date, get_melted_frame

st.set_page_config(page_title="Similar Companies · ESG AI", layout="wide")

# Detect available data window(s)
available_dir = set(os.listdir("Data")) if os.path.isdir("Data") else set()
available_datasets = [d for d in available_dir if "_to_" in d]

# Sidebar controls
with st.sidebar:
	st.header("Similar Companies")
	
	# Prefer synthetic_2021_to_now if available, otherwise use dec30_to_jan12
	if available_datasets:
		default_dataset = "synthetic_2021_to_now" if "synthetic_2021_to_now" in available_datasets else "dec30_to_jan12"
		dataset_choice = st.selectbox(
			"Data window",
			options=sorted(available_datasets),
			index=sorted(available_datasets).index(default_dataset) if default_dataset in available_datasets else 0,
			format_func=lambda x: x.replace("synthetic_", "").replace("_to_", " to ").replace("_", " ").title()
		)
		start_day, end_day = dataset_choice.split("_to_")
	else:
		start_day, end_day = "dec30", "jan12"
	
	# Load data with graceful fallback
	with st.spinner("Loading data…"):
		try:
			data = Data().read(start_day, end_day)
		except NameError:
			st.warning("Selected data window not available. Falling back to dec30_to_jan12.")
			data = Data().read("dec30", "jan12")
	companies = sorted(data["data"].Organization.unique())
	if len(companies) == 0:
		st.stop()
	primary = st.selectbox("Select a company", companies, index=0)

# Prepare base frame
company_cols = ["Organization", "DATE", "Tone", "Polarity", "PositiveTone", "NegativeTone", "SourceCommonName", "ActivityDensity", "WordCount", "URL"]
df = data["data"][company_cols].copy()

# Filter on selected range first
date_min = data["data"].DATE.min()
date_max = data["data"].DATE.max()
start_date, end_date = st.date_input("Analysis range", (date_min, date_max))
df = filter_on_date(df, start_date, end_date, date_col="DATE")

# Compute nearest neighbor from embeddings
emb = data.get("embed")
comparator = None
suggested_list = []
if emb is not None and not emb.empty:
	# Ensure expected columns
	vec_cols = [c for c in emb.columns if c not in ("company",)]
	emb_df = emb.copy()
	# Normalize names to match Organization
	emb_df["company_norm"] = emb_df["company"].str.strip().str.lower()
	primary_norm = primary.strip().lower()
	# Normalize vectors
	vecs = emb_df[vec_cols].to_numpy(dtype=float)
	norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
	vecs_norm = vecs / norms
	# Map company -> normalized vector
	name_to_idx = {n: i for i, n in enumerate(emb_df["company_norm"]) }
	if primary_norm in name_to_idx:
		pi = name_to_idx[primary_norm]
		pvec = vecs_norm[pi]
		sims = vecs_norm @ pvec
		order = np.argsort(-sims)
		# Build suggestions excluding self
		for idx in order:
			if idx == pi:
				continue
			name = emb_df.iloc[idx]["company"]
			score = float(sims[idx])
			suggested_list.append((name, score))
			if len(suggested_list) >= 5:
				break
		if suggested_list:
			# Display suggestions in a more friendly way
			suggestion_text = ", ".join([f"**{n.title()}** ({s:.2f})" for n, s in suggested_list])
			st.caption(f"💡 Similar companies: {suggestion_text}")
			# Get all candidates including suggestions
			company_lower_to_label = {c.lower(): c for c in companies}
			candidates = [company_lower_to_label.get(n.lower(), n) for n, _ in suggested_list if n.lower() in company_lower_to_label]
			if not candidates:
				candidates = [c for c in companies if c != primary]
		else:
			candidates = [c for c in companies if c != primary]
	
	# Multi-select for comparison companies
	comparison_companies = st.multiselect(
		"Select companies to compare",
		options=candidates if 'candidates' in locals() else [c for c in companies if c != primary],
		default=candidates[:min(3, len(candidates))] if 'candidates' in locals() and candidates else [c for c in companies if c != primary][:min(3, len(companies)-1)],
		help="Select up to 5 companies to compare with the primary company"
	)

# Ensure we have at least one company to compare with
if not comparison_companies:
	st.info("Please select at least one company to compare.")
	st.stop()

# Combine primary with comparison companies
selection = [primary] + comparison_companies

# Add a nice header
st.markdown("---")
comparison_text = ", ".join([f"**{c.title()}**" for c in selection])
st.markdown(f"### 📊 Comparing {len(selection)} Companies")
st.caption(f"Analysis period: {start_date.strftime('%B %d, %Y')} - {end_date.strftime('%B %d, %Y')}")

# Tabs
tab_overview, tab_trends, tab_articles, tab_stats = st.tabs(["📈 Overview", "📉 Trends", "📰 Articles", "📊 Stats"]) 

with tab_overview:
	# Build comparison table
	st.markdown("#### 📋 Comparison Table")
	
	# Collect summaries for all companies
	summaries = {}
	for company in selection:
		company_df = df[df.Organization == company]
		summaries[company] = build_company_summary(company_df)
	
	# Create comparison DataFrame
	comparison_data = {
		"Company": [c.title() for c in selection],
		"Articles": [summaries[c]["article_count"] for c in selection],
		"Avg Tone": [round(summaries[c]["avg_tone"], 2) if summaries[c]["avg_tone"] else None for c in selection],
		"Positive Share": [f"{summaries[c]['positive_ratio']*100:.1f}%" if summaries[c]['positive_ratio'] else "N/A" for c in selection],
		"Avg Polarity": [round(summaries[c]["avg_polarity"], 2) if summaries[c]["avg_polarity"] else None for c in selection],
	}
	comparison_df = pd.DataFrame(comparison_data)
	
	# Display table with styling
	st.dataframe(
		comparison_df,
		use_container_width=True,
		hide_index=True,
		column_config={
			"Company": st.column_config.TextColumn("Company", width="large"),
			"Articles": st.column_config.NumberColumn("Articles", format="%,d"),
			"Avg Tone": st.column_config.NumberColumn("Avg Tone", format="%.2f"),
			"Positive Share": st.column_config.TextColumn("Positive Share"),
			"Avg Polarity": st.column_config.NumberColumn("Avg Polarity", format="%.2f"),
		}
	)

with tab_trends:
	st.markdown("#### 📈 Tone Over Time")
	st.caption("Comparison of average tone sentiment for all companies")
	line_cols = {}
	for company in selection:
		cdf = df[df.Organization == company]
		series = cdf.groupby("DATE")["Tone"].mean().rename(company.title())
		line_cols[company.title()] = series
	if line_cols:
		comp = pd.concat(line_cols.values(), axis=1)
		comp.index = pd.to_datetime(comp.index)
		comp = comp.sort_index()
		st.line_chart(comp, height=300)
	
	st.markdown("#### 📊 Article Volume Over Time")
	st.caption("Comparison of article count for all companies")
	vol_cols = {}
	for company in selection:
		cdf = df[df.Organization == company]
		series = cdf.groupby("DATE").size().rename(company.title())
		vol_cols[company.title()] = series
	if vol_cols:
		vol = pd.concat(vol_cols.values(), axis=1).fillna(0).astype(int)
		vol.index = pd.to_datetime(vol.index)
		vol = vol.sort_index()
		st.area_chart(vol, height=300)

with tab_articles:
	st.markdown("#### 📰 Top Positive/Negative Articles")
	
	# Create columns for all companies (up to 4 max for readability)
	num_cols = min(len(selection), 4)
	article_cols = st.columns(num_cols)
	
	for idx, company in enumerate(selection[:num_cols]):
		with article_cols[idx]:
			st.markdown(f"**{company.title()}**")
			company_df = df[df.Organization == company]
			
			st.markdown("- Top positive:")
			for _, row in company_df.sort_values("Tone", ascending=False).head(3).iterrows():
				url = row.get("URL", "")
				date = pd.to_datetime(row.get("DATE")).strftime("%d %b %Y")
				source = row.get("SourceCommonName", "")
				st.markdown(f"- [{date} · {source}<br/>(Tone {row.get('Tone'):.2f})]({url})", unsafe_allow_html=True)
			
			st.markdown("- Top negative:")
			for _, row in company_df.sort_values("Tone", ascending=True).head(3).iterrows():
				url = row.get("URL", "")
				date = pd.to_datetime(row.get("DATE")).strftime("%d %b %Y")
				source = row.get("SourceCommonName", "")
				st.markdown(f"- [{date} · {source}<br/>(Tone {row.get('Tone'):.2f})]({url})", unsafe_allow_html=True)

with tab_stats:
	st.subheader("Per-company quick stats")
	for company in selection:
		cdf = df[df.Organization == company]
		s = build_company_summary(cdf)
		with st.expander(f"📊 {company.title()}", expanded=True):
			# Create a cleaner layout with metrics
			stat_col1, stat_col2 = st.columns(2)
			with stat_col1:
				st.metric("Articles Analyzed", f"{s['article_count']:,d}")
				st.metric("Average Tone", f"{s['avg_tone']:.2f}")
			with stat_col2:
				pos_share = f"{s['positive_ratio']*100:.1f}%" if s['positive_ratio'] else "N/A"
				st.metric("Positive Share", pos_share)
				st.metric("Average Polarity", f"{s['avg_polarity']:.2f}" if s['avg_polarity'] else "N/A") 