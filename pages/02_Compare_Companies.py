import streamlit as st
import pandas as pd
import numpy as np
import os
from download_data import Data
from app import filter_company_data, render_metrics, build_company_summary, filter_on_date, get_melted_frame

st.set_page_config(page_title="Similar Companies · ESG AI", layout="wide")

# Detect available data window(s)
available_dir = set(os.listdir("Data")) if os.path.isdir("Data") else set()
has_dec30_jan12 = "dec30_to_jan12" in available_dir

# Sidebar controls
with st.sidebar:
	st.header("Similar Companies")
	# Default to the known window present in the repo
	start_options = ["dec30"] if has_dec30_jan12 else ["dec30"]
	end_options = ["jan12"]
	start_day = st.selectbox("Start data window", start_options, index=0)
	end_day = st.selectbox("End data window", end_options, index=0)
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
			default_compare = suggested_list[0][0]
			st.caption("Suggested similar companies (cosine): " + ", ".join([f"{n} ({s:.2f})" for n, s in suggested_list]))
			# Offer override selector intersected with available companies (case-insensitive)
			company_lower_to_label = {c.lower(): c for c in companies}
			candidates = [company_lower_to_label.get(n.lower(), n) for n, _ in suggested_list if n.lower() in company_lower_to_label]
			if not candidates:
				candidates = [c for c in companies if c != primary]
			comparator = st.selectbox("Compare with", options=candidates, index=0)

if comparator is None:
	# Fallback comparator if embeddings not available or no match
	others = [c for c in companies if c != primary]
	if others:
		comparator = others[0]
	else:
		st.info("Not enough companies to compare.")
		st.stop()

selection = [primary, comparator]

# Tabs
tab_overview, tab_trends, tab_articles, tab_stats = st.tabs(["Overview", "Trends", "Articles", "Stats"]) 

with tab_overview:
	st.subheader("Suggested pair")
	st.write({"Primary": primary, "Comparator": comparator})
	# KPI cards
	left_df = df[df.Organization == primary]
	right_df = df[df.Organization == comparator]
	left_summary = build_company_summary(left_df)
	right_summary = build_company_summary(right_df)
	col_l, col_r = st.columns(2)
	with col_l:
		st.subheader(primary)
		render_metrics(left_summary)
	with col_r:
		st.subheader(comparator)
		render_metrics(right_summary)

with tab_trends:
	st.subheader("Tone over time")
	line_cols = {}
	for company in selection:
		cdf = df[df.Organization == company]
		series = cdf.groupby("DATE")["Tone"].mean().rename(company)
		line_cols[company] = series
	if line_cols:
		comp = pd.concat(line_cols.values(), axis=1)
		comp.index = pd.to_datetime(comp.index)
		comp = comp.sort_index()
		st.line_chart(comp, height=280)
	st.subheader("Article volume")
	vol_cols = {}
	for company in selection:
		cdf = df[df.Organization == company]
		series = cdf.groupby("DATE").size().rename(company)
		vol_cols[company] = series
	if vol_cols:
		vol = pd.concat(vol_cols.values(), axis=1).fillna(0).astype(int)
		vol.index = pd.to_datetime(vol.index)
		vol = vol.sort_index()
		st.area_chart(vol, height=240)

with tab_articles:
	st.subheader("Top positive/negative articles")
	col1, col2 = st.columns(2)
	with col1:
		st.markdown(f"**{primary}**")
		st.markdown("- Top positive:")
		for _, row in left_df.sort_values("Tone", ascending=False).head(3).iterrows():
			url = row.get("URL", "")
			date = pd.to_datetime(row.get("DATE")).strftime("%d %b %Y")
			source = row.get("SourceCommonName", "")
			st.markdown(f"- [{date} · {source} (Tone {row.get('Tone'):.2f})]({url})")
		st.markdown("- Top negative:")
		for _, row in left_df.sort_values("Tone", ascending=True).head(3).iterrows():
			url = row.get("URL", "")
			date = pd.to_datetime(row.get("DATE")).strftime("%d %b %Y")
			source = row.get("SourceCommonName", "")
			st.markdown(f"- [{date} · {source} (Tone {row.get('Tone'):.2f})]({url})")
	with col2:
		st.markdown(f"**{comparator}**")
		st.markdown("- Top positive:")
		for _, row in right_df.sort_values("Tone", ascending=False).head(3).iterrows():
			url = row.get("URL", "")
			date = pd.to_datetime(row.get("DATE")).strftime("%d %b %Y")
			source = row.get("SourceCommonName", "")
			st.markdown(f"- [{date} · {source} (Tone {row.get('Tone'):.2f})]({url})")
		st.markdown("- Top negative:")
		for _, row in right_df.sort_values("Tone", ascending=True).head(3).iterrows():
			url = row.get("URL", "")
			date = pd.to_datetime(row.get("DATE")).strftime("%d %b %Y")
			source = row.get("SourceCommonName", "")
			st.markdown(f"- [{date} · {source} (Tone {row.get('Tone'):.2f})]({url})")

with tab_stats:
	st.subheader("Per-company quick stats")
	for company in selection:
		cdf = df[df.Organization == company]
		s = build_company_summary(cdf)
		with st.expander(company, expanded=False):
			st.write({
				"Articles": s["article_count"],
				"Avg Tone": s["avg_tone"],
				"Positive Share": s["positive_ratio"],
				"Avg Polarity": s["avg_polarity"],
			}) 