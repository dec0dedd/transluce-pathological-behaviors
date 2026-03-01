import streamlit as st
import pandas as pd
import json
import glob
import math

st.set_page_config(page_title="Red-Teaming Log Viewer", layout="wide")
st.title("🛡️ PRBO Reward Logs Viewer")

# 1. Removed TTL. It now only reloads when you explicitly clear the cache.
@st.cache_data
def load_data():
    all_data = []
    for f_name in glob.glob("reward_logs/*.jsonl"):
        with open(f_name, 'r', encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    if "score" in df.columns:
        df = df.sort_values(by="score", ascending=False)
    return df

# Initialize session state for pagination
if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# --- Sidebar Controls ---
st.sidebar.header("Controls & Filters")

# Manual Refresh Button
if st.sidebar.button("🔄 Refresh Data"):
    load_data.clear()  # Clears the cache
    st.session_state.current_page = 1 # Reset to page 1
    st.rerun()

df = load_data()

if df.empty:
    st.warning("No logs found in `reward_logs/`. Are the files being generated?")
    st.stop()

# Filters
selected_behavior = st.sidebar.selectbox(
    "Filter by Behavior ID:", 
    options=["All"] + df["behavior_id"].unique().tolist()
)

min_score = st.sidebar.slider(
    "Minimum Score", 
    min_value=float(df["score"].min()), 
    max_value=float(df["score"].max()), 
    value=float(df["score"].min())
)

# Apply filters
filtered_df = df[df["score"] >= min_score]
if selected_behavior != "All":
    filtered_df = filtered_df[filtered_df["behavior_id"] == selected_behavior]

total_items = len(filtered_df)
st.sidebar.write(f"**Total matching logs:** {total_items}")

if total_items == 0:
    st.info("No logs match your current filters.")
    st.stop()

# --- Pagination Logic ---
ITEMS_PER_PAGE = 20
total_pages = math.ceil(total_items / ITEMS_PER_PAGE)

# Ensure current page is within bounds after filtering
if st.session_state.current_page > total_pages:
    st.session_state.current_page = 1

# Pagination Controls
st.sidebar.markdown("---")
st.sidebar.subheader("Pagination")
page_selection = st.sidebar.number_input(
    f"Page (1 - {total_pages})", 
    min_value=1, 
    max_value=total_pages, 
    value=st.session_state.current_page,
    step=1
)
st.session_state.current_page = page_selection

# Calculate slice indices
start_idx = (st.session_state.current_page - 1) * ITEMS_PER_PAGE
end_idx = start_idx + ITEMS_PER_PAGE

# Slice the dataframe to only the items for the current page
page_df = filtered_df.iloc[start_idx:end_idx]

st.write(f"Showing items **{start_idx + 1} to {min(end_idx, total_items)}** of **{total_items}**")

# --- Main UI Loop (Only loops through the current page) ---
for idx, row in page_df.iterrows():
    with st.expander(f"Score: {row.get('score', 0):.2f} | Behavior: {row.get('behavior_id', 'Unknown')}", expanded=False):
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Score", f"{row.get('score', 0):.2f}")
        col2.metric("Prompt Score", f"{row.get('prompt_score_value', 0)}")
        col3.metric("Normal/Steered Score", f"{row.get('normal_response_score_value', 0)} / {row.get('steered_response_score_value', 0)}")
        
        st.markdown("### 🎯 Behavior Target")
        st.info(row.get("behavior", "N/A"))
        
        st.markdown("### 📝 Investigator Prompt (Policy Output)")
        st.code(row.get("policy_output", "N/A"), language="text")
        
        st.markdown("### 🤖 Target Model Responses")
        col_norm, col_steer = st.columns(2)
        with col_norm:
            st.markdown("**Normal Response**")
            st.text_area("normal", row.get("normal_response", ""), height=200, label_visibility="collapsed", key=f"norm_{idx}")
        with col_steer:
            st.markdown("**Steered Response**")
            st.text_area("steered", row.get("steered_response", ""), height=200, label_visibility="collapsed", key=f"steer_{idx}")
        
        st.markdown("### ⚖️ Judge Rationale")
        tab1, tab2, tab3 = st.tabs(["Prompt Judge", "Normal Response Judge", "Steered Response Judge"])
        
        with tab1:
            st.write(row.get("prompt_judge_response", "N/A"))
        with tab2:
            st.write(row.get("normal_judge_response", "N/A"))
        with tab3:
            st.write(row.get("steered_judge_response", "N/A"))