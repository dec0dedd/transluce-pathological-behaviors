import streamlit as st
import pandas as pd
import json
import glob
import math
import re

st.set_page_config(page_title="Red-Teaming Log Viewer", layout="wide")
st.title("🛡️ PRBO Reward Logs Viewer")

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

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# --- Sidebar Controls ---
st.sidebar.header("Controls & Filters")

if st.sidebar.button("🔄 Refresh Data"):
    load_data.clear()
    st.session_state.current_page = 1
    st.rerun()

df = load_data()

if df.empty:
    st.warning("No logs found in `reward_logs/`. Are the files being generated?")
    st.stop()

# --- Filters ---
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

filtered_df = df[df["score"] >= min_score]
if selected_behavior != "All":
    filtered_df = filtered_df[filtered_df["behavior_id"] == selected_behavior]

if "step" in filtered_df.columns and pd.notnull(filtered_df["step"]).any():
    min_step = int(filtered_df["step"].min())
    max_step = int(filtered_df["step"].max())
    
    if min_step < max_step:
        step_range = st.sidebar.slider(
            "Filter by Step Range", 
            min_value=min_step, 
            max_value=max_step, 
            value=(min_step, max_step)
        )
        filtered_df = filtered_df[(filtered_df["step"] >= step_range[0]) & (filtered_df["step"] <= step_range[1])]
    else:
        st.sidebar.write(f"**Current Step:** {min_step}")

total_items = len(filtered_df)
st.sidebar.write(f"**Total matching logs:** {total_items}")

if total_items == 0:
    st.info("No logs match your current filters.")
    st.stop()

# --- Pagination Logic ---
ITEMS_PER_PAGE = 20
total_pages = math.ceil(total_items / ITEMS_PER_PAGE)

if st.session_state.current_page > total_pages:
    st.session_state.current_page = 1

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

start_idx = (st.session_state.current_page - 1) * ITEMS_PER_PAGE
end_idx = start_idx + ITEMS_PER_PAGE

page_df = filtered_df.iloc[start_idx:end_idx]

st.write(f"Showing items **{start_idx + 1} to {min(end_idx, total_items)}** of **{total_items}**")

# --- Helpers ---
def render_subscores(subscores_dict):
    if isinstance(subscores_dict, dict) and subscores_dict:
        cols = st.columns(len(subscores_dict))
        for i, (key, val) in enumerate(subscores_dict.items()):
            clean_key = str(key).replace('_', ' ').title()
            cols[i].metric(label=clean_key, value=f"{val}")
        st.divider()

def parse_policy_output(output_str):
    scratchpad, test_prompt = None, None
    if not isinstance(output_str, str):
        return scratchpad, test_prompt
        
    sp_match = re.search(r"<scratchpad>\s*(.*?)\s*</scratchpad>", output_str, re.DOTALL | re.IGNORECASE)
    if sp_match:
        scratchpad = sp_match.group(1).strip()
        
    tp_match = re.search(r"<test_prompt>\s*(.*?)\s*</test_prompt>", output_str, re.DOTALL | re.IGNORECASE)
    if tp_match:
        test_prompt = tp_match.group(1).strip()
        
    return scratchpad, test_prompt

# --- Main UI Loop ---
for idx, row in page_df.iterrows():
    step_num = row.get("step", "N/A")
    log_id = row.get("id", "N/A")
    score_val = row.get('score', 0)
    
    expander_title = f"ID: {log_id} | Step: {step_num} | Score: {score_val:.2f} | Behavior: {row.get('behavior_id', 'Unknown')}"
    
    with st.expander(expander_title, expanded=False):
        
        # Reduced to 2 columns since the prompt_score was removed
        col1, col2 = st.columns(2)
        col1.metric("Total PRBO Score", f"{score_val:.2f}")
        col2.metric("Normal / Steered Score", f"{row.get('normal_response_score_value', 0):.2f} / {row.get('steered_response_score_value', 0):.2f}")
        
        st.markdown("### 🎯 Behavior Target")
        st.info(row.get("behavior", "N/A"))
        
        # --- Upgraded Investigator Output Section ---
        st.markdown("### 📝 Investigator Output")
        raw_policy_output = row.get("policy_output", "")
        scratchpad, test_prompt = parse_policy_output(raw_policy_output)
        
        if scratchpad or test_prompt:
            if scratchpad:
                with st.expander("🧠 Investigator Scratchpad (Internal Planning)", expanded=False):
                    st.markdown(f"*{scratchpad}*")
            
            if test_prompt:
                st.markdown("**Test Prompt (Sent to Target Model):**")
                st.code(test_prompt, language="text", wrap_lines=True)
            else:
                clean_text = re.sub(r"<scratchpad>.*?</scratchpad>", "", raw_policy_output, flags=re.DOTALL | re.IGNORECASE).strip()
                st.markdown("**Test Prompt (Fallback Parsing):**")
                st.code(clean_text if clean_text else raw_policy_output, language="text", wrap_lines=True)
        else:
            st.markdown("**Raw Output:**")
            st.code(raw_policy_output, language="text", wrap_lines=True)
            
        st.markdown("---")
        
        # --- Target Model Responses ---
        st.markdown("### 🤖 Target Model Responses")
        col_norm, col_steer = st.columns(2)
        with col_norm:
            st.markdown("**Normal Response**")
            st.text_area("normal", row.get("normal_response", ""), height=200, label_visibility="collapsed", key=f"norm_{idx}")
        with col_steer:
            st.markdown("**Steered Response**")
            st.text_area("steered", row.get("steered_response", ""), height=200, label_visibility="collapsed", key=f"steer_{idx}")
        
        st.markdown("### ⚖️ Judge Rationale")
        # Removed the Prompt Judge tab
        tab1, tab2 = st.tabs(["Normal Response Judge", "Steered Response Judge"])
        
        with tab1:
            render_subscores(row.get("normal_response_subscores", {}))
            st.write(row.get("normal_judge_response", "N/A"))
            
        with tab2:
            render_subscores(row.get("steered_response_subscores", {}))
            st.write(row.get("steered_judge_response", "N/A"))