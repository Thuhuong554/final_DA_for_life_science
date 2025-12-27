import streamlit as st
import pandas as pd
import time

from pipeline.screening import screen_end_to_end

from models.bioactivity.loader import (
    load_bioactivity, 
    load_rf_baseline, 
    load_cnn_lstm
)
from models.bioactivity.infer import predict_bioactivity
from models.bioactivity.xai import (
    explain_bioactivity,
    visualize_rf_molecule,
    visualize_cnn_lstm_saliency
)

from models.tox21.hf_loader import load_tox_hf
from models.tox21.hf_infer import predict_tox_hf

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAU_BIO = 0.5
TAU_TOX = 0.5

st.set_page_config(
    page_title="Computational Drug Discovery – Multi-Stage Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI/UX
st.markdown("""
    <style>
    /* Main styling improvements */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    h2 {
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #34495e;
        margin-top: 1.5rem;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .status-info {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    
    /* Metric cards */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        margin: 0.5rem 0;
    }
    
    /* Info boxes */
    .stInfo {
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        background-color: #f0f7ff;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    
    /* Button improvements */
    .stButton > button {
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 0.5rem;
        overflow: hidden;
    }
    
    /* Code block styling */
    .stCodeBlock {
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.75rem 1.5rem;
    }
    
    /* Container improvements */
    .stContainer {
        border-radius: 0.75rem;
        padding: 1.5rem;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    bio_model, bio_tokenizer = load_bioactivity(
        model_dir="checkpoints/bioactivity",
        weights_name="best_reference_chemberta_xai.pth",
        device=device
    )
    tox_model, tox_tokenizer = load_tox_hf(
        artifacts_dir="artifacts/admet_chemberta_tox21",
        device=device
    )
    
    # Load additional models for XAI
    rf_model = load_rf_baseline()
    cnn_lstm_model, cnn_lstm_tokenizer_meta = load_cnn_lstm(device=device)
    
    return (
        bio_model, bio_tokenizer, 
        tox_model, tox_tokenizer,
        rf_model, cnn_lstm_model, cnn_lstm_tokenizer_meta
    )

def outs_to_df(outs):
    rows = []
    for r in outs:
        rows.append({
            "smiles": r.smiles,
            "is_valid": r.is_valid,
            "validation_error": r.validation_error,
            "p_active": r.bio.p_active,
            "active": r.bio.active,
            "p_toxic": r.tox.p_toxic,
            "non_toxic": r.tox.non_toxic,
            "keep": r.keep,
            "reason": r.reason
        })
    return pd.DataFrame(rows)

def main():
    # Header with improved styling
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: #1f77b4; border-bottom: 3px solid #1f77b4; padding-bottom: 0.5rem; margin-bottom: 0.5rem;'>
                🧬 Computational Drug Discovery Pipeline
            </h1>
            <p style='color: #666; font-size: 1.1rem; margin-top: 0.5rem;'>
                Multi-Stage Screening System (Track C)
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Decision rule with better visibility
    st.info("📋 **Decision Rule**: KEEP molecule if **P_active > 0.5** AND **P_toxic < 0.5**")

    # Load resources with progress indicator
    with st.spinner("🔄 Loading AI models... This may take a moment on first run."):
        (
            bio_model, bio_tok, 
            tox_model, tox_tok,
            rf_model, cnn_lstm_model, cnn_lstm_tokenizer_meta
        ) = load_resources()
    
    st.success("✅ All models loaded successfully!")

    # Improved sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        st.markdown("#### 📊 Thresholds")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("τ_bio", "= 0.5", help="Bioactivity threshold")
        with col2:
            st.metric("τ_tox", "= 0.5", help="Toxicity threshold")
        
        st.markdown("---")
        
        st.markdown("#### 💻 System Info")
        device_display = "🟢 GPU (CUDA)" if device.type == "cuda" else "🟡 CPU"
        st.write(f"**Device**: {device_display}")
        
        st.markdown("---")
        
        st.markdown("#### 📖 About")
        st.caption("""
        This pipeline screens molecules through:
        1. **Bioactivity Prediction** (ChemBERTa)
        2. **Toxicity Prediction** (Tox21)
        3. **XAI Explanations** (3 methods)
        """)

    def bio_fn(xs):
        return predict_bioactivity(xs, model=bio_model, tokenizer=bio_tok, tau_bio=TAU_BIO)

    def tox_fn(xs):
        return predict_tox_hf(xs, model=tox_model, tokenizer=tox_tok, tau_tox=TAU_TOX)

    tabs = st.tabs([
        "📁 Batch CSV Upload (Recommended)",
        "⚡ Single SMILES (Quick Test)"
    ])

    # CSV Upload + Screening + XAI under Final Candidates
    with tabs[0]:
        st.markdown("### 📤 Upload Dataset")
        st.markdown("Upload a CSV file containing SMILES strings for batch screening.")
        
        uploaded = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            help="CSV file should contain SMILES strings in one of the columns"
        )

        if uploaded is None:
            st.info("👆 **Please upload a CSV file to begin.** The app will let you select which column contains SMILES strings.")
            return

        # Load and preview CSV
        try:
            df = pd.read_csv(uploaded, delimiter=";")
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("💡 **Tip**: Make sure your CSV file uses semicolon (;) as delimiter.")
            return

        # Preview section
        st.markdown("### 📋 Data Preview")
        st.caption(f"Showing first 10 rows of {len(df)} total rows")
        st.dataframe(df.head(10), use_container_width=True, height=300)

        if df.shape[1] == 0:
            st.error("❌ This CSV appears to have no columns.")
            return

        # Configuration section
        st.markdown("### ⚙️ Configuration")
        col_smiles = st.selectbox(
            "Select the SMILES column",
            options=list(df.columns),
            help="Choose the column that contains SMILES strings"
        )

        col1, col2 = st.columns(2)
        with col1:
            max_rows = st.number_input(
                "Max rows to screen",
                min_value=0,
                value=0,
                step=50,
                help="Set 0 to screen ALL rows. Use a smaller number for quick testing."
            )
        with col2:
            total_available = len(df[col_smiles].dropna())
            rows_to_process = min(max_rows, total_available) if max_rows > 0 else total_available
            st.metric("Rows to process", rows_to_process)

        run_btn = st.button("🚀 Run Screening", type="primary", use_container_width=True, key="batch_screening_btn")

        if run_btn:
            series = df[col_smiles].dropna().astype(str)
            if max_rows and max_rows > 0:
                series = series.iloc[:max_rows]

            smiles_list = series.tolist()
            
            # Progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.info(f"🔄 Processing {len(smiles_list)} SMILES strings...")
            progress_bar.progress(0.3)
            
            # Run screening
            with st.spinner("Running bioactivity and toxicity predictions..."):
                outs = screen_end_to_end(smiles_list, bio_fn=bio_fn, tox_fn=tox_fn)
                progress_bar.progress(0.8)
            
            out_df = outs_to_df(outs)
            progress_bar.progress(1.0)
            status_text.success(f"✅ Screening completed for {len(smiles_list)} molecules!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            # Check for invalid SMILES
            invalid_count = (~out_df["is_valid"]).sum()
            valid_count = out_df["is_valid"].sum()
            
            # Statistics summary
            st.markdown("### 📊 Screening Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Processed", len(out_df))
            with col2:
                st.metric("✅ Valid", valid_count, delta=None)
            with col3:
                st.metric("❌ Invalid", invalid_count, delta=None, delta_color="inverse")
            with col4:
                keep_count = out_df["keep"].sum()
                st.metric("🎯 Candidates", keep_count, delta=None)

            if invalid_count > 0:
                st.warning(f"⚠️ Found {invalid_count} invalid SMILES. They will be marked as invalid and skipped in predictions.")

            # Prepare results dataframe
            screened_df = df.loc[series.index].copy()
            screened_df["is_valid"] = out_df["is_valid"].values
            screened_df["validation_error"] = out_df["validation_error"].values
            screened_df["p_active"] = out_df["p_active"].values
            screened_df["active"] = out_df["active"].values
            screened_df["p_toxic"] = out_df["p_toxic"].values
            screened_df["non_toxic"] = out_df["non_toxic"].values
            screened_df["keep"] = out_df["keep"].values
            screened_df["reason"] = out_df["reason"].values

            # Format dataframe for better display
            display_df = screened_df.copy()
            display_df["p_active"] = display_df["p_active"].round(4)
            display_df["p_toxic"] = display_df["p_toxic"].round(4)

            st.markdown("### 📋 Screening Results")
            st.caption("Complete results with predictions for all molecules")
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Show invalid SMILES details
            if invalid_count > 0:
                with st.expander("🔍 View Invalid SMILES Details", expanded=False):
                    invalid_df = screened_df[~screened_df["is_valid"]].copy()
                    display_cols = [col_smiles, "validation_error"]
                    if all(col in invalid_df.columns for col in display_cols):
                        st.dataframe(invalid_df[display_cols], use_container_width=True)
                    else:
                        st.dataframe(invalid_df[["validation_error"]], use_container_width=True)

            # Final candidates
            final_df = screened_df[screened_df["keep"] == True].copy()
            st.markdown("### 🎯 Final Candidate Molecules")
            st.success(f"✨ Found {len(final_df)} candidate molecules that are **Active** (P_active > 0.5) and **Non-Toxic** (P_toxic < 0.5)")
            
            if len(final_df) > 0:
                final_display_df = final_df.copy()
                final_display_df["p_active"] = final_display_df["p_active"].round(4)
                final_display_df["p_toxic"] = final_display_df["p_toxic"].round(4)
                st.dataframe(final_display_df, use_container_width=True, height=300)
            else:
                st.info("No molecules passed the screening criteria.")

            # Download buttons
            st.markdown("### 💾 Download Results")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download All Results",
                    screened_df.to_csv(index=False).encode("utf-8"),
                    file_name="screening_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "📥 Download Final Candidates",
                    final_df.to_csv(index=False).encode("utf-8"),
                    file_name="final_candidates.csv",
                    mime="text/csv",
                    use_container_width=True,
                    disabled=len(final_df) == 0
                )

            # XAI under Final Candidates (Mandatory for Bioactivity)
            st.markdown("---")
            st.markdown("### 🔬 Bioactivity XAI Explanations")
            st.info(
                "**XAI Analysis**: Explanations are computed for all three Bioactivity models. "
                "All three XAI methods (ChemBERTa, CNN-LSTM, and Random Forest) are displayed for molecules that passed the multi-stage filter (KEEP=TRUE)."
            )

            col1, col2 = st.columns([1, 2])
            with col1:
                top_k = st.slider(
                    "Top-K items to display",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Number of top important features to display per model"
                )

            if final_df.shape[0] == 0:
                st.info("ℹ️ No final candidates found. XAI is not computed.")
            else:
                xai_rows = []
                
                # Progress tracking for XAI
                total_candidates = len(final_df)
                st.write(f"📊 Computing XAI for {total_candidates} candidate molecule(s)...")

                # Show XAI per candidate - all 3 types
                for candidate_idx, (idx, row) in enumerate(final_df.iterrows(), 1):
                    smi = str(row[col_smiles])
                    
                    # Candidate header with better styling
                    st.markdown(f"---")
                    st.markdown(f"#### 🧪 Candidate #{candidate_idx} (Row Index: {idx})")
                    
                    # Display molecule info in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("P_active", f"{row['p_active']:.4f}")
                    with col2:
                        st.metric("P_toxic", f"{row['p_toxic']:.4f}")
                    with col3:
                        status = "✅ KEEP" if row['keep'] else "❌ REJECT"
                        st.metric("Status", status)
                    
                    # SMILES display
                    st.code(smi, language="text")

                    # Create tabs for each XAI method
                    xai_tabs = st.tabs([
                        "🔤 ChemBERTa (Token-level)",
                        "📝 CNN-LSTM (Character-level)",
                        "🌳 Random Forest (Atom-level)"
                    ])

                    # Tab 1: ChemBERTa
                    with xai_tabs[0]:
                        with st.spinner("Computing ChemBERTa token-level explanations..."):
                            try:
                                items, scores, _ = explain_bioactivity(
                                    smi,
                                    model=bio_model,
                                    tokenizer=bio_tok,
                                    device=device,
                                    model_type="chemberta"
                                )
                                xdf = pd.DataFrame({"token": items, "importance": scores})
                                xdf = xdf[~xdf["token"].isin(["<pad>", "<s>", "</s>"])].copy()
                                xdf = xdf.sort_values("importance", ascending=False).head(top_k)
                                
                                st.markdown("**Top Important Tokens:**")
                                st.dataframe(xdf, use_container_width=True, height=300)

                                for _, r2 in xdf.iterrows():
                                    xai_rows.append({
                                        "row_index": idx,
                                        "smiles": smi,
                                        "model_type": "chemberta",
                                        "item": r2["token"],
                                        "item_type": "token",
                                        "importance": float(r2["importance"])
                                    })
                            except Exception as e:
                                st.error(f"❌ Error computing ChemBERTa XAI: {str(e)}")
                                st.exception(e)

                    # Tab 2: CNN-LSTM
                    with xai_tabs[1]:
                        with st.spinner("Computing CNN-LSTM character-level explanations..."):
                            try:
                                items, scores, viz_data = explain_bioactivity(
                                    smi,
                                    model=cnn_lstm_model,
                                    tokenizer_meta=cnn_lstm_tokenizer_meta,
                                    device=device,
                                    model_type="cnn_lstm"
                                )
                                
                                # Visualize saliency chart
                                if viz_data is not None:
                                    st.markdown("**Character-level Saliency Visualization:**")
                                    viz_img = visualize_cnn_lstm_saliency(smi, viz_data)
                                    st.image(viz_img, caption="Character-level Saliency Chart", use_container_width=True)
                                
                                st.markdown("**Top Important Characters:**")
                                xdf = pd.DataFrame({"character": items, "importance": scores})
                                xdf = xdf.sort_values("importance", ascending=False).head(top_k)
                                st.dataframe(xdf, use_container_width=True, height=300)

                                for _, r2 in xdf.iterrows():
                                    xai_rows.append({
                                        "row_index": idx,
                                        "smiles": smi,
                                        "model_type": "cnn_lstm",
                                        "item": r2["character"],
                                        "item_type": "character",
                                        "importance": float(r2["importance"])
                                    })
                            except Exception as e:
                                st.error(f"❌ Error computing CNN-LSTM XAI: {str(e)}")
                                st.exception(e)

                    # Tab 3: Random Forest
                    with xai_tabs[2]:
                        with st.spinner("Computing Random Forest atom-level explanations..."):
                            try:
                                items, scores, viz_data = explain_bioactivity(
                                    smi,
                                    model=None,
                                    rf_model=rf_model,
                                    model_type="rf"
                                )
                                
                                # Visualize molecule with highlighted atoms
                                if viz_data is not None:
                                    st.markdown("**Molecule Visualization with Highlighted Atoms:**")
                                    mol, highlight_atoms = viz_data
                                    mol_img = visualize_rf_molecule(mol, highlight_atoms)
                                    st.image(mol_img, caption="Molecule with Highlighted Important Atoms", use_container_width=True)
                                
                                st.markdown("**Top Important Atom Indices:**")
                                xdf = pd.DataFrame({"atom_index": items, "importance": scores})
                                xdf = xdf.sort_values("importance", ascending=False).head(top_k)
                                st.dataframe(xdf, use_container_width=True, height=300)

                                for _, r2 in xdf.iterrows():
                                    xai_rows.append({
                                        "row_index": idx,
                                        "smiles": smi,
                                        "model_type": "rf",
                                        "item": r2["atom_index"],
                                        "item_type": "atom_index",
                                        "importance": float(r2["importance"])
                                    })
                            except Exception as e:
                                st.error(f"❌ Error computing Random Forest XAI: {str(e)}")
                                st.exception(e)

                # Download XAI results
                if xai_rows:
                    st.markdown("---")
                    st.markdown("### 💾 Download XAI Results")
                    xai_df_all = pd.DataFrame(xai_rows)
                    st.download_button(
                        "📥 Download All XAI Explanations",
                        xai_df_all.to_csv(index=False).encode("utf-8"),
                        file_name="bioactivity_xai_final_candidates_all.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    # Single SMILES quick test
    with tabs[1]:
        st.markdown("### ⚡ Quick Single-Molecule Screening")
        st.caption("Enter a single SMILES string for rapid testing and analysis")
        
        smi = st.text_input(
            "Enter SMILES string",
            placeholder="e.g., CCO (ethanol)",
            help="Enter a valid SMILES string to test"
        )

        if st.button("🚀 Run Screening", type="primary", use_container_width=True, key="single_screening_btn") and smi:
            with st.spinner("🔄 Processing molecule..."):
                out = screen_end_to_end([smi], bio_fn=bio_fn, tox_fn=tox_fn)[0]

            # Check if SMILES is valid
            if not out.is_valid:
                st.error(f"❌ **Invalid SMILES**: {out.validation_error}")
                st.info("💡 Please check your SMILES string and try again.")
            else:
                # Display results with better styling
                st.markdown("### 📊 Screening Results")
                
                # Metrics in columns
                col1, col2, c3 = st.columns(3)
                with col1:
                    active_color = "normal" if out.bio.active else "off"
                    st.metric(
                        "P_active",
                        f"{out.bio.p_active:.4f}",
                        delta="✅ Active" if out.bio.active else "❌ Inactive",
                        delta_color=active_color
                    )
                with col2:
                    non_toxic_color = "normal" if out.tox.non_toxic else "off"
                    st.metric(
                        "P_toxic",
                        f"{out.tox.p_toxic:.4f}",
                        delta="✅ Non-Toxic" if out.tox.non_toxic else "⚠️ Toxic",
                        delta_color=non_toxic_color
                    )
                with c3:
                    keep_color = "normal" if out.keep else "off"
                    keep_status = "✅ KEEP" if out.keep else "❌ REJECT"
                    st.metric("Decision", keep_status, delta=None)
                
                # Reason display
                st.info(f"📝 **Reason**: {out.reason}")

                # Raw output in expander
                with st.expander("🔍 View Raw Output JSON", expanded=False):
                    st.json({
                        "smiles": out.smiles,
                        "is_valid": out.is_valid,
                        "validation_error": out.validation_error,
                        "p_active": out.bio.p_active,
                        "p_toxic": out.tox.p_toxic,
                        "keep": out.keep,
                        "reason": out.reason
                    })

                # XAI for single molecule - all 3 types (only show if SMILES is valid)
                st.markdown("---")
                st.markdown("### 🔬 XAI Explanations")
                if out.keep:
                    st.success("✅ This molecule passed the filter (KEEP=TRUE). XAI explanations from all three models are displayed below.")
                else:
                    st.warning("⚠️ This molecule did not pass the filter (KEEP=FALSE). XAI explanations are still available for analysis.")

                # Create tabs for each XAI method
                xai_tabs_single = st.tabs([
                    "🔤 ChemBERTa (Token-level)",
                    "📝 CNN-LSTM (Character-level)",
                    "🌳 Random Forest (Atom-level)"
                ])

                # Tab 1: ChemBERTa
                with xai_tabs_single[0]:
                    with st.spinner("Computing ChemBERTa token-level explanations..."):
                        try:
                            items, scores, _ = explain_bioactivity(
                                smi,
                                model=bio_model,
                                tokenizer=bio_tok,
                                device=device,
                                model_type="chemberta"
                            )
                            xdf = pd.DataFrame({"token": items, "importance": scores})
                            xdf = xdf[~xdf["token"].isin(["<pad>", "<s>", "</s>"])].copy()
                            xdf = xdf.sort_values("importance", ascending=False).head(20)
                            
                            st.markdown("**Top Important Tokens:**")
                            st.dataframe(xdf, use_container_width=True, height=300)
                        except Exception as e:
                            st.error(f"❌ Error computing ChemBERTa XAI: {str(e)}")
                            st.exception(e)

                # Tab 2: CNN-LSTM
                with xai_tabs_single[1]:
                    with st.spinner("Computing CNN-LSTM character-level explanations..."):
                        try:
                            items, scores, viz_data = explain_bioactivity(
                                smi,
                                model=cnn_lstm_model,
                                tokenizer_meta=cnn_lstm_tokenizer_meta,
                                device=device,
                                model_type="cnn_lstm"
                            )
                            
                            # Visualize saliency chart
                            if viz_data is not None:
                                st.markdown("**Character-level Saliency Visualization:**")
                                viz_img = visualize_cnn_lstm_saliency(smi, viz_data)
                                st.image(viz_img, caption="Character-level Saliency Chart", use_container_width=True)
                            
                            st.markdown("**Top Important Characters:**")
                            xdf = pd.DataFrame({"character": items, "importance": scores})
                            xdf = xdf.sort_values("importance", ascending=False).head(20)
                            st.dataframe(xdf, use_container_width=True, height=300)
                        except Exception as e:
                            st.error(f"❌ Error computing CNN-LSTM XAI: {str(e)}")
                            st.exception(e)

                # Tab 3: Random Forest
                with xai_tabs_single[2]:
                    with st.spinner("Computing Random Forest atom-level explanations..."):
                        try:
                            items, scores, viz_data = explain_bioactivity(
                                smi,
                                model=None,
                                rf_model=rf_model,
                                model_type="rf"
                            )
                            
                            # Visualize molecule with highlighted atoms
                            if viz_data is not None:
                                st.markdown("**Molecule Visualization with Highlighted Atoms:**")
                                mol, highlight_atoms = viz_data
                                mol_img = visualize_rf_molecule(mol, highlight_atoms)
                                st.image(mol_img, caption="Molecule with Highlighted Important Atoms", use_container_width=True)
                            
                            st.markdown("**Top Important Atom Indices:**")
                            xdf = pd.DataFrame({"atom_index": items, "importance": scores})
                            xdf = xdf.sort_values("importance", ascending=False).head(20)
                            st.dataframe(xdf, use_container_width=True, height=300)
                        except Exception as e:
                            st.error(f"❌ Error computing Random Forest XAI: {str(e)}")
                            st.exception(e)

if __name__ == "__main__":
    main()