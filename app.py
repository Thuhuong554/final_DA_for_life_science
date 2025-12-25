import streamlit as st
import pandas as pd

from pipeline.screening import screen_end_to_end

from models.bioactivity.loader import load_bioactivity
from models.bioactivity.infer import predict_bioactivity
from models.bioactivity.xai import explain_bioactivity

from models.tox21.hf_loader import load_tox_hf
from models.tox21.hf_infer import predict_tox_hf

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAU_BIO = 0.5
TAU_TOX = 0.5

st.set_page_config(
    page_title="Computational Drug Discovery – Multi-Stage Pipeline",
    layout="wide"
)

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
    return bio_model, bio_tokenizer, tox_model, tox_tokenizer

def outs_to_df(outs):
    rows = []
    for r in outs:
        rows.append({
            "smiles": r.smiles,
            "p_active": r.bio.p_active,
            "active": r.bio.active,
            "p_toxic": r.tox.p_toxic,
            "non_toxic": r.tox.non_toxic,
            "keep": r.keep,
            "reason": r.reason
        })
    return pd.DataFrame(rows)

def main():
    st.title("Computational Drug Discovery – Multi-Stage Pipeline (Track C)")
    st.caption("Decision rule: KEEP if P_active > 0.5 AND P_toxic < 0.5")

    bio_model, bio_tok, tox_model, tox_tok = load_resources()

    st.sidebar.markdown("### Fixed Thresholds")
    st.sidebar.write("τ_bio > 0.5")
    st.sidebar.write("τ_tox = 0.5")

    def bio_fn(xs):
        return predict_bioactivity(xs, model=bio_model, tokenizer=bio_tok, tau_bio=TAU_BIO)

    def tox_fn(xs):
        return predict_tox_hf(xs, model=tox_model, tokenizer=tox_tok, tau_tox=TAU_TOX)

    tabs = st.tabs([
        "Batch CSV Upload (Recommended)",
        "Single SMILES (Quick test)"
    ])

    # CSV Upload + Screening + XAI under Final Candidates
    with tabs[0]:
        st.subheader("Upload your dataset CSV and screen all molecules")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is None:
            st.info("Upload a CSV file to begin. The app will let you select which column contains SMILES.")
            return

        df = pd.read_csv(uploaded, delimiter=";")
        st.write("Preview (first 10 rows):")
        st.dataframe(df.head(10), width='stretch')

        if df.shape[1] == 0:
            st.error("This CSV appears to have no columns.")
            return

        col_smiles = st.selectbox("Select the SMILES column", options=list(df.columns))

        st.markdown("#### Run settings")
        max_rows = st.number_input(
            "Max rows to screen (for quick testing). Set 0 to screen ALL rows.",
            min_value=0,
            value=0,
            step=50
        )

        run_btn = st.button("Run Screening on CSV", type="primary")

        if run_btn:
            series = df[col_smiles].dropna().astype(str)
            if max_rows and max_rows > 0:
                series = series.iloc[:max_rows]

            smiles_list = series.tolist()
            st.write("Total SMILES to screen:", len(smiles_list))

            outs = screen_end_to_end(smiles_list, bio_fn=bio_fn, tox_fn=tox_fn)
            out_df = outs_to_df(outs)

            screened_df = df.loc[series.index].copy()
            screened_df["p_active"] = out_df["p_active"].values
            screened_df["active"] = out_df["active"].values
            screened_df["p_toxic"] = out_df["p_toxic"].values
            screened_df["non_toxic"] = out_df["non_toxic"].values
            screened_df["keep"] = out_df["keep"].values
            screened_df["reason"] = out_df["reason"].values

            st.subheader("Screening Results (CSV + predictions)")
            st.dataframe(screened_df, width='stretch')

            final_df = screened_df[screened_df["keep"] == True].copy()
            st.subheader("Final Candidate Molecules (Active & Non-Toxic)")
            st.dataframe(final_df, width='stretch')

            st.download_button(
                "Download screening_results.csv",
                screened_df.to_csv(index=False).encode("utf-8"),
                file_name="screening_results.csv"
            )
            st.download_button(
                "Download final_candidates.csv",
                final_df.to_csv(index=False).encode("utf-8"),
                file_name="final_candidates.csv"
            )

            # XAI under Final Candidates (Mandatory for Bioactivity)
            st.subheader("Bioactivity XAI for Final Candidates (Mandatory)")
            st.caption(
                "XAI is computed only for the Bioactivity model, as required. "
                "Token-level attribution is shown for molecules that passed the multi-stage filter (KEEP=TRUE)."
            )

            top_k = st.slider(
                "Top-K tokens to display per molecule",
                min_value=5,
                max_value=50,
                value=20,
                step=5
            )

            if final_df.shape[0] == 0:
                st.info("No final candidates found. XAI is not computed.")
            else:
                xai_rows = []

                # Show XAI per candidate
                for idx, row in final_df.iterrows():
                    smi = str(row[col_smiles])

                    tokens, scores = explain_bioactivity(
                        smi,
                        model=bio_model,
                        tokenizer=bio_tok,
                        device=device
                    )

                    xdf = pd.DataFrame({"token": tokens, "importance": scores})
                    xdf = xdf[~xdf["token"].isin(["<pad>"])].copy()
                    xdf = xdf.sort_values("importance", ascending=False).head(top_k)

                    st.markdown(f"### Candidate row index: {idx}")
                    st.code(smi)
                    st.dataframe(xdf, width='stretch')

                    for _, r2 in xdf.iterrows():
                        xai_rows.append({
                            "row_index": idx,
                            "smiles": smi,
                            "token": r2["token"],
                            "importance": float(r2["importance"])
                        })

                xai_df_all = pd.DataFrame(xai_rows)
                st.download_button(
                    "Download bioactivity_xai_final_candidates.csv",
                    xai_df_all.to_csv(index=False).encode("utf-8"),
                    file_name="bioactivity_xai_final_candidates.csv"
                )

    #Single SMILES quick test
    with tabs[1]:
        st.subheader("Quick single-molecule screening")
        smi = st.text_input("Input SMILES")

        if st.button("Run Single Screening") and smi:
            out = screen_end_to_end([smi], bio_fn=bio_fn, tox_fn=tox_fn)[0]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("P_active", f"{out.bio.p_active:.4f}")
                st.write("Active:", out.bio.active)
            with c2:
                st.metric("P_toxic", f"{out.tox.p_toxic:.4f}")
                st.write("Non-Toxic:", out.tox.non_toxic)
            with c3:
                st.metric("KEEP", "YES" if out.keep else "NO")
                st.write("Reason:", out.reason)

            st.subheader("Raw Output")
            st.json({
                "smiles": out.smiles,
                "p_active": out.bio.p_active,
                "p_toxic": out.tox.p_toxic,
                "keep": out.keep,
                "reason": out.reason
            })

if __name__ == "__main__":
    main()