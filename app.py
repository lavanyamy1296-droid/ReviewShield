import streamlit as st
import pandas as pd
from src.predict import ReviewDetector
from utils.visualization import show_distribution

st.set_page_config(page_title="ReviewShield AI", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_detector():
    return ReviewDetector()

detector = load_detector()

st.markdown("""
<style>
.main {background: #f7f9fc;}
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {padding: 1.6rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg,#101b3f,#284b8f); color: white; margin-bottom: 1.2rem;}
.hero h1 {color:white; margin:0;}
.hero p {color:#e5edff; margin:.45rem 0 0;}
.result {padding:1rem 1.2rem; border-radius:14px; border:1px solid #d9e2f2; background:white;}
.small-note {color:#68758a; font-size:.88rem;}
</style>
<div class="hero">
  <h1>🛡️ ReviewShield AI</h1>
  <p>Machine-learning assisted screening for suspicious product reviews.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About this project")
    st.write("ReviewShield uses TF-IDF text features and supervised classifiers to estimate whether a review resembles the labelled examples used during training.")
    st.caption("Educational prototype — not proof that a review is fake.")
    page = st.radio("Navigate", ["Single Review", "Bulk CSV", "Model Insights"])
    st.divider()
    st.caption("Models: Logistic Regression and Random Forest")

if "history" not in st.session_state:
    st.session_state.history = []

if page == "Single Review":
    st.subheader("Analyze a product review")
    st.write("Paste the review text below. The model will return a predicted label and an estimated probability.")
    examples = {
        "Choose an example": "",
        "Likely genuine": "The package arrived two days early. The sound is clear, the battery lasts around six hours, and the buttons are easy to use. Good value for the price.",
        "Suspicious promotional style": "WOW AMAZING BEST PRODUCT EVER!!! BUY NOW!!! 100% PERFECT!!! FIVE STARS!!!"
    }
    choice = st.selectbox("Try a sample", list(examples.keys()))
    review = st.text_area("Review text", value=examples[choice], height=150, placeholder="Type or paste a product review...")
    model_name = st.selectbox("Classifier", ["Logistic Regression", "Random Forest"])
    if st.button("Analyze review", type="primary", use_container_width=True):
        if len(review.strip()) < 8:
            st.warning("Please enter a review with at least 8 characters.")
        else:
            result = detector.predict(review, model_name)
            label = result["label"]
            col1, col2 = st.columns([1, 1])
            with col1:
                color = "#16834a" if label == "Likely genuine" else "#bd3b36"
                st.markdown(f'<div class="result"><h3 style="color:{color};margin:0">{label}</h3><p>Estimated model probability: <b>{result["confidence"]:.1%}</b></p></div>', unsafe_allow_html=True)
            with col2:
                st.metric("Review length", f'{result["word_count"]} words')
                st.metric("Selected model", model_name)
            st.info("This is a text-pattern prediction, not a factual verification. A genuine review can look promotional, and a fake review can look natural.")
            st.session_state.history.insert(0, {"Review": review, "Prediction": label, "Confidence": round(result["confidence"], 3), "Model": model_name})

    if st.session_state.history:
        st.subheader("Recent analyses")
        st.dataframe(pd.DataFrame(st.session_state.history).head(10), use_container_width=True, hide_index=True)

elif page == "Bulk CSV":
    st.subheader("Analyze many reviews")
    st.write("Upload a CSV containing a column named **review**. Other columns are preserved in the downloadable results.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    model_name = st.selectbox("Classifier for bulk analysis", ["Logistic Regression", "Random Forest"])
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            if "review" not in data.columns:
                st.error("The CSV must contain a column named 'review'.")
            else:
                st.write(f"Loaded {len(data)} rows.")
                if st.button("Analyze CSV", type="primary"):
                    results = detector.predict_many(data["review"].fillna("").astype(str).tolist(), model_name)
                    output = data.copy()
                    output["prediction"] = [r["label"] for r in results]
                    output["confidence"] = [round(r["confidence"], 4) for r in results]
                    st.dataframe(output.head(100), use_container_width=True)
                    st.download_button("Download analyzed CSV", output.to_csv(index=False).encode("utf-8"), "reviewshield_results.csv", "text/csv")
                    show_distribution(output["prediction"].value_counts().to_dict())
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    else:
        st.download_button("Download sample CSV", pd.DataFrame({"review": [
            "The stitching is neat and the size fits as expected.",
            "BEST PRODUCT EVER!!! BUY IT NOW!!!",
            "Battery backup is decent, but charging takes a while."
        ]}).to_csv(index=False).encode("utf-8"), "sample_reviews.csv", "text/csv")

else:
    st.subheader("Model insights")
    metrics = detector.metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Training examples", metrics["training_rows"])
    c2.metric("Test examples", metrics["test_rows"])
    c3.metric("TF-IDF features", metrics["features"])
    st.markdown("### Held-out test performance")
    st.dataframe(pd.DataFrame(metrics["scores"]).T.rename_axis("Model"), use_container_width=True)
    st.caption("These metrics are calculated from the small bundled teaching dataset. They are not evidence of real-world performance.")
    st.markdown("### Dataset and limitations")
    st.write("""
    - The included labelled examples are a small starter dataset designed to make the project run immediately.
    - Review authenticity cannot reliably be established from wording alone.
    - For serious evaluation, replace the starter CSV with a larger, independently labelled dataset and split it by product or reviewer to reduce leakage.
    - Do not use this prototype as the sole basis for moderation, refunds, or seller decisions.
    """)
