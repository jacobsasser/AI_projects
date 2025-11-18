import streamlit as st # for web UI creation 
from sentence_transformers import SentenceTransformer # this is for embedding queries into dense vectors 
from pinecone import Pinecone # for accessing pinecone vector DB 
import os # for readhing environment variable 
from langchain_huggingface import HuggingFaceEndpoint # for accessing HuggingFace inference endpoint 
from langchain.prompts import PromptTemplate
import firebase_admin  # for access to firebase 
from firebase_admin import credentials, firestore, auth
from dotenv import load_dotenv
import urllib.parse as urlparse # in relation to FireStore login method 

# === CrossEncoder for reranking ===
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# === Load environment variables ===
load_dotenv(".env.local")

# === CONFIG ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# === Firebase Setup ===
if not firebase_admin._apps:
    cred = credentials.Certificate("service-account-key.json")
    firebase_admin.initialize_app(cred)
else:
    firebase_admin.get_app()  # Access existing app silently

# === Check for Firebase ID token in URL ===
query_params = st.query_params  # updated API
id_token = query_params.get("id_token")

# === Redirect fallback to Firebase login page if not authenticated ===
if not id_token:
    st.warning("No ID token found. Redirecting to login...")
    st.markdown('<meta http-equiv="refresh" content="2;url=https://clinical-trial-app-3ecb4.web.app">', unsafe_allow_html=True)
    st.stop()

# === Verify Firebase token ===
try:
    decoded_token = auth.verify_id_token(id_token)
    USER_ID = decoded_token["uid"]
    USER_EMAIL = decoded_token.get("email", "unknown")
    st.success(f"✅ Logged in as: {USER_EMAIL}")
except Exception as e:
    st.error(f"Authentication failed: {str(e)}")
    st.stop()

# === Firebase Firestore Setup ===
db = firestore.client()

# === Pinecone Setup ===
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "clinical-trials-rag"
index = pc.Index(INDEX_NAME)

# === Embedding Model ===
embed_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb") # BioBERT sentence transformer model

# === CrossEncoder Setup for Reranking ===
cross_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
cross_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_crossencoder(query, results):
    inputs = [f"{query} [SEP] {r['metadata'].get('description', '')}" for r in results]
    tokens = cross_tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        scores = cross_model(**tokens).logits.squeeze().tolist()
    if isinstance(scores, float):
        scores = [scores]
    for i, r in enumerate(results):
        r["rerank_score"] = scores[i]
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)

# === LLM Setup ===
llm = HuggingFaceEndpoint(
    endpoint_url="https://bjgtatzbe3uoqewn.us-east-2.aws.endpoints.huggingface.cloud", # Inference Endpoint Built from Hugging Face. Pay per hour. 
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.7,
    max_new_tokens=256
)
prompt_template = PromptTemplate.from_template(
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# === Session Chat History Setup ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Tabs UI ===
tab1, tab2, tab3 = st.tabs(["🔍 Ask a Question", "⭐ Bookmarked Trials", "📈 Reranker Evaluation"])

# === TAB 1: Question tab page with sample questions to guide the user ===
# === Chat sessions are saved ===
with tab1:
    st.title("Clinical Trial Discovery") 

    st.markdown("""
    💡 **Example question formats:**
    - What clinical trials are available for non-small cell lung cancer in California?
    - List phase 3 trials for Type 1 Diabetes recruiting in 2025.
    - What studies on immunotherapy for melanoma are active in Europe?
    - Are there trials targeting heart disease patients over 65?
    """)

    # === Show chat history ===
    for q, a in st.session_state.chat_history:
        st.markdown(f"**User:** {q}")
        st.markdown(f"**Bot:** {a}")


    user_query = st.text_input("🔍 Enter your clinical trial questions below:") # actual query input part 

    if user_query: # triggers query upon user type action 
        with st.spinner("Retrieving relevant trials..."): # display spinner while pinecone DB being searched 
            vec = embed_model.encode(user_query).tolist() # embed query using the BioBERT sentence transformer 
            raw_results = index.query(vector=vec, top_k=5, include_metadata=True).matches # search pinecone vector DB. Look for 5 most similar vectors 

            # === CrossEncoder rerank part ===
            results = rerank_with_crossencoder(user_query, raw_results)

            # === Logging: Show original and reranked titles for comparison ===
            print("\n🔹 Pinecone Top-K (Unranked):")
            for r in raw_results:
                print(f"- {r['metadata'].get('title', '')[:80]}")

            print("\n🔸 Reranked by CrossEncoder:")
            for r in results:
                print(f"- {r['metadata'].get('title', '')[:80]}  — Score: {r.get('rerank_score', 0):.4f}")
            
            contexts = [r["metadata"]["text"] for r in results]  
            nct_ids = [r["metadata"].get("nct_id", "") for r in results]

        # === Construct LLM prompt with chat history + context ===
        joined_context = "\n".join(contexts)
        chat_history_text = "\n".join(f"User: {q}\nBot: {a}" for q, a in st.session_state.chat_history)
        prompt = prompt_template.format(context=joined_context, question=user_query, history=chat_history_text)

        with st.spinner("Generating answer..."):
            answer = llm(prompt)

        # === Save chat turn ===
        st.session_state.chat_history.append((user_query, answer))

        st.subheader("🧠 Answer:")
        st.write(answer)

        st.markdown("---")
        st.subheader("📋 Related Clinical Trials")

        for i, match in enumerate(results): # loop through pinecone search results and display them 
            meta = match["metadata"]
            nct_id = meta.get("nct_id", f"chunk_{i}") # assigns fallback chuck ID if 'nct_id' is missing 
            chunk_text = meta.get("text", "")[:400] # shows the first 400 characters of the trial chunk 

            with st.expander(f"Trial: {nct_id}"):  # create an expandable block for each trial 
                st.write(chunk_text + "...")  # preview of trial text
                st.caption(f"🔢 Rerank score: {match.get('rerank_score', 0):.4f}")
            
                # === Fetch full clinical trial details from Firestore ===
                trial_doc = db.collection("ClinicalTrials").document(nct_id).get()
                if trial_doc.exists:
                    trial_data = trial_doc.to_dict()
                    for k, v in trial_data.items():
                        st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
                else:
                    st.warning("⚠️ Full trial details not found in Firestore. Showing partial text preview only.")
            
                # === Fetch existing rating (if any) from Firestore ===
                rating_doc = db.collection("Users").document(USER_ID).collection("Ratings").document(nct_id).get()
                if rating_doc.exists:
                    prev_rating = rating_doc.to_dict().get("rating")  # if document exist, convert to dictionary 
                    st.caption(f"📝 You previously rated this trial: {prev_rating}")  # show user their previous input in caption 
            
                # === Create thumbs up/down buttons ===
                col1, col2, col3 = st.columns([1, 1, 4])  # layout: 👍 👎 (spacer)
            
                with col1:
                    if st.button("👍", key=f"thumb_up_{i}"):  # thumbs-up button
                        db.collection("Users").document(USER_ID).collection("Ratings").document(nct_id).set({
                            "rating": "up",
                            "timestamp": firestore.SERVER_TIMESTAMP
                        })
                        st.success(f"You rated trial {nct_id} 👍")
            
                with col2:
                    if st.button("👎", key=f"thumb_down_{i}"):  # thumbs-down button
                        db.collection("Users").document(USER_ID).collection("Ratings").document(nct_id).set({
                            "rating": "down",
                            "timestamp": firestore.SERVER_TIMESTAMP
                        })
                        st.warning(f"You rated trial {nct_id} 👎")
            
                # === Bookmark button ===
                if st.button(f"⭐ Bookmark {nct_id}", key=f"bookmark_{i}"): 
                    db.collection("Users").document(USER_ID).collection("Bookmarks").document(nct_id).set({
                        "nct_id": nct_id,
                        "text": chunk_text
                    })
                    st.success(f"Bookmarked {nct_id} to Firestore.")


# === TAB 2: Bookmarked Trials ===
with tab2:
    st.title("⭐ Your Bookmarked Trials")
    # retrieve bookmarks from firestore 
    docs = db.collection("Users").document(USER_ID).collection("Bookmarks").stream()
    bookmarks = [doc.to_dict() for doc in docs]

    # if no bookmarks, show message. 
    if not bookmarks:
        st.info("You haven't bookmarked any trials yet.")
    else: # otherwise display bookmarked trials in expanders 
        for b in bookmarks:
            with st.expander(f"{b['nct_id']}"):
                st.write(b["text"])

# === TAB 3: Evaluation Using Real User Feedback ===
with tab3:
    st.title("📈 Reranker Evaluation (NDCG@5)")
    if user_query and results:
        relevance_labels = [] # Initializes a list to store actual user feedback relevance labels per trial.
        rerank_scores = [] # This part holds the CrossEncoder scores assigned to each trial.
        rows = []  # for display

        for r in results[:5]: # Loops over the top 5 trials returned by the model for the current query. 
            nct_id = r["metadata"].get("nct_id", "")
            doc = db.collection("Users").document(USER_ID).collection("Ratings").document(nct_id).get() # Looks in Firestore for this specific user’s rating on this trial.
            label = 0 # default lable if no rating exist
            if doc.exists: # This part maps thumbs-up to relevance score 3, thumbs-down to 0 which mimics graded relevance labels used in search evaluation
                rating = doc.to_dict().get("rating")
                if rating == "up":
                    label = 3
                elif rating == "down":
                    label = 0
            relevance_labels.append(label) # Adds the calculated user relevance label to the evaluation list.
            rerank_scores.append(r.get("rerank_score", 0)) # Stores the CrossEncoder reranker score for this trial.
            rows.append({"NCT ID": nct_id, "Rating": label, "Rerank Score": r.get("rerank_score", 0)})

        st.markdown("### 🔎 Trial Ratings & Scores")
        st.dataframe(rows)

        if any(label > 0 for label in relevance_labels): # Proceeds with evaluation only if at least one trial has been rated “up” (relevant).
            from sklearn.metrics import ndcg_score # Uses sklearn to compute Normalized Discounted Cumulative Gain
            import numpy as np
            ndcg = ndcg_score([relevance_labels], [rerank_scores])
            st.metric("NDCG@5 (based on your ratings)", f"{ndcg:.4f}")

            # === Plot NDCG History Over Time ===
            if "ndcg_history" not in st.session_state:
                st.session_state.ndcg_history = []
            st.session_state.ndcg_history.append(ndcg)

            st.line_chart(st.session_state.ndcg_history, use_container_width=True)
        else:
            st.warning("⚠️ No up/down ratings found for current results. Please rate trials in Tab 1 to enable evaluation.")
    else:
        st.info("Run a query in Tab 1 first to evaluate reranker.") # Prevents evaluation if the user hasn’t searched for anything yet.
