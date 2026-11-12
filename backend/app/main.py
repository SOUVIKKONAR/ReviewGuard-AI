from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/verified_dataset.csv"
REPORTS = ROOT / "reports"

app = FastAPI(title="ReviewGuard API", version="1.0.0", description="Hotel review deception research and similarity analysis")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","), allow_methods=["*"], allow_headers=["*"])

class AnalyzeInput(BaseModel):
    review: str = Field(min_length=20, max_length=10000)
    model: str = "SVM"

class CompareInput(BaseModel):
    review_id_1: int = Field(ge=0)
    review_id_2: int = Field(ge=0)

class Store:
    def __init__(self) -> None:
        self.ready = False
        self.error: str | None = None

    def load(self) -> None:
        try:
            self.reviews = pd.read_csv(DATA).drop_duplicates(subset=["review"]).reset_index(drop=True)
            self.reviews["review_id"] = np.arange(len(self.reviews))
            self.models = {}
            for name in ("svm", "knn", "naive_bayes"):
                path = ROOT / "models" / f"{name}.pkl"
                if path.exists(): self.models[name.upper() if name != "naive_bayes" else "Naive Bayes"] = joblib.load(path)
            self.similarity = np.load(REPORTS / "semantic_similarity/similarity_matrix.npy", mmap_mode="r")
            self.embeddings = np.load(REPORTS / "semantic_similarity/review_embeddings.npy", mmap_mode="r")
            self.groups = pd.read_csv(REPORTS / "mutual_knn_validation/group_details.csv")
            self.members = pd.read_csv(REPORTS / "mutual_knn_validation/group_members.csv")
            self.pairs = pd.read_csv(REPORTS / "mutual_knn_validation/group_pairs.csv", usecols=["group_id", "scope", "k", "threshold", "review_id_1", "review_id_2", "hotel_1", "hotel_2", "similarity", "is_direct_mutual_knn_edge"])
            self.linguistic = pd.read_csv(REPORTS / "linguistic_overlap/edge_linguistic_features.csv", usecols=["review_id_1", "review_id_2", "threshold", "hotel_relationship", "word_jaccard", "bigram_jaccard", "trigram_jaccard", "longest_common_phrase", "top_shared_phrases"])
            self.clusters = pd.read_csv(REPORTS / "semantic_clustering/clustered_reviews.csv", usecols=["cluster"])["cluster"]
            self.metrics = pd.read_csv(REPORTS / "hotel_independent/hotel_independent_comparison.csv")
            self.bert_metrics = pd.read_json(REPORTS / "distilbert/overall_metrics.json")
            self.ready = True
        except Exception as exc:
            self.error = str(exc)
            raise

store = Store()

@app.on_event("startup")
def startup() -> None:
    try: store.load()
    except Exception: pass

def require_store() -> Store:
    if not store.ready: raise HTTPException(503, f"ReviewGuard data is unavailable: {store.error or 'startup loading failed'}")
    return store

def label(value: Any) -> str:
    return "Deceptive" if int(value) == 1 else "Truthful"

def review_obj(s: Store, idx: int, include_truth: bool = True) -> dict[str, Any]:
    row = s.reviews.iloc[idx]
    item = {"review_id": idx, "review": str(row.review), "hotel": str(row.hotel), "sentiment": str(row.get("sentiment", "")), "original_filename": str(row.get("original_filename", "")), "cluster": int(s.clusters.iloc[idx]) if idx < len(s.clusters) else None}
    if include_truth: item["ground_truth"] = label(row.csv_label)
    return item

def predict(s: Store, text: str, model_name: str) -> dict[str, Any]:
    if model_name not in s.models: raise HTTPException(400, "Requested model is unavailable")
    model = s.models[model_name]
    pred = int(model.predict([text])[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba([text])[0]))
    elif hasattr(model, "decision_function"):
        score = float(np.ravel(model.decision_function([text]))[0])
        confidence = float(1 / (1 + np.exp(-min(abs(score), 30))))
    return {"model": model_name, "prediction": label(pred), "label": pred, "confidence": round(confidence, 4) if confidence is not None else None, "confidence_note": "Model score, not a calibrated probability."}

@app.get("/api/health")
def health():
    return {"status": "ok" if store.ready else "degraded", "data_loaded": store.ready, "error": store.error}

@app.get("/api/dashboard/summary")
def dashboard_summary():
    s = require_store(); n = len(s.reviews)
    g = s.groups[(s.groups.scope == "all_hotels") & (s.groups.k == 3) & (s.groups.threshold == .85)]
    return {"total_reviews": n, "truthful": int((s.reviews.csv_label == 0).sum()), "deceptive": int((s.reviews.csv_label == 1).sum()), "hotels": int(s.reviews.hotel.nunique()), "embedding_dimension": int(s.embeddings.shape[1]), "candidate_groups": int(len(g)), "grouped_reviews": int(g.group_size.sum()), "group_config": {"scope":"all_hotels","k":3,"threshold":.85,"provisional":True}}

@app.get("/api/dashboard/charts")
def dashboard_charts():
    s = require_store()
    labels = s.reviews.csv_label.value_counts().sort_index()
    hotels = s.reviews.groupby("hotel").size().sort_values(ascending=False)
    metrics = s.metrics.rename(columns={"model":"model"}).to_dict("records")
    return {"labels":[{"name":"Truthful","value":int(labels.get(0,0))},{"name":"Deceptive","value":int(labels.get(1,0))}], "hotels":[{"name":str(k).title(),"reviews":int(v)} for k,v in hotels.items()], "models":metrics}

@app.post("/api/reviews/analyze")
def analyze_review(payload: AnalyzeInput):
    s = require_store(); name = payload.model if payload.model in s.models else payload.model.upper()
    prediction = predict(s, payload.review.strip(), name)
    matches = similar_to_text(s, payload.review, 5)
    prediction["similar_reviews"] = matches
    prediction["explanation"] = ["Prediction comes from the selected supervised model.", "Similarity is contextual evidence and does not establish deception.", "The confidence value is a model score, not a calibrated fraud probability."]
    return prediction

def similar_to_text(s: Store, text: str, limit: int) -> list[dict[str, Any]]:
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        model = getattr(store, "embedder", None)
        if model is None:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=("cuda" if __import__("torch").cuda.is_available() else "cpu"))
            store.embedder = model
        query = model.encode([text], normalize_embeddings=True)[0]
        scores = np.asarray(s.embeddings) @ query
    except Exception:
        # For unavailable embedding runtime, search the precomputed review graph by lexical overlap.
        tokens = set(text.lower().split())
        scores = np.array([len(tokens & set(str(r).lower().split())) / max(1, len(tokens | set(str(r).lower().split()))) for r in s.reviews.review])
    top = np.argsort(scores)[::-1][:limit]
    return [{**review_obj(s, int(i)), "similarity": round(float(scores[i]), 4)} for i in top]

@app.get("/api/reviews")
def reviews(page:int=Query(1,ge=1), page_size:int=Query(20,ge=1,le=100), search:str="", hotel:str="", label_filter:str="", cluster:int|None=None):
    s=require_store(); d=s.reviews.copy()
    if search: d=d[d.review.str.contains(search,case=False,na=False)]
    if hotel: d=d[d.hotel.astype(str).str.lower()==hotel.lower()]
    if label_filter in ("Truthful","Deceptive"): d=d[d.csv_label==(1 if label_filter=="Deceptive" else 0)]
    if cluster is not None: d=d.iloc[np.flatnonzero(s.clusters.to_numpy()==cluster)]
    total=len(d); start=(page-1)*page_size
    rows=[]
    for i,r in d.iloc[start:start+page_size].iterrows(): rows.append({**review_obj(s,int(i)),"ground_truth":label(r.csv_label)})
    return {"items":rows,"page":page,"page_size":page_size,"total":total,"pages":(total+page_size-1)//page_size}

@app.get("/api/reviews/{review_id}")
def get_review(review_id:int):
    s=require_store()
    if review_id<0 or review_id>=len(s.reviews): raise HTTPException(404,"Review not found")
    item=review_obj(s,review_id); item["prediction"]=predict(s,item["review"],"SVM"); return item

@app.get("/api/reviews/{review_id}/similar")
def similar_reviews(review_id:int,limit:int=Query(10,ge=1,le=50)):
    s=require_store()
    if review_id<0 or review_id>=len(s.reviews): raise HTTPException(404,"Review not found")
    scores=np.asarray(s.similarity[review_id]); ids=np.argsort(scores)[::-1][:limit]
    return {"items":[{**review_obj(s,int(i)),"similarity":round(float(scores[i]),4)} for i in ids]}

@app.post("/api/reviews/compare")
def compare_reviews(payload:CompareInput):
    s=require_store(); a,b=payload.review_id_1,payload.review_id_2
    if max(a,b)>=len(s.reviews): raise HTTPException(404,"Review not found")
    x=review_obj(s,a); y=review_obj(s,b)
    rows=s.linguistic[((s.linguistic.review_id_1==min(a,b))&(s.linguistic.review_id_2==max(a,b)))|((s.linguistic.review_id_1==max(a,b))&(s.linguistic.review_id_2==min(a,b)))]
    evidence=rows.sort_values("threshold").drop_duplicates(["review_id_1","review_id_2"])
    e=evidence.iloc[0].to_dict() if len(evidence) else {}
    return {"review_1":x,"review_2":y,"semantic_similarity":float(s.similarity[a,b]),"word_jaccard":e.get("word_jaccard"),"bigram_jaccard":e.get("bigram_jaccard"),"trigram_jaccard":e.get("trigram_jaccard"),"longest_common_phrase":e.get("longest_common_phrase"),"shared_phrases":e.get("top_shared_phrases"),"hotel_relationship":"same hotel" if x["hotel"]==y["hotel"] else "cross hotel","predictions":[predict(s,x["review"],"SVM"),predict(s,y["review"],"SVM")]}

@app.get("/api/groups")
def groups(page:int=1,page_size:int=25,hotel:str="",scope:str="all_hotels",k:int=3,threshold:float=.85,min_size:int=2,max_size:int=1000,risk_level:str=""):
    s=require_store(); d=s.groups[(s.groups.scope==scope)&(s.groups.k==k)&(s.groups.threshold==threshold)&(s.groups.group_size>=min_size)&(s.groups.group_size<=max_size)].copy()
    if hotel: d=d[d.hotels.astype(str).str.contains(hotel,case=False)]
    d["risk_score"]=[risk_indicator(row) for _,row in d.iterrows()]
    d["risk_level"]=d.risk_score.map(risk_label)
    if risk_level: d=d[d.risk_level==risk_level]
    total=len(d); items=d.iloc[(page-1)*page_size:page*page_size].replace({np.nan:None}).to_dict("records")
    return {"items":items,"total":total,"page":page,"page_size":page_size}

def risk_indicator(row):
    # Experimental heuristic; uses only observable graph/text/model indicators.
    density=float(row.density); cross=float(row.hotel_count>1); repeated=float(getattr(row,"svm_deceptive_rate",0) or 0)
    score=round(45*density+25*cross+30*repeated)
    return min(100,max(0,score))
def risk_label(score): return "Low" if score<40 else "Medium" if score<70 else "High"

@app.get("/api/groups/{group_id}")
def group_detail(group_id:str):
    s=require_store(); rows=s.groups[s.groups.group_id==group_id]
    if rows.empty: raise HTTPException(404,"Group not found")
    r=rows.iloc[0]; members=s.members[s.members.group_id==group_id]; pairs=s.pairs[s.pairs.group_id==group_id]
    score=risk_indicator(r)
    return {"group":{**r.to_dict(),"risk_score":score,"risk_level":risk_label(score),"risk_status":"Experimental indicator, not a fraud probability."},"members":members.drop(columns=["actual_label"],errors="ignore").to_dict("records"),"edges":pairs[pairs.is_direct_mutual_knn_edge==1].to_dict("records"),"explanation":["Group membership follows the saved Mutual KNN experiment.","Ground-truth labels are excluded from this risk indicator.","Similarity and shared phrases can occur in truthful reviews."]}

@app.get("/api/models/metrics")
def model_metrics():
    s=require_store(); traditional=s.metrics.to_dict("records"); bert=s.bert_metrics.to_dict("records")
    return {"hotel_independent":traditional,"five_fold_grouped":bert,"methodology":"Leave-One-Group-Out by hotel for traditional models; five-fold hotel-grouped evaluation for DistilBERT and matched SVM. Results are not mixed across methods."}

@app.get("/api/datasets/summary")
def dataset_summary():
    s=require_store(); counts=s.reviews.csv_label.value_counts(); return {"name":"Deceptive Opinion Spam Dataset (Chicago hotel reviews)","rows":len(s.reviews),"truthful":int(counts.get(0,0)),"deceptive":int(counts.get(1,0)),"hotels":sorted(s.reviews.hotel.unique().tolist()),"fields":list(s.reviews.columns),"warning":"Amazon Review Polarity labels represent sentiment and are not used as deception ground truth."}

@app.get("/api/system/status")
def system_status():
    s=require_store();
    try:
        import torch
        gpu=bool(torch.cuda.is_available())
    except Exception: gpu=False
    return {"backend":"FastAPI","data_loaded":s.ready,"model_availability":{k:True for k in s.models},"embedding_model":"sentence-transformers/all-MiniLM-L6-v2","embedding_runtime":"Lazy-loaded on first free-text analysis","gpu_available":gpu,"dataset_rows":len(s.reviews),"grouping":"Mutual K-Nearest Neighbours","risk_engine":"Experimental indicators"}
