# ReviewGuard

ReviewGuard is a research application for deceptive **Chicago hotel review** analysis. It combines supervised text classification with semantic similarity, Mutual K-Nearest Neighbour (Mutual KNN) grouping, and phrase-overlap inspection. Its purpose is to surface signals for investigation, not to declare that a reviewer committed fraud.

> **Interpret results carefully:** Similarity, shared phrases, a semantic cluster, or a model prediction is not proof of deception. The primary dataset is a labeled hotel-review research corpus, not a verified product-review fraud dataset. Amazon Review Polarity labels are sentiment labels and are not used as deception ground truth.

## Start the application

Use two PowerShell terminals from the project root.

### 1. Start the API

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload
```

The API is at `http://localhost:8000`. Interactive API documentation is at `http://localhost:8000/docs`.

### 2. Start the frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`. The frontend defaults to the API at `http://localhost:8000/api`.

For a fresh Python environment, install backend requirements first:

```powershell
.\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

See [context.md](context.md) for a detailed explanation of dataset construction, preprocessing, model training and evaluation, similarity analysis, grouping, and the application pipeline.

## What the application does

- **Dashboard:** corpus counts, hotel coverage, and label distribution from the verified hotel-review data.
- **Analyze a review:** apply a saved SVM, KNN, or Naive Bayes pipeline to submitted text and show nearest corpus reviews.
- **Review explorer:** search, filter, paginate, and inspect corpus records, labels, and saved semantic neighbors.
- **Group analysis:** browse saved Mutual KNN groups by scope, K, similarity threshold, hotel, minimum size, and experimental risk level.
- **Group details:** inspect members and direct Mutual KNN graph edges.
- **Review comparison:** inspect saved semantic similarity and available word, bigram, trigram, and phrase overlap evidence.
- **Model performance:** compare the traditional-model hotel-independent results separately from the DistilBERT five-fold hotel-grouped experiment.
- **Dataset and system pages:** inspect corpus facts, model availability, embedding setup, and API health.

The analysis form serves the saved traditional scikit-learn artifacts. DistilBERT has completed evaluation results, but no serving checkpoint is present in this repository.

## Evaluation snapshot

The application’s model comparison uses the hotel-independent traditional-model experiment. The DistilBERT result is shown separately because it came from a different, five-fold hotel-grouped evaluation.

| Evaluation | Model | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Leave-One-Hotel-Out | KNN | 79.45% | 74.03% | 90.88% | 81.59% |
| Leave-One-Hotel-Out | SVM | 89.85% | 91.00% | 88.50% | **89.73%** |
| Leave-One-Hotel-Out | Naive Bayes | 89.22% | 90.78% | 87.38% | 89.04% |
| Five-fold hotel-grouped | DistilBERT | 87.97% | 85.10% | 92.13% | 88.48% |
| Five-fold hotel-grouped | SVM baseline on same folds | 88.66% | 89.63% | 87.50% | 88.55% |

These are research evaluation metrics, not calibrated probabilities and not a guarantee of performance on other domains. The exact split designs and a second, random-split result set are described in [context.md](context.md#traditional-supervised-models).

## Repository map

```text
backend/app/main.py                       FastAPI API and data/model services
frontend/src/                             React + TypeScript application
DATASET/deceptive_opinion_spam/           Original CSV of 1,600 hotel reviews
DATASET/original_corpus/op_spam_v1.4/     Source review text and class/hotel mapping
DATASET/amazon_review_polarity_csv/       Separate sentiment-only corpus
data/processed/verified_dataset.csv       Mapped data; downstream code deduplicates it
models/                                   Saved KNN, SVM, and Naive Bayes pipelines
reports/hotel_independent/                Leave-One-Hotel-Out predictions and metrics
reports/distilbert/                       DistilBERT and matched SVM evaluation outputs
reports/semantic_similarity/              Embeddings, cosine matrix, and neighbor outputs
reports/semantic_clustering/              K-Means clusters and summaries
reports/mutual_knn_validation/             Group members, group links, and validation
reports/linguistic_overlap/                Pair-level phrase and n-gram evidence
```

The saved experiment files are reused; starting the web application does not retrain models or regenerate reports. Consult [context.md](context.md) before changing data order or labels: row IDs in the experiment files align with the deduplicated review order.

The large Amazon polarity CSVs (also duplicated as `DATASET/train.csv` and `DATASET/test.csv`) are excluded from Git because they are not needed by the application and exceed ordinary GitHub file limits. The Amazon dataset description remains included.
