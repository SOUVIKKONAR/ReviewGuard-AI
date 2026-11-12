# ReviewGuard application

## Architecture assessment

- Reused the verified 1,596-row Chicago hotel review dataset, saved KNN/SVM/Naive Bayes pipelines, 384-dimensional MiniLM embeddings and similarity matrix, hotel-grouped evaluation reports, Mutual KNN group outputs, and linguistic overlap outputs.
- No FastAPI service or frontend existed. The old `app.py` path was an empty directory, and several `src` modules for serving predictions/similarity are empty. A root `README.md` and this detailed `context.md` now document the project.
- The existing group-risk script writes exploratory connected-component reports. It is not a calibrated fraud score. The application labels its transparent heuristic as experimental and excludes ground-truth labels from its calculation.
- The DistilBERT experiment has evaluation metrics, but no serving checkpoint was found. The analysis form therefore serves saved traditional pipelines.

## Start locally

From the project root, install API dependencies in the existing environment and start the API:

```powershell
.\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The UI defaults to `http://localhost:5173` and the API to `http://localhost:8000`. Configure `VITE_API_URL` and `CORS_ORIGINS` when using different addresses.

## Implemented application areas

- Dashboard with dataset-backed label and hotel counts.
- Saved SVM/KNN/Naive Bayes prediction for pasted review text, plus similar corpus records.
- Searchable, filterable, paginated review explorer with detail and similarity views.
- Mutual KNN groups filtered by scope, K, threshold, hotel text, size, and experimental risk level.
- Group detail with members and direct mutual-neighbor edges.
- Pair comparison with saved semantic similarity and deduplicated linguistic edge evidence.
- Separate hotel-independent and five-fold hotel-grouped metric panels.
- Dataset and runtime status pages.

The system uses the Chicago hotel corpus as deception ground truth. Amazon polarity labels are not used as deception labels. Similarity, clusters, shared phrases, and predictions are risk indicators for review, not proof of fraud.
