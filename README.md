# Hybrid Evaluation Framework for Handwritten Exams

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-AI-4285F4?logo=google&logoColor=white)

A hybrid grading pipeline that mitigates LLM hallucination in automated exam evaluation by fusing **Knowledge Graph structural matching** with **generative AI reasoning**.

Instead of relying solely on an LLM (which can confidently assign high scores to incorrect answers), this framework first measures factual coverage against a rubric-defined concept graph using spaCy and sentence-transformers, then combines that signal with Gemini's coherence and correctness assessment through a tuneable weighted formula.

---

## Architecture & Workflow

```
Answer Sheet Image
        │
        ▼
┌───────────────────┐
│  Gemini Vision    │  ── Step 1: OCR extraction
│  (OCR)            │
└────────┬──────────┘
         │  extracted text
         ▼
┌───────────────────┐     ┌───────────────────┐
│  spaCy NLP        │────▶│  Sentence-        │  ── Step 2: Knowledge Graph matching
│  (concept extract)│     │  Transformers     │
└───────────────────┘     │  (cosine sim)     │
                          └────────┬──────────┘
                                   │  KG coverage score
         ┌─────────────────────────┤
         │                         ▼
         │              ┌───────────────────┐
         │              │  Gemini LLM       │  ── Step 3: Subjective quality eval
         │              │  (coherence +     │
         │              │   correctness)    │
         │              └────────┬──────────┘
         │                       │  LLM score
         ▼                       ▼
┌─────────────────────────────────────────┐
│  Hybrid Scoring Engine                  │  ── Step 4: Final grade
│  Final = α × KG + (1 − α) × LLM       │
└─────────────────────────────────────────┘
```

**Pipeline steps:**

1. **Image Upload** — The student's handwritten or printed answer sheet is uploaded as an image.
2. **Gemini Vision OCR** — Google Gemini extracts raw text from the image.
3. **Knowledge Graph Matching** — spaCy extracts noun chunks and named entities from the student text; sentence-transformers computes cosine similarity against rubric concept nodes built with NetworkX. Concepts exceeding a 0.75 similarity threshold are matched.
4. **LLM Reasoning** — Gemini evaluates the extracted text for coherence and correctness relative to the rubric, returning scores and a natural-language justification.
5. **Hybrid Scoring** — The final grade is calculated as `α × KG Score + (1 − α) × LLM Score`, where α is configurable (default 0.5).

---

## Project Structure

```
hybrid-grader-ai/
├── main.py                        # FastAPI app entry point
├── api/
│   └── routers/
│       └── evaluation.py          # All API endpoints
├── schemas/
│   └── api_models.py              # Pydantic request/response models
├── services/
│   ├── gemini_service.py          # Gemini OCR + LLM evaluation
│   ├── graph_service.py           # KG building + coverage scoring
│   └── scoring_service.py         # Hybrid score calculation
├── static/
│   ├── index.html                 # Frontend UI (TailwindCSS)
│   └── app.js                     # Frontend logic
├── requirements.txt
├── .env                           # API keys (not committed)
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hybrid-grader-ai.git
cd hybrid-grader-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure environment variables

Create a `.env` file in the project root (if one doesn't already exist):

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

You can obtain a key from [Google AI Studio](https://aistudio.google.com/apikey).

### 6. Start the server

```bash
uvicorn main:app --reload
```

The application will be available at **http://localhost:8000**.

- **Frontend UI** — http://localhost:8000/
- **Interactive API docs** — http://localhost:8000/docs

---

## API Endpoints

All endpoints are prefixed with `/api/v1`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/extract-text` | Upload an answer-sheet image and receive OCR-extracted text via Gemini Vision. |
| `POST` | `/api/v1/evaluate-graph` | Submit student text and a rubric graph (JSON body) to get knowledge-graph coverage scores, matched concepts, and missing concepts. |
| `POST` | `/api/v1/evaluate-full` | **Full pipeline.** Upload an image with a question, rubric JSON, and alpha weight (multipart form). Returns OCR text, KG results, LLM analysis, and the final hybrid score. |

### Example: Full Evaluation (cURL)

```bash
curl -X POST http://localhost:8000/api/v1/evaluate-full \
  -F "file=@answer_sheet.jpg" \
  -F "question=Explain the process of photosynthesis." \
  -F 'rubric_json={
    "nodes": [
      {"id": "photosynthesis", "label": "Photosynthesis", "weight": 1.5},
      {"id": "chlorophyll",    "label": "Chlorophyll",     "weight": 1.0},
      {"id": "sunlight",       "label": "Sunlight",        "weight": 1.0},
      {"id": "glucose",        "label": "Glucose",         "weight": 1.2},
      {"id": "oxygen",         "label": "Oxygen",          "weight": 0.8}
    ],
    "edges": [
      {"source": "sunlight",       "target": "photosynthesis", "relationship": "drives"},
      {"source": "chlorophyll",    "target": "photosynthesis", "relationship": "enables"},
      {"source": "photosynthesis", "target": "glucose",        "relationship": "produces"},
      {"source": "photosynthesis", "target": "oxygen",         "relationship": "produces"}
    ]
  }' \
  -F "alpha=0.5"
```

### Example Response

```json
{
  "extracted_text": "Photosynthesis is the process by which plants convert sunlight...",
  "kg_result": {
    "coverage_score": 0.82,
    "matched_concepts": ["photosynthesis", "sunlight", "glucose", "oxygen"],
    "missing_concepts": ["chlorophyll"]
  },
  "llm_result": {
    "coherence_score": 0.9,
    "correctness_score": 0.85,
    "justification": "The answer is well-structured and covers most key concepts accurately..."
  },
  "final_score": 0.8475
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, FastAPI, Uvicorn, Pydantic |
| AI / LLM / Vision | Google Gemini API (`google-genai`) |
| NLP & Graphs | spaCy, sentence-transformers, NetworkX |
| Frontend | HTML, TailwindCSS (CDN), Vanilla JavaScript |
| Environment | python-dotenv |

---

## License

This project is for academic and research purposes.
