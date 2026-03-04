from dotenv import load_dotenv

load_dotenv()  # Load .env before any other imports touch os.getenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routers import evaluation

app = FastAPI(
    title="Hybrid Grader AI",
    description="Hybrid Evaluation Framework for Exam Answers using Gemini Vision and Knowledge Graphs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes must be registered BEFORE the static-files catch-all mount so
# that /api/v1/* paths are always resolved by the router first.
app.include_router(evaluation.router, prefix="/api/v1", tags=["Evaluation"])

# Serve the frontend. html=True makes StaticFiles return index.html for
# directory requests (i.e. visiting http://localhost:8000/ loads the UI).
app.mount("/", StaticFiles(directory="static", html=True), name="static")