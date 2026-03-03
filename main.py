from dotenv import load_dotenv

load_dotenv()  # ✅ Load .env file before anything else

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

app.include_router(evaluation.router, prefix="/api/v1", tags=["Evaluation"])


@app.get("/", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint to verify the service is running."""
    return {"status": "ok", "service": "Hybrid Grader AI"}