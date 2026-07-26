"""
OpenAI-compatible Embedding API server
Serves /v1/embeddings at port 8400 using the Qwen3 sentence-transformers model.
"""
import os
import time
import logging
import numpy as np
from typing import List, Union
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
PORT = 8400
HOST = "0.0.0.0"

# Global model reference
_model = None


def load_model():
    global _model
    logger.info(f"Loading SentenceTransformer from {MODEL_PATH} ...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    _model = SentenceTransformer(MODEL_PATH, device=device, trust_remote_code=True)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s. Embedding dim: {_model.get_sentence_embedding_dimension()}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Embedding API", lifespan=lifespan)


class EmbeddingRequest(BaseModel):
    model: str = MODEL_PATH
    input: Union[str, List[str]]
    encoding_format: str = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict


def normalize_l2(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    inputs = request.input if isinstance(request.input, list) else [request.input]

    try:
        embeddings = _model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
    except Exception as e:
        logger.error(f"Encoding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    data = []
    total_tokens = 0
    for i, (text, emb) in enumerate(zip(inputs, embeddings)):
        emb_list = emb.tolist()
        data.append(EmbeddingData(object="embedding", index=i, embedding=emb_list))
        total_tokens += len(text.split())

    return EmbeddingResponse(
        object="list",
        data=data,
        model=request.model,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )


if __name__ == "__main__":
    logger.info(f"Starting Embedding API server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
