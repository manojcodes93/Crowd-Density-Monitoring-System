from fastapi import FastAPI
import threading
from contextlib import asynccontextmanager

from app.engine import start_engine, current_stats


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    thread = threading.Thread(target=start_engine)
    thread.daemon = True
    thread.start()
    yield
    # Shutdown logic (optional)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "Crowd Monitoring API Running"}


@app.get("/stats")
def get_stats():
    return current_stats