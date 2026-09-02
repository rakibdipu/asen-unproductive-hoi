"""
Omni-RecSys Mega Platform - FastAPI Application Entry Point
Production-ready backend integrating 12 algorithms across 6 generations,
5-stage production pipeline, real-time WebSocket event streaming, and Model Arena.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.config import settings
from backend.api.recommendations import router as rec_router, get_domain_models, SHARED_ARTWORK_BANDIT
from backend.api.users import router as users_router
from backend.api.items import router as items_router
from backend.api.events import router as events_router
from backend.api.benchmark import router as benchmark_router
from backend.api.explain import router as explain_router
from backend.api.artwork import router as artwork_router
from backend.streaming.online_updater import OnlineUpdater

online_updater = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global online_updater
    print("[STARTUP] Initializing Omni-RecSys Mega Platform Backend...")
    # Initialize online updater connecting event stream to feature store & bandits
    online_updater = OnlineUpdater(artwork_bandit=SHARED_ARTWORK_BANDIT)
    # Warm up default movies models
    print("[WARMUP] Pre-warming foundational recommendation models for 'movies' domain...")
    try:
        get_domain_models("movies")
        print("[OK] Models warmed up successfully.")
    except Exception as e:
        print(f"[WARN] Warning during warmup: {e}")
    yield
    print("[SHUTDOWN] Shutting down Omni-RecSys Backend.")


app = FastAPI(
    title="Omni-RecSys Mega Platform API",
    description="Unified content recommendation engine synthesizing Netflix, YouTube, TikTok, Spotify, and Meta production architectures.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for React frontend (Vite port 5173, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register All API Routers
app.include_router(rec_router)
app.include_router(users_router)
app.include_router(items_router)
app.include_router(events_router)
app.include_router(benchmark_router)
app.include_router(explain_router)
app.include_router(artwork_router)


from fastapi.responses import HTMLResponse


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Omni-RecSys Mega Platform API</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0d14; color: #f8fafc; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
            .card { background: #111726; border: 1px solid #1f293d; border-radius: 16px; padding: 40px; max-width: 600px; text-align: center; box-shadow: 0 20px 50px rgba(0,0,0,0.6); }
            h1 { font-size: 2rem; margin-bottom: 8px; background: linear-gradient(135deg, #e50914, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            p { color: #94a3b8; font-size: 1rem; line-height: 1.5; margin-bottom: 24px; }
            .btn-group { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
            .btn { display: inline-block; padding: 12px 24px; border-radius: 8px; font-weight: 600; text-decoration: none; font-size: 0.95rem; transition: transform 0.2s; }
            .btn-primary { background: #e50914; color: white; }
            .btn-secondary { background: #1f293d; color: #38bdf8; border: 1px solid #38bdf8; }
            .badge { display: inline-block; background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 700; margin-bottom: 16px; }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="badge" style="background: rgba(229, 9, 20, 0.2); color: #ff4d4d; border: 1px solid rgba(229, 9, 20, 0.4);">🛋️ 1% BATTERY VIBE • 13 MODELS ACTIVE</div>
            <h1>আসেন Unproductive হই</h1>
            <p>Asen Unproductive Hoi • YouTube, Netflix, TikTok ও Spotify-র ২০ বছরের কাটিং-এজ এআই অ্যালগরিদম দিয়ে আপনার মূল্যবান সময় সফলভাবে নষ্ট করার এক মাস্টারপিস ইঞ্জিন।</p>
            <div class="btn-group">
                <a href="http://localhost:5173" class="btn btn-primary" target="_blank">🛋️ Launch Distraction Dashboard &rarr;</a>
                <a href="/docs" class="btn btn-secondary">Interactive Swagger Docs</a>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "omni-recsys-backend",
        "supported_domains": settings.SUPPORTED_DOMAINS,
        "registered_models_count": len(settings.REGISTERED_MODELS),
        "registered_models": settings.REGISTERED_MODELS,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
