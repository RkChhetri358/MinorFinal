from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import segment, depth, reconstruct

app = FastAPI(
    title="AI Vision Reconstruction Lab API",
    description="Computer Vision inference endpoints for segmentation, depth estimation, and 3D reconstruction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(segment.router, prefix="/segment", tags=["Segmentation"])
app.include_router(depth.router, prefix="/depth", tags=["Depth Estimation"])
app.include_router(reconstruct.router, prefix="/reconstruct", tags=["3D Reconstruction"])


@app.get("/health")
async def health_check():
    return {"status": "operational", "service": "AI Vision Lab"}
