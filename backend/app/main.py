from fastapi import FastAPI

app = FastAPI(
    title="Single-Pass Drone 3D Reconstruction",
    description="Hybrid sensor-fusion and confidence-aware 3D reconstruction system",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "Single-Pass Drone 3D Reconstruction API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
