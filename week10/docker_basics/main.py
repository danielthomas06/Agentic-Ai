from fastapi import FastAPI


app = FastAPI(title="Docker Test API")


@app.get("/")
def root():
    return {
        "message": "FastAPI is running inside Docker"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }