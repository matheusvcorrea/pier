from fastapi import FastAPI
from api.route import router
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lottery Accumulation Prediction API",
    description="API for predicting whether the next lottery draw will accumulate",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)