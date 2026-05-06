from fastapi import FastAPI
import joblib
from prometheus_fastapi_instrumentator import Instrumentator
import logging
from prometheus_client import Counter
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Initialize FastAPI app
app = FastAPI()
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
)
instrumentator.instrument(app).expose(app, include_in_schema=False, should_gzip=True)

# Load model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Prometheus counter for log events
log_counter = Counter(
    'app_log_events_total',
    'Total log events by level',
    ['level']
)

class PrometheusLoggingHandler(logging.Handler):
    def emit(self, record):
        log_counter.labels(level=record.levelname.lower()).inc()

# Attach Prometheus logging handler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.addHandler(PrometheusLoggingHandler())

# Optional: log each request
class LogRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"Request: {request.method} {request.url}")
        response = await call_next(request)
        return response

app.add_middleware(LogRequestMiddleware)

# Home route (fixes your 404 issue)
@app.get("/")
def home():
    return {"message": "Content Moderation API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# Prediction route
@app.post("/predict")
def predict(text: str):
    # Convert text to vector
    text_vec = vectorizer.transform([text])

    # Predict
    prediction = model.predict(text_vec)[0]

    return {
        "input": text,
        "toxic": int(prediction)
    }