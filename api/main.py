from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your individual API apps
from .data_extractor import app as data_extractor_app
from .nutrient_analyzer import app as nutrient_analyzer_app

main_app = FastAPI()

# Add CORS middleware
main_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount apps
main_app.mount("/", data_extractor_app)
main_app.mount("/", nutrient_analyzer_app)

# Optional: Add a root endpoint
@main_app.get("/")
async def root():
    return {"message": "Main application"}
