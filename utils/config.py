# my_app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    QDRANT_HOST = os.getenv("QDRANT_HOST")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")

