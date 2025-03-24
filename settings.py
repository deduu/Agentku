import os
from dotenv import load_dotenv

load_dotenv(override=True) 


class AppSettings:


    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMAIL = os.getenv("EMAIL")
    PASSWORD = os.getenv("PASSWORD")
    print(f"Setting: {OPENAI_API_KEY}")

settings = AppSettings()
