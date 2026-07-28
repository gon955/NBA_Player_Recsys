"""
lambda_handler.py — Lambda entrypoint. Wraps the existing FastAPI app so the
same code serves both `uvicorn app:app` locally and API Gateway / a Function URL
in AWS, with no branching inside the routes.
"""

from mangum import Mangum

from app import app

handler = Mangum(app, lifespan="off")
