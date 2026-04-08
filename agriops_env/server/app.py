"""FastAPI server entry point for AgriOpsEnv package path."""

from openenv.core.env_server import create_fastapi_app

from ..models import AgriOpsAction, AgriOpsObservation
from .environment import AgriOpsEnvironment

app = create_fastapi_app(AgriOpsEnvironment, AgriOpsAction, AgriOpsObservation)
