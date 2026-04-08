"""FastAPI server entry point for AgriOpsEnv."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from openenv.core.env_server import create_fastapi_app

from .environment import AgriOpsEnvironment
from models import AgriOpsAction, AgriOpsObservation

app = create_fastapi_app(AgriOpsEnvironment, AgriOpsAction, AgriOpsObservation)


def main() -> None:
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )


if __name__ == "__main__":
    main()
