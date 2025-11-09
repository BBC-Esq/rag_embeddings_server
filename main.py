import os
import sys
import platform
from pathlib import Path

def set_cuda_paths():
    if platform.system() != "Windows":
        return
    
    venv_base = Path(sys.executable).parent.parent
    nvidia_base = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    
    if not nvidia_base.exists():
        return
    
    cuda_path_runtime = nvidia_base / 'cuda_runtime' / 'bin'
    cuda_path_runtime_lib = nvidia_base / 'cuda_runtime' / 'lib' / 'x64'
    cuda_path_runtime_include = nvidia_base / 'cuda_runtime' / 'include'
    cublas_path = nvidia_base / 'cublas' / 'bin'
    cudnn_path = nvidia_base / 'cudnn' / 'bin'
    nvrtc_path = nvidia_base / 'cuda_nvrtc' / 'bin'
    nvcc_path = nvidia_base / 'cuda_nvcc' / 'bin'
    
    paths_to_add = [
        str(cuda_path_runtime),
        str(cuda_path_runtime_lib),
        str(cuda_path_runtime_include),
        str(cublas_path),
        str(cudnn_path),
        str(nvrtc_path),
        str(nvcc_path),
    ]
    
    current_value = os.environ.get('PATH', '')
    new_value = os.pathsep.join(paths_to_add + ([current_value] if current_value else []))
    os.environ['PATH'] = new_value
    
    triton_cuda_path = nvidia_base / 'cuda_runtime'
    current_cuda_path = os.environ.get('CUDA_PATH', '')
    new_cuda_path = os.pathsep.join([str(triton_cuda_path)] + ([current_cuda_path] if current_cuda_path else []))
    os.environ['CUDA_PATH'] = new_cuda_path

set_cuda_paths()

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from sentry_sdk.integrations.fastapi import FastApiIntegration

import app_state
from service_loader import load_embedding_service
from utils import logger

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await load_embedding_service()
    
    try:
        yield
    finally:
        try:
            if app_state.embedding_service:
                app_state.embedding_service = None
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(f"Error during shutdown: {str(e)}")


app = FastAPI(
    title="Inception v0",
    description="Service for generating embeddings from queries and opinions",
    version="0.0.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=os.getenv("ALLOWED_METHODS", "*").split(","),
    allow_headers=os.getenv("ALLOWED_HEADERS", "*").split(","),
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Inception v2",
        version="2.0.0",
        description="Service for generating embeddings from queries and opinions",
        routes=app.routes,
    )

    if "/api/v1/embed/text" in openapi_schema["paths"]:
        openapi_schema["paths"]["/api/v1/embed/text"]["post"][
            "requestBody"
        ] = {
            "content": {
                "text/plain": {
                    "example": "A very long opinion goes here.\nIt can span multiple lines.\nEach line will be preserved."
                }
            },
            "required": True,
        }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

from route_embedding import router as embedding_router
from route_monitoring import router as monitoring_router

app.include_router(embedding_router, tags=["embedding"])
app.include_router(monitoring_router, tags=["monitoring"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005)