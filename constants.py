# constants.py

priority_libs = {
    "cp311": {
        "GPU": [
            "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-win_amd64.whl",
            "https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp311-cp311-win_amd64.whl",
            "triton-windows==3.4.0.post20",
            "nvidia-cuda-runtime-cu12==12.8.90",
            "nvidia-cublas-cu12==12.8.4.1",
            "nvidia-cuda-nvrtc-cu12==12.8.93",
            "nvidia-cuda-nvcc-cu12==12.8.93",
            "nvidia-cufft-cu12==11.3.3.83",
            "nvidia-cudnn-cu12==9.10.2.21",
        ],
        "CPU": [
            # "https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp311-cp311-win_amd64.whl",
            # "https://download.pytorch.org/whl/cpu/torchvision-0.23.0%2Bcpu-cp311-cp311-win_amd64.whl#sha256=51603eb071d0681abc4db98b10ff394ace31f425852e8de249b91c09c60eb19a",
            # "https://download.pytorch.org/whl/cpu/torchaudio-2.8.0%2Bcpu-cp311-cp311-win_amd64.whl#sha256=db37df7eee906f8fe0a639fdc673f3541cb2e173169b16d4133447eb922d1938"
        ],
        "COMMON": [
            # "https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.8.0-tesseract-5.5.0/tesserocr-2.8.0-cp311-cp311-win_amd64.whl",
        ],
    },
    "cp312": {
        "GPU": [
            "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp312-cp312-win_amd64.whl",
            "https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl",
            "triton-windows==3.4.0.post20",
            "nvidia-cuda-runtime-cu12==12.8.90",
            "nvidia-cublas-cu12==12.8.4.1",
            "nvidia-cuda-nvrtc-cu12==12.8.93",
            "nvidia-cuda-nvcc-cu12==12.8.93",
            "nvidia-cufft-cu12==11.3.3.83",
            "nvidia-cudnn-cu12==9.10.2.21",
        ],
        "CPU": [
            # "https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp312-cp312-win_amd64.whl",
            # "https://download.pytorch.org/whl/cpu/torchvision-0.23.0%2Bcpu-cp312-cp312-win_amd64.whl#sha256=a651ccc540cf4c87eb988730c59c2220c52b57adc276f044e7efb9830fa65a1d",
            # "https://download.pytorch.org/whl/cpu/torchaudio-2.8.0%2Bcpu-cp312-cp312-win_amd64.whl#sha256=9b302192b570657c1cc787a4d487ae4bbb7f2aab1c01b1fcc46757e7f86f391e"
        ],
        "COMMON": [
            # "https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.8.0-tesseract-5.5.0/tesserocr-2.8.0-cp312-cp312-win_amd64.whl",
        ]
    }
}

libs = [
    "altair==5.5.0",
    "annotated-doc==0.0.3",
    "annotated-types==0.7.0",
    "anyio==4.11.0",
    "attrs==25.4.0",
    "blinker==1.9.0",
    "cachetools==6.2.1",
    "certifi==2025.10.5",
    "charset-normalizer==3.4.4", # requests requires <4
    "click==8.3.0",
    "colorama==0.4.6",
    "einops==0.8.1",
    "fastapi==0.121.0",
    "filelock==3.20.0",
    "fsspec[http]==2025.9.0", # datasets requires...
    "gitdb==4.0.12",
    "google==3.0.0",
    "h11==0.16.0",
    "huggingface-hub==0.36.0", # tokenizers requires <1.0
    "idna==3.11",
    "Jinja2==3.1.6",
    "joblib==1.5.2",
    "jsonschema", # only required by tiledb-cloud
    "jsonschema-specifications==2025.9.1",
    "MarkupSafe==3.0.3",
    "mpmath==1.3.0", # sympy 1.13.1 requires <1.4
    "narwhals==2.10.2",
    "networkx==3.5",
    "nltk==3.9.2", # not higher; gives unexplained error
    "numpy==2.3.4", # numba 0.61.2 requires <2.3
    "packaging==25.0",
    "pandas==2.3.3",
    "pillow==12.0.0",
    "prometheus_client==0.23.1",
    "protobuf==6.33.0",
    "pyarrow==22.0.0",
    "pydantic==2.12.3",
    "pydantic_core==2.41.4", # pydantic 2.11.7 requires ==2.37.2; CAUTION, package checker is incorrect, check repo instead
    "pydantic-settings==2.11.0", # langchain-community requires >=2.4.0,<3.0.0
    "pydeck==0.9.1",
    "python-dateutil==2.9.0.post0",
    "python-dotenv==1.2.1",
    "pytz==2025.2",
    "PyYAML==6.0.3",
    "referencing==0.37.0",
    "regex==2025.10.23",
    "requests==2.32.5",
    "rpds-py==0.28.0",
    "safetensors==0.6.2",
    "scikit-learn==1.7.2",
    "scipy==1.16.3",
    "sentence-transformers==5.1.2",
    "sentry-sdk==2.43.0",
    "six==1.17.0",
    "smmap==5.0.2",
    "sniffio==1.3.1",
    "starlette==0.49.3",
    "streamlit==1.51.0",
    "sympy==1.13.3", # torch 2.8.0 requires 1.13.3
    "tenacity==9.1.2",
    "threadpoolctl==3.6.0",
    "tokenizers==0.22.1",
    "toml==0.10.2",
    "tornado==6.5.2",
    "tqdm==4.67.1",
    "transformers==4.57.1",
    "typing-inspection==0.4.2", # required by pydantic and pydantic-settings
    "typing_extensions==4.15.0", # unstructured 0.18.15 requires 4.15.0
    "tzdata==2025.2",
    "urllib3==2.5.0", # requests requires <3
    "uvicorn==0.38.0",
    "watchdog==6.0.0",
]

full_install_libs = []