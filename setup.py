import ast
import os
import re
from pathlib import Path
from setuptools import find_packages, setup

# Base directory
this_dir = os.path.dirname(os.path.abspath(__file__))

# Read version from mmfreelm/__init__.py
def get_mmfreelm_version():
    init_file = Path(this_dir) / 'mmfreelm' / '__init__.py'
    if init_file.exists():
        with open(init_file) as f:
            match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
            if match:
                return ast.literal_eval(match.group(1))
    return "0.1.2"  # Fallback if not found

# Read long description from README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "AI Game Engine and Matmul-Free Language Model combined"

setup(
    name="ai_game_engine",
    version=get_mmfreelm_version(),
    description="AI Game Engine with diffusion model, noise latent functions, patch embedding, and Matmul-free LM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sakib Ahmed",
    author_email="sakibahmed2018go@gmail.com",
    url="https://github.com/Sakib323/AI-Game-Engine",
    packages=find_packages(include=["mmfreelm", "mmfreelm.*"]),
    py_modules=["noise_latent", "diffusion_model", "patch_embedding","modeling_hgrn_bit"],
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "tqdm>=4.62.0",
        "triton",
        "transformers",
        "einops",
        "ninja"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
