from setuptools import setup

setup(
    name="ai_game_engine",
    version="0.1.2",  # Updated version for new module
    py_modules=["noise_latent", "diffusion_model", "patch_embedding"],  # Added patch_embedding
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "tqdm>=4.62.0",
    ],
    author="Sakib Ahmed",
    author_email="sakibahmed2018go@gmail.com",
    description="AI Game Engine with diffusion model utilities, noise latent functions, and patch embedding",
    url="https://github.com/Sakib323/AI-Game-Engine",
)