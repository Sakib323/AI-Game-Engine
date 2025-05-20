from setuptools import setup

setup(
    name="ai_game_engine",
    version="0.1.1",  # Updated version to reflect new module
    py_modules=["noise_latent", "diffusion_model"],  # Added diffusion_model
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "tqdm>=4.62.0",  # Added for progress bar support
    ],
    author="Sakib Ahmed",
    author_email="sakibahmed2018go@gmail.com",
    description="AI Game Engine with diffusion model utilities and noise latent functions",
    url="https://github.com/Sakib323/AI-Game-Engine",
)