from setuptools import setup

setup(
    name="ai_game_engine",
    version="0.1.0",
    py_modules=["noise_latent"],  # Include noise_latent.py as a module
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
    author="Sakib Ahmed",
    author_email="sakibahmed2018go@gmail.com",
    description="AI Game Engine with diffusion model utilities",
    url="https://github.com/Sakib323/AI-Game-Engine",
)