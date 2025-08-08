from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="openra-rl-env",
    version="0.1.0",
    author="OpenRA RL Community",
    author_email="contact@example.com",
    description="Reinforcement Learning Environment for OpenRA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/openra-rl-env",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Games/Entertainment :: Real Time Strategy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "requests>=2.28.0",
        "websocket-client>=1.4.0",
        "gymnasium>=0.28.0",
    ],
    extras_require={
        "rl": [
            "stable-baselines3>=2.0.0",
            "torch>=1.13.0",
            "tensorboard>=2.10.0",
        ],
        "vision": [
            "opencv-python>=4.6.0",
            "Pillow>=9.0.0",
            "matplotlib>=3.5.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.991",
        ],
        "all": [
            "stable-baselines3>=2.0.0",
            "torch>=1.13.0",
            "tensorboard>=2.10.0",
            "opencv-python>=4.6.0",
            "Pillow>=9.0.0",
            "matplotlib>=3.5.0",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.991",
        ],
    },
    entry_points={
        "console_scripts": [
            "openra-env-demo=example_usage:main",
        ],
    },
)