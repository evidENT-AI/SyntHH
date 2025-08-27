from setuptools import setup, find_packages

setup(
    name="synthh",
    version="0.1.0",
    description="Synthetic Hearing Health Data Generation and Validation",
    long_description="A research project for generating synthetic audiometric data that preserves statistical properties of real hearing measurements while protecting patient privacy.",
    author="LB",
    author_email="",
    url="https://github.com/liambarrett/SyntHH",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
    ],
    extras_require={
        "deep-learning": [
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
            "torchvision",
            "torchaudio",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="audiometry synthetic-data healthcare machine-learning privacy",
)