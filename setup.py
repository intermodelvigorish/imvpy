"""
IMV: Information Model Vigor Package Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file, excluding comments and optional dependencies"""
    with open(filename) as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip comments, empty lines, and optional deep learning dependencies
            if line and not line.startswith('#') and not line.startswith('torch'):
                # Skip lines that are just separators
                if '=' in line or line.startswith('-r'):
                    requirements.append(line)
        return requirements

setup(
    name="imv",
    version="1.0.0",
    author="Valler Y.",
    author_email="contact@imv-package.org",
    description="Information Model Vigor: A framework for measuring information content in machine learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/intermodelvigorish/imv_ml_package",
    project_urls={
        "Bug Tracker": "https://github.com/intermodelvigorish/imv_ml_package/issues",
        "Documentation": "https://github.com/intermodelvigorish/imv_ml_package/blob/main/docs/TECHNICAL.md",
        "Source Code": "https://github.com/intermodelvigorish/imv_ml_package",
    },
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0,<2.0.0",
        "pandas>=1.3.0,<3.0.0",
        "scipy>=1.7.0,<2.0.0",
        "scikit-learn>=1.0.0,<2.0.0",
        "matplotlib>=3.4.0,<4.0.0",
        "seaborn>=0.11.0,<1.0.0",
        "joblib>=1.1.0,<2.0.0",
        "tqdm>=4.62.0,<5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
            "jupyter>=1.0.0",
        ],
        "deep-learning": [
            "torch>=1.12.0",
            "transformers>=4.20.0",
            "datasets>=2.0.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "torch>=1.12.0",
            "transformers>=4.20.0",
            "datasets>=2.0.0",
            "tqdm-joblib>=0.0.4",
            "ucimlrepo>=0.0.3",
        ],
    },
    keywords=[
        "machine-learning",
        "feature-importance",
        "shapley-values",
        "information-theory",
        "model-evaluation",
        "interpretability",
        "explainable-ai",
        "deep-learning",
        "ablation-study",
        "multi-class-classification",
    ],
    include_package_data=True,
    zip_safe=False,
)
