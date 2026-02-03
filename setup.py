"""
Setup script for AI Grand Prix Drone Racing package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="drone-racing",
    version="0.1.0",
    author="AI Grand Prix Competitor",
    author_email="competitor@aigrandprix.example",
    description="Autonomous drone racing software for AI Grand Prix 2026",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/NOLA-Tech-Ai-Drone-",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "mypy",
            "ruff",
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "drone-race=main:main",
        ],
    },
)
