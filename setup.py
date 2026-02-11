from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-assessment-platform",
    version="0.1.0",
    author="LLM Assessment Team",
    description="A unified LLM quality and security assessment platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NJX-njx/LLM-Quality-and-Security-Assessment-Platform",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.0",
        "jinja2>=3.1.0",
        "matplotlib>=3.7.0",
    ],
    entry_points={
        "console_scripts": [
            "llm-assess=llm_assessment.cli:main",
        ],
    },
)
