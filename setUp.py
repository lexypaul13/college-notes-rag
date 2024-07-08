from setuptools import setup, find_packages

setup(
    name="college_notes_rag",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv",
        "langchain",
        "langchain-community",
        "langchain-anthropic",
        "unstructured",
        "chromadb",
        "tiktoken",
        "pypdf",
        "sentence-transformers",
    ],
    entry_points={
        "console_scripts": [
            "post_install=post_install:main",
        ],
    },
)