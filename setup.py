from setuptools import setup, find_packages

setup(
    name="transformers-patch",
    version="0.1.0",
    author="Xingkai Yu",
    author_email="xingkai@deepseek.com",
    description="patches for huggingface transformers to save memory",
    packages=find_packages(),
    install_requires=[
        "torch>=2.4",
        "triton>=3.0",
        "transformers>=4.51.0"
    ],
    python_requires=">=3.9",
)