from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="simple_openai_requests",
    version="1.0.0",
    author="Le Hoang Anh",
    author_email="lehoanganh29896@gmail.com",
    description="Unify common OpenAI API requests use cases into a simple interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lehoanganh298/simple_openai_requests",
    packages=['simple_openai_requests'],  # Explicitly specify the package
    package_dir={'simple_openai_requests': 'simple_openai_requests'},  # Specify package directory
    install_requires=[
        "openai>=1.0.0",
        "tqdm>=4.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
)