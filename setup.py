from setuptools import setup, find_packages

setup(
    name="yt_assist",
    version="1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.11.5"
)