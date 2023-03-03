import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pool-playing-robot",
    version="0.0.1",
    author="Eduard Almar",
    author_email="eduard.almar.oliva@gmail.com",
    description="A package for pool playing robot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/opticsensors/pool-playing-robot",
    project_urls={
        "Bug Tracker": "https://github.com/opticsensors/pool-playing-robot/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)