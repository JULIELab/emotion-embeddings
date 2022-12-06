import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="emocoder", # Replace with your own username
    version="0.0.2",
    author="Sven Büchel",
    author_email="sven.buechel@uni-jena.de",
    description="Codebase and experimental framework for the EmoCoder architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JULIELab/emotion-encoder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.7.6',
)