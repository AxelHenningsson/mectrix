import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="mectrix",
    version="0.0.1",
    author="Axel Henningsson",
    author_email="nilsaxelhenningsson@gmail.com",
    description="Continuum mechanics tensor transformations for single crystal applications in python.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/AxelHenningsson/mectrix",
    project_urls={
        "Documentation": "https://github.com/AxelHenningsson/mectrix",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=["matplotlib",
                      "numpy",
                      "scipy"
                      ]
)
