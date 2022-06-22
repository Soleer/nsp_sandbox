import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "nsp_sandbox",
    version = "0.0.1.1",
    author = "Niklas Wünstel",
    author_email = "nsp@soleer.de",
    description = "Nurse Scheduling Problem Sandbox package.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Soleer/nsp_sandbox",
    project_urls ={
        "Bug Tracker": "https://github.com/Soleer/nsp_sandbox/issues",
            },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[],
)