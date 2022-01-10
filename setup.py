import setuptools

vers = '0.0.9'

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="mixpython",
    version=vers,
    author="Andrew Astakhov",
    author_email="aw.astakhov@gmail.com",
    description="mix",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sth-v/mixpython",
    project_urls={
        "Bug Tracker": "https://github.com/sth-v/mixpython/issues",
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