from setuptools import setup, find_packages

setup(
    name="sam3",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "sam3": [
            "assets/*.txt",
            "assets/*.gz",
            "assets/**/*",
        ]
    },
)
