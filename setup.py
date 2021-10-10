from pathlib import Path
from typing import List

from setuptools import find_packages, setup

REQUIREMENTS_DIR = Path(__file__).parent / 'requirements'


def get_requirements(filename: str) -> List[str]:
    with open(REQUIREMENTS_DIR / filename) as file:
        return [ln.strip() for ln in file.readlines()]


essential_packages = get_requirements('essential.txt')
# dev_packages = get_requirements('dev.txt')  # TODO add extra requirements
# test_packages = get_requirements('test.txt')
# docs_packages = get_requirements('docx.txt')

setup(
    name='zreader',
    version='0.1.0',
    license='Apache License 2.0',
    author='Alexander Shulga',
    author_email='alexandershulga.sh@gmail.com',
    python_requires='>=3.8',
    packages=find_packages(
        where='src',
        include=['zreader'],
    ),
    package_dir={'': 'src'},
    install_requires=[essential_packages],
    # extras_require={
    #     "test": test_packages,
    #     "dev": test_packages + dev_packages + docs_packages,
    #     "docs": docs_packages,
    # },
    entry_points={
        'console_scripts': [
            'zreader = app.cli.zreadercli:cli',
        ],
    },
)
