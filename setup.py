from pathlib import Path
from typing import List

from setuptools import find_packages, setup

BASE = Path(__file__).parent
REQUIREMENTS_DIR = BASE / 'requirements'


def get_requirements(filename: str) -> List[str]:
    with open(REQUIREMENTS_DIR / filename) as file:
        return [ln.strip() for ln in file.readlines()]


def get_long_description() -> str:
    long_description = readme_file.read_text() if (readme_file := BASE / 'README.md').exists() else ""

    return long_description.replace('../assets/preview_animation.gif?raw=true',
                                    'https://github.com/alex-snd/TRecover/blob/assets/preview_animation.gif?raw=true')


essential_packages = get_requirements('essential.txt')
api_service_packages = get_requirements('docker/api.txt')
dashboard_service_packages = get_requirements('docker/dashboard.txt')
standalone_service_packages = get_requirements('docker/standalone.txt')
worker_service_packages = get_requirements('docker/worker.txt')

demo_packages = get_requirements('demo.txt')
dev_packages = get_requirements('dev.txt')
docs_packages = get_requirements('docs.txt')
train_packages = get_requirements('train.txt')

setup(
    name='trecover',
    version='1.0.3',
    license='Apache License 2.0',
    author='Alexander Shulga',
    author_email='alexandershulga.sh@gmail.com',
    url='https://alex-snd.github.io/TRecover',
    project_urls={
        'Source Code': 'https://github.com/alex-snd/TRecover',
        'Bug Tracker': 'https://github.com/alex-snd/TRecover/issues',

    },
    description='A python library for training a Transformer neural network to solve the'
                ' Running Key Cipher, widely known in the field of cryptography.',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    keywords=[
        'Deep Learning',
        'Machine Learning',
        'Transformers',
        'NLP',
        'Cryptography',
        'Keyless Reading',
        'TRecover',
        'Text Recovery',
        'PyTorch',
    ],
    python_requires='>=3.8',
    packages=find_packages(
        where='src'
    ),
    package_dir={'': 'src'},
    install_requires=[essential_packages],
    extras_require={
        'api': api_service_packages,
        'dashboard': dashboard_service_packages,
        'standalone': standalone_service_packages,
        'worker': worker_service_packages,
        "demo": demo_packages,
        "dev": dev_packages,
        "docs": docs_packages,
        "train": train_packages,
    },
    entry_points={
        'console_scripts': [
            'trecover = trecover.app.cli.trecovercli:cli',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',

        'Environment :: Console',
        'Environment :: GPU :: NVIDIA CUDA',
        'Environment :: Web Environment',

        'Framework :: AsyncIO',
        'Framework :: Celery',
        'Framework :: FastAPI',
        'Framework :: Jupyter',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'License :: OSI Approved :: Apache Software License',

        'Natural Language :: English',

        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',

        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',

        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Security',
        'Topic :: Security :: Cryptography',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
