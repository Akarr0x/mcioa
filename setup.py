from setuptools import setup, find_packages

setup(
    name='mcioa',
    version='0.9.0',
    author='Alessandro Diamanti',
    author_email='alessandrodiamanti1@gmail.com',
    description='An implementation of Multiple Co-Inertia Analysis for multiomis data integration, inspired by the ADE4 package.',
    url='https://github.com/Akarr0x/mcioa',
    packages=find_packages(),  # automatically discover and include all packages in the package directory
    classifiers=[
        'Programming Language :: Python 3',
        'License :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'pandas>=1.5.3',
        'numpy>=1.25.3',
        'scikit-learn>=1.2.2',
        'scipy>=1.11.1'
    ],
    python_requires='>=3.9',
)
