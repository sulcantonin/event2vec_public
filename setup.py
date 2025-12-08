from setuptools import setup, find_packages

setup(
    name='event2vector',
    version='0.1.0.2',
    author='Antonin Sulc',
    description='A geometric approach to learning composable representations of event sequences.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
        'matplotlib',
        'scikit-learn',
        'openTSNE',
        'gensim',
        'seaborn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

