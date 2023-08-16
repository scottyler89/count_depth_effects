from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='count_depth_effects',
    version='0.1.0',
    description='A package to study the effects of depth of sampling and normalization strategies on observed distances.',
    author='Scott Tyler',
    author_email='scottyler89@gmail.com',
    url='https://github.com/yourusername/count_depth_effects',
    packages=find_packages(),
    install_requires = requirements,
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  # Adjust the license as needed
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='bioinformatics, single-cell, gradient-descent, visualization',
    python_requires='>=3.6',
)
