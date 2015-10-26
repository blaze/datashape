import os
from setuptools import setup

import versioneer

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


setup(
    name='datashape',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Continuum Analytics',
    author_email='blaze-dev@continuum.io',
    description='A data description language.',
    license='BSD',
    keywords='data language',
    url='http://datashape.readthedocs.org/en/latest/',
    packages=['datashape', 'datashape.util', 'datashape.tests'],
    install_requires=read('requirements.txt').strip().split('\n'),
    long_description=read('README.rst'),
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'License :: OSI Approved :: BSD License',
    ],
)
