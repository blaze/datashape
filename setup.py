import os
from distutils.core import setup

import versioneer
versioneer.VCS = 'git'
versioneer.versionfile_source = os.path.join('datashape', '_version.py')
versioneer.versionfile_build = versioneer.versionfile_source
versioneer.tag_prefix = ''  # tags are like 1.2.0
versioneer.parentdir_prefix = 'datashape-'  # dirname like 'myproject-1.2.0'

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


setup(
    name='datashape',
    version=versioneer.get_version(verbose=True),
    cmdclass=versioneer.get_cmdclass(),
    author='Continuum Analytics',
    author_email='blaze-dev@continuum.io',
    description='A data description language.',
    license='BSD',
    keywords='data language',
    url='http://packages.python.org/datashape',
    packages=['datashape', 'datashape.tests'],
    install_requires=read('requirements.txt').strip().split('\n'),
    long_description=read('README.rst'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'License :: OSI Approved :: BSD License',
    ],
)
