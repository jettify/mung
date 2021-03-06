import os
import re
import sys
from setuptools import setup, find_packages


PY_VER = sys.version_info

if not PY_VER >= (3, 6):
    raise RuntimeError('mung does not support Python earlier than 3.6')


def read(f):
    return open(os.path.join(os.path.dirname(__file__), f)).read().strip()


install_requires = [
    'scikit-learn==0.20.3',
]
extras_require = {}


def read_version():
    regexp = re.compile(r"^__version__\W*=\W*'([\d.abrc]+)'")
    init_py = os.path.join(os.path.dirname(__file__),
                           'mung', '__init__.py')
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        else:
            msg = 'Cannot find version in mung/__init__.py'
            raise RuntimeError(msg)


classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Operating System :: POSIX',
    'Development Status :: 2 - Pre-Alpha',
    'Framework :: AsyncIO',
]


setup(name='mung',
      version=read_version(),
      description=('mung'),
      long_description='\n\n'.join((read('README.rst'), read('CHANGES.txt'))),
      install_requires=install_requires,
      classifiers=classifiers,
      platforms=['POSIX'],
      author='Nikolay Novik',
      author_email='nickolainovik@gmail.com',
      url='https://github.com/ml-libs/mung',
      download_url='',
      license='Apache 2',
      packages=find_packages(),
      extras_require=extras_require,
      keywords=['mung', 'model explanation'],
      zip_safe=True,
      include_package_data=True)
