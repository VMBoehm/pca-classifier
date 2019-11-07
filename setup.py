from setuptools import setup

setup(name='pca-classifier',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description='(regularized) linear data compression and its application to classification and out of distribution detection',
      url='http://github.com/VMBoehm/pca_classifier',
      author='Vanessa Martina Boehm',
      author_email='vboehm@berkeley.edu',
      license='GNU General Public License v3.0',
      packages=['pca_classifier'],
      )
