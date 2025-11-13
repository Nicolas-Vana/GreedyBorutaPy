from setuptools import setup

setup(name='greedyboruta',
      version='0.1.0',
      description='Python Implementation of GreedyBoruta Feature Selection',
      url='https://github.com/Nicolas-Vana/GreedyBorutaPy',
      download_url='https://github.com/Nicolas-Vana/GreedyBorutaPy/tarball/0.1.5', # TODO update with each release
      author='Nicolas Vana Santos',
      author_email='nicolas.vana@gmail.com',
      license='BSD 3 clause',
      packages=['greedy_boruta'],
      package_dir={'greedy_boruta': 'greedy_boruta'},
      package_data={'greedy_boruta/examples/*csv': ['greedy_boruta/examples/*.csv']},
      include_package_data=True,
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      keywords=['feature selection', 'machine learning', 'random forest'],
      install_requires=['numpy>=1.10.4',
                        'scikit-learn>=0.17.1',
                        'scipy>=0.17.0'
                        ])