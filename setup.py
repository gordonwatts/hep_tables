# Need setuptools even though it isn't used - loads some plugins.
from setuptools import find_packages  # noqa: F401
from distutils.core import setup

# Use the readme as the long description.
with open("README.md", "r") as fh:
    long_description = fh.read()

extras_require = {'test': ['pytest', 'pytest-asyncio', 'pytest-cov', 'pytest-mock', 'flake8',
                           'coverage', 'twine', 'wheel', 'jupyter-book'],
                  'notebook': ['jupyterlab', 'matplotlib']}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

setup(name="hep_tables",
      version='1.0.0b1',
      packages=['hep_tables'],
      scripts=[],
      description="Tables for structured data",
      long_description=long_description,
      long_description_content_type="text/markdown",
      author="G. Watts (IRIS-HEP/UW Seattle)",
      author_email="gwatts@uw.edu",
      maintainer="Gordon Watts (IRIS-HEP/UW Seattle)",
      maintainer_email="gwatts@uw.edu",
      url="https://github.com/gordonwatts/hep_tables",
      license="TBD",
      test_suite="tests",
      install_requires=["func_adl_xaod>=1.1b1", "servicex>=2b1", "dataframe_expressions>=1.0b1",
                        "make_it_sync"],
      extras_require=extras_require,
      classifiers=[
                   # "Development Status :: 3 - Alpha",
                   # "Development Status :: 4 - Beta",
                   # "Development Status :: 5 - Production/Stable",
                   # "Development Status :: 6 - Mature",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Information Technology",
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 3.7",
                   "Topic :: Software Development",
                   "Topic :: Utilities",
      ],
      data_files=[],
      python_requires='>=3.6, <3.8',
      platforms="Any",
      )
