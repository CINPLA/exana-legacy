# exana

Exana is a library with tools and visualisers for analysis of electrophysiological data in neuroscience. Some are built by our team and some are collected from other projects for availability and to reduce the number of dependencies when performing analysis. The philosophy behind this project is not to give polished production ready analysis tools. Rather, we want to provide useful, tested tools and more importantly generate collaboration on development. We recognize that many labs are building many similar tools for in house analysis that is only available to the public via the methods section in publications. By sharing our analysis we hope that rather than building the same tools over and over again we (the neuroscience community) can together build robust analysis methods. 
 
Exana is a repository that contains the analysis tools we use in the lab which is general enough to be used for a wider audience. We do not claim to be the first or the only one to do this, there exist excellent open source tools out there such as `elephant` and `fieldtrip` to mention some. We only want to contribute to the community, by sharing our tools that we cannot find already made out there.

## Installation

**Alternative 1)**

Clone this repository and install with python

```
git clone https://github.com/CINPLA/exana.git
python setup.py develop
```
**Alternative 2)**

Download Anaconda 3 and install with conda

```
conda install exana -c cinpla -c defaults -c conda-forge
```


## Development

Please fork and send pull requests for contributions.

Exana depends on `neo` and `numpy`, to keep the global dependencies to a minimum please import special dependencies inside functions. Please conform to pep8 and write docstrings in the [numpy style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html).

Test your code as much as possible, and preferably make a proper test case runnable with [pytest](https://docs.pytest.org/en/latest/contents.html) as a minimum write runnable examples in the [docstring](https://docs.pytest.org/en/latest/doctest.html) e.g.

```python
def has_ten_spikes(spiketrain):
  """
  Parameters
  ----------
  spiketrain : neo.SpikeTrain
      A neo spike train with spike times.

  Returns
  -------
  bool
      True if successful, False otherwise.
  
  Example
  -------
  >>> import neo
  >>> spiketrain = neo.SpikeTrain(times=list(range(10)), tstop=10, units='s')
  >>> my_analysis_function(spiketrain)
  True
  """
  if len(spiketrain) == 10:
    return True
  else:
    return False
```

To run tests, simply run in the command line
```
py.test -s --doctest-modules
```
