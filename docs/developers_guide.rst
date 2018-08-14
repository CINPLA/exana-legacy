=================
Developers' guide
=================

Please fork and send pull requests for contributions.

Requirements for code contributions
-----------------------------------

Code contributions to exana need to adhere to the following requirements:

- All added code needs to be useful to more than just you. If only you need the functionality in
  your project, then add it to your project instead.
- Each function needs to have a corresponding test.
- Each function needs to have an example in the docstring.
- Plotting functions must be placed in `examples`. Rationale: Everyone has an opinion on how to plot everything.
    - Every function in examples must also be used in the documentation - see the `docs` folder.
- Create specialized functions. Don't create one function that is general and takes 1000 parameters.
  Make 1000 functions instead.
- Prioritize functions over classes - that keeps the scope minimal.
  Use classes only when you really need it for the data or the API.
- Keep it simple. Sometimes it is better to be a bit verbose to make the logic simpler.
  In other words, use simple functions and avoid classes if you can.
- Avoid wrapping simple functions from other libraries to be "helpful".
  Just show the function you wanted to wrap in your examples or docstrings instead.

Please conform to pep8 and write docstrings in `numpy style <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_

Testing
-------
To run tests, simply run in the command line::

    $ py.test --doctest-modules --ignore setup.py

Test your code as much as possible, and preferably make a proper test case
runnable with `pytest <https://docs.pytest.org/en/latest/contents.html>`_, as a
minimum write runnable examples in the `docstring <https://docs.pytest.org/en/latest/doctest.html>`_;
see example below::

    def has_ten_spikes(spiketrain):
        """
        Parameters
        ----------
        spiketrain : neo.SpikeTrain
            A NEO spike train with spike times.

        Returns
        -------
        bool
            True if successful, False otherwise.

        Example
        -------
        >>> import neo
        >>> spiketrain = neo.SpikeTrain(times=list(range(10)), t_stop=10, units='s')
        >>> has_ten_spikes(spiketrain)
        True
        """
        if len(spiketrain) == 10:
            return True
        else:
            return False


Dependencies
------------

Exana depends on `neo` and `numpy`, to keep the global dependencies to a minimum
please import special dependencies inside functions.


Getting the source code
-----------------------

We use the Git version control system. The best way to contribute is through
GitHub_. You will first need a GitHub account, and you should then fork the
repository.

Working on the documentation
----------------------------

The documentation is written in reStructuredText, using the Sphinx
documentation system. To build the documentation::

    $ cd exana/docs
    $ sphinx-apidoc -o . ../exana/
    $ make html

Then open `doc/build/html/index.html` in your browser.

Committing your changes
-----------------------

Once you are happy with your changes, **run the test suite again to check
that you have not introduced any new bugs**. Then you can commit them to your
local repository::

    $ git commit -m 'informative commit message'

If this is your first commit to the project, please add your name and
affiliation/employer to :file:`doc/source/authors.rst`

You can then push your changes to your online repository on GitHub::

    $ git push

Once you think your changes are ready to be included in the main Neo repository,
open a pull request on GitHub (see https://help.github.com/articles/using-pull-requests).

.. _GitHub: http://github.com
