Sparse2Spatial (s2s)
======================================
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3369212.svg
   :target: https://doi.org/10.5281/zenodo.3369212
   :alt: Zenodo DOI

**Sparse2Spatial** contains routines to convert sparse observations into spatially and temporally resolved datasets using machine learning algorithms.

This package uses packages from the existing Python stack (e.g. dask_, xarray_,
pandas_, sklearn_, XGBoost_). `Pull Requests are
welcome! <https://github.com/tsherwen/sparse2spatial/pulls>`_


Installation
------------

**sparse2spatial** is currently only installable from source. To do this, you
can either clone the source directory and manually install::

    $ git clone https://github.com/tsherwen/sparse2spatial.git
    $ cd sparse2spatial
    $ python setup.py install

or, you can install via pip directly from git::

    $ pip install git+https://github.com/tsherwen/sparse2spatial.git


Usage
------------

Example analysis code for using **sparse2spatial** is available in the
scripts folder, along with predictions for sea-surface iodide, CHBr3, and CH2Br2.

A separate package (TreeSurgeon_) is available for plotting output the from sklearn_
RandomForestRegressors models. Scripts are provided in **sparse2spatial** for making
the input `.csv` files required by TreeSurgeon_.

Work using Sparse2Spatial (s2s)
^^^^^^^^^^^^

A subfolder in the scripts folder is present per species for work using s2s. This currently includes predictions for sea-surface iodide, CHBr3, and CH2Br2.

Publications using **sparse2spatial** are detailed below:

+ Research paper on predicting sea-surface iodide using machine learning

For details on this work please see the paper referenced below.

Sherwen, T., Chance, R. J., Tinel, L., Ellis, D., Evans, M. J., and Carpenter, L. J.: A machine learning based global sea-surface iodide distribution, Earth Syst. Sci. Data, https://doi.org/10.5194/essd-2019-40, 1-40, 2019.

A file to process the of the csv file of observational data used by the above paper is also included in the scripts/Iodide folder. The observational data can be found at the archived location below.

Chance R.; Tinel L.; Sherwen T.; Baker A.; Bell T.; Brindle J.; Campos M.L.A.M.; Croot P.; Ducklow H.; He P.; Hoogakker B.; Hopkins F.E.; Hughes C.; Jickells T.; Loades D.; Reyes Macaya D.A.; Mahajan A.S.; Malin G.; Phillips D.P.; Sinha A.K.; Sarkar A.; Roberts I.J.; Roy R.; Song X.; Winklebauer H.A.; Wuttig K.; Yang M.; Zhou P.; Carpenter L.J. (2019). Global sea-surface iodide observations, 1967-2018. British Oceanographic Data Centre - Natural Environment Research Council, UK. https://doi.org/10/czhx


License
-------

Copyright (c) 2018 `Tomas Sherwen`_

This work is licensed under a permissive MIT License.

Contact
-------

`Tomas Sherwen`_ - tomas.sherwen@york.ac.uk

.. _`Tomas Sherwen`: http://github.com/tsherwen
.. _conda: http://conda.pydata.org/docs/
.. _dask: http://dask.pydata.org/
.. _licensed: LICENSE
.. _xarray: http://xarray.pydata.org/
.. _pandas: https://pandas.pydata.org/
.. _sklearn: https://scikit-learn.org/stable/
.. _XGBoost: https://xgboost.readthedocs.io/en/latest/
.. _TreeSurgeon: https://github.com/wolfiex/TreeSurgeon
.. _AC_tools_wiki: https://github.com/tsherwen/AC_tools/wiki





