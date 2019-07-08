# Sparse2Spatial (s2s)

This package contains routines to convert sparse observations into spatially and temporally resolved datasets using machine learning algorithms.

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

Quick Start
-----------

Functions within **AC_Tools** can be used for various tasks for handling model output and observations.

An exmample would be importing NetCDF files or converting ctm.bpch files from a directory of GEOS-Chem_ output (with ``tracerinfo.dat`` and ``diaginfo.dat`` files).


.. code:: python

    import AC_tools as AC
    folder = '<folder containing GEOS-Chem output>'
    # Get the atmospheric ozone burden in Gg O3 as a np.array
    array = AC.get_O3_burden(folder)
    print( "The ozone burden is: {burden}".format(burden=array.sum()))
    # Get surface area for resolution
    s_area = get_surface_area(res)[..., 0]  # m2 land map
    # Get global average surface CO
    spec = 'CO'
    array = AC.get_GC_output(wd=folder, vars=['IJ_AVG_S__{}'.format(spec)])
    ratio = AC.get_2D_arr_weighted_by_X(array, res='4x5', s_area=s_area)
    print( "The global average surface mixing ratio of {spec} (ppbv) is: {ratio}".format(spec=spec, ratio=ratio*1E9))


Usage
------------

Example analysis code for using Sparse2Spatial is available in the
scripts folder.


Work using Sparse2Spatial (s2s)
^^^^^^^^^^^^

A subfolder in the scripts folder is present for the following work using s2s.

+ Research paper *in review* on predicting sea-surface iodide using machine learning

For details on this work please see the paper referenced below.

[Sherwen, T., Chance, R. J., Tinel, L., Ellis, D., Evans, M. J., and Carpenter, L. J.: A machine learning based global sea-surface iodide distribution, Earth Syst. Sci. Data Discuss., https://doi.org/10.5194/essd-2019-40, *in review*, 2019.](https://doi.org/10.5194/essd-2019-40)

A file to process the of the csv file of observational data used by the above paper is also included in the scripts/Iodide folder. The observational data can be found at the archived location below.

Chance R.; Tinel L.; Sherwen T.; Baker A.; Bell T.; Brindle J.; Campos M.L.A.M.; Croot P.; Ducklow H.; He P.; Hoogakker B.; Hopkins F.E.; Hughes C.; Jickells T.; Loades D.; Reyes Macaya D.A.; Mahajan A.S.; Malin G.; Phillips D.P.; Sinha A.K.; Sarkar A.; Roberts I.J.; Roy R.; Song X.; Winklebauer H.A.; Wuttig K.; Yang M.; Zhou P.; Carpenter L.J.(2019). Global sea-surface iodide observations, 1967-2018. British Oceanographic Data Centre - Natural Environment Research Council, UK. doi:10/czhx.


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
.. _AC_tools_wiki: https://github.com/tsherwen/AC_tools/wiki





