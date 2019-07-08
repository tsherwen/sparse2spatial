# Iodide in sea-surface

This folder contains scripts for two separate pieces of work on sea-surface iodide. The first involved the compilation of the first dataset of all published and unpublished sea-surface observations over a period of more than 50 years. The second used this newly compiled dataset to build a high-resolution (12x12km) and monthly-resolved dataset of sea-surface iodide to be used in models, such as atmospheric models used for studying air-quality or climate.

## Compilation of observations

### Overview

Observations of sea-surface iodide gave been compiled and [archived at the British Oceanographic Data Centre (BODC)](https://doi.org/10.5285/7e77d6b9-83fb-41e0-e053-6c86abc069d0). The dataset is described in the [Chance et al (2019)](https://doi.org/10.5285/7e77d6b9-83fb-41e0-e053-6c86abc069d0) data descriptor paper, which is currently in review.

### How to update the compilation

1. Download the latest from British Oceanographic Data Centre (BODC) archive (see reference below)
1. Compile all new observations into the provided template excel file (one per observational dataset - e.g. cruise/campaign)
1. Download this Github package, install the package (see install instructions in main repository README.md), and navigate to this folder (`sparse2spatial/scripts/Iodide`)
1. Place all new datafiles in this directory
1. Run the main driver file process_new_observations.py at command line or interactively

### References

1. Data descriptor paper for sea-surface iodide observations (*in review.*)

[Chance R.; Tinel L.; Sherwen T.; Baker A.; Bell T.; Brindle J.; Campos M.L.A.M.; Croot P.; Ducklow H.; He P.; Hoogakker B.; Hopkins F.E.; Hughes C.; Jickells T.; Loades D.; Reyes Macaya D.A.; Mahajan A.S.; Malin G.; Phillips D.P.; Sinha A.K.; Sarkar A.; Roberts I.J.; Roy R.; Song X.; Winklebauer H.A.; Wuttig K.; Yang M.; Zhou P.; Carpenter L.J. Global sea-surface iodide observations, 1967-2018. *in review*, 2019](https://doi.org/10.5285/7e77d6b9-83fb-41e0-e053-6c86abc069d0)

1. Archived sea-surface iodide data at British Oceanographic Data Centre (BODC)

[Chance R.; Tinel L.; Sherwen T.; Baker A.; Bell T.; Brindle J.; Campos M.L.A.M.; Croot P.; Ducklow H.; He P.; Hoogakker B.; Hopkins F.E.; Hughes C.; Jickells T.; Loades D.; Reyes Macaya D.A.; Mahajan A.S.; Malin G.; Phillips D.P.; Sinha A.K.; Sarkar A.; Roberts I.J.; Roy R.; Song X.; Winklebauer H.A.; Wuttig K.; Yang M.; Zhou P.; Carpenter L.J.(2019). Global sea-surface iodide observations, 1967-2018. British Oceanographic Data Centre - Natural Environment Research Council, UK. doi:10/czhx.](https://doi.org/10.5285/7e77d6b9-83fb-41e0-e053-6c86abc069d0)

## Prediction of a dataset of sea-surface iodide using machine learning

### Overview

Please see the *paper in open-access review* linked below for more details on this work.

### References

+ Research paper on predicting sea-surface iodide using machine learning

[Sherwen, T., Chance, R. J., Tinel, L., Ellis, D., Evans, M. J., and Carpenter, L. J.: A machine learning based global sea-surface iodide distribution, Earth Syst. Sci. Data Discuss., https://doi.org/10.5194/essd-2019-40, *in review*, 2019.](https://doi.org/10.5194/essd-2019-40)

+ Archived NetCDF files of predicted sea-surface iodide

[Sherwen, T.; Chance, R.J.; Tinel, L.; Ellis, D.; Evans, M.J.; Carpenter, L.J. (2019): Global predicted sea-surface iodide concentrations v0.0.1. Centre for Environmental Data Analysis, 30 April 2019. doi:10.5285/6448e7c92d4e48188533432f6b26fe22. http://dx.doi.org/10.5285/6448e7c92d4e48188533432f6b26fe22](http://dx.doi.org/10.5285/6448e7c92d4e48188533432f6b26fe22)


