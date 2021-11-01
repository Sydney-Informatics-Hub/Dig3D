![GRDC Pilot App](https://github.com/Sydney-Informatics-Hub/Dig3D/blob/main/figures/Banner_Dig3D.png?raw=True)


# Machine Learning for Mapping Soil Constraints in 3D: GRDC Pilot App

This is a GRDC Pilot App for machine learning to transform sparse soil measurements and surface measurements into 3D predictions of soil properties and their uncertainties. One of the key features is the probabilistic 3D modeling (cubing), which is data-driven approach and performed via Gaussian Process Priors with a spatial 3D kernel plus multiple mean functions to take into account a diverse range of additional covariates (e.g. terrain, vegetation, top soil properties, depth). The App is built with Python (see [Mlsoil](https://github.com/Sydney-Informatics-Hub/MLsoil_GRDCapp)) and includes multiple options via a graphical user interface (GUI) or as a settings file. The four main functionalities are:
- Feature importance calculation
- Model evaluation and ranking
- Soil predictions in 3D including prediction uncertainties
- Available water capacity predictions based on soil constraints 

<!--For more model details and theoretical background, please see `docs/description_paper/paper.pdf`.-->

Author: Sebastian Haan, Sydney Informatics Hub at The University of Sydney


### Example I/O

Input: sparse soil measurements at irregular and uncertain depth plus surface measurements of multiple covariates (DEM, Rain, NDVI, Soiltype, Slope etc.)
 
Output: Soil properties (e.g., ESP, pH, EC) and their uncertainties as 3D cubes. 


## Table of Contents
- [Introduction](#introduction)
- [Functionality](#functionality)
- [Installation And Requirements](#installation-and-requirements)
	- [Executables for Windows and MacOS](#executables-for-windows-and-macos)
	- [Python Installation](#python_installation)
- [Getting Started](#getting-started) 
- [Options and Customization](#options-and-customization)
  - [Target Soil Properties](#target-soil-properties)
  - [Feature Importance Tests](#feature-importance-tests)
  - [Model Testing via Crossvalidation](#model-testing-via-crossvalidation)
  	- [Mean Model Functions](#mean-model-functions)
  - [Soil Predictions](#mean-model-functions)
  	- [Spatial Prediction Options](#spatial-prediction-options)
  - [AWC Constraints](#awc-constraints)
- [Results and Output Files](#results-and-output-files)
- [Attribution and Acknowledgments](#attribution-and-acknowledgments)
  - [Project Contributors](#project-contributors)
- [License](#license)


## Introduction
The goal of this project is to develop tools to map fine-scale 3D variability of agronomically important soil constraints, and to map the depth at which these chemical/physical barriers become limiting and impact plant available water capacity (PAWC). These data layers aim to improve prediction of crop yield variability pre- and in-season at the within-field scale, which in turn should improve input management and profitability.


## Functionality

The core features are:

- Input: sparse soil measurements at irregular and uncertain depth plus surface measurements of multiple covariates (DEM, Temperature, NDVI, Soiltype, Slope etc).
- Probabilistic prediction of 3D soil distributions:
 	 - Prediction of spatial covariance via Gaussian Process Regression (GPR) with sparse 3D kernels
	 - Predictions possible at any scale and resolution
	 - Output maps:
		- Prediction maps
		- Uncertainty maps
		- Depth constrain maps
		- Probability exceeding treshold maps
	 - Input Uncertainties taken into account: 
		 - estimation of measurement uncertainties
		 - uncertainties of measurement position (e.g. depth intervals) 
	 - Global GP hyperparameter optimisation
- Multiple options for covariate-dependent mean function of GP: 
	- Power-Transformed Bayesian Linear Regression (BLR+GP)
	- Bayesian Neural Networks (BNN+GP)
	- Random Forest (RF+GP)
 - Automatic training and evaluation of multiple models (BLR+GP, BNN+GP, RF+GP)
 	- 10-fold cross-validation
 	- residual error analysis
 	- evaluation: RMSE, R squared, predicted uncertainty accuracy
 	- ranking of models based on RMSE
 - Feature Importance Calculation
 	- Significance of Bayesian Linear Regression Coefficients
 	- Random Forest Permutation test
 - Prediction of soil constraints and available water capacity (AWC)
 - Multiple output formats:
 	- image maps
 	- csv tables
 	- geolocation-referenced tif
 - Optional support for prediction over volume rather than point predictions
 - Graphical User Interface (GUI) as well as support for Python scripting
 - Available as one-click executable App for Windows 10 and MacOS





## Installation And Requirements

### Executables for Windows and MacOS

The App is available as executable windows 10 (`Dig3D.exe`) and MacOS app (`Dig3D.app`). 
For MacOS please download Dig3d_App_MacOS, unzip the folder and click `Dig3D` in folder MacOS.
This will open the GUI to select options and to run the App.

It is also recommended to download the example data (project_example_Uah) in the same folder as the App. 
This will allow you to test the app on some example field data and to check the required input format specifications for soil data.


### Python Installation

MLsoil requires Python 3 (tested with Python 3.7)

For installation it is recommended to setup a virtual environment. This can be done, e.g., via conda:
(replace ENV_NAME with your environment name) 

```sh
conda create --name ENV_NAME python=3.7.11
```
for windows add env name to path: 
```sh
set PATH=C:\Anaconda\envs\ENV_NAME\Scripts;C:\Anaconda\envs\ENV_NAME;%PATH%
```
activate environment:
```sh
conda activate ENV_NAME
```
and install packages:
```sh
conda env update -n ENV_NAME --file environment_OS.yaml
```
To deactivate environment use `conda deactivate`.


Alternatively use virtualenv, e.g. on a mac, to enable python framework for the GUI:
```sh
pip install virtualenv
env PYTHON_CONFIGURE_OPTS="--enable-framework" python3.7 -m venv ENV_NAME
source ENV_NAME/bin/activate
pip install -r requirements_pipfreeze.txt
```

The dependencies are defined in the requirement files and are included in the repository (see `environment_macOS.yaml`, `environment_win10.yaml`, `requirements_pipfreeze_windows10.txt`)


For custom installation, the main required packages are:

- matplotlib
- seaborn
- scipy
- pandas
- numpy
- scikit-learn
- tqdm
- pyyaml
- pyvista (for 3D visualisation)
- wxpython (for gui)
- gooey (for gui)

Required geospatial libaries:

- geopandas>=0.7.0
- rasterio
- shapely
- geos
- gdal

For bayesian neural network:

- tensorflow
- tensorflow_probability



## GETTING STARTED 

The easiest way to get the App running and to test its functionalities is to start the App via the executable `Dig3D.exe`  or `Dig3D.app` (MacOS). 
This will open a GUI window with some predefined settings (here for the sample data Uah as demonstration). To familiarize yourself with the settings and input data  specifications, it is advisable to check the sample data and test multiple run options.


### Manual via Python
Once the software package and dependencies are installed, you can launch the GUI via python: 

```sh
python MLsoil/run_gui.py
```

or run the App via settings file:

1) Specify settings and filenames for input data in settings yaml file (see example settings file in repository.)

2) run script run_mlsoil.py in folder MLsoil with command line argument of name of settings yaml file, e.g.:

```sh
python run_mlsoil.py NAME_OF_SETTINGS_FILE.yaml
```



## Options and Customization

### Target Soil Properties

The app has been tested given the following soil measurements (see soil example data for Uah):

- Sodicity, i.e., Exchangeable Sodium Percentage (ESP)
- pH value (pH)
- Electrical conductivity (EC)  

Given that the underlying machine learning model is a pure data-driven approach, no prior knowledge is required as input. Thus, in principle any soil property can be modeled if measurements are provided.

### Feature Importance Tests

The importance and significance of factors (covariates) can be used for, e.g., feature selection, and multiple methods exists to determine their importance.  This App applies two methods for estimating feature importance: 1) Bayesian Linear Regression via significance given by the ratio of correlation coefficient divided by its standard deviation, 2) Random Forest permutation test (note that permutation test has multiple advantages over Random Forest impurity-based feature importance. However, permutation and impurity test assume that features are not correlated with each other). The features significance is presented in a ranked bar chart (see section Output). 

![Feature Importance Configuration](https://github.com/Sydney-Informatics-Hub/Dig3D/blob/main/figures/gui_feature-importance.png?raw=True)

### Model Testing via Crossvalidation

Multiple models are automatically tested on the input soil data and evaluated in terms of their performance (see for more details section Output). The residual analysis is performed on the validation data with a split in train/validation ratio of 90/10 and 10 cross validations. The residual errors defined as the predicted value minus the ground truth for each data point in the validation set. 
The lower the Root-Mean-Square-Error (RMSE) value, the better is the prediction. To test whether the predicted uncertainties are consistent with the residual error of the prediction, we calculate theta which is the ratio of the residual squared divided by the predicted variance.
The following three mean function models are included in conjunction to the Gaussian Process.


#### Mean Model Functions

- Bayesian Linear Regression: Before performing linear regression, the App standardizes the data and applies a feature-wise power transform scaler via scikit-learn implementation. Power transforms are a family of parametric, monotonic transformations that are applied to make data more like normal distributed. This is useful for modeling issues related to heteroscedasticity (non-constant variance), or other situations where normality is desired. In detail, the Yeo-Johnson transform is applied, which support both positive or negative data. After the feature-wise scaling of data a first Bayesian Ridge regression is performed using scikit-learn implementation `BayesianRidge`. The results of the coefficients and their uncertainty are used to select only significant features (with the ratio correlation coefficient divided by standard deviation larger than one). Then a second Bayesian Ridge regression is made using only the selected features and the final model and coefficients are stored, with non-significant coefficients set to zero. For more implementation details see `blr.py`.

- Probabilistic Neural Network: The probabilistic neural network is implemented by building a custom tensorflow probability model with automatic feature selection for sparsity. For implementation details see `bnn.py`. This method requires a feature-wise standard scaler and includes an automatic feature selection and network pruning.

- Random Forest: The scikit-learn Random Forest model implementation is applied. No data scaler required. Prediction uncertainties are currently estimates by using the standard deviation and Confidence Intervals of all decision trees. For more implementation details and hyper parameter settings see `rf.py`.

### Soil Predictions

![Cube generation Configuration](https://github.com/Sydney-Informatics-Hub/Dig3D/blob/main/figures/gui_soil-predictions.png?raw=True)

Predictions and their uncertainties are generated at any selected resolution (ideally should match resolution of the covariate grid).
The required input files are: 

- soil measurement file including the target soil property (ESP, ph, EC), xy coordinates, lower and upper depth interval of measurements, and other surface covariates such as DEM, NDVI, radiation, EM measurements.
- the grid data file that includes the covariates for the prediction
The selected features should match the feature names in the soil and grid data. An example of the data files and their specification is provided in the folder `project_example_Uah`. 

The user can select for prediction between the three models (BLR+GP, BNN+GP, RF+GP), which can be based on certain preferences (default BLR+GP) or the result of the model testing (see previous section).

The user can select the minimum and maximum threshold for soil properties (see options `Soil Thresholds`), which creates the corresponding depth constraint map for the predicted soil properties. The threshold value is also used to create maps per depth of the probability exceeding the threshold (see Output example figure).

If predictions should be averaged over a certain volume (e.g., to reduce prediction uncertainty) rather than point predictions (default setting), the user can choose in the settings `Optional Volume Averaging` the prediction type `Volume` and the corresponding size of the volume block (horizontal and vertical resolution) for averaging. The volume averaging method takes into account spatial covariance between points for predicting the volume-averaged mean and uncertainty values. 

![Soil Threshold Configuration](https://github.com/Sydney-Informatics-Hub/Dig3D/blob/main/figures/gui_soil-thresholds.png?raw=True)
![Volume Averaging Option](https://github.com/Sydney-Informatics-Hub/Dig3D/blob/main/figures/gui_volume-averaging.png?raw=True)

### AWC Constraints

The estimation of the available water capacity (AWC) integrated over depth is based on the depth constrain maps for the corresponding soil target, which are produced as part of the soil prediction output (see previous section). The required input files are:

- AWC Data: a cube of the available water in .tif format, where each band represents the available water capacity at 1 cm depth intervals
- The depth constraint maps that are generated as part of the 3D soil predictions.

![Volume Averaging Option](https://github.com/Sydney-Informatics-Hub/Dig3D/blob/main/figures/gui_awc_constraints.png?raw=True)


## Results and Output Files

In this use case scenario, the following soil measurements are used (see data in `project_example_Uah`):

- soil measurements of ESP, pH, and EC at positions x,y (Easting, Northing) and upper and lower depth, including a range of surface covariates extracted for the x,y positions (e.g. DEM_30, 'SlopeDeg', 'rad_dose', 'rad_k', 'NDVI_5',  'NDVI_50', 'NDVI_95', 'EM_top','EM_sub', 'RED_5','RED_50','RED_95').
- surface covariate grid data of the same covariates at grid positions x,y (here at 10 m resolution)
- optional: available water capacity (AWC) file: .tif file with multiple bands, where each band represents the available water capacity (in mm) at 1 cm depth intervals.

Some of the main results are shown below for feature-importance, model selection, soil predictions, and AWC-Constraints.

### List of Output files

Results feature importance:

- Feature_Importance_RF_permutation.png (Random Forest Permutation Importance)
- Feature_Significance_linearBLR.png (Bayesian Linear Regression significance)
- Feature_Significance_powerBLR.png (Bayesian Linear Regression significance with power-law scaled input data)

Results model crossvalidation:

- ESPnfold_summary_stats.csv (summary table)
- Xvalidation_Residual_hist_ESP.png (histogram of combined residual and theta)
- for each cross-validation set (default 10, here for  BLR and ESP):
	- Residual_hist_ESP_nfold1.png (Residual Error Histogram)
	- pred_vs_trueESP_nfold1.png (Prediction vs Truth)
	- Hist_ESP_train.png (Histogram of error for training data)
	- ESP_train.png (ESP subtracted by mean model)
	- ESP_results_nfold1.csv (summary stats table)
	- ESP_residualmap.png (map of residual error)
	- ESP_BLR_pred_vs_true.png (Prediction vs Truth for mean function model only)

Results for soil predictions:

- Depth_ConstrainESP10.tif (geo-referenced tif for depth constraint map)
- Depth_Constrain_SigmaESP10.tif (geo-referenced tif for uncertainty of depth constraint map)
- Depth_Constrain_Sigma ESP10.png (map of depth constraint and uncertainty)
- Pred_ESP_coord_y.txt (cartesian y coordinates for prediction)
- Pred_ESP_coord_x.txt (cartesian x coordinates for prediction)
- Pred_ESP_mean.png (Mean of predicted soil property along vertical axis with data location points)
- Pred_ESP_mean2.png (Mean of predicted soil property along vertical axis with value colored data-points as overlay)
- for each depth slice:
	- Pred_ESP_zxxxcm.png (Prediction map of soil property and uncertainty)
	- Pred_ESP_zxxxcm.tif (geo-referenced tif of prediction map)
	- Std_ESP_zxxxcm.tif (geo-referenced map of standard deviation of prediction)
	- Pred_ESP_zxxxcm.txt (values of prediction matching x,y coord location file)
	- Pred_Stddev_ESP_zxxxcm.txt (values of prediction uncertainty matching x,y coord location file)
	- Prob_exceedingESP10_zxxxcm.png (Map of probability exceeding threshold)
	- Prob_exceedingESP10_zxxxcm.png (geo-referenced tif file of probability exceeding threshold)

Results for AWC constraints:
- Depth_SoilCombined-Constraint.png (map of combined soil constraints)
- Depth_SoilCombined-Constraint.tif (geo-referenced tif of combined soil constraints)
- Depth_AWC_SoilCombined-Constraint.png (map of integrated AWC given soil constraints)
- AWC-SoilCombined-Constraint.tif (geo-referenced tif of AWC given soil constraints)
- Depth_AWC_NoConstrain.png (map of integrated AWC w/o soil constraints)
- Combined_AWC-NoConstrain.tif (geo-referenced tif of integrated AWC w/o soil constraints)
- Depth_AWC_ESP-Constraint.png (map of integrated AWC given soil constraints per soil property type, e.g. ESP)
- AWC-ESP-Constraint.tif (geo-referenced tif of integrated AWC given soil constraints per soil property type, e.g. ESP)

![Example feature importance results for ESP. The left panel shows the BLR significance given by the ratio of correlation coefficient to standard deviation. The right panel shows the results of the Random Forest permutation importance.](https://github.com/Sydney-Informatics-Hub/Dig3D/blob/main/figures/Uah_feature-importance.png?raw=True)

![Example cross-validation model ranking for ESP after 10-fold cross-validation run.](https://github.com/Sydney-Informatics-Hub/Dig3D/blob/main/figures/gui_modeltest-results.png?raw=True)

![Overview plots for one cross-validation set out of ten](https://github.com/Sydney-Informatics-Hub/Dig3D/blob/main/figures/Uah_xval-overview-perfold.png?raw=True)

![Example BLR cross-validation results for ESP for 10 cross-validation sets. The columns are defined as Root-Mean-Squared-Error (RMSE), nRMSE (RMSE/standard deviation of training data), Root-Median-Squared-Error (RMEDIANSE), Theta (the mean of ratios of the squared error to predicted variance) for the complete model and indicate with suffix `_fmean` for the results of mean function only ](https://github.com/Sydney-Informatics-Hub/Dig3D/blob/main/figures/Uah_table-xval-ESP.png?raw=True)

![Example residual plot for cross-validation results for ESP. Shown are the residual error for test data (difference between model prediction and data that was unseen by model) at the top and theta (mean ratio of the squared error to predicted variance) at the bottom.](https://github.com/Sydney-Informatics-Hub/Dig3D/blob/main/figures/Uah-xval_Residual_hist_ESP.png?raw=True)

![Results maps (top) and uncertainty (bottom) of ESP predictions at depths of 20, 40, and 60 cm.](https://github.com/Sydney-Informatics-Hub/Dig3D/blob/main/figures/Uah_ESP-pred.png?raw=True)

![Probability maps of exceeding soil threshold ESP=10 at depths of 20, 40, and 60 cm depths.](https://github.com/Sydney-Informatics-Hub/Dig3D/blob/main/figures/Uah_ESP-prob.png?raw=True)

![Example result overview of AWC soil constraint maps.](https://github.com/Sydney-Informatics-Hub/Dig3D/blob/main/figures/Uah_AWC-results.png?raw=True)



## Attribution and Acknowledgments

This software was developed by the Sydney Informatics Hub, a core research facility of the University of Sydney, as a machine learning pilot App for GRDC.

Acknowledgments are an important way for us to demonstrate the value we bring to your research. Your research outcomes are vital for ongoing funding of the Sydney Informatics Hub.

If you make use of this software for your research project, please include the following acknowledgment:

“This research was supported by the Sydney Informatics Hub, a Core Research Facility of the University of Sydney.”

The software development has been funded by the Grains Research & Development Corporation (GRDC) as part of the project for Machine learning to map soil constraint variability and predict crop yield.

### Project Contributors

Key project contributors to this project are:

 - Dr. Sebastian Haan (Sydney Informatics Hub, The University of Sydney): 
 	Main developer and machine learning scientist 
 - Prof. Brett Wheelan (Sydney Institute of Agriculture, The University of Sydney): 
 	Project lead of GRDC project
 - Dr. Liana Pozza (Sydney Institute of Agriculture, The University of Sydney): 
 	Project research associate, software testing and data mapping
 - Dr. Patrick Filippi (Sydney Institute of Agriculture, The University of Sydney): 
 	Project research associate, data extraction and mapping


## License

Copyright 2021 Sebastian Haan, The University of Sydney

This is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License (LGPL version 2.1) as
published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program (see LICENSE.md). If not, see
<https://www.gnu.org/licenses/>.

<!-- Convert to pdf and docx: -->
<!-- pandoc -V geometry:margin=1in README.md -o docs/MANUAL.pdf -->
<!-- pandoc -V geometry:margin=1in README.md -o docs/MANUAL.docx -->