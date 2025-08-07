# biosensor-colorimetric-photothermal
Data files for analysis using colorimetric and photothermal sensing  

**"Enhancing Sensitivity of Commercial Gold Nanoparticle-Based Lateral Flow Assays: A Comparative Study of Colorimetric and Photothermal Approaches"**

## Python Scripts

### 1. 'ML_with_lasso.py'
- Implements three regression models (linear and polynomial with LASSO, sigmoid regressino with L1 loss function)
- Calculates McFadden's pseudo R squared, AIC, and BIC

### 2. 'ML_without_lasso.py'
- Implements similar models as 'ML_with_lasso.py' without LASSO regularization
- Includes bootstrapping for confidence intervals, can be applied for 'with_lasso.py' as well

### 4. 'line_analysis.py'
- Used for analyzing line intensity extracted from assay images

### 4. 'speckle_analysis.py'
- Analyze photothermal speckle images using FFT 

## Folders

### 1. 'augmented_data'
- Includes processed RGB ratio for each Salmonella concentration used in machine learning regression analysis
- Each file corresponds to the concentration
- Columns: 
    - 'Red Ratio', 'Green Ratio', 'Blue Ratio': intensity values extracted from line intensity analysis
    - 'Brightness', 'Contrast', 'Temperature': refers to changed parameter values
    - 'Target': log-scale concentrations

### 2. 'colorimetric_images'
- Includes three sample sets: 'Sample1', 'Sample2', 'Sample3'
- Each sample set contains give assay images labeled by concentration
- Images used to extract line intensity 

### 3. 'photothermal_data'
- Contains example of photothermal data used (one of the test samples out of three)
- Used for FFT-based signal analysis 
