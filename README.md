 **Project Overview: Predicting Friction Factor (f) and Bed Load Transport Rate (B) in Alluvial Channels Using Machine Learning**

This project aims to predict two critical parameters for sediment transport and hydraulic behavior in alluvial channels: the **friction factor (f)** and **bed load transport rate (B)**. Accurate prediction of these parameters is essential for designing sustainable hydraulic structures and understanding sediment dynamics in riverine systems.


**1. Data Preprocessing**

The workflow begins with **data preprocessing** using pandas, NumPy, seaborn, and matplotlib:

* The dataset `dataset_with_f_B.csv` is loaded.
* Non-numeric entries are coerced to numeric types.
* Missing values are imputed using **mean substitution**.
* Outliers are handled by **capping at the 1st and 99th percentiles**, based on boxplot visualizations.
* Finally, **feature scaling** is performed using `StandardScaler` from `sklearn`, which normalizes the data to improve model convergence and performance.

A **correlation heatmap** is generated using Seaborn to visualize relationships between features and identify multicollinearity, aiding in feature selection and interpretation.


 **2. Model Development & Evaluation**

 a. **XGBoost Regressor (MultiOutputRegressor)**

A robust model is built using **XGBoost**, wrapped with `MultiOutputRegressor` to handle simultaneous prediction of both `B` and `f`. The model is trained using an 80-20 train-test split. Key settings include:

* 500 estimators
* 0.05 learning rate
* max depth = 6

Performance is evaluated using **Mean Squared Error (MSE)** and **R² score**, with **scatter plots** of true vs. predicted values for visual inspection.

To assess generalizability, a **5-fold cross-validation** using `KFold` is implemented manually. It shows consistent R² values across folds, ensuring the model's robustness on unseen data.


 b. **Random Forest & Linear Regression**

To benchmark XGBoost, **Random Forest Regressor** and **Linear Regression** models are also applied separately for `f` and `B`:

* Models are trained and tested using the same split and scaling.
* Performance metrics (MSE and R²) indicate that **Random Forest outperforms Linear Regression**, especially for complex, nonlinear relationships in hydraulic behavior.

This comparison helps validate the use of ensemble tree-based models for more accurate hydraulic predictions.

 **3. Result Interpretation and Engineering Implications**

The evaluation metrics and prediction plots across all models demonstrate that:

* **XGBoost delivers superior accuracy and generalizability**, especially when using cross-validation.
* **Random Forest** performs well, though slightly below XGBoost.
* **Linear Regression** offers a basic benchmark but struggles with the nonlinearity in data.

The successful prediction of `f` and `B` from hydraulic variables confirms the feasibility of **data-driven modeling** in sediment transport studies, offering a viable alternative to purely empirical or analytical formulations. These insights can support more efficient river engineering, sediment management, and flood modeling.


 **Project Artifacts**

* `preprocessed_dataset.csv` – final dataset used for training
* `improved_correlation_matrix.png` – visual correlation analysis
* `XGBoost`, `Random Forest`, and `Linear Regression` code – core model implementations and evaluations

