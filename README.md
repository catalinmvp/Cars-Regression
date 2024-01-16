# Cars-Regression
1.	Introducation

The purpose of this project is to train a machine learning alghorithm which based on a datasource P1-cars.csv containing information related to cars (mpg consumption, model, etc), buildS a regression model that predicts the fuel consumption (denoted “mpg”). Show the performance of the model and validate if linear regression can be used. Compare with other regression models https://archive.ics.uci.edu/ml/datasets/Auto+MPG . 

2. Description
The project will contain 3 different regression models, each of them having the result displayed and compared to each other.

The data set contains a number of instances equals to 398, data which will be used to train the modesl. The data set has a number of 9 attributes including the class attribute.
The attribute information which can be found in the data set is as it follows:
    1. mpg:           continuous
    2. cylinders:     multi-valued discrete
    3. displacement:  continuous
    4. horsepower:    continuous
    5. weight:        continuous
    6. acceleration:  continuous
    7. model year:    multi-valued discrete
    8. origin:        multi-valued discrete
    9. car name:      string (unique for each instance)
Also the data set contains some missing attribute values for horsepower (6 missing values)

3.	Process
The first step in the whole process is first to load the data set, enhance the data obtained and which will be used to train the model by handlind missing values and converting data to numeric whenever is the case. Following this, the data will be split between training and testing sets.
 


4.	Results:


For each model used for the regression, we obtained different result from the accuracy point of view which are as it follows:

For Linear Regression Accuracy it was obtained a 84,76%  with a Linear Regression Mean Squared Error: 8.195452104073787.

For Random Forest Regression Accuracy we obtained a 91,45% with a Random Forest Regressor Mean Squared Error: 4.595884124999997.

For SVR Accuracy we obtained 74,45% with a Support Vector Regressor Mean Squared Error: 13.73530677859128.


Lastly, based on the result obtained for each regression, we have created a plot to show the difference between the predicted MPG from the data set vs the actual data.
 

5.	Conclusions

The regression models were evaluated on their ability to predict the target variable in the dataset. Here are some conclusions based on the provided accuracy results:

Linear Regression:

The Linear Regression model achieved a moderate accuracy, suggesting a reasonable fit to the data.
Linear Regression assumes a linear relationship between the features and the target variable, and it performed adequately in capturing this relationship.

Random Forest Regression:

The Random Forest Regression model outperformed both Linear Regression and Decision Tree Regression, achieving a higher accuracy.
Random Forests are an ensemble of Decision Trees, combining their strengths and mitigating individual weaknesses, resulting in a more robust model.

Support Vector Regression (SVR):

SVR performed reasonably well but didn't surpass the accuracy of the Random Forest model.
Support Vector Regression is effective in capturing complex relationships in high-dimensional spaces, but its performance might be influenced by the choice of kernel and hyperparameters.

As final conclusion we can say that the Random Forest Regression model emerged as the top performer among the evaluated models. However, the choice of the best model depends on various factors, including interpretability, computational resources, and specific characteristics of the dataset. It's recommended to consider the overall context and requirements when selecting a regression model for a particular task.
