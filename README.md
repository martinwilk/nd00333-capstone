*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Capstone Project for Machine Learning Engineer on Microsoft Azure

This is the final project of my Udacity Machine Learning Engineer nanodegree course. I will compare the performance of an AutoML model to an hyperparameter-optimized XGBoost classification model. For this comparison, I use the red and the white wine quality dataset and concatenate these dataset, so I get a dataset containing information from red and white wines.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
In this project I use both wine quality datasets which are available in the UCI Machine Learning data repository. One dataset only contains information about white wines from Portugal, the other dataset consists of red wines. To use both datasets for the classification, I add the wine type as an additional feature to both datasets and concatenate them using the pandas concat method.

The dataset contains physiochemical measurements like the pH-value, the amount of sugar or the percentage of alcohol for each wine along with a sensory rating ranging from 3 to 9 and the information whether the wine is white or red.

I convert the original ratings to a discrete feature with the values BAD, MEDIUM and GOOD using panda's cut method.

### Task
The goal of this project is to predict the quality grade (BAD,MEDIUM,GOOD) of a wine. To archive this, we use the 11 attributes from the dataset and the information about the wine type, which was added to the dataset.
### Access
The data are accessed by loading them using the from_delimited_files method of the TabularDatasetFactory class.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
