# Capstone Project for Machine Learning Engineer on Microsoft Azure

This is the final project of my Udacity Machine Learning Engineer nanodegree course. I will compare the performance of an AutoML model to an hyperparameter-optimized XGBoost classification model. For this comparison, I use the red and the white wine quality dataset and concatenate these datasets, so I get a dataset containing information from red and white wines.


## Dataset

### Overview
In this project I use both wine quality datasets which are available in the UCI Machine Learning data repository. One dataset only contains information about white wines from Portugal, the other dataset consists of red wines. To use both datasets for the classification, I add the wine type as an additional feature to both datasets and concatenate them using the pandas concat method.

The dataset contains physiochemical measurements like the pH-value, the amount of sugar or the percentage of alcohol for each wine along with a sensory rating ranging from 3 to 9 and the information whether the wine is white or red.

I convert the original ratings to a discrete feature with the values BAD, MEDIUM and GOOD using panda's cut method with the following scheme.

below 4 => "BAD"
between 4 and 6 => "MEDIUM"
7 and above => "GOOD"

### Task
The goal of this project is to predict the quality grade (BAD,MEDIUM,GOOD) of a wine. To achieve this, we use the 11 attributes from the dataset and the information about the wine type, which was added to the dataset. As a performance metric the AUC_weighted metric is used.

### Access
The data are accessed by searching the key "wine-quality" in the Workspace's dataset attribute. If the dataset is found in the AzureML Studio it is loaded from that resource. Otherwise the datasets for white and red wines are loaded using the from_delimited_files method of the TabularDatasetFactory class. In the clean_data function the TabularDataset are converted into pandas dataframes and the wine type is added as a new feature. Now the dataframes are concatenated and  registered as a TabularDataset in Azure Machine Learning Studio.

## Automated ML
To configure an AutoML run, an AutoMLConfig object is created. To start the AutoML run that config object is submitted as an argument of the experiment's submit method. In the following, I will describe my AutoML settings.
During my AutoML experiment, a timeout is raised after 30 minutes. This is set by the parameter experiment_timeout_minutes of the AutoMLConfig constructor. This is done to limit the training duration.
The other settings are made in a dict which is passed to the constructor as an keyword argument. Using this dictionary the number of cross validations is set to 3. This means that the dataset is partitioned into three parts. Two of these parts are used to train the model, the other part is used to test the trained model. This procedure is executed three times.

As a primary metric I use the AUC_weighted metric because this metric is less influenced by class imbalance. As you could see in the output of the AutoML run displayed in the jupyter notebook, the dataset is imbalanced, because only 246 out of 6491 wines got a bad rating.  

Because I have provisioned a compute cluster with 6 nodes, I set the number of maximum concurrent iterations to 5. This number should be less or equal to the maximum amount of core of the compute cluster.

Next I enable early stopping to save compute time, if more iterations don't lead to an improvement of the performance metric. In this case, the child run is cancelled. During my AutoML experiment I don't use deep learning to train a model, which is the default. By setting the parameter enable_dnn, deep learning models are trained on the dataset. I have decided not to use deep learning techniques in my project, as they need more data to achieve superior performance compared to standard models.


### Results
Like in the other Udacity projects, the best model returned by AutoML is a VotingEnsemble model.
The classification algorithms involved in the VotingEnsemble are XGBoost, LightGBM and logistic regression. The predictions by each model are combined using a optimized set of weights. In the following I will provide some information about the parameters of the classification algorithms XGBoost and LightGBM. XGBoost uses gradient boosting to optimize decision trees. By training new models based on the errors of prior trained models, they obtain a great performance. The parameters of the XGBoost and logistic regression models are optimized during the training process.  More on the exact parameters used by the VotingEnsemble could be found in the jupyter notebook. The best model has an AUC_weighted of 0.8___.

![Screenshot Widget]()
![best_model_run]()


*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
![Screenshot Widget HD]()
![best_model_run_hd]()
*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
The deployed model was created by AutoML. To query the endpoint with a sample input, I used the requests module. The sample input has to be provided as a json serialized dict containing the data under a data key. Each instance is another dict in the array which is the value corresponding to the data key. The sample data is serialized using the dumps method of the json module.

Because I activated authentication for my REST endpoint an authentication header with the string Bearer and the primary key which was created when the endpoint was created has to be included. The URI of the endpoint, the header containing authentication data and the json payload should be passed to the post method of the requests module. This sends a POST request to the endpoint. The endpoint will provide a prediction for the instances to be scored and return it as a json string to the client.

## Screen Recording
You will find my screencast video under https://...
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
