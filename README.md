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

An interesting improvement to the project is to enable the training of deep learning models just to test whether they could outperform the VotingEnsemble model. Another room of improvement deliver the parameters blocked models or allowed models or a manual feature engineering on the dataset.

*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
I chose the XGBoost classification algorithm to perform the outlined task, because these models often have good performance. XGBoost is an abbreviation for eXtreme Gradient Boosting. Gradient Boosting is an ensemble technique. This means that you aggregate the predictions from so-called base learners (here: decision trees). Because of the aggregation of their predictions the error rate will be reduced compared to the error rate from a single decision tree.
Another positive property of XGBoost is its speed.

The XGBoost classification algorithm has many important hyperparameters like the learning rate, sampling size and the parameters for the decision tree like the maximum depth or a lower bound for the loss reduction incurred by a new split of the decision tree. During my experiments, I will optimize the four hyperparameters below:
|Parameter | description | range |
|----------|------------ |-------|
|Maximum Depth | depth of decision tree | random integer between 3 and 8|
|----------|------------ |-------|
| alpha | L1 regularization on weights of XGBoost | random between 0.1 and 10 |
|----------|------------ |-------|
| Gamma | minimal split loss required for next split| between 0.01 and 5 |
|----------|------------ |-------|
| Learning rate | learning rate used to optimize weights | random between 0.05 and 0.25 |
|----------|------------ |-------|

The maximum depth hyperparameter refers to the maximum depth of the decision trees involved in the algorithm. Gamma is another hyperparameter of a decision tree: a split of a leaf of the decision tree is only performed, if the loss is reduced by gamma by splitting the tree.

The next hyperparameter I would like to optimize is alpha, which provides a L1 regularization to the weights. This regularization of the weights is used to prevent overfitting. Another important parameter I will use is the learning rate. The learning rate is a parameter used in the optimization problem to find the right weights for the ensemble learner.

I use BayesianParameterSampling in my experiments because BayesianParameterSampling is able to learn from prior runs which could lead to better hyperparameters. Because I use this sampling strategy a Policy to cancel runs is not required. I use AUC_weighted as a metric to evaluate the performance. To run my train.py script which is used to train my XGBoost model, a SKLearn estimator is used. The estimator object is passed to the HyperdriveConfig object along with a max_total_runs of 80 and a maximum amount of concurrent runs of 5. I chose 80 as the maximum number of runs because if you specify less than 20 * {number of hyperparameters} when using the Bayesian sampling strategy, you will get a warning that you should allow at least 20 * {number of hyperparameters} runs.
The HyperdriveConfig is submitted to a new experiment and the hyperdrive runs are executed automatically.   

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
![Screenshot Widget HD]()
![best_model_run_hd]()
*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
The deployed model was created by AutoML. To query the endpoint with a sample input, I used the requests module. The sample input has to be provided as a json serialized dict containing the data under a data key. Each instance is another dict in the array which is the value corresponding to the data key. The sample data is serialized using the dumps method of the json module.

Because I activated authentication for my REST endpoint an authentication header with the string Bearer and the primary key which was created when the endpoint was created has to be included. The URI of the endpoint, the header containing authentication data and the json payload should be passed to the post method of the requests module. This sends a POST request to the endpoint. The endpoint will provide a prediction for the instances to be scored and return it as a json string to the client. In the screenshot below you could find a sample input along with the response by the deployed model's endpoint.

## Screen Recording
You will find my screencast video under https://...
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
