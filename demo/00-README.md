# Clinical transformer Run Demonstration
In this capsule we display the usage of the clinical transformer for: 
* Survival prediction
* self supervised learning
* fine tunning of SSL model on survival analysis. 
* Exploration of results

* We are using a **toy dataset** generated using simulations of time to event outcomes in two populations. We split the data into pretrain, train and test. Pretrain data is used to pretrain while the train and validation splits are uset to train and test the survival analysis task, with and without transfer learning.  

These notebooks serve as a starting point for running the clinical transformer. 
* **01-Dataset**. This notebook contains the logic for generating the toy dataset. 
* **02-FoundationModel**: This notebook shows how to pre-train the clinical transformer
* **03-Baseline**: This notebook shows how to perform survival analysis using the clinical transformer. 
* **04-TransferLearning**: This notebook shows you how to fine-tune the pretrained SSL model for survival analysis. 
    * For survival analysis we use 30% of the data as internal validation and it is used to visualize the model learning across iterations.
* **05-ModelPerformance**: A simple script comparing the concordance index of the baseline vs fine-tuned models on the internal validation dataset.
* **06-ModelExploration**: Few analysis done in top of the pre-trained and finetuned models. In particular, the survival analysis, and feature importance.


In this toy example: 
* We show how to run the clinical transformer for survival analysis.
* Build "Foundation Models" using the clinical transformer API. 
* Use the "Foundation Model" for fine-tuning on survival analysis.

The term "Foundation Model" is used as an illustration of the capabilities of the clinical transformer for building large models using self-supervised learning. 


Clinical transformer code is available under /environment/. Note that we also included PySurvival code for generating synthetic data. 