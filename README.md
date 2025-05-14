This repository contains the main code for the TR DIMA: "Zooplankton classification using long-tailed dataset techniques."

Hybrid-SC: Based on "Contrastive Learning based Hybrid Networks for Long-Tailed Image Classification" by Wang et al. This code implements the prototypical version of the method described in the paper.

Logit Compensation: Based on "Long-tail learning via logit adjustment" by Menon et al. We implement only the loss version of logit adjustment.

Two-Stage Training: Based on "Decoupling Representation and Classifier for Long-Tailed Recognition" by Kang et al.

Data_cleaning_5_folds.ipynb: Code for the five-fold data cleaning strategy. Warning: The output is a .txt file listing all suspect data.

Norm_weights_resnet50.ipynb: Observes the correlation between the norm of the weight vectors in the classifier layer and the cardinality of the corresponding class.




