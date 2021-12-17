# AML-Assignment-
Applied Machine Learning Systems ELEC0134 (21/22) Assignment 
 

## Abstact
Early detection of brain tumours remain a radiological challenge, both misdiagnoses and late-stage diagnosis result in a decreased survival rate. Despite the various brain imaging techniques and other examinations that help specialists in the decision-making process, diagnostic error in cancers is frequent therefore, there is a need to improve diagnostic accuracy.Deep Learning, a sub-field of machine learning showed a notable performance on classifying and detecting brain tumoursusing MRI images. This paper focuses on identification of the abnormal growth in the brain and classifying brain tumours into three main types: meningioma, glioma, and pituitary tumours using two deep-learning models.The first type of classification model is the binary classification. The system applies the transfer learning techniques and uses a finetuning VGG16 model which achieves a classification accuracy of 97%.The second type of classification model is multiclassification to classify the MRI images into 4 classes, finetuning
VGG16 in an end-to-end training is proposed. The model records a classification accuracy of 94.33%.The paper introduced solutions to handle issues that are related to the
training dataset(limited and imbalanced dataset) such as:class weights and data augmentation techniques.

## Model

![test](/images/Model_Architecture.PNG)

Model Architecture: 

A for binary classification architecture
B for multi-classification architecture

## Prerequisites

To run the code, you need to install the following dependencies:
* <a href="https://www.tensorflow.org/"> Tensorflow </a>.
* <a href="https://keras.io"> Keras </a>. 
* NumPy.
* Matplotlib.
* Pandas_ml.
* Sklearn.

## Data
The dataset used in this paper is given in two parts, and it was
taken from the Brain MRI Images for Brain Tumour Detection
dataset from the Kaggle site.It consists of a collection
of 3000 512x512 pixel gray-scale MRI images and that
are divided into Tumour and No tumour classes for binary
classification, and four types of classes that are no tumour,
meningioma, pituitary, and glioma for multi-classification.The second part will be used
to evaluate the performance of the proposed models, a separate
test set with 200 images.

## To run

- Notebook:  <code> Brain_tumour_project_final.ipynb </code> this is the final notebook which contains all the experimental results.
- Python package: run <code> python main.py </code> to run all the models.
