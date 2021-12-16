# AML-Assignment-
Applied Machine Learning Systems ELEC0134 (21/22) Assignment 
 

## Abstact
Early detection and discrimination of brain tumors remain a radiological challenge, both misdiagnoses and diagnostic at late-stage result in a decreased of survival rate. Despite the different brain imaging techniques and other examinations that help specialists in decision-making process, diagnostic error in cancers is frequent therefore, improving diagnostic accuracy is needed. Deep Learning is a sub field of machine learning has shown a remarkable performance on classifying and detection brain tumors using MRI images. This paper focuses on identification the abnormal growth in brain and classifying brain tumors into three main types: meningioma, glioma, and pituitary tumor using two deep-learning mod- els. The first type of classification models is binary classifica- tion, two distinct classes are used (Tumor, No tumor) during the training phase. The system applies the transfer learning techniques and uses a fine-tuning VGG16 model. The pro- posed system shows a reliable performance and achieves a classification accuracy of 97%. The second type of classi- fication models is multi-classification, four classes (menin- gioma,glioma,pituitary,and no tumor) are the training sam- ples for the second model. Using an improved Fine-tunning VGG16 in end-to-end training, this task accomplished by at- tained an accuracy up to 94.33%. The paper introduced so- lutions to handle issues that are related to the training dataset such as: limited and imbalanced dataset, class weights tech- nique and data augmentation were introduced in this work.


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
