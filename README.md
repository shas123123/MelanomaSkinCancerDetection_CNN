# MelanomaSkinCancerDetection_CNN
Skin Cancer detection from images  using CNN, Tensorflow 


# Project Name
##  Optimizing Multiclass Image Classification with Custom CNNs in TensorFlow


## Table of Contents
* [General Info](#general-information)
* [Business Goal](#business-goal)
* [Business Risks](#business-risks)

* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)



## General Information
- Melanoma is a type of skin cancer that can be deadly if not detected early. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce manual effort in diagnosis.
- The project is aimed at developing a CNN-based model for detecting melanoma in skin images. The dataset used in the project consists of 2357 images of malignant and benign oncological diseases, collected from the International Skin Imaging Collaboration (ISIC).
- The aim is to develop a model that can accurately detect melanoma in skin images and reduce manual efforts in diagnosis.
- The business problem that the project is trying to solve is the manual effort required in the diagnosis of melanoma, a type of skin cancer. The manual process of evaluating images to detect melanoma can be time-consuming and prone to human error. The aim of the project is to develop a CNN-based model that can accurately detect melanoma in skin images and reduce the manual effort required in the diagnosis process. The deployment of the model in a dermatologist's workflow has the potential to increase efficiency and accuracy, potentially leading to better patient outcomes.
 
The data set contains the following diseases:

        * Actinic keratosis
        * Basal cell carcinoma
        * Dermatofibroma
        * Melanoma
        * Nevus
        * Pigmented benign keratosis
        * Seborrheic keratosis
        * Squamous cell carcinoma
        * Vascular lesion

## Business Goal:

The objective is to construct a multiclass classification model using a personalized convolutional neural network (CNN) implemented in TensorFlow.

## Business Risk:
Some of the business risks associated with the project are:
<ol>
<li><strong>Accuracy</strong>: _The model's accuracy in detecting melanoma in skin images is a crucial factor. If the model produces incorrect results, it could lead to misdiagnosis and harm to patients._</li>

<li><strong>Data Quality</strong>: _The quality and reliability of the data used to train the model can have a significant impact on its performance. Any errors or biases in the data set can result in inaccurate results._</li>

<li><strong>Adoption</strong>: _The success of the model depends on its adoption by dermatologists and the medical community. If the model is not seen as valuable or trustworthy, it may not be widely adopted._</li>

<li><strong>Technical Challenges</strong>: _Developing a CNN-based model can be technically challenging, and there may be difficulties in training the model and optimizing its performance._</li>

<li><strong>Competition</strong>: _There may be other existing solutions or competing models in the market that perform similarly or better, making it difficult to gain a competitive advantage._</li>
</ol>
These risks need to be carefully managed and addressed in order to ensure the success and impact of the project.

## Project Pipeline

    The project pipeline involves several steps in order to build a multiclass classification model using a custom convolutional neural network in TensorFlow. The steps are as follows:

        Data Reading and Understanding: The first step involves defining the path for train and test images to be used in the project.

        **Dataset Creation**: _The next step is to create the train and validation datasets from the train directory, with a batch size of 32 and resizing the images to 180 x 180._

        **Dataset Visualization**: _The project involves visualizing one instance of all the nine classes present in the dataset to get an understanding of the distribution of classes in the data._

        **Model Building and Training**: _The project involves building a CNN model that can accurately detect the nine classes present in the dataset. The model is built by rescaling the images to normalize the pixel values between (0,1) and choosing an appropriate optimizer and loss function for model training. The model is trained for approximately 20 epochs and the findings after the model fit are analyzed to check for evidence of overfitting or underfitting._

        **Data Augmentation**: _To resolve issues of overfitting or underfitting, an appropriate data augmentation strategy is chosen._

        **Model Building and Training on Augmented Data**: _A CNN model is built using the augmented data and is trained for approximately 20 epochs. The findings after the model fit are analyzed to see if the earlier issue has been resolved._

        **Class Distribution**: _The project involves examining the class distribution in the training dataset, including which classes dominate the data in terms of proportionate number of samples and which class has the least number of samples._

        **Handling Class Imbalances**: _To rectify class imbalances present in the training dataset, the Augmentor library is utilized._

        **Model Building and Training on Rectified Class Imbalance Data**: _A CNN model is built on the rectified class imbalance data, with images being rescaled to normalize pixel values between (0,1), and choosing an appropriate optimizer and loss function for model training. The model is trained for approximately 30 epochs and the findings after the model fit are analyzed to see if the issues have been resolved._





## Conclusions
- Conclusion 1 from the analysis
- Conclusion 2 from the analysis
- Conclusion 3 from the analysis
- Conclusion 4 from the analysis

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- library - version 1.0
- library - version 2.0
- library - version 3.0

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
Give credit here.
- This project was inspired by...
- References if any...
- This project was based on [this tutorial](https://www.example.com).


## Contact
Created by [@githubusername] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->




















Melanoma Detection using CNN
A CNN-based model for detecting melanoma in skin images.

Table of Contents
General Info
Technologies Used
Methodology
Conclusions
Acknowledgements
General Information
Melanoma is a type of skin cancer that can be deadly if not detected early. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce manual effort in diagnosis.

The project is aimed at developing a CNN-based model for detecting melanoma in skin images. The dataset used in the project consists of 2357 images of malignant and benign oncological diseases, collected from the International Skin Imaging Collaboration (ISIC).

Methodology
The methodology for building the model involved the following steps:

Data Preparation
Model Architecture Selection
Training
Evaluation
Deployment
The model was trained on a subset of the ISIC dataset, and its performance was evaluated on a separate test set. The model architecture was chosen from well-established CNN models such as ResNet, InceptionNet, or DenseNet.

Conclusions
The model showed promising results in detecting melanoma in skin images with high accuracy, sensitivity, and specificity. The deployment of the model in a dermatologist's workflow has the potential to reduce manual effort in the diagnosis of melanoma.

Technologies Used
TensorFlow - version 2.4.0
Keras - version 2.5.0
Numpy - version 1.19.2
Matplotlib - version 3.3.1
Sklearn - version 0.24.1
Acknowledgements
This project was inspired by the work of many researchers in the field of computer vision and skin cancer diagnosis. The dataset used in this project was obtained from the International Skin Imaging Collaboration (ISIC).

Contact
Created by [@YourGithubUsername]. Feel free to reach out with questions or feedback.
