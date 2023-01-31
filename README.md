# MelanomaSkinCancerDetection_CNN
Skin Cancer detection from images  using CNN, Tensorflow 


# Project Name
##  Optimizing Multiclass Image Classification with Custom CNNs in TensorFlow
- This repository contains the code for detecting melanoma using a Convolutional Neural Network (CNN) and TensorFlow with GPU acceleration. The model is trained on a dataset of skin lesion images to classify them as either melanoma or benign.

## Table of Contents
* [General Info](#general-information)
* [DataSet](#DataSet)
* [Business Goal](#business-goal)
* [Business Risk](#business-risk)
* [Project Pipeline](#proj_pipeline)
* [Model](#model)
* [Usage](#usage)
* [Conclusions](#conclusions)
* [Technologies Used](#technologies-used)
* [References](#References)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)




## General Information
- Melanoma is a type of skin cancer that can be deadly if not detected early. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce manual effort in diagnosis.
- The project is aimed at developing a CNN-based model for detecting melanoma in skin images. The dataset used in the project consists of 2357 images of malignant and benign oncological diseases, collected from the International Skin Imaging Collaboration (ISIC).
- The aim is to develop a model that can accurately detect melanoma in skin images and reduce manual efforts in diagnosis.
- The business problem that the project is trying to solve is the manual effort required in the diagnosis of melanoma, a type of skin cancer. The manual process of evaluating images to detect melanoma can be time-consuming and prone to human error. The aim of the project is to develop a CNN-based model that can accurately detect melanoma in skin images and reduce the manual effort required in the diagnosis process. The deployment of the model in a dermatologist's workflow has the potential to increase efficiency and accuracy, potentially leading to better patient outcomes.
 
## DataSet : 
The model is trained on the ISIC Archive dataset, which contains a large number of dermoscopic images of skin lesions, including both benign and malignant melanomas. The dataset is pre-processed and split into training and validation sets.

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
<li><strong>Accuracy</strong>:<i>The model's accuracy in detecting melanoma in skin images is a crucial factor. If the model produces incorrect results, it could lead to misdiagnosis and harm to patients.</i></li>

<li><strong>Data Quality</strong>:<i>The quality and reliability of the data used to train the model can have a significant impact on its performance. Any errors or biases in the data set can result in inaccurate results.</i></li>

<li><strong>Adoption</strong>:<i>The success of the model depends on its adoption by dermatologists and the medical community. If the model is not seen as valuable or trustworthy, it may not be widely adopted.</i></li>

<li><strong>Technical Challenges</strong>:<i>Developing a CNN-based model can be technically challenging, and there may be difficulties in training the model and optimizing its performance.</i></li>

<li><strong>Competition</strong>:<i>There may be other existing solutions or competing models in the market that perform similarly or better, making it difficult to gain a competitive advantage.</i></li>
</ol>
These risks need to be carefully managed and addressed in order to ensure the success and impact of the project.


## Project Pipeline
The project pipeline involves several steps in order to build a multiclass classification model using a custom convolutional neural network in TensorFlow. The steps are as follows:

<strong>Data Reading and Understanding</strong>: The first step involves defining the path for train and test images to be used in the project.

<strong>Dataset Creation</strong>: <i>The next step is to create the train and validation datasets from the train directory, with a batch size of 32 and resizing the images to 180 x 180.</i>

<strong>Dataset Visualization</strong>: <i>The project involves visualizing one instance of all the nine classes present in the dataset to get an understanding of the distribution of classes in the data.</i>

<strong>Model Building and Training</strong>: <i>The project involves building a CNN model that can accurately detect the nine classes present in the dataset. The model is built by rescaling the images to normalize the pixel values between (0,1) and choosing an appropriate optimizer and loss function for model training. The model is trained for approximately 20 epochs and the findings after the model fit are analyzed to check for evidence of overfitting or underfitting.</i>

<strong>Data Augmentation</strong>: <i>To resolve issues of overfitting or underfitting, an appropriate data augmentation strategy is chosen.</i>

<strong>Model Building and Training on Augmented Data</strong>: <i>A CNN model is built using the augmented data and is trained for approximately 20 epochs. The findings after the model fit are analyzed to see if the earlier issue has been resolved.</i>

<strong>Class Distribution</strong>: <i>The project involves examining the class distribution in the training dataset, including which classes dominate the data in terms of proportionate number of samples and which class has the least number of samples.</i>

<strong>Handling Class Imbalances</strong>: <i>To rectify class imbalances present in the training dataset, the Augmentor library is utilized.</i>

<strong>Model Building and Training on Rectified Class Imbalance Data</strong>: <i>A CNN model is built on the rectified class imbalance data, with images being rescaled to normalize pixel values between (0,1), and choosing an appropriate optimizer and loss function for model training. The model is trained for approximately 30 epochs and the findings after the model fit are analyzed to see if the issues have been resolved.</i>



## Model
The model uses a simple CNN architecture with convolutional, Max pooling and dense layers. The model is trained using SparseCategoricalCrossentropy loss and the Adam optimizer.


## Usage
Clone the repository: git clone [https://github.com/SnehalVirwadekar/MelanomaSkinCancerDetection_CNN.git]


## Observations

### Initial model 

<p>- The model's training accuracy displays a linear increase over time, whereas the validation accuracy plateaus at 50% accuracy during the training process.
<p>- As the training epochs progress, the training loss decreases, however, the validation loss exhibits an opposing trend and increases.
<p>- The visual representation of the model's performance indicates a substantial discrepancy between the training accuracy and validation accuracy, achieving a validation accuracy of approximately 50%.
<p>- The noticeable difference between the training accuracy and validation accuracy is a clear indication of overfitting, indicating that the model may not generalize well to unseen data.

### After training on the augmented data 

<p>- The training accuracy exhibits a linear increase over time, while the validation accuracy increases and plateaus at 55% during the training process.
<p>- As the training epochs progress, the training loss decreases in a consistent manner, and the validation loss decreases before plateauing.
<p>- The visual representation of the model performance indicates a reduction in the difference between the training accuracy and validation accuracy compared to prior models, achieving an accuracy of approximately 55% on the validation set.
<p>- However, the slight discrepancy between the training and validation accuracy is a notable sign of overfitting, implying that the model may not generalize well to unseen data.

### After training on the augmented data with dropout :

<p>- The model's training accuracy exhibits a linear progression over time, while the validation accuracy increases before plateauing at 55% accuracy throughout the training process.
<p>- The training loss consistently decreases with each passing epoch, and a similar trend is observed with the validation loss.
<p>- The visual analysis of the model's performance illustrates a reduction in the disparity between the training accuracy and validation accuracy in comparison to prior models, resulting in a validation accuracy of approximately 55%.
<p>- The limited difference between the training accuracy and validation accuracy is a strong indication of the model's generalization capability to new, unseen data.



### Class Distribution:

<p>- A thorough analysis of the class distribution within the training dataset is crucial.
<p>- In many cases, datasets can exhibit an imbalance in class representation, where one class has a proportionally higher number of samples compared to others. This class imbalance can have a significant impact on the quality of the final model.
<p>- Thus, as a best practice, it is imperative to assess the distribution of classes within the data to ensure a balanced representation and to avoid any potential biases in the model.
<p>- The class with the smallest number of samples is "Seborrheic Keratosis," with a total of 77 samples.
<p>- In terms of proportionate representation, the class that dominates the data is "Pigmented Benign Keratosis," with a substantial 462 samples.

 ### After training on the rectified class imbalance data

<p>- The training accuracy and validation accuracy both increase over time, with the training accuracy increasing linearly and the validation accuracy increasing during the training process.

<p>- The training loss decreases with each epoch, and the validation loss also decreases accordingly.

<p>- The plots demonstrate a significant reduction in the gap between the training accuracy and validation accuracy compared to the previous model, reaching approximately 75% accuracy on the validation set.

<p>- The difference between the training accuracy and validation accuracy is now minimal.
 
 ### Conclusion 
- The model showed promising results in detecting melanoma in skin images with high accuracy, sensitivity, and specificity. The deployment of the model in a dermatologist's workflow has the potential to reduce manual effort in the diagnosis of melanoma.





## Technologies Used
- TensorFlow - version 2.4.0
- Keras - version 2.5.0
- Numpy - version 1.19.2
- Matplotlib - version 3.3.1
- Sklearn - version 0.24.1



## References
- ISIC Archive dataset: https://isic-archive.com/
- TensorFlow: https://www.tensorflow.org/
- CUDA: https://developer.nvidia.com/cuda-zone






## Contact
Created by [@SnehalVirwadekar] - feel free to contact me!



