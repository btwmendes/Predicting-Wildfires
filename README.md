# Predicting-Wildfires

This project was developed by Brian Mendes. Please see the individual notebooks to follow the processes.

## Table of contents
### I. [Background](#Background)
### II. [Datasets used](#Datasets)
### III. [Processing](#Processing)
### V. [Exploratory analysis](#Exploratory-analysis)
### VI. [Modeling](#Modeling)
### IV. [Unsupervised Learning Experiments](#Unsupervised Learning Experiments)
### VII. [Conclusion](#Conclusion)

## Background

The promise of the internet was to democratize information. News was no longer controlled by institutions. An individual could start a blog and broadcast a message to the world. With this leveling of the playing field came unforeseen consequences. Traditional news sources lost revenue streams, fired journalists and lost control of editorial review. Media became decentralized and moved online. As a result, “news” was driven by clickoncomics rather than journalistic principles. Content providers needed to attract attention vis-à-vis clicks and eyeballs by any means necessary. Terms such as “truthiness”, “infotainment” and “fake news” captured the spirit of this transformation of news in the information age. The dilemma we face in today’s world is distinguishing between facts and deception. We need help curating the content to cut through the noise and obtain reliable information. Due to the sheer volume of content produced each day, computers and machine learning could be tools that help us accomplish this goal. 

## Datasets

The datasets are from Kaggle.com and can be accessed from the following link:
https://www.kaggle.com/rtatman/188-million-us-wildfires

### Original data:
The website has a SQlite file called "FPA_FOD_20170508.sqlite". The file is 758.92 MB and holds over 1.88 million rows of wildfires.  The website has a detailed data dictionary that includes the context, explanation of the columns, and sources. 

### Data used in final models:

 | Column | Data type | Description | 
 | --- | --- | --- |
 | fire_year | int64 | The year the fire started |
 | discovery_doy | int64 | The consecutive day of the year the fire was discovered. For example, February 9th is day 40. |
 | discovery_day_of_week | int64 | The day of week the fire was discovered. Monday is 0 and Sunday is 7 |
 | discovery_month | int64 | The month the fire was discovered. January is 1. |
 | fire_size | int64 | Estimate of acres within the final perimeter of the fire. |
 | latitude | int64 | Latitude (NAD83) for point location of the fire (decimal degrees). |
 | longitude | int64 | Longitude (NAD83) for point location of the fire (decimal degrees). |



## Processing

Libraries necessary for this section: 

```
import pandas as pd
import sqlite3
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
```
### a. [Importing](./code/I-Import-Data.ipynb)

The dataset can be found at Kaggle: https://www.kaggle.com/rtatman/188-million-us-wildfires

The data is saved as a sqlite file, which requires the "sqlite3" library to extract the data into a readable format for python. I extracted the data and saved it to a csv copy. 

### b. [Cleaning](./code/II-Data-Cleaning.ipynb)

While the dataset included fires across the U.S., the scope of this project was for California. As such, I filtered the dataset to California only which reduced the number or rows from 1.88 million to about 200,000. Additionally, I reduced the features to cover the time, location and identifying information for each fire. The other columns were not relevant to the model. I also checked and changed data types from objects to integers or date/time. The results were saved to a csv file for modeling. 
 

## [Exploratory analysis](./code/III-EDA.ipynb)

Libraries necessary for this section:
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
```

The goal of the exploratory data analysis was to get a general understanding of the data and find observations that could help the modeling. I started by loading the “california.csv” file, which only included wildfire data located in California. The dataset has about 200,000 rows and started with 23 columns. Checking for null values, I noticed that 10
 out of the 23 features have null values. The categories with missing values are fire and complex name, discovery time, any information on containment, and county. It makes sense that not all fires have a name or are part of a complex. Additionally, not all fires will have an exact discovery time or a definitive containment time. Small fires do not require this granular information. The top three causes of fires are miscellaneous, equipment use, and lightning. Unfortunately, the data dictionary does not describe examples of miscellaneous causes. By plotting the location of each fire by cause it appeared that the cause of the fire did not predict its location. All fires were distributed throughout the state, except for fires caused by railroads, structures and children. The dataset showed that wildfires show a pattern of seasonality. June, July and August accounted for the most fires in the dataset, while the winter months showed the least activity. This was consistent for every cause category as well. The top ten largest wildfires by acres burned have occurred after the year 2000. 



## [Modeling](./code/V-Modeling.ipynb)

Libraries necessary for this section:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
```

### Baseline Model

The baseline model is the percentage of the majority class. This is the percentage the model needs to beat. When looking at the fire causes, Miscellaneous is the majority class at 27.4%. When evaluating the more generic labels (natural, accidental, malicious, other), the baseline score is 41.3%.

### Logistic Regression

The model accuracy and cross validation scores are about 34% for both the training and testing set. This is about 7 percentage points better than the baseline model which has a score of 27%. Since the training and testing scores are essentially the same, I am not concerned about high variance error. However, the model is not very good at predicting the cause of a wildfire.
In an attempt to create a more accurate model, I consolidated the target classes. My theory is that the model does not have enough data in order to discern between 13 classes nor is the logistic regression model strong at handling 13 classes. As a result, I consolidated the causes of wildfires into the following four classes: natural, accidental, malicious and other. This simplification of the classes shows an improvement of the accuracy scores. The model accuracy and cross validation scores increased to about 50% for both the training and testing set. While it's only 9 percentage points better than the baseline, overall it is a higher accuracy rate.
This change also helps with interpretability, especially when looking at the confusion matrix. The model best predicts accidental fires (66%), other fires (47%), and natural fires (44%). A weakness in the model is that it views malicious fires and other fires as accidental 70% and 47% of the time respectively. In fact, the model does not predict one act of arson. This reveals a short coming of the model.

### KNN

The KNN model shows improvement over Logistic Regression. Following the same format as logistic regression, I tested the KNN model on 13 classes and 4 classes. When considering the accuracy scores, both models appear to suffer from high variance bias, meaning the model is overfit to the training data and will not generalize to new data. The cross validation scores are much closer together but show a lower accuracy rate. The accuracy scores are better than the baseline but not by much.

When evaluating 13 classes, the KNN model correctly predicted fires caused by lightning (80%), miscellaneous (52%), equipment use (38%) and arson (33%). This model does a much better job of predicting arson than the logistic regression even if it's only a 33% accuracy rate. When evaluating 4 classes, the KNN model correctly predicts natural (81%), accidental (62%), other (42%) and malicious (31%). The model's strength is predicting lighting fires. The model's weakness is predicting arson, which incorrectly is assigned to accidental 42% of the time. Similarly, 47% of other causes are also categorized as accidental.

### Random Forest

The Random Forest model shows improvement over Logistic Regression and KNN, with one major caveat. Following the same format as before, I tested the Random Forest model on 13 classes and 4 classes. When considering the accuracy scores, both models appear to suffer from high variance bias, meaning the model is overfit to the training data and will not generalize to new data. The training data has a virtually perfect score while the testing score is at least 40 percentage points lower. The cross validation scores are much closer together and show a lower, and most likely more correct, accuracy rate. The accuracy scores beat the baseline better than the other two models. The 4 class Random Forest appears to be the winning model with a cross validated score of about 60%.
The confusion matrix reveals what we sacrifice for a higher score. The model predicts that all wildfires are caused by "other". This approach does not make the model useful.

### Recommended Use

**The best and most useful model is the KNN model.**

Though the Random Forest model was the most accurate it comes at a great cost. It predicts that every fire is caused by "other", which incorrectly predicts every other category. The Logistic Regression model competes with the KNN model by posting comparable accuracy scores; however, it cannot predict arson. 

The KNN model does a better job predicting all categories. The model can best be used by focusing on its strengths - predicting lightning caused fires. As stated above, the KNN model predicted 80% of the naturally occurring fires. This could give investigators a high level of assurance if they need to verify a naturally occuring fire. Conversely, this model should not be relied upon to predict arson, but rather could be one input into their investigation.

All model scores are listed below:

| Model            | Baseline Score | Accuracy Score - Train | Accuracy Score - Test | CrossValScore - Train | CrossValScore - Test |
|------------------|----------------|------------------------|-----------------------|-----------------------|----------------------|
| Logistic Regression - 13 Classes | 0.274          | 0.340                  | 0.342                 | 0.340                 | 0.344                |
| Logistic Regression - 4 Classes  | 0.413          | 0.500                  | 0.500                 | 0.500                 | 0.495 
| KNN - 13 Classes | 0.274          | 0.600                  | 0.420                 | 0.340                 | 0.344                |
| KNN - 4 Classes  | 0.413          | 0.683                  | 0.546                 | 0.497                 | 0.495                |
| Random Forest - 13 Classes | 0.274          | 0.999                  | 0.489                 | 0.483                 | 0.457                |
| Random Forest - 4 Classes  | 0.413          | 0.999                  | 0.614                 | 0.607                 | 0.582                |


## Unsupervised Learning Experiments

The object of this clustering analysis is to see if any interesting relationships exist in the wildfire dataset. Unlike supervised learning, in unsupervised learning there is no target to predict but rather we look for cateogries from the structure in our data. This approach is often used in marketing research.

My methodology includes the following steps:
- Standardize the data
- Consolidate the features using principal component analysis
- Determine the optimal number of K clusters using the elbow method
- Instantiate a K Means model
- Observe the results with a scatter plot
- Evaluate the result with a cluster report

### a. [Clustering A](./code/VI-Unsupervised-Learning-Clustering-A.ipynb)

This notebook has the purpose to run a model with 2 PCA features and 6 K clusters.

### b. [Clustering B](./code/VI-Unsupervised-Learning-Clustering-B.ipynb)

This notebook has the purpose to run a model without PCA and instead use all the scaled features.

### c. [Clustering C](./code/VI-Unsupervised-Learning-Clustering-C.ipynb)

This notebook has the purpose of trying three PCA features instead of two and comparing all three models.

**Comparing the models:**

Out of the three unsupervised clustering model, Model C appears to be the best. As the table below shows, Model C has the highest silhouette score, uses the least number of clusters and captures 80% of the variance with 3 PCA features. These factors make for cleaner and easier to interpret clusters 

| Clustering Model Version | # of PCA Features | Variance Explained by PCA | # of K CLusters | Silhoutee Score |
|--------------------------|-------------------|---------------------------|-----------------|-----------------|
| A                        | 2                 | 60%                       | 6               | 0.358           |
| B                        | 0                 | n/a                       | 6               | 0.232           |
| C                        | 3                 | 80%                       | 4               | 0.366           |



**Interpreting Model C:**

The pair plot and cluster report are excellent resources that help us understand the clusters. The K means model created 4 clusters which were primarily categorized by location and time of fire. The scatter plot with latitude and longitude as the x and y axis shows a good visual of cluster 1 and 3. California is bisected by a line running roughly from Santa Barbara to Kings Canyon. Cluster 1 is located in the southern region and Cluster 3 is located in the northern region. The cluster report confirms this observation by detailing the rules of each cluster. The 58,838 fires in Cluster 1 are 100% categorized by their location. As a result, Cluster 1 includes old and new fires as shown in the "fire_year" and "latitude" plot. 

Cluster 0 is primarily defined by a westward longitude position and the day of discovery depending on the year. Fires before 2003 are more welcome into Cluster 3, which is composed of fires from May to December. However, fires between 2003 and 2009 are included only from September to December.

Like Cluster 1, Cluster 3 is also primarily defined by location but also adds nuances such as the year and discovery day. Cluster 3 can be thought as the other side of the coin to Cluster 0. As noted above, Cluster 0 included fires later in the year. Conversely, Cluster 1 includes the fires earlier in the year. The "discovery_doy" and "latitude" plot clearly shows this separation.

**Cluster Summary:**
- Cluster 0: These are fires before 2003 and discovered in the second half of their year. They are primarily located in the north and west parts of the state.
- Cluster 1: These fires are all located in southern California. Location is its only attribute.
- Cluster 2: This is a very small cluster of only 40 fires that did not fit with the other clusters. They represent the largest fires.
- Cluster 3: These are primarily fires located in northern California that were discovered earlier in the year.



## Conclusion

In the end, we were able to create a highly accurate classification model to differentiate between real and fake news. We are skeptical that we'd get similar results when applied to news stories from outside of this Kaggle dataset but believe we found some significant patterns nonetheless. This bodes well for social media platforms where news articles are shared prolifically; with more time and subject matter expertise, we believe machine learning can be an incredible tool in the battle against misinformation. 



