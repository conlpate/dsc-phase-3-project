
# Is There a Problem with Mental Health in US Police Response? 

![Wapo 2020](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/wapo%20header.png)

![Mental Health Stats](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/header%202.png)


## Introduction

"*Defund the Police*" is not a new rallying cry. From [The Guardian, June 2020](https://www.nytimes.com/2020/06/05/your-money/houses-prices-coronavirus.html), "For years, community groups have advocated for defunding law enforcement - taking money away from police and prisons - and reinvesting those funds in services." They estimate the movement has existed for **five years or more.** Perhaps stemming from the shooting of Michael Brown in Ferguson. [Timeline of Events in Ferguson, AP News, 2019](https://apnews.com/article/9aa32033692547699a3b61da8fd1fc62).

However, since the murder of George Floyd, the call to "Defund the Police" has increased in both frequency and volume. As quoted above, rarely does the movement advocate for a dismantling of police departments; rather, a reallocation of funds toward necessary services.

Shortly after the aforementioned shooting of Michael Brown and the emergence of "Defund the Police" to the mainstream, *The Washington Post* began tracking fatal police shootings in the U.S starting in 2015 ([Washington Post Database Overview](https://www.washingtonpost.com/graphics/investigations/police-shootings-database/?itid=lk_inline_manual_3)).

Perhaps one of the sharpest illustrations of a need to broaden, re-prioritize, defund, whatever one wants to call it, came with the death of Daniel Prude.

![daniel prude](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/daniel%20prude%20treated%20like.png)


[On Daniel Prude, NYTimes, Oct 2020](https://www.washingtonpost.com/graphics/investigations/police-shootings-database/?itid=lk_inline_manual_3)

Daniel Prude, a 41 year old man in Rochester, NY, was hooded and restrained on the ground while suffering a mental episode. After an investigation into his death, the officers on the scene were cleared of any misconduct.

While Daniel Prude's name is not in *The Washington Post's* database, as he was not shot, his death is a tragic illustration of fatal force used against those suffering from a mental illness.

## Objectives

- Using *The Washington Post's* police shooting database, analyze the associated variables with police shootings involving a person suffering from a mental illness.
  - Per *The Washington Post's* breakdown of their dataset, their variable, "signs of mental illness," covers, **"News reports [that] have indicated the victim had a history of mental health issues, expressed suicidal intentions or was experiencing mental distress at the time of the shooting"**
- Build a comprehensive classification model predicting whether or not a person is suffering from mental distress when shot by police.
- Present findings and solutions.

## Steps

1. Import necessary datasets (*Washington Post* and U.S. regions)
2. Clean and scrub the dataframes
3. Explore the datasets
4. Visualize and analyze relevant mental illness variables
5. Build and compare models
6. Tune Best Performing Models

*For clarity and comprehension, this readme will focus primarily on exploration, visualization, and modeling.* The attached notebook follows a CRISP-DM approach.

### Questions, Analysis, and Visualizations

1. What categorical data provides the best insight into true/false signs of mental illness?
   1. How do these interact with each other?
   2. What can we learn from interactions?
2. Which continuous variables (primarily location and age data) provide the best insight into true/false signs of mental illness?
   1. How do these interact with each other?
   2. How do our continuous variables shine light on our categorical data?
3. How can we best visualize the categorical and continuous data to illustrate our findings?

#### Areas of Focus and Analysis
### **Gender**

![Gender/ MI/age](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/signs%20gender%20mi.png)


Comparing the two genders (*this dataset does not account for those who identify as non-binary or transgender*), 1/3 of women fatally shot by police were reported having/displaying signs of mental illness (MI signs)

![gender/MI/comparison ](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/Titled%20Age%20Dist.png)

Unlike men, who seem to peak (RE: MI signs) earlier, **earlier (20s-40s)**, there is a **consistent distribution of MI signs across age brackets for women.**

### **Race**

![race/MI](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/signs%20of%20mental%20illness%20race%20pd.png)



**Asian Americans** and **White (non-Hispanic)** Americans show a consistent trend, with the former peaking closer to middle age and above (>45).

![MI/age](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/Race%20Dist%20Title.png)

**Black** and **Hispanic Americans** seem to have a higher rate of mental illness peaking at middle age (<40).

**Black women** and **White (non-Hispanic) women** are likelier than other ethnicities to display MI signs.

![race age gender](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/Mental%20Illness%20Dist%20Titled.png)

### **Location**

![Signs of mental illness by Division](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/signs%20of%20mi%20division%20pd.png)

The **Middle Atlantic Division** (New Jersey, New York, and Pennsylvania) contains the highest amount of fatalities with signs of mental illness.

**New England** (Conn., ME, MA, NH, RI, and VT) follows the Mid Atlantic Division in positive MI signs.

**East South Central** and the **Mountain** Divisions contain the lowest amount.  


*Are we looking at a population density issue? A higher confirmed rate in urban areas? A combination of the two?*

![mental illness division plot](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/Division%20MI%20titled.png)

New England’s suspected cases of mental illness peak into middle age (>40), while the Mountain and W. South Central divisions peak in the 20s and 30s.

The Mid Atlantic Division, perhaps due to its share of cases, has a steady line throughout age brackets.

![region gender](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/Region%20MI%20titled.png)


**We need to be aware of gender distribution among regions and divisions.**

### Modeling

![Model Overview](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/model%20overview%20title.png)

#### Where to Go?

Before diving into model building, a cursory overview of scores will give some insight into how our data is performing.

Due to the distribution of scores for LogReg, our RFC (random forest classifier), and our DTC (decision tree classifier), we'll start there. 

#### Methods of Model Analysis

**For each model, we'll use the following steps to build and analyze performance:**

1. Assign Test/Train Data to Model
2. Check Test/Train Performance
3. Confusion Matrix
4. ROC/AUC scores 
5. Investigate scores 

**For our DTC and RFC models, we'll use a gridsearch to tune our results.**

#### LogReg

![ROC](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/Logreg%20Roc.png)

- Checking our mean AUC score gives us: 
![AUC](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/logreg%20auc%20mean.png)
	- This is not ideal - less than 20% better than a coin toss at a 50% AUC score. 

**Checking other metrics**

![CM lG](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/Logreg%20CM.png)

- Highly imbalanced. As we proceed in tuning, we will abandon LogReg in favor of DTC and RFC.. 

![logreg scores](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/Logreg%20Scores.png)

- Our overall scores are decent, but the model is lacking in analysis for MI True.
- Accuracy for our **test set** is 75%.
- As we start to see our data is imbalanced, we will rely on our macro F1 score. Our average is just over 50% and its MI True percentage is abysmal. 

#### Decision Tree Classifier

![DTC CM](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/Initial%20DT%20CM.png)

- We will need to tune our data more thoroughly. **Highly imbalanced.**

![DTC scores](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/DT%20and%20logreg%20first.png)

- Our model is failing entirely at this point. It is not predicting MI True cases in the slightest. 


#### Random Forest Classifier

-Based on our preliminary analysis, we'd expect our RFC to perform in a similar fashion as our DTC and LogReg. In some ways, yes. It outperforms the DTC model, but still struggles with MI True scores. 

![RFC CM](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/first%20RFC%20.png)

- We're seeing a consistent lack of balance across our models. 

![RFC scores](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/first%20RFC%20with%20other%20scores.png)

- At this moment, our RFC is performing in a similar fashion as our LogReg model. 

## Best Models (FS and SMOTE)

#### Decision Tree Classifier (Best with SMOTE and FS)

![bestDTC](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/best%20DTC.png)

- After using feature selection prior to our gridsearch and oversampling our target class, our DTC model is doing much better. 

![bestDTCsc](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/best%20DTC%20and%20methods.png)

- Compared to the other methods of tuning we've employed on our DTC models, our best scores are achieved using the above methods. While our accuracy has gone down, our balance issues have been addressed; however, more can be done. We're still underperforming with our target class. 
- **Accuracy**: 69%
- **F1 Score**: 57%

#### Random Forest Classifier (Best with SMOTE, no FS)

![rocrfcsmote](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/best%20RFC%20auc.png)

- As we can see, an initial examination of our RFC SMOTE model gives us our best AUC score to date. 
- Checking the mean, we get an even higher score. 

![rocrfcmean](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/rfc%20auc.png)

- This is miles ahead of our previous AUC scores. While our best performing model isn't the model with the highest AUC score (more below), it's good to note our RFC is doing much better. 

![rfccm](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/best%20rfc%20CM.png)

- Much better! More balanced all around. Still not ideal. 

![rfcscor](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/best%20RFC%20and%20methods.png)

- As we can see, our RFC model with SMOTE (no FS) is giving us our best metrics. 
- **Accuracy**: 67%
- **F1 Score**: 60%

## Analysis and Future Steps

#### Analysis
1. Logistic Regression
    - Accuracy: 75%
    - F1 Score (macro average): 54%
    - **Highly imbalanced**, not a good fit for this data. 
2. Decision Tree Classifier with FS, Gridsearch, and SMOTE
    - Accuracy: 69%
    - F1 Score (macro average): 57%
3. Random Forest Classifier with Gridsearch, SMOTE, no FS
    - Accuracy: 67%
    - F1 Score (macro average): 60%
    

    
#### Future Steps 
1. Expand data. 
    - While adding in regions and divisions has eased project visualization, the added data doesn't address our imbalance issues. We need more data on casualties suffering from mental illness. 
    - Expand the time range? 
    - Build a new dataset taking information from local and state datasets? 
2. Balance and Fit
    - Oversampling our target class has improved our metrics, but could more be done by undersampling our features?
    - Across the board, our F1 scores could be improved. Is this possible with our current data? 
    - Our best performing model, RFC with SMOTE (no FS) is currently overfitting. More work needs to be done to minimize fit issues. 
3. Model Efficiency 
    - At this moment, our models take quite a bit of time to run. Using a proportioned selection of the data will improve processing time and, depending on the method, may assist with our sampling issue. 
    - This is especially true running a gridsearch for the DTC and RFC models. 
4. ROC/AUC
    - Need to further explore ROC/AUC scores for DTC and RFC. Need to further visualize the mean AUC scores of our high performing RFC models. 
    
#### Notes on the Models
- While sampling has fixed much of our initial issues, our models are still imbalanced. We need to consider whether the provided data can accurately and comprehensively answer our initial question. 
- A threshold across models should be put in place for future modeling. Set metrics for underperforming, average, and exceptional models to ease future tuning issues and questions. 



## Business Deliverables
![Defund](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/defund.png)

1. Reallocation of funds for crisis response should be funneled toward those most in need. Young men, women of all ages, Asian Americans, Black women. 
2. Different regions require different support strategies. Larger urban areas could benefit from a more nuanced crisis approach. 
3. More data is needed on: rural areas, gender identification (non-binary, transgender), funding for crisis calls, non-fatal injuries. 
    
