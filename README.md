
# The Problem with Mental Health in US Police Response

![Wapo 2020](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/wapo%20header.png)

![Mental Health Stats](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/header%202.png)


## Introduction

"*Defund the Police*" is not a new rallying cry. Per [The Guardian, June 2020](https://www.theguardian.com/us-news/2020/jun/05/defunding-the-police-us-what-does-it-mean), "For years, community groups have advocated for defunding law enforcement - taking money away from police and prisons - and reinvesting those funds in services." They estimate the movement has existed for five years or more, increasing after the fatal shoooting of Michael Brown in 2014.

Perhaps one of the clearest examples in recent years of a need to reassess police funding and services came with the death of Daniel Prude in October, 2020. 

![daniel prude](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/daniel%20prude%20treated%20like.png)


[On Daniel Prude, NYTimes, Oct 2020](https://www.washingtonpost.com/graphics/investigations/police-shootings-database/?itid=lk_inline_manual_3)

Daniel Prude, a 41 year old man in Rochester, NY, with a history of mental illness, died of asphyxia after being hooded and physically incapacitated by local police officers. At the time of his arrest and subsequent restraint, Daniel was suffering from a mental health episode, triggered by the ingestion of PCP. Daniel's death was ruled a homicide by the medical examiner conducting his autopsy. 

As calls to defund the police gained traction, *The Washington Post* began tracking fatal police shootings in the U.S starting in 2015 ([Washington Post Database Overview](https://www.washingtonpost.com/graphics/investigations/police-shootings-database/?itid=lk_inline_manual_3)).

While Daniel Prude's name is not in *The Washington Post's* database, as he was not shot, his death is a tragic illustration of fatal force used against those in most need of an alternative, crisis-oriented response. 

# Table of contents

- [Objectives and Steps](#objectives)
- [EDA](#eda)
    - [Gender](#gender)
    - [Race](#race)
    - [Location](#location)
- [Model Analysis, RFC](#modeling)	
    - [Vanilla](#vanilla)
    - [Final](#final)
- [Recommendations](#recommendations)

# Objectives

- Provide law enforcement agencies, their representatives and  other relevant bodies the necessary data to understand and contextualize the relationship between fatal shootings and mental illness.  
- Using *The Washington Post's* police shooting database, analyze fatal police shootings involving a person suffering from a mental illness and their associated variables.
  - Per *The Washington Post's* breakdown of their dataset, their variable, "signs of mental illness," covers, **"News reports [that] have indicated the victim had a history of mental health issues, expressed suicidal intentions or was experiencing mental distress at the time of the shooting"**
  - Build a binary classification model to predict mental illness presence among police shooting fatalities. 


### Modeling and Notebook Approach

1. Import necessary datasets (*Washington Post* and U.S. Regions)
2. Clean and scrub
3. Explore the data
4. Visualize and analyze relevant mental illness variables
5. Build and compare models
6. Present model and recommendations

[Washington Post Github Database](https://github.com/washingtonpost/data-police-shootings)


### Questions, Analysis, and Visualizations

1. What categorical data provides the best insight into true/false signs of mental illness?
   1. How do these interact with each other? What can we learn from these interactions? 
2. Which continuous variables (primarily location and age data) provide the best insight into true/false signs of mental illness?
   1. How do these interact with each other?
   2. How do our continuous variables align with our categorical data?
3. How can we best visualize the categorical and continuous data to illustrate our findings and provide recommendations?

# EDA 
## **Gender**

![Gender/ MI/age](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/signs%20gender%20mi.png)


Comparing the two genders (*this dataset does not account for those who identify as non-binary or transgender*), 1/3 of women fatally shot by police were reported having/displaying signs of mental illness (MI signs)

![gender/MI/comparison ](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/Titled%20Age%20Dist.png)

Unlike men, who seem to peak (RE: MI signs) earlier, **earlier (20s-40s)**, there is a **consistent distribution of MI signs across age brackets for women.**

## **Race**

![race/MI](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/signs%20of%20mental%20illness%20race%20pd.png)



**Asian Americans** and **White (non-Hispanic)** Americans show a consistent trend, with the former peaking closer to middle age and above (>45).

![MI/age](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/Race%20Dist%20Title.png)

**Black** and **Hispanic Americans** seem to have a higher rate of mental illness peaking at middle age (<40).

**Black women** and **White (non-Hispanic) women** are likelier than other ethnicities to display MI signs.

![race age gender](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/Mental%20Illness%20Dist%20Titled.png)

## **Location**

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

# Modeling

## Vanilla


![ConfusionMatrixVan](https://github.com/conlpate/dsc-phase-3-project/blob/main/images/vanilla%20model.png)

### After looking at our vanilla random forest classifier's confusion matrix, one thing becomes apparent: 
	- Our initial model, highly imbalanced, cannot predict true cases of mental illness. 

#### Where to Go?

As we want to minimize type II errors (inaccurate classification of mental illness), the classifier needs to primarily focus on false negatives. With that in mind, we'll iterate through multiple models, all while focusing on increasing recall. 

#### Methods of Model Analysis

**For each model, we'll use the following metrics to analyze performance:**

1. Confusion Matrix
2. Mean AUC
3. Train/Test Sccore

## Final

![SMOTECM](https://github.com/conlpate/dsc-phase-3-project/blob/main/images/RFCsmote%20model.png)

### Insights 

Our final model has a few added and altered parameters compared to the vanilla model, but the primary increase in performance can be attributed to oversampling the minority class via SMOTE. 

In the attached notebook, you'll see various random forest classifiers fit with feature importance, hyperparameter tuning and a combination of the two with SMOTE. However, the best performing model with the highest recall score was also, as mentioned above, the simplest. 

![SMOTECM](https://github.com/conlpate/dsc-phase-3-project/blob/main/images/CM:AUC.png)

![roc/auc](https://github.com/conlpate/dsc-phase-3-project/blob/main/images/ROC%20Curve%20Mean%20AUC.png)

**Our SMOTE RFC model's recall score is twenty points higher than the next best model. Furthermore, our SMOTE RFC model's mean AUC score is tied with the other best performing models.** 

### Continuing Data Analysis and Model Improvement

1. Expand timeframe, region information.
    - While adding in regions and divisions has eased project visualization, the added data doesn't address the lack of comprehensive mental health/mental illness data across disparate regions. We need more data on casualties suffering from mental illness. 
    - Expanding the collection timeframe may show trends/spikes that a 5 year period cannot. 
2. Combine local/state/federal databases indicating a mental illness presence, regardless of fatalities. 
    - Are there commonalities across other types of police response aside from fatal shootings? Would we see the same indicators in arrests or non-lethal police responses? 


# Recommendations

![Defund](https://github.com/conlpate/dsc-mod-3-project-v2-1-onl01-dtsc-pt-052620/blob/master/images3/defund.png)

1. **Young men, women of all ages, Asian Americans, and Black women all indicate a higher mental illness presence in fatal encounters with the police.** These demographics need to be cross checked against other mental illness variables present in police shootings: **wielding toy weapons and a stationary response (not fleeing).**  	 	
2. **Different regions may require different support strategies.** The aforementioned demographics need to be tested explicitly across both rural and urban environments. 
3. **More data is needed on:** rural areas, gender identification (non-binary, transgender), regional funding for crisis calls, fatal injuries not involving firearms (e.g., Daniel Prude). 

If the above items are found consistent across rural and urban areas, response teams should be aware of the variables most closely aligned with mental illness or a mental health episode. This could be accomplished through enhanced training or a reallocation of resources toward crisis response teams. Furthermore, as a preventative step, outreach should be focused on the aforementioned demographics, those most likely to have signs of mental health issues. 
    
