![CANPL](https://pbs.twimg.com/profile_images/1191405045788676097/vk_lsh7F_200x200.jpg)

[CANPLES Heroku App LINK](https://canples.herokuapp.com/index)

## canpl-es
A (WIP) expert system to analyse the satistics and results of the Canadian Premier League

Currently using GaussianNB() and RandomForestClassifier() to make the match predictions between teams as games are listed in the schedule.

### How Match Prediction is Deployed
This app is using the following models (due to size limitations) for predictions;
- [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [Bagging Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
- [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

Although the size contraints did play a role in choosing the final models, the Bagging Classifier and Random Forest Regressor did perform very well. If not equally well to the other top performing models listed below.

#### Comparison Results of Classifiers

| results | knn | adaboost | bagged | random forest | voting classifier |
| --- | --- | --- | --- | --- | --- |
|count | 10 tests | 10 tests | 10 tests | 10 tests | 10 tests |
| mean | 0.348000 |	0.4140 |	0.432000 |	0.422000 |	0.454000 |
| std |	0.048259 |	0.0924 |	0.084958 |	0.035839 |	0.073666 |
| min |	0.280000 |	0.3000 |	0.300000 |	0.360000 |	0.360000 |
| 25% |	0.320000 |	0.3300 |	0.400000 |	0.400000 |	0.405000 |
| 50% |	0.340000 |	0.4100 |	0.430000 |	0.430000 |	0.440000 |
| 75% |	0.375000 |	0.4600 |	0.475000 |	0.455000 | 0.525000 |
| max |	0.440000 |	0.5600 |	0.580000 | 0.460000 |	0.560000 |

#### Comparison Results of Regressors

| results | voting regressor | decision tree | random forest | keras sequential |
| --- | --- | --- | --- | --- |
| count | 10 tests | 10 tests | 10 tests | 10 tests |
| mean |	0.30800 |	0.286000 |	0.320000 |	0.344000 |
| std |	0.04341 |	0.060406 |	0.054975 |	0.095592 |
| min |	0.26000 |	0.220000 |	0.200000 |	0.220000 |
| 25% |	0.28000 |	0.240000 |	0.300000 |	0.285000 |
| 50% |	0.29000 |	0.280000 |	0.320000 |	0.310000 |
| 75% |	0.34000 |	0.315000 |	0.350000 |	0.410000 |
| max |	0.38000 |	0.400000 |	0.400000 |	0.540000 |

Ideally I would have preferred to use the following, which were providing better results;
- [Voting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
- [Voting Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html)

I will look into the Large File support here on GitHub and try to get those models working. It meant better score prediction with a more varied final score result.


### The League
[canpl.ca](https://canpl.ca/)

[Wikipedia Article](https://en.wikipedia.org/wiki/Canadian_Premier_League)

### The Beautiful Game in Canada
The intention here is to make a robust useable system, built with python to replace this [CPL Data - Google Spreadsheet](https://docs.google.com/spreadsheets/d/1B2ZqJczaT9k8b9ik3MUnKWIDggo_oX5M1O5lkf9d0bw/edit#gid=780793363)

I hope to incorporate Machine Learning and provide a great resource for Canadian fans of the beautiful game.
