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
| count | 10 tests | 10 tests | 10 tests | 10 tests | 10 tests |
| mean | 0.344000 | 0.404000 | 0.430000 | 0.448000 | 0.440000 |
| std | 0.074117 | 0.100133 | 0.062004 | 0.070048 | 0.077172 |
| min | 0.240000 | 0.260000 | 0.300000 | 0.360000 | 0.300000 |
| 25% | 0.275000 | 0.345000 | 0.405000 | 0.380000 | 0.395000 |
| 50% | 0.350000 | 0.400000 | 0.450000 | 0.470000 | 0.460000 |
| 75% | 0.395000 | 0.470000 | 0.475000 | 0.515000 | 0.480000 |
| max | 0.460000 | 0.580000 | 0.500000 | 0.520000 | 0.540000 |

#### Comparison Results of Regressors

| results | voting regressor | decision tree | random forest | keras sequential |
| --- | --- | --- | --- | --- | --- | --- |
| count | 10 tests | 10 tests | 10 tests | 10 tests
| mean | 0.334000 | 0.278000 | 0.310000 | 0.25600 |
| std | 0.088969 | 0.064944 | 0.041366 | 0.04402 |
| min | 0.180000 | 0.200000 | 0.260000 | 0.18000 |
| 25% | 0.280000 | 0.240000 | 0.280000 | 0.22500 |
| 50% | 0.350000 | 0.250000 | 0.300000 | 0.25000 |
| 75% | 0.390000 | 0.340000 | 0.355000 | 0.29500 |
| max | 0.460000 | 0.380000 | 0.360000 | 0.32000 |

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
