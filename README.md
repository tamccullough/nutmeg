![CANPL](https://pbs.twimg.com/profile_images/1191405045788676097/vk_lsh7F_200x200.jpg)

[CANPL Heroku App LINK](https://canpl.herokuapp.com/)

## WOW. That's Different

It may not seem like much to you. But the main.py file and the accompanying cpl_main.py file have both gone through some incredible clean ups. Why? Because I've gotten better. You learn and you progress, and you should always go back and clean up bad code.

## Canpl

Nutmeg is a CanPL Fan Site designed to provide stat tables and charts based on user input

Currently using MultinomialNB(), RandomForestRegressor() and RandomForestClassifier() to make the match predictions between teams as games are listed in the schedule.

### How Match Prediction is Deployed

This app is using the following models for predictions;
- [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

After a lot of retooling. It has all changed.

Likehood table is created before a match with the results being fed into
- MultinomialNB()

Using the predictions from that, those results are then fed into

- RandomForestRegressor() to generate a score prediction using a 14 men roster and other features

To which those predictions are fed into

- RandomForestClassifier() to generate a match prediction using both 14 men rosters and other features

The final output predicts the home teams probability and result label

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

| results | voting regressor | decision tree | random forest |
| --- | --- | --- | --- |
| count | 10 tests | 10 tests | 10 tests |
| mean | 0.334000 | 0.278000 | 0.310000 |
| std | 0.088969 | 0.064944 | 0.041366 |
| min | 0.180000 | 0.200000 | 0.260000 |
| 25% | 0.280000 | 0.240000 | 0.280000 |
| 50% | 0.350000 | 0.250000 | 0.300000 |
| 75% | 0.390000 | 0.340000 | 0.355000 |
| max | 0.460000 | 0.380000 | 0.360000 |

### The League
[canpl.ca](https://canpl.ca/)

[Wikipedia Article](https://en.wikipedia.org/wiki/Canadian_Premier_League)

### The Beautiful Game in Canada
I hope this becomes a great resource for the fans of the CanPL.
I hope to incorporate Machine Learning and provide a great resource for Canadian fans of the beautiful game.
