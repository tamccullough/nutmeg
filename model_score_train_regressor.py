# Todd McCullough
# 2020
import re
import numpy as np
import pandas as pd
import cpl_main as cpl
import random

results = pd.read_csv(f'datasets/soccer-nn-train.csv')
print(results.head(2))
print(results.shape)

def pump_it_up(db):
    df = db.copy()
    dc = df.copy()
    m = df['p1'].copy()
    n = df['p2'].copy()
    o = df['p3'].copy()
    p = df['p4'].copy()
    q = df['p5'].copy()
    r = df['p6'].copy()
    df['p1'] = dc.pop('p8')
    df['p2'] = dc.pop('p10')
    df['p3'] = dc.pop('p12')
    df['p4'] = dc.pop('p9')
    df['p5'] = dc.pop('p11')
    df['p6'] = dc.pop('p13')
    df['p7'] = m
    df['p8'] = n
    df['p9'] = o
    df['p10'] = p
    df['p11'] = q
    df['p12'] = r
    df['p13'] = dc.pop('p7')
    dc = df.copy()
    db = pd.concat([db,df])
    df = dc.copy()
    m = df['p13'].copy()
    n = df['p12'].copy()
    o = df['p11'].copy()
    p = df['p10'].copy()
    q = df['p9'].copy()
    r = df['p8'].copy()
    df['p13'] = dc.pop('p8')
    df['p12'] = dc.pop('p10')
    df['p11'] = dc.pop('p12')
    df['p10'] = dc.pop('p9')
    df['p9'] = dc.pop('p11')
    df['p8'] = dc.pop('p13')
    df['p7'] = m
    df['p6'] = n
    df['p5'] = o
    df['p4'] = p
    df['p3'] = q
    df['p2'] = r
    df['p1'] = dc.pop('p7')
    #dc = df.copy()
    db = pd.concat([db,df])
    db = cpl.index_reset(db)
    return db

df = pump_it_up(results)
print(df.shape)
db = pump_it_up(df)
print(db.shape)
db = results.copy()
db.pop('game')
db.pop('team')
y = db.pop('s')
db.pop('r')
X = db

X['all'] = round(X.sum(axis = 1, skipna = True) / 13,2)
print(X.head(2))
from math import sqrt
from sklearn.metrics import mean_squared_error

#importing libraries from sklearn
from sklearn import tree
#from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler#,Imputer
from sklearn import metrics
# import algorithm modules
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
seed = 7

#Linear Regression Model
def linearRegression():
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

lr = linearRegression()

print('Linear Regression Model')
print('\nRMSE: ', sqrt(mean_squared_error(y_test,lr.predict(X_test))))
print('\nScore',round(lr.score(X_test, y_test)*100,2))

#DecisionTreeRegressor
def decisionTree():
    model = DecisionTreeRegressor(criterion='mse', splitter='random', max_depth=8, max_features='log2')
    model.fit(X_train, y_train)
    return model
dt = decisionTree()

print('Decision Tree Regression Model')
print('\nRMSE: ', sqrt(mean_squared_error(y_test, dt.predict(X_test))))
print('\nScore',round(dt.score(X_test, y_test)*100,2))

#Random Forest Regression
def forestRegression():
    model = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    model.fit(X_train, y_train)
    return model

rf = forestRegression()

print('Random Forest Regression Model')
print('\nRMSE: ', sqrt(mean_squared_error(y_test,rf.predict(X_test))))
print('\nScore',round(rf.score(X_test, y_test)*100,2))

from sklearn.ensemble import VotingRegressor
vr = VotingRegressor(estimators=[('lr', lr), ('dt', dt), ('rf', rf)])
vr = vr.fit(X_train, y_train)

print('Voting Regressor Model')
print('\nRMSE: ', sqrt(mean_squared_error(y_test,vr.predict(X_test))))
print('\nScore',round(vr.score(X_test, y_test)*100,2))

def kerasSequential():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss = 'mse',
                optimizer = tf.keras.optimizers.RMSprop(0.1),
                metrics = ['mae', 'mse'])

    return model
ks = kerasSequential()
print(ks.summary())
trained_weight = ks.get_weights()[0]
trained_bias = ks.get_weights()[1]

EPOCHS = 450
history = ks.fit(X_train,
                 y_train,
                 epochs = EPOCHS,
                 batch_size = 128,
                 validation_split = 0.2,
                 verbose = 1)

hist = pd.DataFrame(history.history)
mse = hist['mse']
epochs = history.epoch

from matplotlib import pyplot as plt
#@title Define the plotting functions
def plot_the_model(trained_weight, trained_bias, feature, label):
    """Plot the trained model against the training feature and label."""
    # Label the axes.
    plt.xlabel("feature")
    plt.ylabel("label")
    # Plot the feature values vs. label values.
    plt.scatter(feature, label)
    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    plt.plot(feature.tolist(), label.tolist(), c='r')
    # Render the scatter plot and the red line.
    plt.show()

def plot_the_loss_curve(epochs, mse):
    """Plot the loss curve, which shows loss vs. epoch."""
    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('mse')
    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min()*0.97, mse.max()])
    plt.show()

feature = X['all'].copy()
label = y.copy()

#plot_the_model(trained_weight, trained_bias, feature, label)
plot_the_loss_curve(epochs, mse)

print(hist.describe())

def check(a,b):
    #print('pred :',a,'actual :',b)
    if a == b:
        result = 1
    else:
        result = 0
    return result

print('lr',lr.score(X_train, y_train))
print('dt',dt.score(X_train, y_train))
print('rf',rf.score(X_train, y_train))
print('vr',vr.score(X_train, y_train))
ks_test = ks.evaluate(X_train, y_train,verbose=0)
print('ks',ks_test[1])

from matplotlib import pyplot as plt

def cycle_pred_test(num,model):
    score = check(result,y_test.loc[num])
    return score

def model_pred_test(model):
    pred = []
    numbers = X_test.index
    random_nums = random.choices(numbers, k=50)
    for i in random_nums:
        p = X_test.loc[i].tolist()
        result = model.predict([p]).flatten().round()
        prediction = print_pred_results(int(result[0]),i)
        pred.append(prediction)
    dz = pd.DataFrame(pred)
    #df = pd.DataFrame(prob)
    c = str(float(dz.sum().values / 50))
    return c

test_results = pd.DataFrame(index=range(10),columns=['vr','dt','rf','ks'])
test_results = test_results.fillna(0.0)

for i in range(10):
    v_s = model_pred_test(vr)
    test_results.at[i,'vr'] = v_s
    d_s = model_pred_test(dt)
    test_results.at[i,'dt'] = d_s
    r_s = model_pred_test(rf)
    test_results.at[i,'rf'] = r_s
    k_s = model_pred_test(ks)
    test_results.at[i,'ks'] = k_s

print(test_results.describe())

import pickle
filename = 'models/cpl_score_regressor.sav'
pickle.dump(dt, open(filename, 'wb'))

import pandas as pd
import pickle
import cpl_main as cpl

year = '2020'
team_ref = pd.read_csv('datasets/teams.csv')
results = pd.read_csv(f'datasets/{year}/cpl-{year}-results.csv')
stats = pd.read_csv(f'datasets/{year}/cpl-{year}-stats.csv')
player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')
results_brief = pd.read_csv(f'datasets/{year}/cpl-{year}-results_brief.csv')
team_stats = pd.read_csv(f'datasets/{year}/cpl-{year}-team_stats.csv')
schedule = pd.read_csv(f'datasets/{year}/cpl-{year}-schedule.csv')
rated_forwards = pd.read_csv(f'datasets/{year}/cpl-{year}-forwards.csv')
rated_midfielders = pd.read_csv(f'datasets/{year}/cpl-{year}-midfielders.csv')
rated_defenders = pd.read_csv(f'datasets/{year}/cpl-{year}-defenders.csv')
rated_keepers = pd.read_csv(f'datasets/{year}/cpl-{year}-keepers.csv')

filename = 'models/cpl_score_regressor.sav'
cpl_score_model = pickle.load(open(filename, 'rb'))

#model_pred_test(cpl_classifier_model)

# home side
q1 = schedule.iloc[0]['home']
# away side
q2 = schedule.iloc[0]['away']
print(q1,q2)

compare = cpl.get_team_comparison(results_brief,q1,q2)

t1_x, t1_y = cpl.get_NB_data(compare,q1)
t2_x, t2_y = cpl.get_NB_data(compare,q2)

game_info = schedule[schedule['home'] == q1]
game_info = game_info[game_info['away'] == q2]
game_info

game = game_info.iloc[0]['game']
game_h = cpl.get_home_away_comparison(stats,game,q1)
game_a = cpl.get_home_away_comparison(stats,game,q2)

home_roster = cpl.get_compare_roster(results,q1,team_stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info)
away_roster = cpl.get_compare_roster(results,q2,team_stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info)

def get_overall_roster(game_roster):
    b = []
    for i in range(game_roster.shape[0]):
        b.append(game_roster.iloc[i]['overall']) # get the player overall score for each player in the game
    if len(b) < 16:
        i = int(16 - len(b))
        for j in range(0,i):
            b.append(0)
    db = pd.DataFrame(b[0:14])
    db = db.T
    return db

q1_roster = get_overall_roster(home_roster)
q1_roster

q2_roster = get_overall_roster(away_roster)
q2_roster

def roster_regressor_pred(model,array):
    prediction = model.predict([array]).flatten()
    df = pd.DataFrame(prediction)
    return df

home_win, draw, away_win = cpl.get_match_prediction(q1,q2,t1_x,t1_y,t2_x,t2_y)
print(home_win)
print(away_win)

classifier = 'models/cpl_roster_classifier.sav'
cpl_classifier_model = pickle.load(open(classifier, 'rb'))

home_win_new, away_win_new, draw_new = cpl.get_final_game_prediction(cpl_classifier_model,q1_roster,q2_roster,home_win,away_win,draw)

print(home_win_new)
print(away_win_new)

print(q1,'\nwin probability: ', round(home_win_new,2))

print(q2,'\nwin probability: ', round(away_win_new,2))

print('Draw probability: ', round(draw_new,2))

def get_final_score_prediction(model,q1_roster,q2_roster,home_win_new,away_win_new):

    def final_score_fix(home_score,away_score,home_win_new,away_win_new):
        if home_win_new > away_win_new: # fix the score prediction - if the probability of home win > away win
            home_score = away_score + 1 # change the predicted score to reflect that
            return home_score,away_score
        elif home_win_new < away_win_new: # else the probability of home win < away win
            away_score = home_score + 1 # change the predicted score to reflect that
            return home_score,away_score
        else:
            return home_score,away_score

    def score(num): #improve this later for greater predictions
        new_score = int(round(num,0)) # convert the float value to int and round it
        return new_score

    q1_pred = roster_pred(model,q1_roster)
    q1_s = score(q1_pred.iloc[0][0])
    q2_pred = roster_pred(model,q2_roster)
    q2_s = score(q2_pred.iloc[0][0])
    home_score, away_score = final_score_fix(q1_s, q2_s,home_win_new,away_win_new)
    return home_score, away_score

def roster_pred(model,array):
    prediction = model.predict([array]).flatten()
    df = pd.DataFrame(prediction)
    #print('score :',prediction)
    return df

home_score, away_score = get_final_score_prediction(cpl_score_model,q1_roster,q2_roster,home_win_new,away_win_new)

print(home_score, away_score)

print(home_score)

print(away_score)
