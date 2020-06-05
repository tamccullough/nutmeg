# Todd McCullough
# 2020
import re
import numpy as np
import pandas as pd
import cpl_main as cpl
import random

results = pd.read_csv(f'datasets/soccer-nn-train.csv')

print(results)

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

db = df.copy()

db.pop('game')
db.pop('team')
y = db.pop('r')
db.pop('s')
X = db

print(X.head(2))

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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
seed = 7

#K Neighbors Classifier
# algorithm = ‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’
def kNeighbors(x,y):
    model = KNeighborsClassifier(algorithm = 'auto',
                                 leaf_size=30,
                                 metric='minkowski',
                                 metric_params=None,
                                 n_jobs=1,
                                 n_neighbors=5,
                                 p=2,
                                 weights='uniform')
    model.fit(x, y)
    return model

knn = kNeighbors(X_train, y_train)

#Random Forest Regression
def forestRegression_1(x,y):
    model = RandomForestClassifier(n_estimators = 100,
                                   min_samples_leaf = 5,
                                   min_samples_split = 12,
                                   random_state = 0,
                                   max_depth = 80)
    model.fit(x, y)
    return model

rf1 = forestRegression_1(X_train, y_train)

# Bagged Decision Trees for Classification - necessary dependencies
def baggedTree(x,y,seed):
    model = BaggingClassifier(base_estimator=rf1,
                              n_estimators=10,
                              random_state=seed)
    model.fit(x, y)
    return model

bag = baggedTree(X_train, y_train,seed)

# AdaBoost Classification
def adaBoost(x,y,seed):
    model = AdaBoostClassifier(n_estimators=70,
                               random_state=seed,
                               algorithm='SAMME')
    model.fit(x, y)
    return model

ada = adaBoost(X_train, y_train,seed)

# Voting Ensemble for Classification
def ensembleClassifier(x,y):
    # create the sub models
    estimators = []
    estimators.append(('rf1', rf1))
    estimators.append(('knn', knn))
    estimators.append(('bag', bag))

    # create the ensemble model
    model = VotingClassifier(estimators,voting='soft',weights=[0.83,0.79,0.82])
    model.fit(x,y)
    return model

ens = ensembleClassifier(X_train, y_train)

def check(a,b):
    if a == b:
        result = '<'
    else:
        result = '-'
    return result

print('rf1',rf1.score(X_train, y_train))
print('ens',ens.score(X_train, y_train))
print('bag',bag.score(X_train, y_train))
print('knn',knn.score(X_train, y_train))
print('ada',ada.score(X_train, y_train))

def print_pred_results(model,result,num):
    print('model : ', result, check(result,y_test.loc[num]))

def predictionTest(num,model1,model2,model3):
    p = X_test.loc[num].tolist()
    result1 = model1.predict([p]).flatten()
    result2 = model2.predict([p]).flatten()
    result3 = model3.predict([p]).flatten()
    print('\nActual       : ',y_test.loc[num])
    print_pred_results(model1,result1,num)
    print_pred_results(model2,result2,num)
    print_pred_results(model3,result3,num)

def cycle_prob_test(num,model):
    p = X_test.loc[num].tolist()
    e = model.predict_proba([p]).flatten()
    return e.tolist()

def cycle_pred_test(num,model):
    p = X_test.loc[num].tolist()
    e = model.predict([p]).flatten()
    if e[0] == y_test.loc[num]:
        a = 1
    else:
        a = 0
    return a

def model_pred_test(model):
    pred = []
    prob = []
    numbers = X_test.index
    random_nums = random.choices(numbers, k=50)
    for i in random_nums:
        pred.append(cycle_pred_test(i,model)) # check to see if the values are correct and score it
        #prob.append(cycle_prob_test(i,model))
    dz = pd.DataFrame(pred)
    #df = pd.DataFrame(prob)
    c = str(float(dz.sum().values / 50))
    print('score :',c)

test_results = pd.DataFrame(index=range(10),columns=['knn','ada','bag','rf1','ens'])
test_results = test_results.fillna(0.0)

for i in range(10):
    k_s = model_pred_test(knn)
    test_results.at[i,'knn'] = k_s
    a_s = model_pred_test(ada)
    test_results.at[i,'ada'] = a_s
    b_s = model_pred_test(bag)
    test_results.at[i,'bag'] = b_s
    r_s = model_pred_test(rf1)
    test_results.at[i,'rf1'] = r_s
    e_s = model_pred_test(ens)
    test_results.at[i,'ens'] = e_s

print(test_results.describe())

import pickle
filename = 'models/cpl_roster_classifier.sav'
pickle.dump(rf1, open(filename, 'wb'))

cpl_classifier_model = pickle.load(open(filename, 'rb'))

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

# home side
q1 = schedule.iloc[3]['home']
# away side
q2 = schedule.iloc[3]['away']
print(q1,q2)

compare = cpl.get_team_comparison(results_brief,q1,q2)

t1_x, t1_y = cpl.get_NB_data(compare,q1)
t2_x, t2_y = cpl.get_NB_data(compare,q2)

game_info = schedule[schedule['home'] == q1]
game_info = game_info[game_info['away'] == q2]

game = game_info.iloc[0]['game']
game_h = cpl.get_home_away_comparison(stats,game,q1)
game_a = cpl.get_home_away_comparison(stats,game,q2)

h1_roster = cpl.get_compare_roster(results,q1,team_stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info)
h2_roster = cpl.get_compare_roster(results,q2,team_stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info)

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

q1_roster = get_overall_roster(h1_roster)
print(q1_roster)

q2_roster = get_overall_roster(h2_roster)
print(q2_roster)

def roster_pred(model,array):
    prediction = model.predict_proba([array]).flatten()
    df = pd.DataFrame(prediction)
    #print('score :',prediction)
    return df

home_win, draw, away_win = cpl.get_match_prediction(q1,q2,t1_x,t1_y,t2_x,t2_y)

home_win_new, away_win_new, draw_new = cpl.get_final_game_prediction(cpl_classifier_model,q1_roster,q2_roster,home_win,away_win,draw)

print(q1,'\nwin probability: ', round(home_win_new,2))

print(q2,'\nwin probability: ', round(away_win_new,2))

print('Draw probability: ', round(draw_new,2))

q1_prediction = roster_pred(cpl_classifier_model,q1_roster)
q2_prediction = roster_pred(cpl_classifier_model,q2_roster)

q_draw = (q1_prediction.iloc[1][0] + q2_prediction.iloc[2][0]) /2

q1_p = round(q1_prediction.iloc[2][0],2)
q2_p = round(q2_prediction.iloc[2][0],2)

if q1_p > q2_p:
    print(q1,'predicted to win ',q1_p)
elif q2_p > q1_p:
    print(q2,'predicted to win ',q2_p)
else:
    print('A Draw is predicted ',q1_p,' ',q2_p)

total_ = q1_p + home_win + q2_p + away_win + q_draw + draw
print(round((q1_p + home_win) / total_, 2))

print(round((q2_p + away_win) / total_, 2))

print(round((q_draw + draw) / total_, 2))

def get_final_game_prediction(model,q1_roster,q2_roster,home_win,away_win,draw):
    q1_prediction = roster_pred(model,q1_roster)
    q1_p = round(q1_prediction.iloc[2][0],2)
    q2_prediction = roster_pred(model,q2_roster)
    q2_p = round(q2_prediction.iloc[2][0],2)
    q_draw = (q1_prediction.iloc[1][0] + q2_prediction.iloc[2][0]) / 2
    total_ = q1_p + home_win + q2_p + away_win + q_draw + draw
    h_w = round((q1_p + home_win) / total_, 2)
    a_w = round((q2_p + away_win) / total_, 2)
    g_d = round((q_draw + draw) / total_, 2)
    return h_w, a_w, g_d

home_win_new, away_win_new, draw_new = get_final_game_prediction(cpl_classifier_model,q1_roster,q2_roster,home_win,away_win,draw)

print(home_win_new)
print(away_win_new)
print(draw_new)
