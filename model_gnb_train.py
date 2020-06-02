# Todd McCullough
# 2020
# ### Gaussian Naive Bayes Model Likelihood

import re
import numpy as np
import pandas as pd
#import cpl_main as cpl

year = '2020'

results = pd.read_csv(f'datasets/{year}/cpl-{year}-results.csv')
stats = pd.read_csv(f'datasets/{year}/cpl-{year}-stats.csv')
team_ref = pd.read_csv('datasets/teams.csv')

if year == '2019':
    team_ref = team_ref[1:]

results_brief = pd.read_csv(f'datasets/{year}/cpl-{year}-results_brief.csv')
schedule = pd.read_csv(f'datasets/{year}/cpl-{year}-schedule.csv')

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB,BernoulliNB
import statistics

# home side
q1 = schedule.iloc[2]['home']
# away side
q2 = schedule.iloc[2]['away']
print(q1,q2)

def get_team_comparison(data,q1,q2):
    # getting games with q1 in both home or away
    db = data[data['team'] == q1]
    db = db.reset_index()
    db.pop('index')
    # filering down more to get only the games against q2
    db = db.sort_values(by=['m','d'])
    db = db[(db['home'] == q2) | (db['away'] == q2)]
    db = db.reset_index()
    db.pop('index')
    if db.empty == True:
        db = pd.DataFrame([(0,0,0,0,q1,'D',q2,'D','empty',q1)],columns=['d','m','hs','as','home','hr','away','ar','summary','team'])
    return db


compare = get_team_comparison(results_brief,q1,q2)

def get_match_tables(data,query):
    db = data[data['home'] == query]
    db = pd.concat([db,data[data['away'] == query]])
    db = db.sort_values(by=['m','d'])
    return db

def likelihood_input(array,a_list):
    b = a_list[0]
    c = a_list[1]
    d = a_list[2]
    array.append(b)
    array.append(c)
    array.append(d)
    return array

def likelihood_table(data,query):
    df = get_match_tables(data,query)
    array = []
    cols = data.columns
    for row in range(0,df.shape[0]):
        if df.iloc[row]['home'] == query:
            if df.iloc[row]['hr'] == 'W':
                array = likelihood_input(array,[[1,2,1],[1,0,0],[1,1,0]])
            if df.iloc[row]['hr'] == 'L':
                array = likelihood_input(array,[[1,2,0],[1,0,1],[1,1,0]])
            if df.iloc[row]['hr'] == 'D':
                array = likelihood_input(array,[[1,2,0],[1,0,0],[1,1,1]])
        if df.iloc[row]['away'] == query:
            if df.iloc[row]['ar'] == 'W':
                array = likelihood_input(array,[[2,2,1],[2,0,0],[2,1,0]])
            if df.iloc[row]['ar'] == 'L':
                array = likelihood_input(array,[[2,2,0],[2,0,1],[2,1,0]])
            if df.iloc[row]['ar'] == 'D':
                array = likelihood_input(array,[[2,2,1],[2,0,0],[2,1,1]])
    db= pd.DataFrame(array,columns=['h/a','w/l/d','y/n'])
    return db

def get_NB_data(data,query):
    db = likelihood_table(data,query)
    dy = db.pop('y/n').to_list()
    dx = [tuple(x) for x in db.values]
    return dx, dy

t1_x, t1_y = get_NB_data(compare,q1)
t2_x, t2_y = get_NB_data(compare,q2)

def get_match_prediction_result(query,x,y,array):
    prediction = get_gnb_prediction(query,x,y,array)
    return prediction

def get_match_prediction(q1,q2,x1,y1,x2,y2):
    if len(x1) == 0:
        x = round(1/3,2)
        home_win, away_win,draw = x,x,x
        return home_win,away_win,draw
    home_win = get_match_prediction_result(q1,x1,y1,[1,2])
    draw = get_match_prediction_result(q1,x1,y1,[1,1])
    away_win = get_match_prediction_result(q2,x2,y2,[2,2])
    return home_win, draw, away_win

def gnb_mod():

    gnb = GaussianNB()
    bnb = BernoulliNB()
    # Train the model using the training sets

    gnb.fit(x,y)
    bnb.fit(x,y)

    # use below instead of predicted = model.predict([result]) because we want the probability
    gnb_pred = np.round(gnb.predict_proba([result])[:, 1],decimals=2)
    bnb_pred = np.round(bnb.predict_proba([result])[:, 1],decimals=2)

    pred = round((gnb_pred[0] + bnb_pred[0]) / 2,2)

    return pred


def get_gnb_prediction(query,x,y,result):

    gnb = GaussianNB()
    bnb = BernoulliNB()
    # Train the model using the training sets

    gnb.fit(x,y)
    bnb.fit(x,y)

    # use below instead of predicted = model.predict([result]) because we want the probability
    gnb_pred = np.round(gnb.predict_proba([result])[:, 1],decimals=2)
    bnb_pred = np.round(bnb.predict_proba([result])[:, 1],decimals=2)

    pred = round((gnb_pred[0] + bnb_pred[0]) / 2,2)

    return pred

def get_match_prediction_result(query,x,y,array):
    prediction = get_gnb_prediction(query,x,y,array)
    return prediction

def get_match_prediction(q1,q2,x1,y1,x2,y2):
    if len(x1) == 0:
        x = round(1/3,2)
        home_win, away_win,draw = x,x,x
        return home_win,away_win,draw
    home_win = get_match_prediction_result(q1,x1,y1,[1,2])
    draw = get_match_prediction_result(q1,x1,y1,[1,1])
    away_win = get_match_prediction_result(q2,x2,y2,[2,2])
    return home_win, draw, away_win


home_win, draw, away_win = get_match_prediction(q1,q2,t1_x,t1_y,t2_x,t2_y)

print(q1,'\nwin probability: ', round(home_win,2))

print(q2,'\nwin probability: ', round(away_win,2))

print('Draw probability: ', round(draw,2))
