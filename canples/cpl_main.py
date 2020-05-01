# canpl statistics
# Todd McCullough 2020
import pandas as pd
import numpy as np
import os

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB,BernoulliNB
import statistics

year = str(2019)

results = pd.read_csv(f'canples/datasets/{year}/cpl-{year}-results.csv')
stats = pd.read_csv(f'canples/datasets/{year}/cpl-{year}-stats.csv')

teams = results['home'].unique()
teams = np.sort(teams,axis=-1)

if year == '2019':
    teams_short = ['CFC', 'FCE', 'FFC', 'HFX', 'PFC', 'VFC', 'Y9']
    colours = ['w3-2019-fiesta', 'w3-2019-princess-blue', 'w3-2019-turmeric', 'w3-vivid-blue', 'w3-vivid-reddish-purple', 'w3-2019-biking-red', 'w3-vivid-yellowish-green']
else:
    teams_short = ['AO','CFC', 'FCE', 'FFC', 'HFX', 'PFC', 'VFC', 'Y9']
    colours = ['w3-vivid-red','w3-2019-fiesta', 'w3-2019-princess-blue', 'w3-2019-turmeric', 'w3-vivid-blue', 'w3-vivid-reddish-purple', 'w3-2019-biking-red', 'w3-vivid-yellowish-green']

results_old = results.loc[0:93].copy() # this is a temporary solution

def get_team_names(data,df,dc):
    db = data
    db = np.sort(db,axis=-1)
    a =[]
    for team in db:
        a.append(team)
    db = a
    db = pd.DataFrame({'team': db, 'short': df,'colour': dc})
    return db

team_ref = get_team_names(results['home'].unique(),teams_short,colours)

def index_reset(data):
    data = data.reset_index()
    data.pop('index')
    return data

def get_schedule(data):
    db = data.copy()
    db = db.sort_values(by=['game'])
    db = db.tail(4)
    db = db[['game','home','away']]
    db = index_reset(db)
    return db

def compare_standings(db,df):
    a = []
    for team in team_ref['team']:
        rank1 = df[df['team'] == team]
        rank2 = db[db['team'] == team]
        change = rank1.iloc[0]['rank'] - rank2.iloc[0]['rank']
        a.append([team,change])
    db = pd.DataFrame(a)
    db = pd.DataFrame({'team': db.iloc[:][0], 'change': db.iloc[:][1]})
    db = db.sort_values(by=['change'],ascending=False)
    db = index_reset(db)
    return db

def get_team_results(data,query):
    db = data[data['home'] == query]
    da = data[data['away'] == query]
    db = pd.concat([db,da])
    db = index_reset(db)
    return db

def get_team_brief(data,query):
    db = get_team_results(data,query)
    cols = ['game','s','csh','csa','combined','venue','links']
    for col in cols:
        db.pop(col)
    db = db.sort_values(by=['m','d'])
    db = index_reset(db)
    db['summary'] = '0'
    for i in range(0,db.shape[0]):
        if db.iloc[i]['home'] == query:
            x = team_ref[team_ref['team'] == db.iloc[i]['away']].reset_index()
            opponent = x.iloc[0]['short']
            outcome = db.iloc[i]['hr'] + ' A'
        else:
            x = team_ref[team_ref['team'] == db.iloc[i]['home']].reset_index()
            opponent = x.iloc[0]['short']
            outcome = db.iloc[i]['ar'] + ' A'
        score = str(db.iloc[i]['hs']) + ' - ' + str(db.iloc[i]['as'])
        db.loc[i,'summary'] = outcome + ' ' + score +  ' ' + opponent
    db['team'] = query
    return db

def get_results_brief(data):
    db = pd.DataFrame()
    for team in team_ref['team']:
        df = get_team_brief(data,team)
        db = pd.concat([db,df])
    db = index_reset(db)
    return db

def get_club_statistics(data,query):
    df = data
    a = []
    cols = df.columns
    for row in range(0,df.shape[0]):
        if df.iloc[row]['home'] == query:
            if df.iloc[row]['hr'] == 'W':
                vals = [1,3,3,1,0,0,df.iloc[row]['hs'],df.iloc[row]['as'],df.iloc[row]['hs'],df.iloc[row]['as'],0,0]
                a.append(vals)
            if df.iloc[row]['hr'] == 'L':
                vals = [1,0,3,0,1,0,df.iloc[row]['hs'],df.iloc[row]['as'],df.iloc[row]['hs'],df.iloc[row]['as'],0,0]
                a.append(vals)
            if df.iloc[row]['hr'] == 'D':
                vals = [1,1,3,0,0,1,df.iloc[row]['hs'],df.iloc[row]['as'],df.iloc[row]['hs'],df.iloc[row]['as'],0,0]
                a.append(vals)
        if df.iloc[row]['away'] == query:
            if df.iloc[row]['ar'] == 'W':
                vals = [1,3,3,1,0,0,df.iloc[row]['as'],df.iloc[row]['hs'],0,0,df.iloc[row]['as'],df.iloc[row]['hs']]
                a.append(vals)
            if df.iloc[row]['ar'] == 'L':
                vals = [1,0,3,0,1,0,df.iloc[row]['as'],df.iloc[row]['hs'],0,0,df.iloc[row]['as'],df.iloc[row]['hs']]
                a.append(vals)
            if df.iloc[row]['ar'] == 'D':
                vals = [1,1,3,0,0,1,df.iloc[row]['as'],df.iloc[row]['hs'],0,0,df.iloc[row]['as'],df.iloc[row]['hs']]
                a.append(vals)
    db= pd.DataFrame(a,columns=['gp','pts','tpp','w','l','d','gf','ga','gfh','gah','gfa','gaa'])
    db = pd.DataFrame(db.sum())
    db = db.T
    return db

def get_standings(data,season):
    db = pd.DataFrame()
    if season == 1:
        data = data[data['s'] <= 1]
    if season == 2:
        data = data[data['s'] > 1]
    teams = data['home'].unique()
    teams = np.sort(teams,axis=-1)
    for team in teams:
        df = get_team_brief(data,team)
        df = get_club_statistics(df,team)
        ppg = round(df['pts']/df['gp'],2)
        gd = df['gf'] - df['ga']
        df.insert(0,'team',team)
        df.insert(4,'ppg',ppg)
        df.insert(8,'gd',gd)
        db = pd.concat([db,df])
    db = db.sort_values(by=['pts','w','gf'],ascending=False)
    db = index_reset(db)
    db = db.reset_index()
    db = db.rename(columns={'index':'rank'})
    db['rank'] = db['rank'] + 1
    return db

def get_team_stats(data,query):
    db = data[data['team'] == query]
    names = db['name'].unique()
    information = data.copy()
    db.pop('number')
    db = db.groupby(['name']).sum()
    db.insert(0,'last','empty')
    db.insert(0,'first','empty')
    db.insert(0,'position','empty')
    db.insert(0,'number',0)
    #db.insert(0,'team',team)
    i = 0
    for name in names:
        player = information[information['name'] == name].head(1)
        db.at[name,'first'] = player.iloc[0]['first']
        db.at[name,'last'] = player.iloc[0]['last']
        db.at[name,'number'] = int(player.iloc[0]['number'])
        db.at[name,'position'] = player.iloc[0]['position']
        db.at[name,'pass accuracy'] = player.iloc[0]['pass accuracy'].mean()
        db.at[name,'cross accuracy'] = player.iloc[0]['cross accuracy'].mean()
    db = db.reset_index()
    return db

def get_stats_all(data):
    db = pd.DataFrame()
    for team in teams:
        df = get_team_stats(data,team)
        df.insert(0,'team',team)
        db = pd.concat([db,df])
    db = index_reset(db)
    return db

# get associated information for players league wide and calculate an overall score for each position
def get_evaluation(db,df):
    names = db.name.unique() # grab the list of names at the specified position
    eval_ = db.describe().T # get the evalution scores
    checks = db.columns[3:] # slice away the first three columns (name,number,postion) not needed
    db['overall'] = 0.0 # create the final column overall
    db = db.set_index('name') # set the index to the player name to search for a specific player
    for name in names: # iterate through the names in the lisst
        player = df[df['name'] == name].head(1) # get the players details
        a = [] # create an empty array to store the scores
        for check in checks: # iterate through the columns of remaining data
            result = player.iloc[0][check] / eval_['max'][check] # calculate the score for the value found value/max
            a.append(result) # append the result into the list
            overall = round(sum(a) / len(checks),2) #calculate the final score sum(list) / num of checks
            db.at[name,'overall'] = overall # assign the value as the overall score
    db = db.reset_index() # reset the index, making the name column a column again
    db = db.sort_values(by=['overall'],ascending=False) # sort using overall, descending
    return db

def top_goalscorers(data):
    df = data.copy()
    cols = ['team','name','number','minutes','goals']
    db = df[cols]
    db = db.sort_values(by=['goals'],ascending=False)
    db = db.reset_index()
    db.pop('index')
    team = db.pop('team')
    db.insert(0,'team',team)
    db = db[db['goals'] >= 1]
    return db

def top_assists(data):
    df = data.copy()
    cols = ['team','name','number','minutes','assists']
    db = df[cols]
    db = get_evaluation(db,df)
    db = db.sort_values(by=['assists'],ascending=False)
    db = db.reset_index()
    db.pop('index')
    team = db.pop('team')
    db.insert(0,'team',team)
    db = db[db['assists'] >= 1]
    return db

def top_forwards(data): # get the forwards in the league
    player_information = data.copy() # load player information
    cols = ['team','name','number','minutes','goals','chances','assists','shots','shots on target','passes','crosses','duels','tackles']
    df = player_information[player_information['position'] == 'f'] # get the forwards where position = f
    db = df[cols] # select specific columns associated with the evaluation
    db = get_evaluation(db,df)
    db = index_reset(db)
    names = db.name.unique() # get the names of the players who fit the criteria
    db = db.set_index('name') # set the index to the name column to make the search possible
    for name in names:
        player = df[df['name'] == name].head(1) # forwards main purpose is to score goals
        '''if (player.iloc[0]['goals'] <= 2.0 and player.iloc[0]['minutes'] >= 1000.0): # if player scores less than 2 & has minutes greater than 1000
            db.at[name,'overall'] = db.at[name,'overall'] - 0.2
        if player.iloc[0]['goals'] >= 8.0: # reward forwards scoring greater than 8 goals
            db.at[name,'overall'] = db.at[name,'overall'] + 0.1'''
    db = db.sort_values(by=['overall'],ascending=False)
    db = db.reset_index()
    team = db.pop('team')
    db.insert(0,'team',team)
    return db

def top_midfielders(data): # get the midfielders in the league
    player_information = data.copy()
    cols = ['team','name','number','minutes','goals','assists','touches','passes','pass accuracy','crosses','cross accuracy','chances','duels','tackles']
    df = player_information[player_information['position'] == 'm'] # get the midfields where position = m
    db = df[cols]
    db = get_evaluation(db,df)
    db = index_reset(db)
    names = db.name.unique()
    db = db.set_index('name')
    for name in names:
        player = df[df['name'] == name].head(1)
        '''if player.iloc[0]['goals'] <= 2.0:
            db.at[name,'overall'] = db.at[name,'overall'] - 0.2
        if player.iloc[0]['goals'] >= 8.0:
            db.at[name,'overall'] = db.at[name,'overall'] + 0.1'''
    db = db.sort_values(by=['overall'],ascending=False)
    db = db.reset_index()
    team = db.pop('team')
    db.insert(0,'team',team)
    return db

def top_defenders(data):  # get the defenders in the league
    player_information = data.copy()
    cols = ['team','name','number','minutes','tackles','tackles won','clearances','interceptions','duels','duels won']
    df = player_information[player_information['position'] == 'd'] # get the defenders where position = d
    db = df[cols]
    db = get_evaluation(db,df)
    db = index_reset(db)
    team = db.pop('team')
    db.insert(0,'team',team)
    return db

def top_keepers(data):  # get the keepers in the league
    player_information = data.copy()
    cols = ['team','name','number','minutes','clean sheets','saves','shots faced','claimed crosses']
    df = player_information[player_information['position'] == 'g'] # get the goalkeepers where position = g
    db = df[cols]
    db = get_evaluation(db,df)
    db = index_reset(db)
    team = db.pop('team')
    db.insert(0,'team',team)
    return db

def top_offenders(data):  # get the offences handed out in the league
    player_information = data.copy()
    cols = ['team','name','number','minutes','yellow','red','fouls conceded']
    df = player_information
    db = df[cols]
    db = get_evaluation(db,df)
    db = db.sort_values(by=['red','yellow'],ascending=False)
    db = db.reset_index()
    db.pop('index')
    team = db.pop('team')
    db.insert(0,'team',team)
    return db

def get_match_tables(data,query):
    db = data[data['home'] == query]
    db = pd.concat([db,data[data['away'] == query]])
    db = db.sort_values(by=['m','d'])
    return db

def likelihood_table(data,query):
    df = get_match_tables(data,query)
    a = []
    cols = data.columns
    for row in range(0,df.shape[0]):
        if df.iloc[row]['home'] == query:
            if df.iloc[row]['hr'] == 'W':
                b = [1,2,1]
                c = [1,0,0]
                d = [1,1,0]
                a.append(b)
                a.append(c)
                a.append(d)
            if df.iloc[row]['hr'] == 'L':
                b = [1,2,0]
                c = [1,0,1]
                d = [1,1,0]
                a.append(b)
                a.append(c)
                a.append(d)
            if df.iloc[row]['hr'] == 'D':
                b = [1,2,0]
                c = [1,0,0]
                d = [1,1,1]
                a.append(b)
                a.append(c)
                a.append(d)
        if df.iloc[row]['away'] == query:
            if df.iloc[row]['ar'] == 'W':
                b = [2,2,1]
                c = [2,0,0]
                d = [2,1,0]
                a.append(b)
                a.append(c)
                a.append(d)
            if df.iloc[row]['ar'] == 'L':
                b = [2,2,0]
                c = [2,0,1]
                d = [2,1,0]
                a.append(b)
                a.append(c)
                a.append(d)
            if df.iloc[row]['ar'] == 'D':
                b = [2,2,0]
                c = [2,0,0]
                d = [2,1,1]
                a.append(b)
                a.append(c)
                a.append(d)
    db= pd.DataFrame(a,columns=['h/a','w/l/d','y/n'])
    return db

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
    return db

def get_NB_data(data,query):
    db = likelihood_table(data,query)
    dy = db.pop('y/n').to_list()
    dx = [tuple(x) for x in db.values]
    return dx, dy

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
    #print(gnb_pred[0], bnb_pred[0], pred)

    return pred

def get_match_prediction_home(query,x,y):
    home_win = get_gnb_prediction(query,x,y,[1,2])
    draw = get_gnb_prediction(query,x,y,[1,1])
    return home_win, draw

def get_match_prediction_away(query,x,y):
    away_win = get_gnb_prediction(query,x,y,[2,2])
    return away_win

def get_match_prediction(q1,q2,x1,y1,x2,y2):
    home_win, draw = get_match_prediction_home(q1,x1,y1)
    away_win = get_match_prediction_away(q2,x2,y2)
    return home_win, draw, away_win

def get_team_form(data,query):
    db = data[data['team'] == query]
    db = pd.DataFrame(db['summary'])
    return db

def get_form_results(data):
    db = pd.DataFrame()
    form = get_results_brief(data[data['s'] <= 1])
    teams = data.home.unique()
    teams = np.sort(teams,axis=-1)
    for team in teams:
        df = get_team_form(form,team)
        db[team] = df['summary'].values
    db = db.T
    db = db.reset_index()
    return db

def get_team_history(data,query):
    df = data[data['away'] == query].copy()
    df = df[['d','m','as','hs','away','ar','home','hr']]
    df = df.rename(columns={'as':'hs','hs':'as','away':'home','ar':'hr','home':'away','hr':'ar'})
    db = data[data['home'] == query].copy()
    db = db[['d','m','hs','as','home','hr','away','ar']]
    db = pd.concat([db,df])
    db = db.tail(5)
    db = db.sort_values(by=['m','d'],ascending=False)
    return db

def get_five_game_form(data,query):
    db = get_team_history(data,query)
    db = db.pop('hr')
    a = []
    for i in db:
        if i == 'W':
            j = [1,0,0]
            a.append(j)
        if i == 'L':
            j = [0,1,0]
            a.append(j)
        if i == 'D':
            j = [0,0,1]
            a.append(j)
    db = pd.DataFrame(a,columns=['w','l','d'])
    db = pd.DataFrame(db.sum())
    return db

def clean_team_game(data,db,check):
    if check == 0:
        df = data.iloc[0]['team']
    else:
        df = data.iloc[-1]['team']
    df = db[(db['home'] == df) | (db['away'] == df)]
    db = index_reset(df)
    db = pd.DataFrame(db.iloc[0][['home','hs','away','as']])
    db = db.T
    return db

def get_short_name(data):
    for team in team_ref['team']:
        name = team_ref[team_ref['team'] == team]
        if data.iloc[0]['home'] == team:
            data['home'] = name.iloc[0]['short']
        if data.iloc[0]['away'] == team:
            data['away'] = name.iloc[0]['short']
    return data

def get_long_name(string):
    for short in team_ref['short']:
        row = team_ref[team_ref['short'] == short]
        if string == short:
            string = row.iloc[0]['team']
    return string

def get_weeks_results(data,standings):
    df = data
    month = df.iloc[-1]['m']
    week = df.iloc[-1]['d']
    db = df[df['m'] == month]
    db = db[db['d'] >= week - 7]
    db = db.sort_values(by=['d'],ascending=False)
    goals = db['hs'].sum() + db['as'].sum()
    max_home_win = db[db['hs'] == db['hs'].max()]
    big_win = max_home_win[['home','hs','away','as']]
    big_win = index_reset(big_win)
    big_win = get_short_name(big_win)
    top_team = clean_team_game(standings,db,0)
    top_team = get_short_name(top_team)
    low_team = clean_team_game(standings,db,1)
    low_team = get_short_name(low_team)
    return db,goals,big_win,top_team, low_team

def get_home_away_comparison(data,game,query):
    db = data[data['game'] == game].copy()
    db = db[db['team'] == query]
    db = db.sort_values(by=['minutes'],ascending=False)
    db = db[0:11]
    db = db['name']
    return db

team_stats = get_stats_all(stats)

rated_forwards = top_forwards(team_stats)
rated_midfielders = top_midfielders(team_stats)
rated_defenders = top_defenders(team_stats)
rated_goalscorers = top_goalscorers(team_stats)
rated_keepers = top_keepers(team_stats)
rated_offenders = top_offenders(team_stats)

def get_compare_roster(data):
    a = []
    for name in data:
        for f in range(rated_forwards.shape[0]):
            if rated_forwards.loc[f]['name'] == name:
                player = rated_forwards.loc[f]
                a.append(player)
        for f in range(rated_midfielders.shape[0]):
            if rated_midfielders.loc[f]['name'] == name:
                player = rated_midfielders.loc[f]
                a.append(player)
        for f in range(rated_defenders.shape[0]):
            if rated_defenders.loc[f]['name'] == name:
                player = rated_defenders.loc[f]
                a.append(player)
        for f in range(rated_keepers.shape[0]):
            if rated_keepers.loc[f]['name'] == name:
                player = rated_keepers.loc[f]
                a.append(player)
    db = pd.DataFrame(a)
    db = db[['name','number','overall']]
    db = db.sort_values(by=['overall'],ascending=False)
    return db
