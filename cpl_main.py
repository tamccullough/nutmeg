# canpl statistics
# Todd McCullough 2020
import pandas as pd
import numpy as np
import os

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB,BernoulliNB
import statistics

def get_long_name(string,team_ref):
    for short in team_ref['short']:
        row = team_ref[team_ref['short'] == short]
        if string == short:
            string = row.iloc[0]['team']
    return string

def get_shortest_name(string,team_ref):
    for team in team_ref['team']:
        row = team_ref[team_ref['team'] == team]
        if string == team:
            string = str(row.iloc[0]['short'])
    return string

def get_schedule(data):
    db = data.copy()
    db = db[db['s'] <= 1]
    #db = db.tail(4)
    db = db[['game','home','away']]
    db = index_reset(db)
    db = db.sort_values(by=['game'])
    return db

def get_team_names(data,df,dc):
    db = data
    db = np.sort(db,axis=-1)
    a =[]
    for team in db:
        a.append(team)
    db = pd.DataFrame()
    db['short'] = pd.Series(df)
    db['colour'] = pd.Series(dc)
    db.insert(0,'team',pd.Series(a))
    return db

def fix_db_na(data):
    db = data.copy()
    if db['team'].isnull().values.any():
        for row in range(db.shape[0]):
            if pd.isna(db.iloc[row]['team']) == True:
                print(True)
                db.iloc[row]['team'] = get_long_name(db.iloc[row]['team'],data)
    return db

def index_reset(data):
    data = data.reset_index()
    data.pop('index')
    return data

def get_team_results(data,query):
    db = data[data['home'] == query]
    da = data[data['away'] == query]
    db = pd.concat([db,da])
    db = index_reset(db)
    return db

def get_team_brief(data,query,df):
    db = get_team_results(data,query)
    cols = ['game','s','csh','csa','combined','venue','links']
    for col in cols:
        db.pop(col)
    db = db.sort_values(by=['m','d'])
    db = index_reset(db)
    db['summary'] = '0'
    for i in range(0,db.shape[0]):
        if db.iloc[i]['home'] == query:
            x = df[df['team'] == db.iloc[i]['away']]#.reset_index()
            opponent = x.iloc[0]['short']
            outcome = db.iloc[i]['hr'] + ' A'
        else:
            x = df[df['team'] == db.iloc[i]['home']]#.reset_index() NOT SURE WHY THIS BROKE SUDDENLY
            opponent = x.iloc[0]['short']
            outcome = db.iloc[i]['ar'] + ' A'
        score = str(db.iloc[i]['hs']) + ' - ' + str(db.iloc[i]['as'])
        db.loc[i,'summary'] = outcome + ' ' + score +  ' ' + opponent
    db['team'] = query
    return db

def get_results_brief(data,dc):
    db = pd.DataFrame()
    for team in dc['team']:
        df = get_team_brief(data,team,dc)
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

def get_standings(data,season,ref):
    db = pd.DataFrame()
    if season == 1:
        data = data[data['s'] <= 1]
    if season == 2:
        data = data[data['s'] > 1]
    teams = ref['team']
    #teams = np.sort(teams,axis=-1)
    for team in teams:
        df = get_team_brief(data,team,ref)
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
    db = db.fillna(0)
    return db

def compare_standings(db,df,dc):
    a = []
    for team in dc['team']:
        rank1 = df[df['team'] == team]
        rank2 = db[db['team'] == team]
        if rank1.iloc[0]['rank'] == rank2.iloc[0]['rank']:
            change = 0
        else:
            change = rank1.iloc[0]['rank'] - rank2.iloc[0]['rank']
        a.append([team,change])
    db = pd.DataFrame(a)
    db = pd.DataFrame({'team': db.iloc[:][0], 'change': db.iloc[:][1]})
    db = db.sort_values(by=['change'],ascending=False)
    db = index_reset(db)
    return db

def clean_team_game(data,db,check): # Fix this section for teams that haven't played yet
    if check == 0:
        df = data.iloc[0]['team'] # Getting the name of the top team
    else:
        df = data.iloc[-1]['team'] # Getting the name of the bottom placed team
    if data.iloc[-1]['gp'] == 0 and check == 1:
        db = pd.DataFrame([(df,0,df,0)],columns=['home','hs','away','as']) # make an empty set if the game is empty
    else:
        df = db[(db['home'] == df) | (db['away'] == df)] # get appropirate game results for specified team
        db = index_reset(df)
        db = db.iloc[0][['home','hs','away','as']]
        db = pd.DataFrame(db)
        db = db.T
    return db

def get_short_name(data,dc):
    for team in data['home']:
        row = dc[dc['team'] == team]
        data['home'] = row.iloc[0]['short']
    for team in data['away']:
        row = dc[dc['team'] == team]
        data['away'] = row.iloc[0]['short']
    return data

def get_weeks_results(data,standings,dc):
    if data.iloc[0]['hr'] == 'E':
        db = pd.DataFrame([('NA',0,'NA',0)],columns=['home','hs','away','as'])
        big_win, top_team, low_team = db,db,db
        goals = 0
        return db,goals,big_win,top_team,low_team
    df = data
    month = df.iloc[-1]['m']
    week = df.iloc[-1]['d']
    db = df[df['m'] == month]
    db = db[db['d'] >= week - 6]
    db = db.sort_values(by=['game'],ascending=False)
    goals = db['hs'].sum() + db['as'].sum()
    max_home = db[db['hs'] == db['hs'].max()]
    max_away = db[db['as'] == db['as'].max()]
    if max_home.iloc[0]['hs'] > max_away.iloc[0]['as']:
        max_home_win = max_home
    else:
        max_home_win = max_away
    big_win = max_home_win[['home','hs','away','as']]
    big_win = index_reset(big_win)
    print('big_win', big_win)
    big_win = get_short_name(big_win,dc)
    big_win = pd.DataFrame(big_win.loc[0])
    big_win = big_win.T
    top_team = clean_team_game(standings,db,0)
    top_team = get_short_name(top_team,dc)
    low_team = clean_team_game(standings,db,1)
    low_team = get_short_name(low_team,dc)
    return db,goals,big_win,top_team,low_team

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

def get_stats_all(data,dc):
    db = pd.DataFrame()
    for team in dc['team']:
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
    if data.minutes.sum() == 0:
        db = pd.DataFrame([('NA',0,0,0,0)],columns=['team','name','number','minutes','goals'])
        return db
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
    if data.minutes.sum() == 0:
        db = pd.DataFrame([('NA',0,0,0,0)],columns=['team','name','number','minutes','assists'])
        return db
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
    if data.minutes.sum() == 0:
        db = pd.DataFrame([('NA',0,0,0,0,0,0,0,0,0,0,0,0)],columns=['team','name','number','minutes','goals','chances','assists','shots','shots on target','passes','crosses','duels','tackles'])
        return db
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
    #db = db.fillna(0)
    # .apply(lambda x: value if condition else new_value)
    #db['overall'] = db['overall'].apply(lambda x: [ y if y < 1.0 else 0.0 for y in x ])
    #db[db['overall'] > 1] = 0
    return db

def top_midfielders(data): # get the midfielders in the league
    if data.minutes.sum() == 0:
        db = pd.DataFrame([('NA',0,0,0,0,0,0,0,0,0,0,0,0,0)],columns=['team','name','number','minutes','goals','assists','touches','passes','pass accuracy','crosses','cross accuracy','chances','duels','tackles'])
        return db
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
    #db = db.fillna(0)
    #db[db['overall'] > 1] = 0
    return db

def top_defenders(data):  # get the defenders in the league
    if data.minutes.sum() == 0:
        db = pd.DataFrame([('NA',0,0,0,0,0,0,0,0,0)],columns=['team','name','number','minutes','tackles','tackles won','clearances','interceptions','duels','duels won'])
        return db
    player_information = data.copy()
    cols = ['team','name','number','minutes','tackles','tackles won','clearances','interceptions','duels','duels won']
    df = player_information[player_information['position'] == 'd'] # get the defenders where position = d
    db = df[cols]
    db = get_evaluation(db,df)
    db = index_reset(db)
    team = db.pop('team')
    db.insert(0,'team',team)
    #db = db.fillna(0)
    #db[db['overall'] > 1] = 0
    return db

def top_keepers(data):  # get the keepers in the league
    if data.minutes.sum() == 0:
        db = pd.DataFrame([('NA',0,0,0,0,0,0,0)],columns=['team','name','number','minutes','clean sheets','saves','shots faced','claimed crosses'])
        return db
    player_information = data.copy()
    cols = ['team','name','number','minutes','clean sheets','saves','shots faced','claimed crosses']
    df = player_information[player_information['position'] == 'g'] # get the goalkeepers where position = g
    db = df[cols]
    db = get_evaluation(db,df)
    db = index_reset(db)
    team = db.pop('team')
    db.insert(0,'team',team)
    #db = db.fillna(0)
    #db[db['overall'] > 1] = 0
    return db

def top_offenders(data):  # get the offences handed out in the league
    if data.minutes.sum() == 0:
        db = pd.DataFrame([('NA',0,0,0,0,0,0)],columns=['team','name','number','minutes','yellow','red','fouls conceded'])
        return db
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
    #db = db.fillna(0)
    #db[db['overall'] > 1] = 0
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
    if db.empty == True:
        db = pd.DataFrame([(0,0,0,0,q1,'D',q2,'D','empty',q1)],columns=['d','m','hs','as','home','hr','away','ar','summary','team'])
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
    home_win = round(get_gnb_prediction(query,x,y,[1,2]),2)
    draw = round(get_gnb_prediction(query,x,y,[1,1]),2)
    return home_win, draw

def get_match_prediction_away(query,x,y):
    away_win = get_gnb_prediction(query,x,y,[2,2])
    return away_win

def get_match_prediction(q1,q2,x1,y1,x2,y2):
    if len(x1) == 0:
        x = round(1/3,2)
        home_win, away_win,draw = x,x,x
        return home_win,away_win,draw
    home_win, draw = get_match_prediction_home(q1,x1,y1)
    away_win = get_match_prediction_away(q2,x2,y2)
    return home_win, draw, away_win

def get_team_form(data,query):
    db = data[data['team'] == query]
    db = pd.DataFrame(db['summary'])
    return db

def get_form_results(data,dc):
    db = pd.DataFrame()
    form = get_results_brief(data[data['s'] <= 1],dc)
    teams = data.home.unique()
    teams = np.sort(teams,axis=-1)
    for team in teams:
        df = get_team_form(form,team)
        #print(team,'\n',df)
        db[team] = pd.Series(df['summary'].values)
    db = db.T
    db = db.reset_index()
    db = db.fillna('E')
    return db

def get_roster(query,stats,team_ref): # use team stats to get the player information
    roster = get_stats_all(stats,team_ref)
    roster = roster[roster['team'] == query]
    roster = roster[['name','number']]
    roster.insert(2,'overall',0)
    roster = roster.head(11)
    return roster

def get_home_away_comparison(stats,game,team):
    db = stats[stats['game'] == game].copy()
    db = db[db['team'] == team]
    db = db.sort_values(by=['minutes'],ascending=False)
    db = db#[0:11]
    db = db['name']
    return db

def get_compare_roster(query,game,for_,mid_,def_,keep_,results,stats,team_ref):
    # going through the rated players to get the best players for each position
    # using game_h,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,results,team_stats
    if results.iloc[0]['hr'] == 'E': # check if games haven't been played
        db = get_roster(query,stats,team_ref)
        return db
    a = []
    for name in game:
        for f in range(for_.shape[0]):
            if for_.loc[f]['name'] == name:
                player = for_.loc[f]
                a.append(player)
        for f in range(mid_.shape[0]):
            if mid_.loc[f]['name'] == name:
                player = mid_.loc[f]
                a.append(player)
        for f in range(def_.shape[0]):
            if def_.loc[f]['name'] == name:
                player = def_.loc[f]
                a.append(player)
        for f in range(keep_.shape[0]):
            if keep_.loc[f]['name'] == name:
                player = keep_.loc[f]
                a.append(player)
    db = pd.DataFrame(a)
    db = db[['name','number','overall']]
    db = db.sort_values(by=['overall'],ascending=False)
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

def get_game_roster_prediction(stats,get_games,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,results,team_stats,team_ref):
    a = []
    for game in get_games: # cycle through the available games
        row = results[results['game'] == game] # select specific game results
        for team in row.iloc[0][['home','away']]: # cycle through the teams for each result
            if row.iloc[0]['home'] == team:
                result = row.iloc[0]['hr'] # get the appropriate result for each team
            else:
                result = row.iloc[0]['ar']
            if result == 'W': # alter the value for the model classifier
                result = 3
            elif result == 'D':
                result = 2
            else:
                result = 1
            game_check = get_home_away_comparison(stats,game,team)# get the roster for the team in the game
            # get the player overall score for each player in the game
            game_roster = get_compare_roster(team,game_check,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,results,team_stats,team_ref)
            game_roster = index_reset(game_roster)
            b = []
            b.append(game) # collecting all the information in the list
            b.append(team)
            for i in range(game_roster.shape[0]):
                b.append(game_roster.iloc[i]['overall']) # get the player overall score for each player in the game
            if len(b) < 16:
                i = int(16 - len(b))
                for j in range(0,i):
                    b.append(0)
            b.append(int(result))
            a.append(b)
    return a

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

def roster_pred(model,array):
    prediction = model.predict_proba([array]).flatten()
    df = pd.DataFrame(prediction)
    #print('score :',prediction)
    return df

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
