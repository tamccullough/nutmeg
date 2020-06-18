# canpl statistics
# Todd McCullough 2020
from datetime import date, timedelta
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

def get_longest_name(da,db,dc,team_ref):
    def get_long(data,dd):
        db = data.copy()
        for team in db['home']:
            row = dd[dd['short'] == team]
            db.at[0,'home'] = row.iloc[0]['team']
        for team in db['away']:
            row = dd[dd['short'] == team]
            db.at[0,'away'] = row.iloc[0]['team']
        return db
    da = get_long(da,team_ref)
    db = get_long(db,team_ref)
    dc = get_long(dc,team_ref)
    teams_in = pd.DataFrame([da.iloc[0]['home'],da.iloc[0]['away'],db.iloc[0]['home'],db.iloc[0]['away'],dc.iloc[0]['home'],dc.iloc[0]['away']],columns=['teams'])
    teams_in = teams_in.teams.unique()
    return teams_in

def get_short_name(data,dc):
    for team in data['home']:
        row = dc[dc['team'] == team]
        data.at[0,'home'] = row.iloc[0]['short']
    for team in data['away']:
        row = dc[dc['team'] == team]
        data.at[0,'away'] = row.iloc[0]['short']
    return data

def get_schedule(data):
    db = data.copy()
    db = db[db['s'] <= 1]
    #db = db.tail(4)
    db = db[['game','home','away']]
    db = index_reset(db)
    #db = db.sort_values(by=['game'])
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

def get_team_results(results_db,query): # get all the games played by the specific team
    home_games = results_db[results_db['home'] == query] # all home games
    away_games = results_db[results_db['away'] == query] # all away games
    team_results = pd.concat([home_games,away_games]) # create dataframe from the combined search
    team_results = index_reset(team_results)
    return team_results # return the dataframe

def get_team_brief(results_db,query,df):
    team_brief = get_team_results(results_db,query) # get team games function
    cols = ['game','s','csh','csa','combined','venue','links'] # create list to pop specific unecessary columns
    for col in cols:
        team_brief.pop(col)
    team_brief = team_brief.sort_values(by=['m','d']) # sort the values by month and day
    team_brief = index_reset(team_brief) # reset the index and drop the column named index that is created  from the old index
    team_brief['summary'] = '0' # create summary column holding 0 values
    for i in range(0,team_brief.shape[0]): # sort through the games
        if team_brief.iloc[i]['home'] == query:
            away_team = df[df['team'] == team_brief.iloc[i]['away']] # get the opponents name
            opponent = away_team.iloc[0]['short'] # convert the name to the short name
            outcome = team_brief.iloc[i]['hr'] + ' H' # combine home result with H
        else:
            home_team = df[df['team'] == team_brief.iloc[i]['home']] # get the opponents name
            opponent = home_team.iloc[0]['short'] # convert the name to the short name
            outcome = team_brief.iloc[i]['ar'] + ' A' # combine away result with A
        score = str(team_brief.iloc[i]['hs']) + ' - ' + str(team_brief.iloc[i]['as'])
        team_brief.loc[i,'summary'] = outcome + ' ' + score +  ' ' + opponent
    team_brief['team'] = query # create team column holding the team's name in all rows
    return team_brief # return the dataframe

def get_results_brief(data,dc):
    db = pd.DataFrame()
    for team in dc['team']:
        df = get_team_brief(data,team,dc)
        db = pd.concat([db,df])
    db = index_reset(db)
    return db

def get_club_statistics(team_results,query):

    df = team_results
    a = []
    cols = df.columns # get the columns of the dataframe

    def get_game_results(result_check):
        if result_check == 'W':
            r,w,l,d = 3,1,0,0
        elif result_check == 'L':
            r,w,l,d = 0,0,1,0
        elif result_check == 'D':
            r,w,l,d = 1,0,0,1
        else:
            r,w,l,d = 0,0,0,0
        return r,w,l,d
    for row in range(0,df.shape[0]):
        # cycling through to get the appropiate data for each game
        # depending on the results of, w,l or d
        # game played, win/loss/draw/ points, possible points,win,loss,draw,home-score,away-score,home-score,away-score
        if df.iloc[row]['home'] == query:
            if df.iloc[row]['hr'] != 'E':
                points,w,l,d = get_game_results(df.iloc[row]['hr'])
                vals = [1,points,3,w,l,d,df.iloc[row]['hs'],df.iloc[row]['as'],df.iloc[row]['hs'],df.iloc[row]['as'],0,0]
                a.append(vals)
        if df.iloc[row]['away'] == query:
            if df.iloc[row]['ar'] != 'E':
                points,w,l,d = get_game_results(df.iloc[row]['ar'])
                vals = [1,points,3,w,l,d,df.iloc[row]['hs'],df.iloc[row]['as'],df.iloc[row]['hs'],df.iloc[row]['as'],0,0]
                a.append(vals)
    db= pd.DataFrame(a,columns=['gp','pts','tpp','w','l','d','gf','ga','gfh','gah','gfa','gaa'])
    db = pd.DataFrame(db.sum())
    db = db.T
    return db


def get_standings(results,season_number,team_ref):
    standings = pd.DataFrame()
    # select the appropriate season, regular/championship
    if season_number == 1:
        results_db = results[results['s'] <= 1]
    if season_number == 2:
        results_db = results[results['s'] > 1]
    teams = team_ref['team']
    #teams = np.sort(teams,axis=-1)
    for team in teams: # loop through the teams of the league
        team_results = get_team_brief(results_db,team,team_ref)
        team_results = get_club_statistics(team_results,team)
        ppg = round(team_results['pts']/team_results['gp'],2) # calculate points per game
        gd = team_results['gf'] - team_results['ga'] #  calculate goal differential
        team_results.insert(0,'team',team)
        team_results.insert(4,'ppg',ppg)
        team_results.insert(8,'gd',gd)
        standings = pd.concat([standings,team_results])
    standings = standings.sort_values(by=['pts','w','gf'],ascending=False)
    standings = index_reset(standings)
    standings = standings.reset_index()
    standings = standings.rename(columns={'index':'rank'})
    standings['rank'] = standings['rank'] + 1
    standings = standings.fillna(0)

    columns = standings.select_dtypes(include=['float']).columns
    for column in columns:
        if column == 'ppg':
            continue
        standings[column] = standings[column].astype(int)

    return standings

def get_team_graphs(stats):
    g_cols = ['chances','goals','assists','pass-acc', 'cross-acc','shots', 's-target', 's-box','s-out-box','clearances','interceptions', 'yellow','shots faced','claimed crosses', 'cs']
    team_mean = stats.copy()
    goals = stats[['team','goals']]
    assists = stats[['team','assists']]
    team_mean = team_mean.select_dtypes(include=['float'])
    team_mean.insert(0,'team',stats['team'])
    team_mean = team_mean.groupby(['team']).mean()
    team_mean = team_mean[g_cols]
    team_mean['claimed crosses'] = team_mean['claimed crosses'] * 15
    team_mean['cs'] = team_mean['cs'] * 30
    team_mean['goals'] = goals.groupby(['team']).sum()
    team_mean['assists'] = assists.groupby(['team']).sum()
    team_mean['big chances'] = (team_mean['goals'] + 2) / team_mean['chances']
    team_mean['attacking plays'] = (team_mean['assists'] + 2) / team_mean['chances']
    team_mean['combination plays'] = team_mean['assists'] / team_mean['goals']
    team_mean['accuracy'] = team_mean['pass-acc'] + team_mean['cross-acc']
    team_mean['defending'] = team_mean['clearances'] * team_mean['interceptions']
    team_mean['chance creation'] = (team_mean['shots'] + team_mean['s-box'] + team_mean['s-out-box']) * team_mean['s-target']
    team_mean['finishing'] = team_mean['chance creation'] * team_mean['goals']
    team_mean = team_mean.rename(columns={'cs':'clean sheets'})

    for col in team_mean.columns:
        if team_mean[col].max() > 1.0:
            team_mean[col] = team_mean[col] / team_mean[col].max()
        if team_mean[col].max() < 0.2:
            team_mean[col] = team_mean[col] * 5
        else:
            continue

    for col in team_mean.columns:
        team_mean[col] = team_mean[col] - 0.1

    team_mean = team_mean[['clean sheets','big chances', 'attacking plays', 'combination plays', 'accuracy','defending', 'chance creation', 'finishing']]
    team_mean = team_mean.reset_index()
    return team_mean

def make_radar(data,team_ref,year):
    team = data['team']
    info = team_ref[team_ref['team'] == team]
    colour1 = info['colour1'].values
    colour1 = colour1[0]
    colour2 = info['colour2'].values
    colour2 = colour2[0]
    # number of variable
    categories=list(team_graphs)[1:]
    N = len(categories)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = data.drop('team').values.flatten().tolist()
    values += values[:1]
    values

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=14)

    # Draw ylabels
    ax.set_title(data['team'], color=colour1, size=24)
    ax.set_rlabel_position(0)
    plt.yticks([0.25,0.5,0.75], ["0.25","0.50","0.75"], color='silver', size=12)
    plt.ylim(0,1)
    # Plot data
    ax.plot(angles, values, linewidth=8, linestyle='solid', color=colour1)

    # Fill area
    ax.fill(angles, values, colour2, alpha=0.4)
    filename = f'static/images/{year}/cpl-{year}-{team}-radar.png'
    plt.savefig(filename)

def compare_standings(standings_current,standings_old,team_ref):
    # getting the change in team standings between current week and previous week
    a = []
    for team in team_ref['team']:
        rank1 = standings_old[standings_old['team'] == team] # get team's previous rank
        rank2 = standings_current[standings_current['team'] == team] # get teams current rank
        # calculate the change in team's ranking
        if rank1.iloc[0]['rank'] == rank2.iloc[0]['rank']:
            change = 0
        else:
            change = rank1.iloc[0]['rank'] - rank2.iloc[0]['rank']
        a.append([team,change])
    current_rankings = pd.DataFrame(a)
    current_rankings = pd.DataFrame({'team': current_rankings.iloc[:][0], 'change': current_rankings.iloc[:][1]})
    current_rankings = current_rankings.sort_values(by=['change'],ascending=False) # sort by change
    current_rankings = index_reset(current_rankings)
    return current_rankings

def clean_team_game(data,db,check):
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

def get_weeks_results(data,standings,team_ref):
    if data.iloc[0]['hr'] == 'E':
        db = pd.DataFrame([('NA',0,'NA',0)],columns=['home','hs','away','as'])
        big_win, top_team, low_team,other_team = db,db,db,db
        goals = 0
        return db,goals,big_win,top_team,low_team,other_team
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
    big_win = get_short_name(big_win,team_ref)
    big_win = pd.DataFrame(big_win.loc[0])
    big_win = big_win.T
    top_team = clean_team_game(standings,db,0)
    top_team = get_short_name(top_team,team_ref)
    low_team = clean_team_game(standings,db,1)
    low_team = get_short_name(low_team,team_ref)
    teams_in = get_longest_name(big_win,top_team,low_team,team_ref)
    other_team = db[(~db['home'].isin(teams_in)) | (~db['away'].isin(teams_in))]
    other_team = index_reset(other_team)
    other_team = pd.DataFrame(other_team.loc[0][['home','hs','away','as']])
    other_team = other_team.T
    other_team = get_short_name(other_team,team_ref)
    return db,goals,big_win,top_team,low_team,other_team

def get_team_stats(data,query):
    team_stats = data[data['team'] == query]
    names = team_stats['name'].unique()
    information = data.copy()
    team_stats.pop('number')
    team_stats = team_stats.groupby(['name']).sum()
    team_stats.insert(0,'last','empty')
    team_stats.insert(0,'first','empty')
    team_stats.insert(0,'position','empty')
    team_stats.insert(0,'number',0)
    i = 0
    for name in names:
        player = information[information['name'] == name].head(1)
        team_stats.at[name,'first'] = player.iloc[0]['first']
        team_stats.at[name,'last'] = player.iloc[0]['last']
        team_stats.at[name,'number'] = int(player.iloc[0]['number'])
        team_stats.at[name,'position'] = player.iloc[0]['position']
        team_stats.at[name,'pass-acc'] = player.iloc[0]['pass-acc'].mean()
        team_stats.at[name,'cross-acc'] = player.iloc[0]['cross-acc'].mean()
    team_stats = team_stats.reset_index()
    return team_stats

def get_stats_all(stats,team_ref):
    stats_all = pd.DataFrame()
    for team in team_ref['team']:
        df = get_team_stats(stats,team)
        short_team = get_shortest_name(team,team_ref)
        df.insert(0,'team',short_team)
        stats_all = pd.concat([stats_all,df])
    stats_all = index_reset(stats_all)
    return stats_all

# get associated information for players league wide and calculate an overall score for each position
def get_evaluation(condensed_player_info,full_player_info):
    names = condensed_player_info.name.unique() # grab the list of names at the specified position
    eval_ = condensed_player_info.describe().T # get the evalution scores
    checks = condensed_player_info.columns[4:] # slice away the first three columns (name,number,postion) not needed
    condensed_player_info['overall'] = 0.0 # create the final column overall
    condensed_player_info = condensed_player_info.set_index('name') # set the index to the player name to search for a specific player
    for name in names: # iterate through the names in the lisst
        player = full_player_info[full_player_info['name'] == name].head(1) # get the players details
        a = [] # create an empty array to store the scores
        for check in checks: # iterate through the columns of remaining data
            result = player.iloc[0][check] / eval_['max'][check] # calculate the score for the value found value/max
            a.append(result) # append the result into the list
            overall = str(sum(a) / len(checks)) #calculate the final score sum(list) / num of checks
            overall = overall[0:4]
            condensed_player_info.at[name,'overall'] = overall # assign the value as the overall score
    condensed_player_info = condensed_player_info.reset_index() # reset the index, making the name column a column again
    condensed_player_info = condensed_player_info.sort_values(by=['overall'],ascending=False) # sort using overall, descending
    return condensed_player_info

def top_tracked(team_stats,tracked):
    cols = ['team','name','position','number','minutes',tracked]
    if team_stats.minutes.sum() == 0:
        lst = ['rank']
        lst.extend(cols)
        tracked_player_stat = pd.DataFrame([(0,'NA','NA',0,0,0,0)],columns=lst)
        return tracked_player_stat
    player_information = team_stats.copy()
    tracked_player_stat = player_information[cols]
    tracked_player_stat = tracked_player_stat.sort_values(by=[tracked],ascending=False)
    tracked_player_stat = tracked_player_stat.reset_index()
    tracked_player_stat.pop('index')
    team = tracked_player_stat.pop('team')
    tracked_player_stat.insert(0,'team',team)
    tracked_player_stat = tracked_player_stat[tracked_player_stat[tracked] >= 1]
    rank = tracked_player_stat.index + 1
    tracked_player_stat.insert(0,'rank',rank)

    columns = tracked_player_stat.select_dtypes(include=['float']).columns
    for column in columns:
        if column == 'overall':
            continue
        tracked_player_stat[column] = tracked_player_stat[column].astype(int)

    return tracked_player_stat

def top_position(team_stats,position): # get the forwards in the league
    colf = ['team','name','number','position','minutes','goals','chances','assists','shots','s-target','passes','crosses','duels','tackles']
    colm = ['team','name','number','position','minutes','goals','assists','touches','passes','pass-acc','crosses','cross-acc','chances','duels','tackles']
    cold = ['team','name','number','position','minutes','tackles','t-won','clearances','interceptions','duels','d-won']
    colg = ['team','name','number','position','minutes','cs','saves','shots faced','claimed crosses']

    def create_blank_frame(columns):
        lst = [np.zeros(len(columns), dtype=int)]
        dataframe = pd.DataFrame(lst,columns=columns)
        dataframe['team'] = 'NA'
        dataframe['name'] = 'NA'
        dataframe['overall'] = 0
        return dataframe

    if team_stats.minutes.sum() == 0:
        if position == 'f':
            condensed_player_info = create_blank_frame(colf)
        if position == 'm':
            condensed_player_info = create_blank_frame(colm)
        if position == 'd':
            condensed_player_info = create_blank_frame(cold)
        if position == 'g':
            condensed_player_info = create_blank_frame(colg)
        return condensed_player_info

    if position == 'f':
        cols = colf
    if position == 'm':
        cols = colm
    if position == 'd':
        cols = cold
    if position == 'g':
        cols = colg

    player_information = team_stats.copy() # load player information
    full_player_info = player_information[player_information['position'] == position] # filter the players by selected position
    condensed_player_info = full_player_info[cols] # select specific columns associated with the evaluation
    condensed_player_info = get_evaluation(condensed_player_info,full_player_info) # condensed Dataframe and full Dataframe being passes
    condensed_player_info = index_reset(condensed_player_info)
    names = condensed_player_info.name.unique() # get the names of the players who fit the criteria
    condensed_player_info = condensed_player_info.set_index('name') # set the index to the name column to make the search possible

    for name in names:
        player = full_player_info[full_player_info['name'] == name].head(1) # forwards main purpose is to score goals
        if player.iloc[0]['assists'] > 2.0: # reward getting more than 3 assists
            condensed_player_info.at[name,'overall'] = condensed_player_info.at[name,'overall'] + 0.1
        if position == 'm':
            if player.iloc[0]['goals'] >= 5.0: # reward scoring greater than 5 goals
                condensed_player_info.at[name,'overall'] = condensed_player_info.at[name,'overall'] + 0.1
            if player.iloc[0]['pass-acc'] >= 0.85: # reward scoring greater than 5 goals
                condensed_player_info.at[name,'overall'] = condensed_player_info.at[name,'overall'] + 0.1
        if position == 'f':
            if (player.iloc[0]['goals'] <= 2.0 and player.iloc[0]['minutes'] >= 1000.0): # if player scores less than 2 & has minutes greater than 1000
                condensed_player_info.at[name,'overall'] = condensed_player_info.at[name,'overall'] - 0.1
            if player.iloc[0]['goals'] >= 8.0: # reward scoring greater than 8 goals
                condensed_player_info.at[name,'overall'] = condensed_player_info.at[name,'overall'] + 0.1
        if position == 'd':
            if (player.iloc[0]['interceptions'] > 200.0 and player.iloc[0]['minutes'] >= 1000.0): # if player scores less than 2 & has minutes greater than 1000
                condensed_player_info.at[name,'overall'] = condensed_player_info.at[name,'overall'] + 0.1
            if player.iloc[0]['d-won'] > 110.0: # reward scoring greater than 8 goals
                condensed_player_info.at[name,'overall'] = condensed_player_info.at[name,'overall'] + 0.1

    condensed_player_info = condensed_player_info.sort_values(by=['overall'],ascending=False)
    condensed_player_info = condensed_player_info.reset_index()
    team = condensed_player_info.pop('team')
    condensed_player_info.insert(0,'team',team)

    columns = condensed_player_info.select_dtypes(include=['float']).columns
    for column in columns:
        if column == 'overall':
            condensed_player_info[column] = condensed_player_info[column].round(4).astype(str)
            condensed_player_info[column] = condensed_player_info[column].astype(float)
            continue
        condensed_player_info[column] = condensed_player_info[column].astype(int)

    return condensed_player_info

def top_offenders(data):  # get the offences handed out in the league
    cols = ['team','name','position','number','minutes','yellow','red','f-conceded']
    if data.minutes.sum() == 0:
        top_offenders = pd.DataFrame([('NA','NA','NA',0,0,0,0,0)],columns=cols)
        return top_offenders
    player_information = data.copy()
    top_offenders = player_information[cols]
    top_offenders = get_evaluation(top_offenders,player_information)
    top_offenders = top_offenders.sort_values(by=['red','yellow'],ascending=False)
    top_offenders = top_offenders.reset_index()
    top_offenders.pop('index')
    team = top_offenders.pop('team')
    top_offenders.insert(0,'team',team)

    columns = top_offenders.select_dtypes(include=['float']).columns
    for column in columns:
        if column == 'overall':
            continue
        top_offenders[column] = top_offenders[column].astype(int)

    return top_offenders

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

def get_team_form(data,query):
    db = data[data['team'] == query]
    db = pd.DataFrame(db['summary'])
    return db

def get_form_results(data,dc):
    db = pd.DataFrame()
    form = get_results_brief(data[data['s'] <= 1],dc)
    teams = dc['team']
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
    roster = roster[['name','number','position']]
    roster.insert(3,'overall',0)
    roster = index_reset(roster)
    return roster

def get_roster_overall(query,stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info): # use team stats to get the player information
    team = get_shortest_name(query,team_ref)
    def get_score(data,name):
        db = data[data['name'] == name]
        if db.empty:
            previous = player_info[player_info['name'] == name]
            if previous.empty:
                db = 0
            else:
                previous = previous['overall'].values
                db = previous[0]
        else:
            db = db['overall'].values
            db = db[0]
        return db
    def get_image(data,name):
        db = data[data['name'] == name]
        if db['image'].empty:
            db = 'empty.jpg'
        else:
            db = db['image'].values
            db = db[0]
        return db
    def get_link(data,name):
        db = data[data['name'] == name]
        if db['link'].empty:
            db = 'https://en.wikipedia.org/wiki/Canadian_Premier_League'
        else:
            db = db['link'].values
            db = db[0]
        return db
    def get_flag(data,name):
        db = data[data['name'] == name]
        if db['flag'].empty:
            db = 'empty.png'
        else:
            db = db['flag'].values
            db = db[0]
        return db
    roster = get_stats_all(stats,team_ref)
    roster = roster[roster['team'] == team].copy()
    roster = roster[['name','first','last','number','position']] # scale the dataframe down to what we need
    #roster.insert(3,'overall',a)
    a = []
    b = []
    c = []
    d = []
    for i in range(0,roster.shape[0]):
        if roster.iloc[i]['position'] == 'f':
            score = str(get_score(rated_forwards,roster.iloc[i]['name']))
            a.append(score[0:4])
            b.append(get_image(player_info,roster.iloc[i]['name']))
            c.append(get_flag(player_info,roster.iloc[i]['name']))
            d.append(get_link(player_info,roster.iloc[i]['name']))
        if roster.iloc[i]['position'] == 'm':
            score = str(get_score(rated_midfielders,roster.iloc[i]['name']))
            a.append(score[0:4])
            b.append(get_image(player_info,roster.iloc[i]['name']))
            c.append(get_flag(player_info,roster.iloc[i]['name']))
            d.append(get_link(player_info,roster.iloc[i]['name']))
        if roster.iloc[i]['position'] == 'd':
            score = str(get_score(rated_defenders,roster.iloc[i]['name']))
            a.append(score[0:4])
            b.append(get_image(player_info,roster.iloc[i]['name']))
            c.append(get_flag(player_info,roster.iloc[i]['name']))
            d.append(get_link(player_info,roster.iloc[i]['name']))
        if roster.iloc[i]['position'] == 'g':
            score = str(get_score(rated_keepers,roster.iloc[i]['name']))
            a.append(score[0:4])
            b.append(get_image(player_info,roster.iloc[i]['name']))
            c.append(get_flag(player_info,roster.iloc[i]['name']))
            d.append(get_link(player_info,roster.iloc[i]['name']))
    roster['overall'] = a
    roster['flag'] = c
    roster['link'] = d
    roster.insert(0,'image',b)
    #roster['image'] = b
    roster = index_reset(roster)
    roster.pop('name')
    return roster

def get_home_away_comparison(stats,game,team):
    db = stats[stats['game'] == game].copy()
    db = db[db['team'] == team]
    db = db.sort_values(by=['minutes'],ascending=False)
    db = db#[0:11]
    db = db['name']
    return db

def get_compare_roster(results,query,stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info):
    roster = get_roster_overall(query,stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info)
    def get_player(data,string):
        dz = data[data['position'] == string]
        dz = dz[['first','last','number','position','overall']]
        dz.insert(0,'name',dz['first'] + ' ' + dz['last'])
        dz.pop('first')
        dz.pop('last')
        return dz
    dk = get_player(roster,'g')
    dk = dk.sort_values(by=['overall'],ascending=False)
    dd = get_player(roster,'d')
    dd = dd.sort_values(by=['overall'],ascending=False)
    dm = get_player(roster,'m')
    dm = dm.sort_values(by=['overall'],ascending=False)
    df = get_player(roster,'f')
    df = df.sort_values(by=['overall'],ascending=False)
    db = pd.concat([dk[0:1],dd[0:4],dm[0:4],df[0:2]])
    db = index_reset(db)
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

'''def get_game_roster_prediction(stats,get_games,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,results,team_stats,team_ref,player_info):
    a = []
    for game in get_games: # cycle through the available games
        row = results[results['game'] == game] # select specific game results
        for team in row.iloc[0][['home','away']]: # cycle through the teams for each result
            if row.iloc[0]['home'] == team:
                result = row.iloc[0]['hr'] # get the appropriate result for each team
                score = row.iloc[0]['hs']
            else:
                result = row.iloc[0]['ar']
                score = row.iloc[0]['as']
            if result == 'W': # alter the value for the model classifier
                result = 3
            elif result == 'D':
                result = 2
            else:
                result = 1
            game_check = get_home_away_comparison(stats,game,team)# get the roster for the team in the game
            # get the player overall score for each player in the game
            game_roster = get_compare_roster(results,team,team_stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info)
            game_roster = index_reset(game_roster)
            b = []
            b.append(game) # collecting all the information in the list
            b.append(team)
            for i in range(game_roster.shape[0]):
                overall = game_roster.iloc[i]['overall']
                b.append(float(overall)) # get the player overall score for each player in the game
            if len(b) < 16:
                i = int(16 - len(b))
                for j in range(0,i):
                    b.append(0)
            b.append(int(result))
            b.append(int(score))
            a.append(b)
    return a'''

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

def get_power_rankings(standings,standings_old,team_ref,results,previous_rankings):
    a = []
    for team in team_ref['team']:
        old_rank = previous_rankings[previous_rankings['team'] == team]
        old_rank = old_rank['rank'].values
        old_rank = old_rank[0]
        form = get_five_game_form(results,team)
        form = str(round(form.at['w',0],1))+'-'+str(form.at['l',0])+'-'+str(form.at['d',0])
        crest = team_ref[team_ref['team'] == team]
        colour = crest['colour'].values
        colour = colour[0]
        crest = crest['crest'].values
        crest = crest[0]

        rank1 = standings_old[standings_old['team'] == team]
        rank2 = standings[standings['team'] == team]

        if rank1.iloc[0]['rank'] == 1:
            bonus = 4
        elif rank1.iloc[0]['rank'] == 2:
            bonus = 3
        elif rank1.iloc[0]['rank'] == 3:
            bonus = 2
        else:
            bonus =0

        if standings.iloc[0]['gp'] == 0:
            bonus = 0

        if rank1.iloc[0]['rank'] == rank2.iloc[0]['rank']:
            change = 0
        else:
            change = (rank1.iloc[0]['rank'] - rank2.iloc[0]['rank']) * - 1

        if rank1.iloc[0]['gd'] == rank2.iloc[0]['gd']:
            gd_bonus = 0
        else:
            gd_bonus = (rank1.iloc[0]['gd'] - rank2.iloc[0]['gd']) * - 1

        if rank1.iloc[0]['ga'] == rank2.iloc[0]['ga']:
            ga_nerf = 0
        else:
            ga_nerf = (rank1.iloc[0]['ga'] - rank2.iloc[0]['ga']) * - 1

        if rank1.iloc[0]['w'] == rank2.iloc[0]['w']:
            w_bonus = 0
        else:
            w_bonus = (rank1.iloc[0]['w'] - rank2.iloc[0]['w']) * - 1

        goal_bonus = gd_bonus - ga_nerf
        change = change + bonus + goal_bonus + w_bonus

        a.append([team,form,old_rank,change,goal_bonus,w_bonus,crest,colour])
    power_rankings = pd.DataFrame(a,columns = ['team','form','old_rank','change','goal_bonus','w_bonus','crest','colour'])
    power_rankings = power_rankings.sort_values(by=['change'],ascending=False)
    power_rankings = index_reset(power_rankings)
    rank = power_rankings.index + 1
    power_rankings.insert(0,'rank',rank)
    power_rankings['previous'] = (power_rankings['rank'] - power_rankings['old_rank'])*-1
    return power_rankings

def get_best_eleven(team_stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info):
    def get_image(data,name):
        db = data[data['name'] == name]
        if db['image'].empty:
            db = 'empty.jpg'
        else:
            db = db['image'].values
            db = db[0]
        return db
    def get_link(data,name):
        db = data[data['name'] == name]
        if db['link'].empty:
            db = 'https://en.wikipedia.org/wiki/Canadian_Premier_League'
        else:
            db = db['link'].values
            db = db[0]
        return db
    def get_flag(data,name):
        db = data[data['name'] == name]
        if db['flag'].empty:
            db = 'empty.png'
        else:
            db = db['flag'].values
            db = db[0]
        return db

    check = team_stats.describe()
    if check.loc['max']['minutes'] == 0:
        best_eleven = pd.read_csv('datasets/2019/cpl-2019-best_eleven.csv')
        #best_eleven = pd.DataFrame([['empty.jpg','empty.png',0,'NA',0,'NA','NA','https://canpl.ca/']],columns=['image','flag','number','position','overall','first','last','link'])
        #best_eleven = pd.concat([best_eleven]*11)
        return best_eleven
    else:
        roster = team_stats.copy()
        roster = roster[['name','first','last']]

        top_keeper = rated_keepers.head(1)
        top_keeper = top_keeper[['name','number','position','overall']]
        top_defenders = rated_defenders.iloc[0:3][['name','number','position','overall']]
        top_midfielders = rated_midfielders.iloc[0:5][['name','number','position','overall']]
        top_forwards = rated_forwards.iloc[0:2][['name','number','position','overall']]
        best_eleven = pd.DataFrame(columns=['name','number','position','overall'])
        best_eleven = pd.concat([best_eleven,top_keeper,top_defenders,top_midfielders,top_forwards])
        a,b,c,d,e = [],[],[],[],[]


        names = best_eleven['name'].values

        for i in range(0,best_eleven.shape[0]):
            player = roster[roster['name'] == best_eleven.iloc[i]['name']]
            player= index_reset(player)
            first = player.iloc[0]['first']
            last = player.iloc[0]['last']
            a.append(first)
            b.append(last)
            c.append(get_image(player_info,best_eleven.iloc[i]['name']))
            d.append(get_flag(player_info,best_eleven.iloc[i]['name']))
            e.append(get_link(player_info,best_eleven.iloc[i]['name']))

        best_eleven.insert(0,'image',c)
        best_eleven.insert(1,'first',a)
        best_eleven.insert(2,'last',b)
        best_eleven.insert(3,'flag',d)
        best_eleven['link'] = e
        best_eleven.pop('name')
        best_eleven = index_reset(best_eleven)
        return best_eleven

def roster_regressor_pred(model,array):
    prediction = model.predict([array]).flatten()
    df = pd.DataFrame(prediction)
    return df

def get_final_score_prediction(model,q1_roster,q2_roster,home_win_new,away_win_new):

    def roster_pred(model,array):
        prediction = model.predict([array]).flatten()
        return prediction

    def final_score_fix(home_score,away_score,home_win_new,away_win_new):
        if home_win_new > away_win_new and home_score < away_score: # fix the score prediction - if the probability of home win > away win and score doesn't reflect it
            print(home_score,away_score)
            old = home_score
            home_score = away_score # change the predicted score to reflect that
            away_score = old
            return home_score,away_score
        elif home_win_new < away_win_new and home_score > away_score: # else the probability of home win < away win
            print(home_score,away_score)
            old = away_score
            away_score = home_score
            home_score = old # change the predicted score to reflect that
            return home_score,away_score
        elif home_win_new < away_win_new and home_score == away_score:
            print(home_score,away_score)
            home_score = home_score - 1
            return home_score,away_score
        elif home_win_new > away_win_new and home_score == away_score:
            print(home_score,away_score)
            away_score = away_score - 1
            return home_score,away_score
        elif home_win_new == away_win_new:
            away_score = home_score
            print(home_score,away_score)
            return home_score,away_score
        else:
            print(home_score,away_score)
            return home_score,away_score

    def score(num): #improve this later for greater predictions
        new_score = int(round(num,0)) # convert the float value to int and round it
        return new_score

    q1_pred = roster_pred(model,q1_roster)
    q1_s = score(q1_pred[0])
    q2_pred = roster_pred(model,q2_roster)
    q2_s = score(q2_pred[0])
    home_score, away_score = final_score_fix(q1_s, q2_s,home_win_new,away_win_new)
    return home_score, away_score


def get_game_roster_prediction(get_games,results,stats,team_ref,player_info):
    def index_reset(data):
        data = data.reset_index()
        data.pop('index')
        return data

    a = []
    for game in get_games: # cycle through the available games
        row = results[results['game'] == game] # select specific game results
        players = stats[stats['game'] == game]
        for team in row.iloc[0][['home','away']]: # cycle through the teams for each result
            if row.iloc[0]['home'] == team:
                result = row.iloc[0]['hr'] # get the appropriate result for each team
                score = row.iloc[0]['hs']
            else:
                result = row.iloc[0]['ar']
                score = row.iloc[0]['as']
            if result == 'W': # alter the value for the model classifier
                result = 3
            elif result == 'D':
                result = 2
            else:
                result = 1
            game_roster = players[players['team'] == team]
            game_roster = index_reset(game_roster)
            game_roster = game_roster['name']
            b = []
            b.append(game) # collecting all the information in the list
            b.append(team)
            for i in range(game_roster.shape[0]):
                name = game_roster.iloc[i]
                overall = player_info[player_info['name']==name]['overall']
                overall = overall.values
                b.append(float(overall[0])) # get the player overall score for each player in the game
            if len(b) < 16:
                i = int(16 - len(b))
                for j in range(0,i):
                    b.append(0)
            b.append(int(result))
            b.append(int(score))
            a.append(b)
    return a

def get_team_files(schedule,team_ref):
    team1 = get_shortest_name(schedule.iloc[0]['home'],team_ref)
    team2 = get_shortest_name(schedule.iloc[0]['away'],team_ref)
    team3 = get_shortest_name(schedule.iloc[1]['home'],team_ref)
    team4 = get_shortest_name(schedule.iloc[1]['away'],team_ref)
    team5 = get_shortest_name(schedule.iloc[2]['home'],team_ref)
    team6 = get_shortest_name(schedule.iloc[2]['away'],team_ref)
    team7 = get_shortest_name(schedule.iloc[3]['home'],team_ref)
    team8 = get_shortest_name(schedule.iloc[3]['away'],team_ref)

    return team1, team2, team3, team4, team5, team6, team7, team8

def update_player_info(year,player_info,rated_forwards,rated_midfielders,rated_defenders,rated_keepers):
    today = date.today() - timedelta(5)
    day = today.strftime("%d_%m_%Y")
    print(day)
    rated_forwards.to_csv(f'datasets/{year}/cpl-{year}-forwards-{day}.csv',index=False)
    rated_midfielders.to_csv(f'datasets/{year}/cpl-{year}-midfielders-{day}.csv',index=False)
    rated_defenders.to_csv(f'datasets/{year}/cpl-{year}-defenders-{day}.csv',index=False)
    rated_keepers.to_csv(f'datasets/{year}/cpl-{year}-keepers-{day}.csv',index=False)

    def get_player_score(data,name):
        name = [name]
        if data[data['name'].isin(name)].empty:
            pass
        else:
            overall = data[data['name'].isin(name)]
            new_overall = overall['overall'].values
            return new_overall

    combine = [rated_forwards,rated_midfielders,rated_defenders,rated_keepers]
    names = player_info['name'].values
    a = []
    for name in names:
        j = 1
        for i in range(0,4):
            score = get_player_score(combine[i],name)
            if score == None:
                j += 1
                pass
            if score != None:
                overall = score[0]
                a.append(overall)
            if j == 5:
                overall = 0.0
                a.append(overall)
    player_info['overall'] = a
    return player_info
