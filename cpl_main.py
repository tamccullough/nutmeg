# canpl statistics
# Todd McCullough 2020
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from math import pi
#import matplotlib.pyplot as plt
import os
import random
import re

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
import statistics

current_year = date.today().strftime('%Y')

ccd_names = {'Atlético Ottawa' : 'Atlético Ottawa',
            'Cavalry' : 'Cavalry FC',
            'Edmonton' : 'FC Edmonton',
            'Forge' : 'Forge FC',
            'HFX Wanderers' : 'HFX Wanderers FC',
            'Pacific' : 'Pacific FC',
            'Valour' : 'Valour FC',
            'York United' : 'York United FC',
            'York United' : 'York9 FC','York9' : 'York9 FC'}

team_names = {'Atlético Ottawa' : 'ato',
              'Cavalry FC' : 'cav','Cavalry' : 'cav',
              'FC Edmonton' : 'fce','Edmonton' : 'fce',
              'Forge FC' : 'for','Forge' : 'for',
              'HFX Wanderers FC' : 'hfx','HFX Wanderers' : 'hfx',
              'Pacific FC' : 'pac','Pacific' : 'pac',
              'Valour FC' : 'val','Valour' : 'val',
              'York United FC' : 'yor','York United' : 'yor',
              'York9 FC' : 'y9','York9' : 'y9'}
name_fix = {'United' : 'York United FC'}

def get_weekday():
    weekDays = ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')
    current_year = date.today().strftime('%Y')
    month = datetime.today().strftime('%B')
    today = datetime.today()
    day = datetime.today().strftime('%d')

    weekday_num = today.weekday()
    weekday = weekDays[weekday_num]
    return month, day, weekday

def get_shortest_name(team_name):
    return team_names[team_name]

def index_reset(data):
    data = data.reset_index(drop=True)
    return data

def get_schedule(data):
    db = data.copy()
    #db = db[db['s'] <= 1]
    #db = db.tail(4)
    db = db[['game','home','away']]
    db = index_reset(db)
    #db = db.sort_values(by=['game'])
    return db

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

def power_rankings(standings,standings_old,prev_rankings,results,team_ref):
    # apply the sum aggregate to the gd values of the dataframe
    agg_func_math = {'gd':'sum'}
    # compare current standings to previous standings to see team results change
    compare = pd.concat([standings,standings_old]).groupby(['team']).agg(agg_func_math).round(2)
    compare['gd'] = compare['gd'].apply(lambda x: round((x + abs(min(compare['gd'])))/(abs(min(compare['gd']))+abs(max(compare['gd']))),2))
    values = []
    for c,r in compare.sort_values(by='gd',ascending=False).iterrows():
        values.append([c,r['gd'].item()])
    power = pd.DataFrame(values,columns=['team','rank'])
    power['rank'] = power.index+1
    def f(x):
        return prev_rankings[prev_rankings['team'] == x['team']]['rank'].values[0] - x['rank']
    power['move'] = power.apply(f,axis=1)
    power['form'] = power['team'].apply(lambda x: '-'.join([str(i) for i in get_five_game_form(results,x,0).T.values[0]]))
    return power

def get_results_brief(results,year):

    team_change = {'Atlético Ottawa' : 'ATO',
              'Cavalry FC' : 'CAV','Cavalry' : 'CAV',
              'FC Edmonton' : 'EDM','Edmonton' : 'EDM',
              'Forge FC' : 'FOR','Forge' : 'FOR',
              'HFX Wanderers FC' : 'HFX','HFX Wanderers' : 'HFX',
              'Pacific FC' : 'PAC','Pacific' : 'PAC',
              'Valour FC' : 'VAL','Valour' : 'VAL',
              'York United FC' : 'YOR','York United' : 'YOR'}

    results_brief = pd.DataFrame()
    temp = results[['d','m','hs','as','home','hr','away','ar']]
    for team in sorted(results['home'].unique()):
        db = temp[temp['home'] == team].reset_index(drop=True)
        db['summary'] = ''
        db['team'] = team
        db['summary'] = db['hr'] + ' h '+ db['hs'].apply(lambda x:str(x)) + ' ' + db['as'].apply(lambda x: str(x)) + ' ' + db['away'].apply(lambda x: team_change[x])
        results_brief = pd.concat([results_brief,db])

    for team in sorted(results['away'].unique()):
        df = temp[temp['away'] == team].reset_index(drop=True)
        df['summary'] = ''
        df['team'] = team
        df['summary'] = df['ar'] + ' h '+ df['hs'].apply(lambda x:str(x)) + ' ' + df['as'].apply(lambda x: str(x)) + ' '+ df['home'].apply(lambda x: team_change[x])
        results_brief = pd.concat([results_brief,df])

    results_brief['scatter'] = results_brief['home'].apply(lambda x: team_change[x]) + ' vs '+ results_brief['away'].apply(lambda x: team_change[x]) + f' ({year}' + results_brief['m'].apply(lambda x: '-'+str(x) if len(str(x)) > 1 else '-0'+str(x)) + results_brief['d'].apply(lambda x: '-'+str(x)+")" if len(str(x)) > 1 else '-0'+str(x)+")")

    results_brief['result'] = results_brief['summary'].apply(lambda x: x[:1])

    return results_brief

def get_club_statistics(team_results,query):

    df = team_results.copy()
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
                vals = [1,points,3,w,l,d,df.iloc[row]['as'],df.iloc[row]['hs'],0,0,df.iloc[row]['as'],df.iloc[row]['hs']]
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

# graphing related functions


def get_90(dataframe):
    data = dataframe.copy()
    cols = data.columns.tolist()
    r = ['team','minutes','name','number','position','overall']
    for x in r:
        cols.remove(x)
    for col in cols:
        if (col == 'pass-acc') or (col == 'cross-acc'):
            pass
        else:
            col_max = data[col].max()
            data[col] = [round(x/(y/90),2) if (x > 0) & (y > 0) else 0 for x,y in zip(data[col],data['minutes'])]
    data['new_overall'] = 0.0
    for i in range(data.shape[0]):
        summed = data.iloc[i][cols].sum()
        data.at[i,'new_overall'] = summed
    data['new_overall'] = round((data['new_overall'] / data['new_overall'].max()) - 0.1,2)
    data = data.sort_values(by='new_overall',ascending = False)
    data['new_overall'] =[ 0.0 if x <= 0.0 else x for x in data['new_overall']]
    data = data.fillna(0.0)
    data['minutes'] = 90
    return data

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

def clean_team_game(standings,game_week,check):
    team_names = ['Atlético Ottawa','Cavalry FC','FC Edmonton','Forge FC','HFX Wanderers FC','Pacific FC','Valour FC','York9 FC','York United FC']
    db_tail = game_week.copy()
    shape = db_tail.shape[0]
    db_tail = db_tail.tail(shape)
    if check == 0:
        team = standings.iloc[0]['team'] # Getting the name of the top team
        team = [x for x in team_names if team in x][0]
    else:
        team = standings.iloc[-1]['team'] # Getting the name of the bottom placed team
        team = [x for x in team_names if team in x][0]
    if standings.iloc[-1]['matches'] == 0 and check == 1:
        db = pd.DataFrame([('TBD',0,'TBD',0)],columns=['home','hs','away','as']) # make an empty set if the game is empty
    else:
        try:
            df = db_tail[(db_tail['home'] == team) | (db_tail['away'] == team)].tail(1) # get appropirate game results for specified team
            db = index_reset(df)
            db = db.iloc[0][['home','hs','away','as']]
            db = pd.DataFrame(db)
            db = db.T
        except:
            print(team)
            print('ERROR**************************************')

    return db

def get_weeks_results(year,results,standings,stats,team_ref,team_names):
    if results.at[0,'hr'] == 'E':
        game_week = pd.DataFrame([('TBD',0,'TBD',0)],columns=['home','hs','away','as'])
        big_win, top_team, low_team,other_team = game_week,game_week,game_week,game_week
        goals, assists, yellows, reds = 0,0,0,0
        return game_week,goals,big_win,top_team,low_team,other_team, assists, yellows, reds
    elif results.tail(1)['hr'].values[0] != 'E':
        if year == '2019':
            played_games = results[results['s'] == 2]
        else:
            played_games = results[results['s'] == 1]

        max_home = played_games[(played_games['hs'] == played_games['hs'].max()) & (played_games['hr'] == "W")]
        max_home = played_games[played_games['as'] == played_games['as'].min()]
        max_home = index_reset(max_home)

        max_away = played_games[(played_games['as'] == played_games['as'].max()) & (played_games['ar'] == "W")]
        max_away = index_reset(max_away)
        if max_away.empty:
            max_away = played_games[played_games['as'] == played_games['as'].max()]
            max_away = index_reset(max_away)

        if max_home.at[0,'hs'] > max_away.at[0,'as']:
            max_home_win = max_home
        else:
            max_home_win = max_away

        big_win = max_home_win[['home','hs','away','as']]
        big_win = index_reset(big_win)
        big_win = pd.DataFrame(big_win.loc[0])
        big_win = big_win.T

        top_result = played_games[(played_games['home'] == standings.iloc[0]['team']) | (played_games['away'] == standings.iloc[0]['team'])][['hr','home','hs','away','as','ar']]#clean_team_game(standings,game_week,0) # finding top team
        top_result = top_result[(top_result['hs'] == top_result['hs'].min()) | (top_result['as'] == top_result['as'].min())]
        top_result['abs'] = abs(top_result['hs'] - top_result['as'])
        top_result = top_result[(top_result['abs'] == top_result['abs'].max())]
        top_result = top_result[((top_result['hr'] == "W") & (top_result['home'] == standings.iloc[0]['team'])) | ((top_result['ar'] == "W") & (top_result['away'] == standings.iloc[0]['team']))]
        top_result = index_reset(top_result)

        low_result = played_games[(played_games['home'] == standings.iloc[-1]['team']) | (played_games['away'] == standings.iloc[-1]['team'])][['hr','home','hs','away','as','ar']]#clean_team_game(standings,game_week,1) # finding bottom team
        low_result = low_result[(low_result['hs'] == low_result['hs'].min()) | (low_result['as'] == low_result['as'].min())]
        low_result['abs'] = abs(low_result['hs'] - low_result['as'])
        low_result = low_result[(low_result['abs'] == low_result['abs'].max())]
        low_result = index_reset(low_result)

        teams_in = pd.concat([big_win,top_result,low_result,team_ref])#get_longest_name(big_win,top_team,low_team,team_ref)
        other_team = played_games[(~played_games['home'].isin(teams_in)) | (~played_games['away'].isin(teams_in))]
        other_team = index_reset(other_team)
        other_team = pd.DataFrame(other_team.loc[0][['home','hs','away','as']])
        other_team = other_team.T

        goals = standings['Goal'].sum()
        assists = stats['assists'].sum()
        yellows = stats['yellow'].sum()
        reds = stats['red'].sum()

        return played_games, goals, big_win, top_result, low_result, other_team, int(assists), int(yellows), int(reds)
    else:
        print('SEASON ONGOING')
        print('=====================================')
        if year == '2019':
            played_games = results[results['s'] == 2]
        else:
            played_games = results[(results['s'] == 1)&(results['hr']!='E')]
        if year == current_year:
            w = played_games['w'].tail(1).values[0]
            game_week = played_games[played_games['w'] == w]
        else:
            game_week = played_games[played_games['w'] == played_games['w'].tail(1).values[0]]

        game_week = game_week.sort_values(by='hs',ascending=False)
        game_week = game_week.reset_index(drop=True)

        print(game_week)

        max_home = game_week[(game_week['hs'] == game_week['hs'].max()) & (game_week['hr'] == "W")]
        if max_home.empty:
            max_home = game_week[(game_week['hs'] == game_week['hs'].max()) & ((game_week['hr'] == "D")|(game_week['hr'] == "L"))]

        max_home = max_home[max_home['as'] == max_home['as'].min()]
        max_home = max_home.reset_index(drop=True)

        max_away = game_week[(game_week['as'] == game_week['as'].max()) & (game_week['ar'] == "W")]
        if max_home.empty:
            max_away = game_week[(game_week['as'] == game_week['as'].max()) & ((game_week['ar'] == "D")|(game_week['ar'] == "L"))]
        max_away = max_away.reset_index(drop=True)
        if max_away.empty:
            max_away = game_week[game_week['as'] == game_week['as'].max()]
            max_away = max_away.reset_index(drop=True)

        if max_home.at[0,'hs'] > max_away.at[0,'as']:
            max_home_win = max_home
        else:
            max_home_win = max_away

        big_win = max_home_win[['home','hs','away','as']]
        big_win = big_win.reset_index(drop=True)
        big_win = pd.DataFrame(big_win.loc[0])
        big_win = big_win.T

        top_result = game_week[(game_week['home'] == standings.iloc[0]['team']) | (game_week['away'] == standings.iloc[0]['team'])][['home','hr','hs','away','ar','as']]#clean_team_game(standings,game_week,0) # finding top team
        try:
            top_result = top_result[(top_result['hs'] == top_result['hs'].min()) | (top_result['as'] == top_result['as'].min())]
        except:
            top_result = top_result[(top_result['hs'] >= 0) | (top_result['as'] >= 0)]
        top_result = top_result[((top_result['hr'] == "W") & (top_result['home'] == standings.iloc[0]['team'])) | ((top_result['ar'] == "W") & (top_result['away'] == standings.iloc[0]['team']))]
        if top_result.empty:
            top_result = game_week[((game_week['hr'] == "W") | (game_week['ar'] == "W"))][['home','hr','hs','away','ar','as']]
        top_result['abs'] = abs(top_result['hs'] - top_result['as'])
        top_result = top_result[(top_result['abs'] == top_result['abs'].max())]
        top_result = top_result.reset_index(drop=True)

        low_result = game_week[(game_week['home'] == standings.iloc[-1]['team']) | (game_week['away'] == standings.iloc[-1]['team'])][['home','hr','hs','away','ar','as']]#clean_team_game(standings,game_week,1) # finding bottom team
        if low_result.empty:
            temp_standings = standings[standings['matches'] > 0]
            low_result = game_week[(game_week['home'] == temp_standings.iloc[-1]['team']) | (game_week['away'] == temp_standings.iloc[-1]['team'])][['home','hr','hs','away','ar','as']]
        low_result = low_result[(low_result['hs'] == low_result['hs'].min()) | (low_result['as'] == low_result['as'].min())]
        low_result['abs'] = abs(low_result['hs'] - low_result['as'])
        low_result = low_result[(low_result['abs'] == low_result['abs'].max())]
        low_result = low_result.reset_index(drop=True)

        teams_in = []
        for x in [big_win,top_result,low_result]:
            for i in x[['home','away']].values[0]:
                teams_in.append(i)
        teams_in = set(teams_in)

        other_team = game_week[(~game_week['home'].isin(teams_in)) | (~game_week['away'].isin(teams_in))]
        if other_team.empty:
            manual_standings = pd.read_csv(f'datasets/{year}/league/{year}-regular-standings-manual.csv')
            bt = manual_standings.at[7,'team']
            other_team = played_games[(played_games['w'] == w-1)&((played_games['home']==bt)|(played_games['away']==bt))]
        other_team = other_team.reset_index(drop=True)
        other_team = pd.DataFrame(other_team.loc[0][['home','hs','away','as']])
        other_team = other_team.T

        goals = standings['Goal'].sum()
        assists = stats['assists'].sum()
        yellows = stats['yellow'].sum()
        reds = stats['red'].sum()

        return played_games, goals, big_win, top_result, low_result, other_team, assists, yellows, reds

def get_team_stats(stats,query):
    team_stats = stats[stats['team'] == query]
    names = team_stats['name'].unique()
    information = stats.copy()
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
        short_team = get_shortest_name(team)
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
            result = player.iloc[0][check] / (eval_['max'][check]+(eval_['max'][check]*0.15)) # calculate the score for the value found value/max
            result = np.nan_to_num(result)
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
    tracked_player_stat = tracked_player_stat.reset_index(drop=True)
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
    cold = ['team','name','number','position','minutes','goals','tackles','t-won','clearances','interceptions','duels','d-won']
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
            if player.iloc[0]['pass-acc'] >= 0.85: # reward high passing rate
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
        if column == 'pass-acc':
            condensed_player_info[column] = condensed_player_info[column].round(4).astype(str)
            condensed_player_info[column] = condensed_player_info[column].astype(float)
            continue
        if column == 'cross-acc':
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
    top_offenders = top_offenders.reset_index(drop=True)
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
                array = likelihood_input(array,[[2,2,0],[2,0,0],[2,1,1]])
    db= pd.DataFrame(array,columns=['h/a','w/l/d','y/n'])
    return db

def get_NB_data(data,query):
    db = likelihood_table(data,query)
    dy = db.pop('y/n').to_list()
    dx = [tuple(x) for x in db.values]
    return dx, dy

def get_team_comparison(data,q1,q2):
    # getting games with q1 in both home or away
    db = data[(data['home'] == q1)|(data['away'] == q1)]#data[data['team'] == q1]
    db = db.reset_index(drop=True)
    # filering down more to get only the games against q2
    db = db.sort_values(by=['m','d'])
    db = db[(db['home'] == q2) | (db['away'] == q2)]
    db = db.reset_index(drop=True)
    if db.empty == True:
        db = pd.DataFrame([(0,0,0,0,q1,'D',q2,'D','empty',q1)],columns=['d','m','hs','as','home','hr','away','ar','summary','team'])
    return db

def get_match_history(compare,query):
    db = likelihood_table(compare,query)#get_NB_data(compare,q1)
    db = db[db['y/n'] == 1]
    w_count = [1 for x in db[db['w/l/d'] == 2].values]
    l_count = [1 for x in db[db['w/l/d'] == 0].values]
    d_count = [1 for x in db[db['w/l/d'] == 1].values]
    df = pd.DataFrame(columns=['w','l','d'])
    df.at[0,'w'] = sum(w_count)
    df['l'] = sum(l_count)
    df['d'] = sum(d_count)
    return df

def get_nb_match_prediction(q1,q2,x1,y1,x2,y2):

    def get_nb_prediction(query,x,y,result):
        mnb = MultinomialNB()
        # Train the model using the training sets
        mnb.fit(x,y)
        # use below instead of predicted = model.predict([result]) because we want the probability
        mnb_pred = np.round(mnb.predict_proba([result])[:, 1],decimals=2)
        pred = round(mnb_pred[0],2)
        return pred

    def get_match_prediction_result(query,x,y,array):
        prediction = get_nb_prediction(query,x,y,array)
        return prediction

    def norm_data(x,y,z):
        total = round(x,2)+round(y,2)+round(z,2)
        if total > 1.0:
            diff = round(total - 1.0,2)
            total = total - diff
            x = round(x / total - (diff/3),2)
            return x
        x = round(x / total,2)
        return x

    if (len(x1) == 0) or (len(x2) == 0):
        x = round(1/3,2)
        home_win, away_win, draw = x,x,x
        return home_win,away_win,draw

    home_win = get_match_prediction_result(q1,x1,y1,[1,2])
    draw = get_match_prediction_result(q1,x1,y1,[1,1])
    away_win = get_match_prediction_result(q2,x2,y2,[2,2])
    home_win_new = norm_data(home_win, draw, away_win)
    draw_new = norm_data(draw, home_win, away_win)
    away_win_new = norm_data(away_win, draw, home_win)
    return home_win_new, draw_new, away_win_new

def get_team_form(data,query):
    db = data[data['team'] == query].sort_values(by=['m','d'])
    db = pd.DataFrame(db['summary'])
    return db

def get_form_results(data,team_ref):
    db = pd.DataFrame()
    form = get_results_brief(data[data['s'] <= 1],team_ref)
    teams = team_ref['team'].unique()
    for team in teams:
        df = get_team_form(form,team)
        db[team] = pd.Series(df['summary'].values)
    db = db.fillna('-')
    for col in db.columns:
        for i in range(db.shape[0]):
            if 'nan' in db.at[i,col]:
                db.at[i,col] = 'E'
    db = db.T
    db = db.reset_index()
    return db

def best_roster(team_name, player_info):

    formation = {'d':3,'m':4,'f':3,'g':1}
    # cycle through the positions and get the appropriate number for each position
    # get the appropriate number from the formation that the coach typically uses
    #**** the above needs to be implemented ****
    positions = {}
    for pos in ['d','f','g','m']:
        positions[pos] = player_info[(player_info['team'] == team_name)&(player_info['position'] == pos)][['display','number','position','overall']].sort_values(by=['overall'],ascending=False).head(formation[pos])
    for k in ['d','f','m']:
        if len(positions[k]) < formation[k]:
            change = formation[k] - len(positions[k])
            positions['m'] = player_info[(player_info['team'] == team_name)&(player_info['position'] == 'm')][['display','number','position','overall']].sort_values(by=['overall'],ascending=False).head(formation['m']+change)
        else:
            pass
    # once we have all 11 players combine them and return the dataframe
    roster = pd.concat([positions['g'],positions['d'],positions['m'],positions['f']])
    roster = roster.reset_index(drop=True)
    return roster

def prediction_roster(team_name,player_info):
    # the prediction needs more players to work with than just the best_roster
    # add players to it
    # get the best roster
    roster = best_roster(team_name,player_info)
    formation = {'d':1,'m':1,'f':1}
    # get best_roster player names
    current_players = [x for x in roster['display'].unique()]
    # grab appropriate number of remaining players
    # check to make sure they are not in the existing best_roster
    positions = {}
    for pos in ['d','m','f']:
        positions[pos] = player_info[(~player_info['display'].isin(current_players))&(player_info['team'] == team_name)&(player_info['position'] == pos)][['display','number','position','overall']].sort_values(by=['overall'],ascending=False).head(1)
        # check if potentially the dataframe is empty
        # grab players from other positions if so
        if positions[pos].empty:
            lst = ['d','m','f']
            lst.remove(pos)
            new_pos = random.choice(lst)
            positions[pos] = player_info[(~player_info['display'].isin(current_players))&(player_info['team'] == team_name)&(player_info['position'] == new_pos)][['display','number','position','overall']].sort_values(by=['overall'],ascending=False).head(1)
        else:
            new_names = [x for x in positions[pos]['display'].unique()]
            current_players = current_players+new_names
    roster = pd.concat([roster,positions['d'],positions['m'],positions['f']])

    return roster.reset_index(drop=True)

def get_roster(query,stats,team_ref): # use team stats to get the player information
    roster = get_stats_all(stats,team_ref)
    roster = roster[roster['team'] == query]
    roster = roster[['name','number','position']]
    roster.insert(3,'overall',0)
    roster = index_reset(roster)
    return roster

def get_player_card(name,stats,stats_seed,player_info):
    player_stats = stats[stats['name'] == name].groupby('name').sum()
    if player_stats.empty:
        player_stats = stats_seed[stats_seed['name'] == name].groupby('name').sum()
    player_stats = player_stats.reset_index()

    name = player_stats['name'].values[0]
    player_information = player_info[player_info['name'] == name]
    count = stats[stats['name'] == name].groupby('name').count()
    if count.empty:
        count = stats_seed[stats_seed['name'] == name].groupby('name').count()
    count = int(count['game'].values[0])
    for col in player_stats.columns:
        if col == 'name':
            pass
        else:
            player_stats[col] = player_stats[col].astype('int32')
    player_stats['pass-acc'] = player_stats['pass-acc'].apply(lambda x: round(x / count,2))
    player_stats['cross-acc'] = player_stats['cross-acc'].apply(lambda x: round(x / count,2))
    player_stats['number']= int(player_information['number'].values[0])
    player_stats.insert(0,'image',player_information['image'].values[0])
    player_stats.insert(1,'flag',player_information['flag'].values[0])
    player_stats.insert(2,'overall',player_information['overall'].values[0])
    player_stats.insert(31,'wiki',player_information['link'].values[0])
    player_stats.insert(0,'team',player_information['team'].values[0])
    minutes = player_stats['minutes'].values[0]
    player_stats['xG'] = player_stats['goals'].apply(lambda x: round(x / (minutes/90),2))
    player_stats['xA'] = player_stats['assists'].apply(lambda x: round(x / (minutes/90),2))
    position = player_information['position'].values[0]
    if position == 'f':
        position = 'Forward'
    elif position == 'm':
        position = 'Midfield'
    elif position == 'd':
        position = 'Defender'
    else:
        position = 'Goal Keeper'
    player_stats.insert(5,'position',position)
    player_stats= player_stats.fillna(0)
    return player_stats

def get_player_card_previous(name,stats,player_info):
    player_stats = stats[stats['last'] == name].groupby('name').sum()
    if player_stats.empty:
        player_stats = stats_seed[stats_seed['last'] == name].groupby('name').sum()
    player_stats = player_stats.reset_index()
    name = player_stats['name'].values[0]
    player_information = player_info[player_info['name'] == name]
    count = stats[stats['name'] == name].groupby('name').count()
    if count.empty:
        count = stats_seed[stats_seed['name'] == name].groupby('name').count()
    count = int(count['game'].values[0])
    for col in player_stats.columns:
        if col == 'name':
            pass
        else:
            player_stats[col] = player_stats[col].astype('int32')
    player_stats['pass-acc'] = player_stats['pass-acc'].apply(lambda x: round(x / count,2))
    player_stats['cross-acc'] = player_stats['cross-acc'].apply(lambda x: round(x / count,2))
    player_stats['number']= int(player_information['number'].values[0])
    player_stats.insert(0,'image',player_information['image'].values[0])
    player_stats.insert(1,'flag',player_information['flag'].values[0])
    player_stats.insert(2,'overall',player_information['overall'].values[0])
    player_stats.insert(31,'wiki',player_information['link'].values[0])
    player_stats.insert(0,'team',player_information['team'].values[0])
    minutes = player_stats['minutes'].values[0]
    player_stats['xG'] = player_stats['goals'].apply(lambda x: round(x / (minutes/90),2))
    player_stats['xA'] = player_stats['assists'].apply(lambda x: round(x / (minutes/90),2))
    position = player_information['position'].values[0]
    if position == 'f':
        position = 'Forward'
    elif position == 'm':
        position = 'Midfield'
    elif position == 'd':
        position = 'Defender'
    else:
        position = 'Goal Keeper'
    player_stats.insert(5,'position',position)
    player_stats= player_stats.fillna(0)
    return player_stats

def get_roster_overall(query,stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info): # use team stats to get the player information
    team = get_shortest_name(query)
    def get_score(data,name):
        db = data[data['name'] == name]
        try:
            return db['overall'].values[0]
        except:
            try:
                previous = player_info[player_info['name'] == name]['overall'].values[0]
                return previous
            except:
                return 0.0
    def get_image(player_info,name):
        db = player_info[player_info['name'] == name]
        if db['image'].empty:
            db = 'empty.jpg'
        else:
            db = db['image'].values[0]
        return db
    def get_link(player_info,name):
        db = player_info[player_info['name'] == name]
        if db['link'].empty:
            db = 'https://en.wikipedia.org/wiki/Canadian_Premier_League'
        else:
            db = db['link'].values[0]
        return db
    def get_flag(player_info,name):
        db = player_info[player_info['name'] == name]
        if db['flag'].empty:
            db = 'empty.png'
        else:
            db = db['flag'].values[0]
        return db
    def get_name(player_info,name):
        try:
            db = player_info[player_info['name'] == name]
            return db['display'].values[0]
        except:
            return name
    roster = get_stats_all(stats,team_ref)
    roster = roster[roster['team'] == team].copy()
    roster = roster[['name','first','last','number','position']] # scale the dataframe down to what we need
    a, b, c, d, e = [], [], [], [], []
    for i in range(0,roster.shape[0]):
        if roster.iloc[i]['position'] == 'f':
            score = str(get_score(rated_forwards,roster.iloc[i]['name']))
            a.append(score[0:4])
            b.append(get_image(player_info,roster.iloc[i]['name']))
            c.append(get_flag(player_info,roster.iloc[i]['name']))
            d.append(get_link(player_info,roster.iloc[i]['name']))
            e.append(get_name(player_info,roster.iloc[i]['name']))
        if roster.iloc[i]['position'] == 'm':
            score = str(get_score(rated_midfielders,roster.iloc[i]['name']))
            a.append(score[0:4])
            b.append(get_image(player_info,roster.iloc[i]['name']))
            c.append(get_flag(player_info,roster.iloc[i]['name']))
            d.append(get_link(player_info,roster.iloc[i]['name']))
            e.append(get_name(player_info,roster.iloc[i]['name']))
        if roster.iloc[i]['position'] == 'd':
            score = str(get_score(rated_defenders,roster.iloc[i]['name']))
            a.append(score[0:4])
            b.append(get_image(player_info,roster.iloc[i]['name']))
            c.append(get_flag(player_info,roster.iloc[i]['name']))
            d.append(get_link(player_info,roster.iloc[i]['name']))
            e.append(get_name(player_info,roster.iloc[i]['name']))
        if roster.iloc[i]['position'] == 'g':
            score = str(get_score(rated_keepers,roster.iloc[i]['name']))
            a.append(score[0:4])
            b.append(get_image(player_info,roster.iloc[i]['name']))
            c.append(get_flag(player_info,roster.iloc[i]['name']))
            d.append(get_link(player_info,roster.iloc[i]['name']))
            e.append(get_name(player_info,roster.iloc[i]['name']))
    roster['overall'] = a
    roster['flag'] = c
    roster['link'] = d
    roster.insert(0,'image',b)
    roster = index_reset(roster)
    roster.pop('name')
    roster['name'] = e
    #roster['first'] = roster['name'].apply(lambda x: re.split('\W+',x)[0])
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
    roster['position'] = roster['position'].astype('object')
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
    db = pd.concat([dk[0:1],dd[0:4],dm[0:4],df[0:2],dd[4:7],dm[4:7],df[2:5]])
    db = index_reset(db)
    return db

def get_team_history(results,query,period = 0):
    away = results[results['away'] == query].copy()
    away = away[['d','m','as','hs','away','ar','home','hr']]
    away = away.rename(columns={'as':'hs','hs':'as','away':'home','ar':'hr','home':'away','hr':'ar'})
    home = results[results['home'] == query].copy()
    home = home[['d','m','hs','as','home','hr','away','ar']]
    if period == 0:
        return pd.concat([home,away]).sort_values(by=['m','d']).tail()
    else:
        return pd.concat([home,away]).sort_values(by=['m','d'])

def get_five_game_form(results,query,period):
    team_form = get_team_history(results,query,period)
    team_form = team_form.pop('hr')
    convert = {'W':[1,0,0],'L':[0,0,1],'D':[0,1,0]}
    a = []
    for i in team_form:
        a.append(convert[i])
    form = pd.DataFrame(a,columns=['w','l','d'])
    return pd.DataFrame(form.sum()).astype('int')

def get_overall_roster(game_roster,player_info):
    b = []
    for i in range(game_roster.shape[0]):
        name = game_roster.iloc[i]['name']
        try:
            overall = player_info[player_info['name']==name]['overall'].values[0]
        except:
            try:
                overall = player_info[player_info['display']==name]['overall'].values[0]
            except:
                overall = 0.00
        try:
            position = player_info[player_info['name']==name]['position'].values[0]
        except:
            try:
                position = player_info[player_info['display']==name]['position'].values[0]
            except:
                position = 'm'
        if position == 'f':
            overall = round(overall,2) + 4
        elif position == 'm':
            overall = round(overall,2) + 3
        elif position == 'd':
            overall = round(overall,2) + 2
        else:
            overall = round(overall,2) + 1
        overall = str(overall)
        b.append(float(overall[0:4])) # get the player overall score for each player in the game
    if len(b) < 16:
        i = int(16 - len(b))
        for j in range(0,i):
            b.append(0)
    else:
        if len(b) > 16:
            k = 16 - len(b)
            b = b[:k]
    db = pd.DataFrame(b[0:14])
    db = db.T
    return db

def add_features(data,score,team_pred):
    cols = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13']
    data = pd.DataFrame(data.values,columns=cols)
    data = data.astype('float32')
    data['score'] = score
    data['pred'] = team_pred
    data['sum'] = round(data[cols].sum(axis=1),2)
    data['count'] = (data[data[cols]>0].count(axis=1))-10
    data['result'] = round(data['sum'] / (data['count']+0.1),2)
    data['plus'] = data['sum'] + data['score']
    data['test'] = round(data['sum']  / data['sum'].max() - 0.05,3)
    if data['count'].values[0] == 0:
        data['test2'] = 0
    else:
        data['test2'] = round(data['count']  / data['count'].max() - 0.05,3)
    data['diff'] = round(data['score'] * data[cols].max(axis=1)).astype('int64') + 1
    return data

def get_classification(home_win, away_win ,draw):
    if home_win > away_win:
        if home_win > draw:
            hp, ap = 3, 1
        else:
            hp, ap = 2, 2
    elif away_win > home_win:
        if away_win > draw:
            hp, ap = 3, 1
        else:
            hp, ap = 2, 2
    else:
        hp, ap = 2, 2
    return hp, ap

def roster_pred(model,array):
    prediction = model.predict_proba([array.values[0]]).flatten()
    df = pd.DataFrame(prediction)
    return df

def get_final_game_prediction(model,home_array,home_score,away_score):

    def random_draw(x,y):
        if x > y:
            end = x
        else:
            end = y
        if end == 0:
            return x,y
        else:
            x = random.choice(range(end))
            y = x
            return x,y
    # get the model prediction probability of W, L, D

    ###### need to be sure of the order of these probabilities

    def roster_pred(model,array):
        prediction = model.predict([array]).flatten()
        if prediction == 0:
            outcome = 'L'
        elif prediction == 1:
            outcome = 'D'
        else:
            outcome = 'W'
        return outcome

    def roster_prob(model,array):
        probability = model.predict_proba([array]).flatten()
        return probability

    def norm(x,y,z):
        norm = x + y + z
        x, y, z = round(x / norm,2), round(y / norm,2), round(z / norm,2)
        return x, y, z

    def round_results(lst):
        return [round(x,2) for x in lst]

    # get the prediction from the classification model
    prediction = roster_pred(model,home_array)
    probability = roster_prob(model,home_array)
    probability = round_results(probability)

    p_l, p_d, p_w = probability[0],probability[1],probability[2]#norm(q1_w, q1_l, q1_d)

    #FIX THE SCORE ####
    # adjust score depending on the outcome from the prediction
    if prediction == 'W' and (home_score == away_score) or prediction == 'W' and (home_score < away_score):
        if home_score > 1:
            away_score = random.choice(range(home_score-1))
        else:
            away_score = 0
    if prediction == 'L' and (home_score == away_score) or prediction == 'L' and (home_score > away_score):
        if away_score > 1:
            home_score = random.choice(range(away_score-1))
        else:
            home_score = 0
    if prediction == 'D':
        home_score, away_score = random_draw(home_score, away_score)

    return p_w, p_l, p_d, home_score, away_score, prediction

def get_best_eleven(team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info):

    year_check = team_ref['year'].unique()[0]#.values[0]
    if year_check != '2019':
        year_check = str(int(year_check)-1)

    check = rated_forwards.describe()
    if check.loc['max']['Min'] == 0:
        best_eleven = pd.read_csv(f'datasets/{year_check}/cpl-{year_check}-best_eleven.csv')
        return best_eleven
    else:
        top_keeper = rated_keepers.head(1)
        top_keeper = top_keeper[['name','number','overall']]
        top_keeper['position'] = 'g'
        top_defenders = rated_defenders.iloc[0:3][['name','number','overall']]
        top_defenders['position'] = 'd'
        top_midfielders = rated_midfielders.iloc[0:5][['name','number','overall']]
        top_midfielders['position'] = 'm'
        top_forwards = rated_forwards.iloc[0:2][['name','number','overall']]
        top_forwards['position'] = 'f'
        best_eleven = pd.DataFrame(columns=['name','number','overall','position'])
        best_eleven = pd.concat([best_eleven,top_keeper,top_defenders,top_midfielders,top_forwards])
        a,b,c,d,e,f = [],[],[],[],[],[]

        names = best_eleven['name'].values

        for i in range(0,best_eleven.shape[0]):
            name = best_eleven.iloc[i]['name']
            try:
                player = player_info[player_info['name'] == name]
            except:
                player = player_info[player_info['display'] == name]
            try:
                team = player[player['name'] == name]['team'].values[0]
            except:
                team = player[player['display'] == name]['team'].values[0]
            #player= index_reset(player)
            try:
                #a.append(name.split(' ')[0])
                #b.append(' '.join(name.split(' ')[1:]))
                c.append(player[player['name'] == name]['image'].values[0])
                d.append(player[player['name'] == name]['flag'].values[0])
                e.append(player[player['name'] == name]['link'].values[0])
                f.append(team_ref[team_ref['team'] == team]['colour'].values[0])
            except 'Exception' as e:
                #a.append(name.split(' ')[0])
                #b.append(name.split(' ')[1:])
                c.append(player[player['display'] == name]['image'].values[0])
                d.append(player[player['display'] == name]['flag'].values[0])
                e.append(player[player['display'] == name]['link'].values[0])
                f.append(team_ref[team_ref['team'] == team]['colour'].values[0])
                print(e)


        best_eleven.insert(0,'image',c)
        best_eleven.insert(3,'flag',d)
        best_eleven['link'] = e
        best_eleven['colour'] = f
        #best_eleven.pop('name')
        best_eleven = index_reset(best_eleven)
        return best_eleven

def roster_regressor_pred(model,array):
    prediction = model.predict([array]).flatten()
    df = pd.DataFrame(prediction)
    return df

def get_final_score_prediction(model,home_roster,away_roster,home_win,away_win):
    # get a prediction for the amount of goals that the team will score
    def roster_pred(model,array):
        return int(model.predict([array]).flatten()[0])

    def fix_score(home_score,away_score,home_win,away_win):
        if home_win > away_win and home_score < away_score: # fix the score prediction - if the probability of home win > away win and score doesn't reflect it
            home_score, away_score = away_score, home_score # change the predicted score to reflect that
            return home_score,away_score,home_win,away_win
        elif home_win < away_win and home_score > away_score: # else the probability of home win < away win
            home_score, away_score =  away_score, home_score # change the predicted score to reflect that
            return home_score,away_score,home_win,away_win
        elif home_win < away_win and home_score == away_score:
            home_win = away_win
            return home_score,away_score,home_win,away_win
        elif home_win > away_win and home_score == away_score:
            home_win = away_win
            return home_score,away_score,home_win,away_win
        else:
            return home_score,away_score,home_win,away_win

    home_score = roster_pred(model,home_roster)
    away_score = roster_pred(model,away_roster)

    home_score, away_score, home_win_new, away_win_new = fix_score(home_score, away_score,home_win,away_win)

    return home_score,away_score, home_win_new, away_win_new

def get_game_roster_prediction(get_games,results,stats,team_ref,player_info):
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
    team1 = get_shortest_name(schedule.iloc[0]['home'])
    team2 = get_shortest_name(schedule.iloc[0]['away'])
    try:
        team3 = get_shortest_name(schedule.iloc[1]['home'])
        team4 = get_shortest_name(schedule.iloc[1]['away'])
    except:
        team3 = 1
        team4 = 1
    try:
        team5 = get_shortest_name(schedule.iloc[2]['home'])
        team6 = get_shortest_name(schedule.iloc[2]['away'])
        team7 = get_shortest_name(schedule.iloc[3]['home'])
        team8 = get_shortest_name(schedule.iloc[3]['away'])
    except:
        team5 = 1
        team6 = 1
        team7 = 1
        team8 = 1

    return team1, team2, team3, team4, team5, team6, team7, team8

def update_player_info(year,week,player_info,rated_forwards,rated_midfielders,rated_defenders,rated_keepers):

    def get_player_score(data,name):
        name = [name]
        if data[data['name'].isin(name)].empty:
            pass
        else:
            overall = data[data['name'].isin(name)]
            try:
                new_overall = overall['overall'].values[0]
            except:
                new_overall = 0.0
            return new_overall

    game_week = 'week' + week
    rated_forwards.to_csv(f'datasets/{year}/week/cpl-{year}-forwards-{game_week}.csv',index=False)
    rated_midfielders.to_csv(f'datasets/{year}/week/cpl-{year}-midfielders-{game_week}.csv',index=False)
    rated_defenders.to_csv(f'datasets/{year}/week/cpl-{year}-defenders-{game_week}.csv',index=False)
    rated_keepers.to_csv(f'datasets/{year}/week/cpl-{year}-keepers-{game_week}.csv',index=False)
    player_info.to_csv(f'datasets/{year}/week/player-{year}-info-{game_week}.csv',index=False)

    combine = [rated_forwards,rated_midfielders,rated_defenders,rated_keepers]
    names = player_info['name'].values
    a,b = [],[]
    for name in names:
        j = 1
        for i in range(0,4):
            # for the start of the season load last season's overall if empty
            old_overall = player_info[(player_info['name'] == name) | (player_info['display'] == name)]
            old_o = old_overall['overall'].values[0]
            if j == 1:
                b.append(old_o)

            score = get_player_score(combine[i],name)
            if score == None: # if score is none, increase j and move to the next dataset
                j += 1
                pass
            if score != None:
                overall = score
                a.append(overall)
            if j == 5:
                overall = old_o # save the old result if there is not a current overall
                a.append(overall)
    player_info['overall'] = a
    player_info[f'g-{week}-o'] = b
    return player_info

## this can be combined
####

def add_regressor_features(data):
    cols = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
    data = pd.DataFrame(data.values,columns=cols)
    data = data.astype('float32')
    # new features
    data['sum'] = round(data[cols].sum(axis=1),2)
    data['count'] = (data[data[cols]>0].count(axis=1))-11
    data['result'] = round(data['sum'] / (data['count']+0.1),2)
    data['test'] = round(data['sum']  / data['sum'].max() - 0.05,3)
    if data['count'].values[0] == 0:
        data['test2'] = 0
    else:
        data['test2'] = round(data['count']  / data['count'].max() - 0.05,3)
    cols2=[]
    for col in cols:
        string = 'n' + col
        cols2.append(string)
        data[string] = data[col].apply(lambda x: int(x))
    # get player position from first cols
    data['sum'] = data['sum'] - data[cols2].sum(axis=1)
    data['f'] = (data[data[cols2]==4].count(axis=1))
    data['m'] = (data[data[cols2]==3].count(axis=1))
    data['d'] = (data[data[cols2]==2].count(axis=1))
    data['g'] = (data[data[cols2]==1].count(axis=1))
    # remove cols2 integers from cols - retaining only the overall float value
    for i in range(len(cols)):
        data[cols[i]] = data[cols[i]] - data[cols2[i]]
    return data

def add_classifier_features(data,score,team_pred):
    cols = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
    data = pd.DataFrame(data.values,columns=cols)
    data = data.astype('float32')
    # getting the predicted score and predicted result
    data['score'] = score
    data['pred'] = team_pred
    # new features
    data['sum'] = round(data[cols].sum(axis=1),2)
    data['count'] = (data[data[cols]>0].count(axis=1))-11
    data['result'] = round(data['sum'] / (data['count']+0.1),2)
    data['plus'] = data['sum'] + data['score']
    data['test'] = round(data['sum']  / data['sum'].max() - 0.05,3)
    if data['count'].values[0] == 0:
        data['test2'] = 0
    else:
        data['test2'] = round(data['count']  / data['count'].max() - 0.05,3)
    data['diff'] = round(data['score'] * data[cols].max(axis=1)).astype('int64') + 1
    cols2=[]
    for col in cols:
        string = 'n' + col
        cols2.append(string)
        data[string] = data[col].apply(lambda x: int(x))
    # get player position from first cols
    data['sum'] = data['sum'] - data[cols2].sum(axis=1)
    data['f'] = (data[data[cols2]==4].count(axis=1))
    data['m'] = (data[data[cols2]==3].count(axis=1))
    data['d'] = (data[data[cols2]==2].count(axis=1))
    data['g'] = (data[data[cols2]==1].count(axis=1))
    # remove cols2 integers from cols - retaining only the overall float value
    for i in range(len(cols)):
        data[cols[i]] = data[cols[i]] - data[cols2[i]]
    return data

######################
### make the player_graphs

def get_team_graphs(stats,standings):
    comparing = standings.sort_values(by=['team'])

    def get_column_overall(lst):
        data = stats[lst]
        data = data.groupby(['team']).sum()
        data['overall'] = data.sum(axis=1) / data.shape[1]
        data['overall'] = data['overall'] / data['overall'].max()
        data['overall'] = data['overall'] - 0.1
        data = data[['overall']]
        data = data.reset_index()
        data.pop('team')
        return data['overall']

    offense = get_column_overall(['team','goals','chances','assists','shots','s-target','passes','crosses','duels','tackles'])
    central = get_column_overall(['team','goals','assists','touches','passes','pass-acc','crosses','cross-acc','chances','duels','tackles'])
    defense = get_column_overall(['team','tackles','t-won','clearances','interceptions','duels','d-won'])
    keeping = get_column_overall(['team','cs','saves','shots faced','claimed crosses'])

    g_cols = ['chances','goals','assists','pass-acc','cross-acc','shots','s-target','s-box','s-out-box','clearances','interceptions','yellow','shots faced','claimed crosses','cs']
    team_mean = stats.copy()
    goals = stats[['team','goals']]
    assists = stats[['team','assists']]
    team_mean = team_mean.select_dtypes(include=['float'])
    team_mean.insert(0,'team',stats['team'])
    try:
        team_mean = team_mean.groupby(['team']).mean()
    except:
        teams = stats.team.unique()
        team_mean = pd.DataFrame(columns=['team','clean sheets','big chances','attacking plays','combination plays','accuracy','defending','chance creation','finishing'])
        team_mean['team'] = teams
        for col in team_mean.columns:
            if col == 'team':
                continue
            else:
                team_mean[col] = 0.5
        return team_mean

    team_mean = team_mean[g_cols]
    team_mean['claimed crosses'] = team_mean['claimed crosses'] * 15
    team_mean['cs'] = team_mean['cs'] * 100
    team_mean['goals'] = goals.groupby(['team']).sum()
    team_mean['assists'] = assists.groupby(['team']).sum()
    team_mean['big chances'] = (team_mean['goals'] + 2) / team_mean['chances']
    team_mean['attacking plays'] = (team_mean['assists'] + 2) / team_mean['chances']
    team_mean['combination plays'] = team_mean['assists'] / team_mean['goals'] * 100
    team_mean['offense'] = comparing['gf'].values / offense.values
    team_mean['midfield'] = comparing['gd'].values * central.values + comparing['gd'].max()
    team_mean['defending'] = 100 - (comparing['ga'].values * defense.values)
    team_mean['chance creation'] = (team_mean['shots'] + team_mean['s-box'] + team_mean['s-out-box']) * team_mean['s-target'] * 100
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

    team_mean = team_mean[['clean sheets','big chances','attacking plays','combination plays','offense','midfield','defending','chance creation','finishing']]
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
    #plt.figure(figsize=(8,8))
    plt.figure(figsize=(8,8), dpi=80, facecolor=FACE,edgecolor=EDGE)
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor(FACE)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='white', size=14)

    # Draw ylabels
    #ax.set_title(data['team'], color=colour2, size=24)
    ax.set_rlabel_position(0)
    plt.yticks([0.25,0.5,0.75], ["0.25","0.50","0.75"], color='white', size=12)
    plt.ylim(0,1)
    # Plot data
    ax.plot(angles, values, linewidth=8, linestyle='solid', color=colour2)

    # Fill area
    ax.fill(angles, values, colour2, alpha=0.5)

    filename = f'static/images/{year}/cpl-{year}-{team}-radar.png'
    plt.savefig(filename, facecolor= FACE,edgecolor=EDGE)
