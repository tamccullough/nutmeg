from flask import Flask
from flask import Blueprint, flash, g, redirect, render_template, request, url_for
import cpl_main

## GETTING DATE AND TIME
from datetime import date
today = date.today()
this_year = date.today().strftime('%Y')

from time import sleep

import numpy as np
import pandas as pd
import re

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

team_colour = {'Atlético Ottawa' : '#fff',
              'Cavalry FC' : '#fff',
              'FC Edmonton' : '#fff',
              'Forge FC' : '#fff',
              'HFX Wanderers FC' : '#052049',
              'Pacific FC' : '#fff',
              'Valour FC' : '#fff',
              'York United FC' : '#003b5c',
              'York9 FC' : '#003b5c'}

col_check = {'overall':'O',
            'Aerials':'air',
            'BgChncFace':'BgChcon',
            'ChncOpnPl':'Ch-c',
            'CleanSheet':'CS',
            'Clrnce':'Clr',
            'DefTouch':'defTch',
            'Disposs':'dis',
            'DuelLs':'DlL',
            'DuelsW%':'DlW',
            'ExpA':'xA',
            'ExpG':'xG',
            'ExpGAg':'xGc',
            'FlComA3':'FlA3',
            'FlSufM3':'Flsuf',
            'Goal':'G',
            'GoalCnIBx':'GcBox',
            'GoalCncd':'Gc',
            'Int':'int',
            'Off':'shtTrg',
            'Pass%':'Pc%',
            'PsAtt':'Pat',
            'PsCmpA3':'Pata',
            'PsCmpM3':'Patm',
            'PsOnHfFl':'Pfd',
            'PsOpHfFl':'Pfa',
            'PsOpHfScs':'Psuca',
            'Recovery':'Rec',
            'Saves':'SV',
            'SucflDuels':'sDl',
            'SucflTkls':'sTkl',
            'SvClct':'SVc',
            'SvDive':'dvS',
            'SvHands':'hdS',
            'SvInBox':'SVi',
            'SvOutBox':'SVo',
            'SvPrdSaf':'SVps',
            'SvStand':'SVstn',
            'TchsA3':'Tcha',
            'TchsM3':'Tchm',
            'TouchOpBox':'TchoB',
            'Touches':'Tch'}

def new_col(data):#,chart=''
    for col in data.columns:
        try:
            data[col_check[col]] = data[col]
            if col_check[col]:
                data.pop(col)
        except:
            pass
    for col in data.columns:
        try:
            if col in ['display','number']:
                temp = data.pop(col)
                data[col] = temp
        except:
            pass
    return data

def convert_num_str(num):
    num = str(num*100)
    return num[0:4]

def get_year():
    global year

    try:
        year = request.form['year']
    except:
        pass

    return year

def load_main_files(year):

    results = pd.read_csv(f'datasets/{year}/league/{year}-results.csv')

    try:
        week = results[results['hr'] == 'E']['w'].head(1).values[0]
        schedule = results[results['w'] == week].copy()
        schedule = schedule.fillna(0)
    except:
        week = results['w'].tail(1).values[0]
        schedule = results[results['w'] == week].copy()

    stats = pd.read_csv(f'datasets/{year}/cpl-{year}-stats.csv')
    stats_seed = pd.read_csv(f'datasets/{year}/cpl-{year}-stats-seed.csv')

    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')

    results_old = results[results['hr'] != 'E'].copy()
    results_diff = pd.concat([results, results_old]).drop_duplicates(keep=False)

    if results_diff.empty:
        results_diff = results_old.tail(1)

    colours = team_ref['colour']

    results_brief = cpl_main.get_results_brief(results,year)

    return results, team_ref, player_info, results_old, results_diff, schedule, results_brief


def load_player_files(year):

    # get all rated player information based on position and calculate and overall score for the individual player
    rated_assists = pd.read_csv(f'datasets/{year}/playerstats/{year}-assists.csv')
    rated_goalscorers = pd.read_csv(f'datasets/{year}/playerstats/{year}-goalscorers.csv')
    rated_offenders = pd.read_csv(f'datasets/{year}/playerstats/{year}-discipline.csv')

    stats = {'goals' : rated_goalscorers['Goal'].sum(),'assists' : rated_assists['Ast'].sum(),'yellows' : rated_offenders['Yellow'].sum(),'reds' : rated_offenders['Red'].sum()}

    rated_forwards = pd.read_csv(f'datasets/{year}/playerstats/{year}-forwards.csv')
    rated_midfielders = pd.read_csv(f'datasets/{year}/playerstats/{year}-midfielders.csv')
    rated_defenders = pd.read_csv(f'datasets/{year}/playerstats/{year}-defenders.csv')
    rated_keepers = pd.read_csv(f'datasets/{year}/playerstats/{year}-keepers.csv')

    return rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists, stats

canples = Flask(__name__)
# get current day and set time variables
month, day, weekday = cpl_main.get_weekday()
games_played = {1:28,2:7}
# set the year - which will change based on user choice
year = '2020'

@canples.context_processor
def inject_user():

    return dict(today = today, day = day, weekday = weekday, month = month, theme = 'mono')

@canples.route('/', methods=['GET','POST'])
def index():
    na = 'TBD'

    get_year()

    results, team_ref, player_info, results_old, results_diff, schedule, results_brief = load_main_files(year)
    rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists, stats = load_player_files(year)

    championship = pd.read_csv(f'datasets/{year}/league/{year}-championship.csv')
    playoffs = pd.read_csv(f'datasets/{year}/league/{year}-playoffs.csv')
    standings = pd.read_csv(f'datasets/{year}/league/{year}-regular-standings.csv')
    standings_old = pd.read_csv(f'datasets/{year}/league/{year}-regular-standings-prev.csv')
    rankings = pd.read_csv(f'datasets/{year}/league/{year}-power_rankings.csv')

    if championship.empty:
        top_team = standings.iloc[0]['team']
        top_team_info = team_ref[team_ref['team'].str.contains(top_team)]
        first_colour = top_team_info.iloc[0]['colour']
        first_crest = top_team_info.iloc[0]['crest']
        top_mover = rankings[rankings['power'] == 1]['team']
        top_crest = team_ref[team_ref['team'].str.contains(top_mover)]
        top_crest = top_crest.iloc[0]['crest']
        top_dropper = rankings[rankings['power'] == 8]['team']
        bot_crest = team_ref[team_ref['team'].str.contains(top_dropper)]
        bot_crest = bot_crest.iloc[0]['crest']
    else:
        top_team = championship.iloc[0]['team']
        top_team_info = team_ref[team_ref['team'].str.contains(top_team)]
        first_colour = top_team_info.iloc[0]['colour']
        first_crest = top_team_info.iloc[0]['crest']
        top_mover = standings.iloc[0]['team']
        top_crest = team_ref[team_ref['team'].str.contains(top_mover)]
        top_crest = top_crest.iloc[0]['crest']

        if year == '2019':
            top_dropper = playoffs.iloc[-1]['team']
        else:
            top_dropper = standings.iloc[-1]['team']

        bot_crest = team_ref[team_ref['team'].str.contains(top_dropper)]
        bot_crest = bot_crest.iloc[0]['crest']
    #game_week, goals, big_win, top_team, low_team, other_team, assists, yellows, reds
    game_week, goals, big_win, top_result, low_result, other_result, assists, yellows, reds = cpl_main.get_weeks_results(year,results,standings,stats,team_ref,team_names)
    assists, yellows, reds = int(assists), int(yellows), int(reds)

    top_scorer = rated_goalscorers.loc[0]
    top_scorer['overall'] = player_info[player_info['name'] == top_scorer['name']]['overall'].values[0]
    top_assist = rated_assists.loc[0]
    top_assist['overall'] = player_info[player_info['name'] == top_assist['name']]['overall'].values[0]
    top_forward = rated_forwards.loc[0]
    top_midfielder = rated_midfielders.loc[0]
    top_defender = rated_defenders.loc[0]
    top_keeper = rated_keepers.loc[0]
    top_offender = rated_offenders.loc[0]
    top_offender['overall'] = player_info[player_info['name'] == top_scorer['name']]['overall'].values[0]

    if results.iloc[0]['hr'] == 'E':
        top_team, top_mover, top_dropper, first_crest, top_crest, bot_crest, first_colour = na, na, na, 'CPL-Crest-White.png', 'oneSoccer_nav.png', 'canNat_icon.png', 'w3-indigo'

    suspended = ['none']
    if championship.empty:
        headline = f'{year} Season Underway'
    else:
        headline = f"{year} Season Completed"

    return render_template('cpl-es-index.html',top_mover = top_mover, top_dropper = top_dropper,
    goals = goals,  assists = assists, yellows = yellows, reds = reds,
    big_win = big_win, top_result = top_result, low_result = low_result, other_result = other_result,
    top_team = top_team, top_keeper = top_keeper,top_forward = top_forward,
    top_midfielder = top_midfielder, top_defender = top_defender,
    top_scorer = top_scorer, top_assist = top_assist, top_offender = top_offender, suspended = suspended,
    first_crest = first_crest, first_colour = first_colour, top_crest = top_crest, bot_crest = bot_crest,
    headline = headline)

@canples.route('/standings', methods=['GET','POST'])
def standings():

    get_year()

    championship = pd.read_csv(f'datasets/{year}/league/{year}-championship.csv')
    playoffs = pd.read_csv(f'datasets/{year}/league/{year}-playoffs.csv')
    standings = pd.read_csv(f'datasets/{year}/league/{year}-regular-standings.csv')

    team_form_results = pd.read_csv(f'datasets/{year}/cpl-{year}-team_form.csv')
    team_ref = pd.read_csv(f'datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]

    def get_crest(data,column):
        data['crest'] = '-'
        for i in range(data.shape[0]):
            data.at[i,'crest'] = team_ref[team_ref['team'] == data.at[i,column]]['crest'].values[0]
        return data

    championship = get_crest(championship,'team')
    playoffs = get_crest(playoffs,'team')
    standings = get_crest(standings,'team')
    team_form_results = get_crest(team_form_results,'index')

    columns = standings.columns

    return render_template('cpl-es-standings.html',columns = columns,
    championship_table = championship, standings_table = standings, playoffs_table = playoffs,
    form_table = team_form_results, year = year,
    headline = 'Standings')

@canples.route('/best11', methods=['GET','POST'])
def eleven():

    get_year()

    best_eleven = pd.read_csv(f'datasets/{year}/playerstats/{year}-best_eleven.csv')
    #best_eleven = pd.read_csv(f'datasets/{year}/cpl-{year}-best_eleven.csv')
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')

    attackers = best_eleven[best_eleven['position'] == 'f']
    midfield = best_eleven[best_eleven['position'] == 'm']

    defenders = best_eleven[best_eleven['position'] == 'd']
    keeper = best_eleven[best_eleven['position'] == 'g']

    return render_template('cpl-es-best_eleven.html',
    html_table = best_eleven,  headline = 'Best Eleven', year = year,
    attackers = attackers, defenders = defenders, midfield = midfield, keeper = keeper)

@canples.route('/power', methods=['GET','POST'])
def power():

    get_year()

    if year != this_year:
        headline = f'Final Power Rankings for '
    else:
        headline = 'Power Rankings '

    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]

    power = pd.read_csv(f'datasets/{year}/league/{year}-power_rankings.csv')

    for i in range(power.shape[0]):
        power.at[i,'crest'] = team_ref[team_ref['team'] == power.at[i,'team']]['crest'].values[0]
        power.at[i,'colour'] = team_ref[team_ref['team'] == power.at[i,'team']]['colour'].values[0]

    return render_template('cpl-es-power.html',html_table = power, year = year, headline = headline)

@canples.route('/versus', methods=['GET','POST'])
def versus():

    import pickle
    classifier = 'models/cpl_MATCH_classifier-08-21-rf1-2.sav' # BEST so far (25.0, 39.29, 35.71)
    cpl_classifier_model = pickle.load(open(classifier, 'rb'))
    regressor = 'models/cpl_score_regressor-07-20-vr---34.sav' # good results somewhat HIGH
    cpl_score_model = pickle.load(open(regressor, 'rb'))

    year = '2020'

    try:
        matches_predictions = pd.read_csv(f'datasets/{year}/cpl-{year}-match_predictions.csv')
    except:
        matches_predictions = pd.DataFrame()
    try:
        game_form = pd.read_csv(f'datasets/{year}/cpl-{year}-game_form.csv')
    except:
        game_form = pd.DataFrame()
    '''try:
        team_rosters = pd.read_csv(f'datasets/{year}/cpl-{year}-team_rosters.csv')
    except:
        team_rosters = pd.DataFrame()'''

    results, team_ref, player_info, results_old, results_diff, schedule, results_brief = load_main_files(year)
    rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists, stats = load_player_files(year)


    stats_season = pd.read_csv(f'datasets/{year}/cpl-{year}-stats.csv')
    stats_seed = pd.read_csv(f'datasets/{year}/cpl-{year}-stats-seed.csv')

    results_old = pd.read_csv(f'datasets/{year}/cpl-{year}-results_old.csv')
    stats_old = pd.read_csv(f'datasets/{year}/cpl-{year}-stats_old.csv')

    # home side
    q1 = matches_predictions.iloc[0]['home']
    home_team_info = team_ref[team_ref['team'] == q1]
    home_colour = home_team_info.iloc[0]['colour']
    home_fill = home_team_info.iloc[0]['colour2']
    home_crest = home_team_info.iloc[0]['crest']

    game_info = schedule[schedule['home'] == q1]
    game = game_info.iloc[0]['game']

    # away side
    q2 = game_info.iloc[0]['away']
    away_team_info = team_ref[team_ref['team'] == q2]
    away_colour = away_team_info.iloc[0]['colour']
    away_fill = away_team_info.iloc[0]['colour2']
    away_crest = away_team_info.iloc[0]['crest']

    home_win = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['home_p'].values[0]
    draw = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['draw_p'].values[0]
    away_win = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['away_p'].values[0]
    home_form = game_form[q1]
    away_form = game_form[q2]

    home_roster = cpl_main.best_roster(q1, rated_keepers, rated_defenders, rated_midfielders, rated_forwards, player_info)
    away_roster = cpl_main.best_roster(q2, rated_keepers, rated_defenders, rated_midfielders, rated_forwards, player_info)

    home_score = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['hs'].values[0]
    away_score = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['as'].values[0]

    # THIS IS OLD HERE BELOW
    results_brief = cpl_main.get_results_brief(results,year)
    results_brief_old = cpl_main.get_results_brief(results_old,year)
    results_brief = pd.concat([results_brief,results_brief_old])

    compare = cpl_main.get_team_comparison(results_brief,q1,q2)
    q1_r = cpl_main.get_match_history(compare,q1)
    q2_r = cpl_main.get_match_history(compare,q2)


    team1, team2, team3, team4, team5, team6, team7, team8 = cpl_main.get_team_files(schedule,team_ref)

    if (home_win < draw) and (away_win < draw):
        home_win, away_win = draw, draw

    home_win = round(round(home_win,3)*100,3)
    away_win = round(round(away_win,3)*100,3)
    draw = round(round(draw,3)*100,3)

    group1 = team1 + '-' + team2
    if team3 == 1:
        group2 = 1
        group3 = 1
        group4 = 1
    else:
        group2 = team3 + '-' + team4
        if team5 == 1:
            group3 = 1
            group4 = 1
        else:
            group3 = team5 + '-' + team6
            group4 = team7 + '-' + team8

    radar = pd.read_csv(f'datasets/{year}/league/{year}-radar.csv')
    homet = max(q1.split(' '), key=len)
    awayt = max(q2.split(' '), key=len)
    home_radar = radar[radar['team'].str.contains(homet)]
    home_radar = home_radar.reset_index()
    home_radar.pop('index')
    away_radar = radar[radar['team'].str.contains(awayt)]
    away_radar = away_radar.reset_index()
    away_radar.pop('index')

    home_sum = home_roster['overall'].sum()
    away_sum = away_roster['overall'].sum()

    #DEFINE GAME_WEEK FOR THIS headline = f'Week {game_week} Matches:'

    if results.iloc[-2]['hr'] != 'E':
        headline = f'Finals: {q1} vs {q2}'
    else:
        #headline = f'Week {game_week} Matches:'
        headline = f'Week __ Matches:'

    return render_template('cpl-es-comparison.html',
    home_team = q1, home_table = home_roster.head(11), home_win = home_win, home_history = q1_r,
    home_crest = home_crest, home_colour = home_colour, home_fill = home_fill, home_radar = home_radar,
    away_team = q2, away_table = away_roster.head(11), away_win = away_win, away_history = q2_r,
    away_crest = away_crest, away_colour = away_colour, away_fill = away_fill, away_radar = away_radar,
    draw = draw, home_form = home_form, away_form = away_form,
    headline = headline, year = year,
    home_score = home_score, away_score = away_score,
    team1 = team1, team2 = team2, team3 = team3, team4 = team4, team5 = team5, team6 = team6, team7 = team7, team8 = team8,
    group1 = group1, group2 = group2, group3 = group3, group4 = group4)

@canples.route('/teams', methods=['GET','POST'])
def teams():

    get_year()
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    columns = team_ref.columns

    return render_template('cpl-es-teams.html',columns = columns, headline = 'Club Information',
    html_table = team_ref, year = year, roster = roster)

@canples.route('/radar', methods=['GET','POST'])
def radar():

    get_year()

    geegle = ['#000','#fff','#ffd700','#ff00ff','#4a86e8','#0000ff','#9900ff','#ff00ff']

    team_standings = pd.read_csv(f'datasets/{year}/cpl-{year}-standings_current.csv')
    radar = pd.read_csv(f'datasets/{year}/league/{year}-radar.csv')
    team_standings = team_standings.sort_values(by='team')
    team_standings['xg'] = round((team_standings['gf'] / team_standings['gp'])*7,2)
    team_standings['xp'] = round(team_standings['pts'] / team_standings['gp'],2)
    team_standings['xt'] = team_standings['xp'] * 7
    team_standings['xg'] = team_standings['xg'].astype('int')
    team_standings['xt'] = team_standings['xt'].astype('int')
    team_standings = team_standings.reset_index()
    team_standings.pop('index')

    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]

    team_list = sorted([x for x in team_ref['team'].unique()])

    ### GET TEAM LINE CHARTS
    team_dict = {}
    team_stats = {}
    for team in team_ref['team'].unique():
        team_stats[team] = []
        team_dict[team] = pd.read_csv(f'datasets/{year}/league/{year}-{team}-line.csv')

        team_stats[team].append(team_dict[team].loc[team_dict[team].shape[0]-1]['ExpG'])
        team_stats[team].append(team_dict[team].loc[team_dict[team].shape[0]-1]['ExpA'])
        team_stats[team].append(round( team_dict[team]['Goal'].sum() / team_dict[team].shape[0],2))
        team_dict[team] = team_dict[team][team_dict[team].columns[2:]].T
        team_dict[team] = cpl_main.index_reset(team_dict[team]).values.tolist()

    columns = team_ref.columns

    return render_template('cpl-es-radar.html',columns = columns, html_table = team_ref,
    team_list = team_list, team_dict = team_dict, team_stats = team_stats,
    stats = team_standings, radar = radar, year = year, headline = 'Radar Charts')

@canples.route('/roster', methods=['GET','POST'])
def roster():

    team = request.form['team']

    radar = pd.read_csv(f'datasets/{year}/league/{year}-radar.csv')
    radar = radar[radar['team'] == team]
    radar = cpl_main.index_reset(radar)

    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')
    roster = player_info[player_info['team'] == team][['name','image','position','number','flag','overall','link']]
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    coach = team_ref[team_ref['team'] == team][['cw','cl','cd','coach','country','image','w','l','d','year']]

    roster_team_info = team_ref[team_ref['team'] == team].copy()
    roster_team_info = cpl_main.index_reset(roster_team_info)
    roster_colour = roster_team_info.iloc[0]['colour']
    crest = roster_team_info.iloc[0]['crest']
    colour1 = roster_team_info.iloc[0]['colour1']
    colour2 = roster_team_info.iloc[0]['colour2']

    ### GET TEAM LINE CHARTS
    team_stats = []
    for team_name in team_ref['team'].unique():
        team_line = pd.read_csv(f'datasets/{year}/league/{year}-{team_name}-line.csv')
        team_stats.append(team_line.loc[team_line.shape[0]-1]['ExpG'])
        team_stats.append(team_line.loc[team_line.shape[0]-1]['ExpA'])
        team_stats.append(round( team_line['Goal'].sum() / team_line.shape[0],2))
        team_line = team_line[team_line.columns[2:]].T
        team_line = cpl_main.index_reset(team_line).values.tolist()
    ### COMPLETE THIS

    return render_template('cpl-es-roster.html',team_name = team, coach = coach, radar = radar, year = year, team_line = team_line, team_stats = team_stats,
    crest = crest, colour1 = colour1, colour2 = colour2, html_table = roster, team_colour = roster_colour)

@canples.route('/player', methods=['POST'])
def player():

    get_year()

    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')

    name = request.form['name']

    if name in player_info['name'].unique():
        print('NAME present')
        pass
    else:
        print('NAME not present')
        try:
            name = player_info[player_info['name'] == name]['name'].values[0]
        except Exception as e:
            print(e)
            sleep(2)
            name = player_info[player_info['display'] == name]['name'].values[0]

    player = player_info[player_info['name'] == name]
    team = player['team'].values[0]
    colour3 = team_colour[team]
    roster_team_info = team_ref[team_ref['team'] == team]
    roster_colour = roster_team_info.iloc[0][4]
    crest = roster_team_info.iloc[0]['crest']
    colour1 = roster_team_info.iloc[0]['colour1']
    colour2 = roster_team_info.iloc[0]['colour2']
    pos = player['position'].values[0]
    position = {'d':'defenders','f':'forwards','g':'keepers','m':'midfielders'}
    db = pd.read_csv(f'datasets/{year}/playerstats/{year}-{position.get(pos)}.csv')
    db90 = pd.read_csv(f'datasets/{year}/playerstats/{year}-{position.get(pos)}90.csv')
    discipline = pd.read_csv(f'datasets/{year}/playerstats/{year}-discipline.csv')

    db = new_col(db)
    db90 = new_col(db)

    radar_chart = db.copy()
    radar_chart.pop('display')
    radar_chart = new_col(radar_chart)
    for col in radar_chart.columns[5:-1]:
        radar_chart[col] = round((radar_chart[col] / (radar_chart[col].max()+0.05)),2)
    radar_chart = radar_chart[radar_chart['name'] == name][radar_chart.columns[6:-1]]
    radar_chart = cpl_main .index_reset(radar_chart)
    radar_chart_cols = "'"+"', '".join(radar_chart.columns)+"'"
    db = db[db['name'] == name][db.columns]
    db90 = db90[db90['name'] == name][db90.columns]
    discipline = discipline[discipline['name'] == name]

    player_line_db = pd.read_csv(f'datasets/{year}/playerstats/{year}-{position.get(pos)}-line.csv')
    try:
        player_line_db = player_line_db[['name','display','Touches','Goal','PsAtt','PsCmpM3','TchsM3']]
    except:
        try:
            player_line_db = player_line_db[['name','display','Touches','Goal','PsAtt','TchsA3']]
        except:
            try:
                player_line_db = player_line_db[['name','display','Touches','Goal','TchsM3','DefTouch']]
            except:
                player_line_db = player_line_db[['name','display','CleanSheet','Saves','SvDive','Recovery','ExpGAg']]

    player_line_db = new_col(player_line_db)

    def get_norm(data):
        df = data.copy()
        cols = [x for x in data.columns if x not in ['name','display','G','CS']]
        for col in cols:
            df[col] = round(df[col]/ df[col].max() ,2)*2
        return df

    player_line_db = get_norm(player_line_db)

    def percentage_check(x):
        try:
            if '%' in x:
                return float(re.sub('%','',x))
            elif '-' in x:
                return 0.0
        except:
            return x

    def build_player_line(player_line_db,name,string='name'):
        if 'G' in player_line_db.columns.values:
            g = player_line_db.pop('G')
            player_line_db.insert(1,'G',g)
        if string == 'display':
            name = player_info[player_info['name'] == name]['display'].values[0]
        else:
            pass
        data = player_line_db[player_line_db[string] == name][player_line_db.columns[1:-1]].copy().T
        line_columns = [x for x in player_line_db.columns[1:-1]]

        for col in data.columns:
            data[col] = data[col].apply(lambda x: percentage_check(x))
        return data, line_columns

    player_line, line_columns = build_player_line(player_line_db,name)
    player_line = cpl_main.index_reset(player_line).values.tolist()
    player_line_length = len(player_line) - 1
    player_line_end = len(player_line) - 1

    if player_line[0]:
        print('Dataset filled')
        print('+++++++++++++++++++++++++++++++++++')
    else:
        print('Dataset empty')
        print('-----------------------------------')
        player_line, line_columns = build_player_line(player_line_db,name,string='display')
        player_line = cpl_main.index_reset(player_line).values.tolist()

    chart_team_colour_list = ['#fa5252','#ffa64a','#88de62','#67bccf','#508ceb','#5199db','#6c51bd','#bf5c90']*2
    cav = ["#007f5f","#2b9348","#55a630","#80b918","#aacc00","#bfd200","#d4d700","#dddf00","#eeef20","#ffff3f"]
    ato = ["#03045e","#023e8a","#0077b6","#0096c7","#00b4d8","#48cae4","#90e0ef","#ade8f4","#caf0f8"]
    forge = ["#03071e","#370617","#6a040f","#9d0208","#d00000","#dc2f02","#e85d04","#f48c06","#faa307","#ffba08"]
    pastels = ["#7400b8","#6930c3","#5e60ce","#5390d9","#4ea8de","#48bfe3","#56cfe1","#64dfdf","#72efdd","#80ffdb"]
    pastels2 = ["#f72585","#b5179e","#7209b7","#560bad","#480ca8","#3a0ca3","#3f37c9","#4361ee","#4895ef","#4cc9f0"]
    oranges = ["#ff4800","#ff5400","#ff6000","#ff6d00","#ff7900","#ff8500","#ff9100","#ff9e00","#ffaa00","#ffb600"]
    val = ["#ff7b00","#ff8800","#ff9500","#ffa200","#ffaa00","#ffb700","#ffc300","#ffd000","#ffdd00","#ffea00"]
    grays = ["#eaeaea","#c0c1c4","#abadb1","#96989e","#81848b","#6c6f78","#63666f","#535761","#424651"]
    blues2 = ["#0466c8","#0353a4","#023e7d","#002855","#001845","#001233","#33415c","#5c677d","#7d8597","#979dac"]
    geegle = ['#ff9900','#ffff00','#4a86e8','#ff00ff','#4a86e8','#0000ff','#9900ff','#ff00ff']
    geegle = ['#000','#fff','#ffd700','#ff00ff','#4a86e8','#0000ff','#9900ff','#ff00ff']
    google = ['#df0772','#fe546f','#ff9e7d','#ffd080','#01cbcf','#0188a5','#6450a6']

    chart_team_colour_list = {
    '#102f52' : ["#03045e","#023e8a","#0077b6","#0096c7","#00b4d8","#48cae4","#90e0ef","#ade8f4","#caf0f8"],
    '#335525' : ["#007f5f","#2b9348","#55a630","#80b918","#aacc00","#bfd200","#d4d700","#dddf00","#eeef20","#ffff3f"],
    '#0c2340' : ["#0466c8","#0353a4","#023e7d","#002855","#001845","#001233","#33415c","#5c677d","#7d8597","#979dac"],
    '#53565a' : ["#eaeaea","#c0c1c4","#abadb1","#96989e","#81848b","#6c6f78","#63666f","#535761","#424651"],
    '#052049' : ['#006666','#3300CC','#3333FF','#0033CC','#003399','#006699','#006666','#3399CC'],
    '#00b7bd' : ["#7400b8","#6930c3","#5e60ce","#5390d9","#4ea8de","#48bfe3","#56cfe1","#64dfdf","#72efdd","#80ffdb"],
    '#b9975b' : ["#ff7b00","#ff8800","#ff9500","#ffa200","#ffaa00","#ffb700","#ffc300","#ffd000","#ffdd00","#ffea00"],
    '#003b5c' : ["#03045e","#023e8a","#0077b6","#0096c7","#00b4d8","#48cae4","#90e0ef","#ade8f4","#caf0f8"]
    }

    details = {}
    for word in ['display','number','nationality','graph','radar','team']:
        if word in ['graph','radar']:
            try:
                details[word+'_image'] = player[player['display'] == name][word].values[0]
            except:
                details[word+'_image'] = ''
        else:
            try:
                details[word] = player[player['display'] == name][word].values[0]
            except:
                details[word] = player[player['name'] == name][word].values[0]
    #line_back_colour = chart_team_colour_list[details[team]]

    if position.get(pos)[:1] == 'f':
        length = len(db.columns)
        half = int(length/2)
        full = int(len(db.columns) - 2)
        col_nums = [half,full]
    else:
        length = len(db.columns)
        half = int(length/2)
        full = int(len(db.columns) - 2)
        col_nums = [half,full]

    return render_template('cpl-es-player.html', name = details['display'], graph = details['graph_image'], player_line_length = player_line_length, player_line_end = player_line_end,
    radar = details['radar_image'], nationality = details['nationality'], team_name = team, player_info = player, full_name = name, year = year,
    team_colour = roster_colour, crest = crest, position = position.get(pos)[:-1], number = details['number'], chart_team_colour_list = geegle,
    stats = db, stats90 = db90, discipline = discipline, radar_chart = radar_chart, radar_chart_cols = radar_chart_cols,
    colour1 = colour1, colour2 = colour2, colour3 = colour3, col_nums = col_nums, player_line = player_line,line_columns = line_columns)

@canples.route('/goals', methods=['GET','POST'])
def goals():

    get_year()

    #rated_goalscorers = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_goalscorers.csv')
    rated_goalscorers = pd.read_csv(f'datasets/{year}/playerstats/{year}-goalscorers.csv')
    rated_g10 = rated_goalscorers.head(10)
    #rated_assists = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_assists.csv')
    rated_assists = pd.read_csv(f'datasets/{year}/playerstats/{year}-assists.csv')
    rated_a10 = rated_assists.head(10)

    columns_g = rated_goalscorers.columns
    columns_a = rated_assists.columns

    return render_template('cpl-es-goals.html',columns_g = columns_g, columns_a = columns_a, year = year,
    html_table = rated_g10, assists_table = rated_a10, headline = 'Top 10 Goals / Assists')

@canples.route('/forwards', methods=['GET','POST'])
def forwards():
    get_year()

    rated_forwards = pd.read_csv(f'datasets/{year}/playerstats/{year}-forwards.csv')
    columns = rated_forwards.columns

    rated_forwards = new_col(rated_forwards)

    return render_template('cpl-es-position.html', year = year,
    columns = columns,html_table = rated_forwards)

@canples.route('/forwardsP90', methods=['GET','POST'])
def forwards_90():

    get_year()

    rated_forwards_90 = pd.read_csv(f'datasets/{year}/playerstats/{year}-forwards90.csv')
    rated_forwards = rated_forwards_90.sort_values(by='overall',ascending=False)
    columns = rated_forwards.columns

    rated_forwards = new_col(rated_forwards)

    return render_template('cpl-es-position.html', year = year,
    columns = columns,html_table = rated_forwards)

@canples.route('/midfielders', methods=['GET','POST'])
def midfielders():

    get_year()

    rated_midfielders = pd.read_csv(f'datasets/{year}/playerstats/{year}-midfielders.csv')
    columns = rated_midfielders.columns

    rated_midfielders = new_col(rated_midfielders)

    return render_template('cpl-es-position.html', year = year,
    columns = columns,html_table = rated_midfielders)

@canples.route('/midfieldersP90', methods=['GET','POST'])
def midfielders_90():

    get_year()

    rated_midfielders_90 = pd.read_csv(f'datasets/{year}/playerstats/{year}-midfielders90.csv')
    rated_midfielders = rated_midfielders_90.sort_values(by='overall',ascending=False)
    columns = rated_midfielders.columns

    rated_midfielders = new_col(rated_midfielders)

    return render_template('cpl-es-position.html', year = year,
    columns = columns,html_table = rated_midfielders)

@canples.route('/defenders', methods=['GET','POST'])
def defenders():

    get_year()

    rated_defenders = pd.read_csv(f'datasets/{year}/playerstats/{year}-defenders.csv')
    columns = rated_defenders.columns

    rated_defenders = new_col(rated_defenders)

    return render_template('cpl-es-position.html', year = year,
    columns = columns,html_table = rated_defenders)

@canples.route('/defendersP90', methods=['GET','POST'])
def defenders_90():

    get_year()

    rated_defenders_90 = pd.read_csv(f'datasets/{year}/playerstats/{year}-defenders90.csv')
    rated_defenders = rated_defenders_90.sort_values(by='overall',ascending=False)
    columns = rated_defenders.columns

    rated_defenders = new_col(rated_defenders)

    return render_template('cpl-es-position.html', year = year,
    columns = columns,html_table = rated_defenders)

@canples.route('/keepers', methods=['GET','POST'])
def keepers():

    get_year()

    rated_keepers = pd.read_csv(f'datasets/{year}/playerstats/{year}-keepers.csv')
    columns = rated_keepers.columns

    rated_keepers = new_col(rated_keepers)

    return render_template('cpl-es-position.html', year = year,
    columns = columns,html_table = rated_keepers)

@canples.route('/keepersP90', methods=['GET','POST'])
def keepers_90():

    get_year()

    rated_keepers_90 = pd.read_csv(f'datasets/{year}/playerstats/{year}-keepers90.csv')
    rated_keepers = rated_keepers_90.sort_values(by='overall',ascending=False)
    columns = rated_keepers.columns

    rated_keepers = new_col(rated_keepers)

    return render_template('cpl-es-position.html', year = year,
    columns = columns,html_table = rated_keepers)

@canples.route('/discipline', methods=['GET','POST'])
def discipline():

    get_year()

    rated_offenders = pd.read_csv(f'datasets/{year}/playerstats/{year}-discipline.csv')
    columns = rated_offenders.columns

    return render_template('cpl-es-discipline.html',columns = columns,
    html_table = rated_offenders, year = year, headline = 'Discipline')

@canples.route('/hell')
def hello():
    return 'Welcome to HELL world!'

if __name__ == "__main__":
    canples.run()
