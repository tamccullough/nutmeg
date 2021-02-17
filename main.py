from flask import Flask
from flask import Blueprint, flash, g, redirect, render_template, request, url_for
import cpl_main

## GETTING DATE AND TIME
from datetime import date
today = date.today()
this_year = date.today().strftime('%Y')

import numpy as np
import pandas as pd

import pickle
classifier = 'models/cpl_MATCH_classifier-08-21-rf1-2.sav' # BEST so far (25.0, 39.29, 35.71)
cpl_classifier_model = pickle.load(open(classifier, 'rb'))
regressor = 'models/cpl_score_regressor-07-20-vr---34.sav' # good results somewhat HIGH
cpl_score_model = pickle.load(open(regressor, 'rb'))

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

canples = Flask(__name__)

def convert_num_str(num):
    num = str(num*100)
    return num[0:4]

def get_year():
    global year
    try:
        year = request.form['year']
    except:
        year = current_year
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

    team_stats = pd.read_csv(f'datasets/{year}/cpl-{year}-team_stats.csv')
    colours = team_ref['colour']

    results_brief = cpl_main.get_results_brief(results)
    try:
        matches_predictions = pd.read_csv(f'datasets/{year}/cpl-{year}-match_predictions.csv')
    except:
        matches_predictions = pd.DataFrame()
    try:
        game_form = pd.read_csv(f'datasets/{year}/cpl-{year}-game_form.csv')
    except:
        game_form = pd.DataFrame()
    try:
        team_rosters = pd.read_csv(f'datasets/{year}/cpl-{year}-team_rosters.csv')
    except:
        team_rosters = pd.DataFrame()

    return results, stats, stats_seed, team_ref, player_info, results_old, results_diff, schedule, team_stats, results_brief, matches_predictions, game_form, team_rosters


def load_player_files(year):
    # get all rated player information based on position and calculate and overall score for the individual player
    rated_goalscorers = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_goalscorers.csv')
    rated_offenders = pd.read_csv(f'datasets/{year}/cpl-{year}-discipline.csv')
    rated_assists = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_assists.csv')

    rated_forwards = pd.read_csv(f'datasets/{year}/playerstats/{year}-forwards.csv')
    rated_midfielders = pd.read_csv(f'datasets/{year}/playerstats/{year}-midfielders.csv')
    rated_defenders = pd.read_csv(f'datasets/{year}/playerstats/{year}-defenders.csv')
    rated_keepers = pd.read_csv(f'datasets/{year}/playerstats/{year}-keepers.csv')

    rated_forwards_90 = pd.read_csv(f'datasets/{year}/playerstats/{year}-forwards.csv')
    rated_midfielders_90 = pd.read_csv(f'datasets/{year}/playerstats/{year}-midfielders.csv')
    rated_defenders_90 = pd.read_csv(f'datasets/{year}/playerstats/{year}-defenders.csv')
    rated_keepers_90 = pd.read_csv(f'datasets/{year}/playerstats/{year}-forwards.csv')

    return rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists


# get current day and set time variables
month, day, weekday = cpl_main.get_weekday()
current_year = '2020'
games_played = {1:28,2:7}
# set the year - which will change based on user choice
year = current_year

results, stats, stats_seed, team_ref, player_info, results_old, results_diff, schedule, team_stats, results_brief, matches_predictions, game_form, team_rosters  = load_main_files(year)
rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists = load_player_files(year)

@canples.context_processor
def inject_user():

    return dict(today = today, day = day, weekday = weekday, month = month, year = year, theme = 'bland')

@canples.route('/', methods=['GET','POST'])
def index():
    na = 'TBD'

    #year = current_year
    get_year()

    results, stats, stats_seed, team_ref, player_info, results_old, results_diff, schedule, team_stats, results_brief, matches_predictions, game_form, team_rosters  = load_main_files(year)

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
        print('********************************')
        print('YEAR:',year)
        print('********************************')
        if year == '2019':
            top_dropper = playoffs.iloc[-1]['team']
            print('********************************')
            print('TOP TEAM: \n',playoffs.iloc[0]['team'])
            print('********************************')
            print('BOTTOM TEAM: \n',playoffs.iloc[-1]['team'])
        else:
            top_dropper = standings.iloc[-1]['team']
            print('********************************')
            print('TOP TEAM: \n',standings.iloc[0]['team'])
            print('********************************')
            print('BOTTOM TEAM: \n',standings.iloc[-1]['team'])
        print('********************************')
        print('********************************')
        bot_crest = team_ref[team_ref['team'].str.contains(top_dropper)]
        bot_crest = bot_crest.iloc[0]['crest']
    #game_week, goals, big_win, top_team, low_team, other_team, assists, yellows, reds
    game_week, goals, big_win, top_result, low_result, other_result, assists, yellows, reds = cpl_main.get_weeks_results(year,results,standings,stats,team_ref,team_names)
    assists, yellows, reds = int(assists), int(yellows), int(reds)

    top_forward = rated_forwards.loc[0]
    top_midfielder = rated_midfielders.loc[0]
    top_defender = rated_defenders.loc[0]
    top_scorer = rated_goalscorers.loc[0]
    top_scorer['overall'] = player_info[player_info['name'] == top_scorer['name']]['overall'].values[0]
    top_assist = rated_assists.loc[0]
    top_assist['overall'] = player_info[player_info['name'] == top_assist['name']]['overall'].values[0]
    top_keeper = rated_keepers.loc[0]
    top_offender = rated_offenders.loc[0]

    if results.iloc[0]['hr'] == 'E':
        top_team, top_mover, top_dropper, first_crest, top_crest, bot_crest, first_colour = na, na, na, 'CPL-Crest-White.png', 'oneSoccer_nav.png', 'canNat_icon.png', 'w3-indigo'

    suspended = 'none'
    if championship.empty:
        headline = f'{year} Season Underway'
    else:
        headline = f"{year} Champions - {championship.iloc[0]['team']}"

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

    columns = standings.columns

    return render_template('cpl-es-standings.html',columns = columns,
    championship_table = championship, standings_table = standings, playoffs_table = playoffs,
    form_table = team_form_results,
    headline = 'Standings')

@canples.route('/best11', methods=['GET','POST'])
def eleven():

    get_year()
    best_eleven = pd.read_csv(f'datasets/{year}/playerstats/{year}-best_eleven.csv')
    #best_eleven = pd.read_csv(f'datasets/{year}/cpl-{year}-best_eleven.csv')
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')
    '''names=[]
    for i in range(best_eleven.shape[0]):
        name = best_eleven.iloc[i]['first']+' '+best_eleven.iloc[i]['last']
        replace = player_info[player_info['display'] == name]['name'].values[0]
        names.append(replace)'''
    #best_eleven['full_name'] = names
    attackers = best_eleven[best_eleven['position'] == 'f']
    midfield = best_eleven[best_eleven['position'] == 'm']
    '''midfield = midfield.sort_values(by='overall',ascending=False)
    midfield = cpl_main.index_reset(midfield)
    print(midfield)'''
    defenders = best_eleven[best_eleven['position'] == 'd']
    keeper = best_eleven[best_eleven['position'] == 'g']

    return render_template('cpl-es-best_eleven.html',
    html_table = best_eleven,  headline = 'Best Eleven',
    attackers = attackers, defenders = defenders, midfield = midfield, keeper = keeper)

@canples.route('/power')
def power():
    get_year()
    power = pd.read_csv(f'datasets/{year}/cpl-{year}-power_rankings.csv')
    return render_template('cpl-es-power.html',html_table = power)

@canples.route('/versus', methods=['GET','POST'])
def comparison():

    get_year()

    print('\nON THE PREDICTIONS PAGE\n')

    results_old = pd.read_csv(f'datasets/{year}/cpl-{year}-results_old.csv')
    stats_old = pd.read_csv(f'datasets/{year}/cpl-{year}-stats_old.csv')

    # home side
    q1 = matches_predictions.iloc[0]['home']
    home_team_info = team_ref[team_ref['team'] == q1]
    home_colour = home_team_info.iloc[0]['colour']
    home_fill = home_team_info.iloc[0]['colour2']
    home_crest = home_team_info.iloc[0]['crest']

    game_info = schedule[schedule['home'] == q1]
    print(game_info)
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

    home_roster = cpl_main.best_roster(q1,results,results_old,stats,stats_old,stats_seed,player_info,rated_forwards)
    away_roster = cpl_main.best_roster(q2,results,results_old,stats,stats_old,stats_seed,player_info,rated_forwards)
    home_score = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['hs'].values[0]
    away_score = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['as'].values[0]

    # THIS IS OLD HERE BELOW
    results_brief = cpl_main.get_results_brief(results)
    results_brief_old = cpl_main.get_results_brief(results_old)
    results_brief = pd.concat([results_brief,results_brief_old])
    print('*********************************************')
    print(results_brief)
    print('*********************************************')
    compare = cpl_main.get_team_comparison(results_brief,q1,q2)
    q1_r = cpl_main.get_match_history(compare,q1)
    q2_r = cpl_main.get_match_history(compare,q2)


    team1, team2, team3, team4, team5, team6, team7, team8 = cpl_main.get_team_files(schedule,team_ref)
    #print(team1, team2, team3, team4, team5, team6, team7, team8)

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

    print('*********************************************')
    #print('Week:\n',game_week)
    print('*********************************************')
    print('*********************************************')
    print('HOME TEAM:\n',homet)
    print('HOME RADAR:\n',home_radar)
    print('*********************************************')
    print('*********************************************')
    print('HOME TEAM:\n',awayt)
    print('HOME RADAR:\n',away_radar)
    print('*********************************************')

    home_sum = home_roster['overall'].sum()
    away_sum = away_roster['overall'].sum()
    print('HOME FORM:\n',home_form,'\n')
    print('HOME SHAPE:\n',home_form.shape[0],'\n')

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
    headline = headline,
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
    html_table = team_ref,  roster = roster)

@canples.route('/radar', methods=['GET','POST'])
def radar():

    get_year()
    page = '/radar'
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
    print(team_standings)
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    columns = team_ref.columns

    print('=============================================')
    print('*********************************************')
    print('Standings:\n',team_standings)
    print('RADAR:\n')
    print(radar)
    print('*********************************************')
    print('=============================================')

    return render_template('cpl-es-radar.html',columns = columns, html_table = team_ref,
    stats = team_standings, radar = radar, headline = 'Radar Charts')

@canples.route('/roster', methods=['GET','POST'])
def roster():

    team = request.form['team']

    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')
    roster = player_info[player_info['team'] == team][['name','image','position','number','flag','overall','link']]
    coach = team_ref[team_ref['team'] == team][['cw','cl','cd','coach','country','image','w','l','d','year']]

    print(roster.columns)
    print(roster)
    print(coach.columns)
    print(coach)
    print('COLOUR:',team_ref[team_ref['team'] == team]['colour'].values[0])

    return render_template('cpl-es-roster.html',team_name = team, coach = coach,
    html_table = roster, team_colour = team_ref[team_ref['team'] == team]['colour'].values[0])

@canples.route('/player', methods=['POST'])
def player():

    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')

    print('=============================================')
    print('*********************************************')
    print('NAME REQUEST:')
    print(request.form['name'])
    print('*********************************************')
    print('=============================================')
    name = request.form['name']

    player = player_info[player_info['name'] == name]
    team = player['team'].values[0]
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
    radar_chart = db.copy()
    radar_chart.pop('display')
    for col in radar_chart.columns[5:-1]:
        print('*********************************************')
        print(col)
        print('*********************************************')
        radar_chart[col] = round((radar_chart[col] / (radar_chart[col].max()+0.05)),2)
    radar_chart = radar_chart[radar_chart['name'] == name][radar_chart.columns[6:-1]]
    radar_chart = cpl_main .index_reset(radar_chart)
    radar_chart_cols = "'"+"', '".join(radar_chart.columns)+"'"
    db = db[db['name'] == name][db.columns]
    print('\n',name,'\n',db)
    db90 = db90[db90['name'] == name][db90.columns]
    discipline = discipline[discipline['name'] == name]
    print('\n',discipline,'\n')

    details = {}
    for word in ['display','number','nationality','graph','radar']:
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

    print('=============================================')
    print('*********************************************')
    print('Player:\n',details['display'])
    print('nationality:\n',details['nationality'])
    print('\n')
    print(radar_chart_cols)
    print(', '.join([str(x) for x in radar_chart.loc[0]]))
    print('\n')
    print(colour1)
    print(colour2)
    print(int(len(db.columns)))
    print(5+int((len(db.columns)-5)/2))
    print('*********************************************')
    print('=============================================')
    if position.get(pos)[:-1] == 'f':
        col_nums = [8,8+int((len(db.columns)-8)/2)]
    else:
        col_nums = [8,8+int((len(db.columns)-8)/2)]
    print(col_nums)
    print('*********************************************')
    print('=============================================')


    return render_template('cpl-es-player.html', name = details['display'], graph = details['graph_image'],
    radar = details['radar_image'], nationality = details['nationality'], team_name = team, player_info = player,
    team_colour = roster_colour, crest = crest, position = position.get(pos)[:-1], number = details['number'],
    stats = db, stats90 = db90, discipline = discipline, radar_chart = radar_chart, radar_chart_cols = radar_chart_cols,
    colour1 = colour1, colour2 = colour2,col_nums = col_nums)

@canples.route('/goals', methods=['GET','POST'])
def goals():
    get_year()
    rated_goalscorers = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_goalscorers.csv')
    rated_g10 = rated_goalscorers.head(10)
    rated_assists = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_assists.csv')
    rated_a10 = rated_assists.head(10)
    columns_g = rated_goalscorers.columns
    columns_a = rated_assists.columns

    return render_template('cpl-es-goals.html',columns_g = columns_g, columns_a = columns_a,
    html_table = rated_g10, assists_table = rated_a10, headline = 'Top Goal Scorers')

@canples.route('/forwards', methods=['GET','POST'])
def forwards():
    get_year()

    rated_forwards = pd.read_csv(f'datasets/{year}/playerstats/{year}-forwards.csv')
    columns = rated_forwards.columns

    return render_template('cpl-es-forwards.html',
    columns = columns,html_table = rated_forwards)

@canples.route('/forwardsP90', methods=['GET','POST'])
def forwards_90():
    get_year()

    rated_forwards_90 = pd.read_csv(f'datasets/{year}/playerstats/{year}-forwards90.csv')
    rated_forwards = rated_forwards_90.sort_values(by='overall',ascending=False)
    columns = rated_forwards.columns

    return render_template('cpl-es-forwards.html',
    columns = columns,html_table = rated_forwards)

@canples.route('/midfielders', methods=['GET','POST'])
def midfielders():
    get_year()

    rated_midfielders = pd.read_csv(f'datasets/{year}/playerstats/{year}-midfielders.csv')
    columns = rated_midfielders.columns

    return render_template('cpl-es-midfielders.html',
    columns = columns,html_table = rated_midfielders)

@canples.route('/midfieldersP90', methods=['GET','POST'])
def midfielders_90():
    get_year()

    rated_midfielders_90 = pd.read_csv(f'datasets/{year}/playerstats/{year}-midfielders90.csv')
    rated_midfielders = rated_midfielders_90.sort_values(by='overall',ascending=False)
    columns = rated_midfielders.columns

    return render_template('cpl-es-midfielders.html',
    columns = columns,html_table = rated_midfielders)

@canples.route('/defenders', methods=['GET','POST'])
def defenders():
    get_year()

    rated_defenders = pd.read_csv(f'datasets/{year}/playerstats/{year}-defenders.csv')
    columns = rated_defenders.columns

    return render_template('cpl-es-defenders.html',
    columns = columns,html_table = rated_defenders)

@canples.route('/defendersP90', methods=['GET','POST'])
def defenders_90():
    get_year()

    rated_defenders_90 = pd.read_csv(f'datasets/{year}/playerstats/{year}-defenders90.csv')
    rated_defenders = rated_defenders_90.sort_values(by='overall',ascending=False)
    columns = rated_defenders.columns

    return render_template('cpl-es-defenders.html',
    columns = columns,html_table = rated_defenders)

@canples.route('/keepers', methods=['GET','POST'])
def keepers():
    get_year()

    rated_keepers = pd.read_csv(f'datasets/{year}/playerstats/{year}-keepers.csv')
    columns = rated_keepers.columns

    return render_template('cpl-es-keepers.html',
    columns = columns,html_table = rated_keepers)

@canples.route('/keepersP90', methods=['GET','POST'])
def keepers_90():
    get_year()

    rated_keepers_90 = pd.read_csv(f'datasets/{year}/playerstats/{year}-keepers90.csv')
    rated_keepers = rated_keepers_90.sort_values(by='overall',ascending=False)
    columns = rated_keepers.columns

    return render_template('cpl-es-keepers.html',
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
