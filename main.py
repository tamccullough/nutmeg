from flask import Flask
from flask import Blueprint, flash, g, redirect, render_template, request, url_for
import cpl_main
from datetime import date
today = date.today()
current_year = date.today().strftime('%Y')

import numpy as np
import pandas as pd

import pickle
classifier = 'models/cpl_MATCH_classifier-08-21-rf1-2.sav' # BEST so far (25.0, 39.29, 35.71)
cpl_classifier_model = pickle.load(open(classifier, 'rb'))
regressor = 'models/cpl_score_regressor-07-20-vr---34.sav' # good results somewhat HIGH
cpl_score_model = pickle.load(open(regressor, 'rb'))

canples = Flask(__name__)

theme = 'bland'
month, day, weekday = cpl_main.get_weekday()
year = '2020'

def convert_num_str(num):
    num = str(num*100)
    return num[0:4]

def load_main_files(year):
    results = pd.read_csv(f'datasets/{year}/cpl-{year}-results.csv')
    stats = pd.read_csv(f'datasets/{year}/cpl-{year}-stats.csv')
    stats_seed = pd.read_csv(f'datasets/{year}/cpl-{year}-stats-seed.csv')
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')
    current_teams = team_ref['team']
    results_old = results[results['hr'] != 'E'].copy()
    results_diff = pd.concat([results, results_old]).drop_duplicates(keep=False)
    if results_diff.empty:
        results_diff = results_old.tail(1)
    print('THE DIFF \n',results_diff,'\n')
    schedule = cpl_main.get_schedule(results_diff) # from results create the schedule dataset
    team_stats = pd.read_csv(f'datasets/{year}/cpl-{year}-team_stats.csv')
    colours = team_ref['colour']
    results_brief = pd.read_csv(f'datasets/{year}/cpl-{year}-results_brief.csv')
    matches_predictions = pd.read_csv(f'datasets/{year}/cpl-{year}-match_predictions.csv')
    game_form = pd.read_csv(f'datasets/{year}/cpl-{year}-game_form.csv')
    team_rosters = pd.read_csv(f'datasets/{year}/cpl-{year}-team_rosters.csv')

    return results, stats, stats_seed, team_ref, player_info, results_old, results_diff, schedule, stats, team_stats, results_brief, matches_predictions, game_form, team_rosters

def load_main_files_old(year):
    results = pd.read_csv(f'datasets/{year}/cpl-{year}-results.csv')
    stats = pd.read_csv(f'datasets/{year}/cpl-{year}-stats.csv')
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')
    current_teams = team_ref['team']
    results_old = results[:-7].copy()
    results_diff = pd.concat([results, results_old]).drop_duplicates(keep=False)
    schedule = cpl_main.get_schedule(results_diff) # from results create the schedule dataset
    team_stats = pd.read_csv(f'datasets/{year}/cpl-{year}-team_stats.csv')
    colours = team_ref['colour']
    results_brief = pd.read_csv(f'datasets/{year}/cpl-{year}-results_brief.csv')

    return results, stats, team_ref, player_info, results_old, results_diff, schedule, stats, team_stats, results_brief


def load_player_files(year):
    # get all rated player information based on position and calculate and overall score for the individual player
    rated_forwards = pd.read_csv(f'datasets/{year}/cpl-{year}-forwards.csv')
    rated_midfielders = pd.read_csv(f'datasets/{year}/cpl-{year}-midfielders.csv')
    rated_defenders = pd.read_csv(f'datasets/{year}/cpl-{year}-defenders.csv')
    rated_goalscorers = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_goalscorers.csv')
    rated_keepers = pd.read_csv(f'datasets/{year}/cpl-{year}-keepers.csv')
    rated_offenders = pd.read_csv(f'datasets/{year}/cpl-{year}-discipline.csv')
    rated_assists = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_assists.csv')
    return rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists

@canples.route('/')
def index():
    na = 'TBD'

    year = '2020'
    other_year = '2019'

    results, stats, stats_seed, team_ref, player_info, results_old, results_diff, schedule, stats, team_stats, results_brief, matches_predictions, game_form, team_rosters = load_main_files(year)
    rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists = load_player_files(year)

    standings = cpl_main.get_standings(results,1,team_ref)
    standings_old =pd.read_csv(f'datasets/{year}/cpl-{year}-standings_previous.csv')
    compare_standings = cpl_main.compare_standings(standings,standings_old,team_ref)

    top_team = 'Forge FC'#standings.iloc[0]['team']
    top_team_info = team_ref[team_ref['team'] == top_team]
    first_colour = top_team_info.iloc[0][4]
    first_crest = top_team_info.iloc[0][5]
    top_mover = 'Cavalry FC'#compare_standings.iloc[0]['team']
    top_crest = team_ref[team_ref['team'] == top_mover]
    top_crest = top_crest.iloc[0][5]
    top_dropper = 'FC Edmonton'#compare_standings.iloc[-1]['team']
    bot_crest = team_ref[team_ref['team'] == top_dropper]
    bot_crest = bot_crest.iloc[0][5]

    game_week, goals, big_win, top_result, low_result, other_result, assists, yellows, reds = cpl_main.get_weeks_results(results[results['s'] <= 1],standings,stats,team_ref)
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

    champs = f'{top_team} - {year} Champions'
    suspended = 'none'

    return render_template('cpl-es-index.html',top_mover = top_mover, top_dropper = top_dropper,
    goals = goals,  assists = assists, yellows = yellows, reds = reds,
    big_win = big_win, top_result = top_result, low_result = low_result, other_result = other_result,
    top_team = top_team, top_keeper = top_keeper,top_forward = top_forward,
    top_midfielder = top_midfielder, top_defender = top_defender,
    top_scorer = top_scorer, top_assist = top_assist, top_offender = top_offender, suspended = suspended,
    first_crest = first_crest, first_colour = first_colour, top_crest = top_crest, bot_crest = bot_crest,
    today = today, year = year, other_year = other_year, champs = champs,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/year')
def index_19():
    na = 'TBD'

    year = '2019'
    other_year = '2020'

    results, stats, team_ref, player_info, results_old, results_diff, schedule, stats, team_stats, results_brief = load_main_files_old(year)
    rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers,rated_assists = load_player_files(year)

    standings = pd.read_csv(f'datasets/{year}/cpl-{year}-standings.csv')
    standings_old = cpl_main.get_standings(results_old,1,team_ref)
    compare_standings = cpl_main.compare_standings(standings,standings_old,team_ref)

    top_team = 'Forge FC'
    top_team_info = team_ref[team_ref['team'] == top_team]
    first_colour = top_team_info.iloc[0][4]
    first_crest = top_team_info.iloc[0][5]
    top_mover = 'Cavalry FC'
    top_crest = team_ref[team_ref['team'] == top_mover].iloc[0][5]
    top_dropper = 'HFX Wanderers FC'
    bot_crest = team_ref[team_ref['team'] == top_dropper].iloc[0][5]

    game_week, goals, big_win, top_result, low_result, other_result, assists, yellows, reds = cpl_main.get_weeks_results(results[results['s'] <= 1],standings,stats,team_ref)
    assists, yellows, reds = int(assists), int(yellows), int(reds)

    top_forward = rated_forwards.loc[0]
    top_midfielder = rated_midfielders.loc[0]
    top_defender = rated_defenders.loc[0]
    top_scorer = rated_goalscorers.loc[0]
    top_scorer['overall'] = player_info[player_info['name'] == top_scorer['name']]['overall'].values[0]
    top_assist = rated_assists.loc[0]
    top_assist['overall'] = player_info[player_info['name'] == top_assist['name']]['overall'].values[0]
    top_keeper = rated_keepers.sort_values(by=['cs'],ascending=False)
    top_keeper = top_keeper.reset_index().loc[0]
    top_offender = rated_offenders.loc[0]

    champs = f'{top_team} - {year} Champions'
    suspended = 'none'

    return render_template('cpl-es-index.html',top_mover = top_mover, top_dropper = top_dropper,
    goals = goals,  assists = assists, yellows = yellows, reds = reds,
    big_win = big_win, top_result = top_result, low_result = low_result, other_result = other_result,
    top_team = top_team, top_keeper = top_keeper,top_forward = top_forward,
    top_midfielder = top_midfielder, top_defender = top_defender,
    top_scorer = top_scorer, top_assist = top_assist, top_offender = top_offender, suspended = suspended,
    first_crest = first_crest, first_colour = first_colour, top_crest = top_crest, bot_crest = bot_crest,
    today = today, year = year, other_year = other_year, champs = champs,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/standings')
def standings():
    year = '2020'
    page = '/standings-'
    championship = pd.read_csv(f'datasets/{year}/cpl-{year}-championship.csv')
    standings = pd.read_csv(f'datasets/{year}/cpl-{year}-standings_current.csv')
    team_form_results = pd.read_csv(f'datasets/{year}/cpl-{year}-team_form.csv')

    team_ref = pd.read_csv(f'datasets/teams.csv')
    championship['crest'] = championship['team'].apply(lambda x: team_ref[team_ref['team'] == x]['crest'].values[0])
    standings['crest'] = standings['team'].apply(lambda x: team_ref[team_ref['team'] == x]['crest'].values[0])
    team_form_results['crest'] = team_form_results['index'].apply(lambda x: team_ref[team_ref['team'] == x]['crest'].values[0])

    columns = standings.columns
    if championship.at[0,'gp'] == 0:
        check = 0
    else:
        check = 1
    return render_template('cpl-es-standings.html',columns = columns,
    championship_table = championship, check = check,
    standings_table = standings, form_table = team_form_results,
    year = year, page = page,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/standings-', methods=['POST'])
def standingsY():
    year = request.form['year']
    page = '/standings-'
    championship = pd.read_csv(f'datasets/{year}/cpl-{year}-championship.csv')
    if year == '2019':
        standings = pd.read_csv(f'datasets/{year}/cpl-{year}-standings.csv')
    else:
        standings = pd.read_csv(f'datasets/{year}/cpl-{year}-standings_current.csv')
    team_form_results = pd.read_csv(f'datasets/{year}/cpl-{year}-team_form.csv')

    team_ref = pd.read_csv(f'datasets/teams.csv')
    championship['crest'] = championship['team'].apply(lambda x: team_ref[team_ref['team'] == x]['crest'].values[0])
    standings['crest'] = standings['team'].apply(lambda x: team_ref[team_ref['team'] == x]['crest'].values[0])
    team_form_results['crest'] = team_form_results['index'].apply(lambda x: team_ref[team_ref['team'] == x]['crest'].values[0])

    columns = standings.columns
    if championship.at[0,'gp'] == 0:
        check = 0
    else:
        check = 1
    return render_template('cpl-es-standings.html',columns = columns,
    championship_table = championship, check = check,
    standings_table = standings, form_table = team_form_results,
    year = year, page = page,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/best11')
def eleven():
    year = '2020'
    page = '/best11-'
    best_eleven = pd.read_csv(f'datasets/{year}/cpl-{year}-best_eleven.csv')
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')
    names=[]
    for i in range(best_eleven.shape[0]):
        name = best_eleven.iloc[i]['first']+' '+best_eleven.iloc[i]['last']
        replace = player_info[player_info['display'] == name]['name'].values[0]
        names.append(replace)
    best_eleven['full_name'] = names
    attackers = best_eleven[best_eleven['position'] == 'f']
    midfield = best_eleven[best_eleven['position'] == 'm']
    midfield = midfield.sort_values(by='overall')
    print(midfield[['full_name','overall']])
    defenders = best_eleven[best_eleven['position'] == 'd']
    keeper = best_eleven[best_eleven['position'] == 'g']

    return render_template('cpl-es-best_eleven.html',
    html_table = best_eleven, year = year, page = page,
    attackers = attackers, defenders = defenders, midfield = midfield, keeper = keeper,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/best11-', methods=['POST'])
def elevenY():
    year = request.form['year']
    page = '/best11-'
    best_eleven = pd.read_csv(f'datasets/{year}/cpl-{year}-best_eleven.csv')
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')
    names=[]
    for i in range(best_eleven.shape[0]):
        name = best_eleven.iloc[i]['first']+' '+best_eleven.iloc[i]['last']
        replace = player_info[player_info['display'] == name]['name'].values[0]
        names.append(replace)
    best_eleven['full_name'] = names
    attackers = best_eleven[best_eleven['position'] == 'f']
    midfield = best_eleven[best_eleven['position'] == 'm']
    midfield = midfield.sort_values(by='overall')
    print(midfield[['full_name','overall']])
    defenders = best_eleven[best_eleven['position'] == 'd']
    keeper = best_eleven[best_eleven['position'] == 'g']

    return render_template('cpl-es-best_eleven.html',
    html_table = best_eleven, year = year, page = page,
    attackers = attackers, defenders = defenders, midfield = midfield, keeper = keeper,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/power')
def power():
    year = '2020'
    other_year = '2019'
    power = pd.read_csv(f'datasets/{year}/cpl-{year}-power_rankings.csv')
    return render_template('cpl-es-power.html',html_table = power, year = year, other_year = other_year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/versus')
def comparison1():
    year = '2020'
    other_year = '2019'
    headline = 'Finals - Forge VS Wanderers'
    results, stats, stats_seed, team_ref, player_info, results_old, results_diff, schedule, stats, team_stats, results_brief, matches_predictions, game_form, team_rosters  = load_main_files(year)
    rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists = load_player_files(year)

    results_old = pd.read_csv(f'datasets/{year}/cpl-{year}-results_old.csv')
    stats_old = pd.read_csv(f'datasets/{year}/cpl-{year}-stats_old.csv')

    # home side
    q1 = matches_predictions.iloc[0]['home']
    home_team_info = team_ref[team_ref['team'] == q1]
    home_colour = home_team_info.iloc[0][4]
    home_crest = home_team_info.iloc[0][5]

    game_info = schedule[schedule['home'] == q1]
    print(game_info)
    game = game_info.iloc[0]['game']

    # away side
    q2 = game_info.iloc[0]['away']
    away_team_info = team_ref[team_ref['team'] == q2]
    away_colour = away_team_info.iloc[0][4]
    away_crest = away_team_info.iloc[0][5]

    home_win = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['home_p'].values[0]
    draw = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['draw_p'].values[0]
    away_win = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['away_p'].values[0]
    home_form = game_form[q1]
    away_form = game_form[q2]

    home_roster = cpl_main.best_roster(q1,results,results_old,stats,stats_old,stats_seed,player_info,rated_forwards)
    away_roster = cpl_main.best_roster(q2,results,results_old,stats,stats_old,stats_seed,player_info,rated_forwards)
    home_score = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['hs'].values[0]
    away_score = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['as'].values[0]

    results_brief = cpl_main.get_results_brief(results,team_ref)
    results_brief_old = cpl_main.get_results_brief(results_old,team_ref)
    results_brief = pd.concat([results_brief,results_brief_old])
    compare = cpl_main.get_team_comparison(results_brief,q1,q2)
    q1_r = cpl_main.get_match_history(compare,q1)
    q2_r = cpl_main.get_match_history(compare,q2)


    team1, team2, team3, team4, team5, team6, team7, team8 = cpl_main.get_team_files(schedule,team_ref)
    print(team1, team2, team3, team4, team5, team6, team7, team8)

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

    home_sum = home_roster['overall'].sum()
    away_sum = away_roster['overall'].sum()
    #print(home_sum,away_sum)

    return render_template('cpl-es-comparison.html',home_table = home_roster.head(11), away_table = away_roster.head(11), home_win = home_win,
    home_history = q1_r, away_history = q2_r,
    home_team = q1, away_team = q2, away_win = away_win, draw = draw, home_form = home_form, away_form = away_form, schedule = schedule, year = year,
    home_crest = home_crest, home_colour = home_colour, away_crest = away_crest, away_colour = away_colour, headline = headline, home_score = home_score, away_score = away_score,
    team1 = team1, team2 = team2, team3 = team3, team4 = team4, team5 = team5, team6 = team6, team7 = team7, team8 = team8, other_year = other_year,
    group1 = group1, group2 = group2, group3 = group3, group4 = group4,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/versus-', methods=['POST'])
def comparison2():
    year = '2020'
    other_year = '2019'
    headline = 'Last 3 Matches - 1st Potential Match of Second Round'
    results, stats, stats_seed, team_ref, player_info, results_old, results_diff, schedule, stats, team_stats, results_brief, matches_predictions, game_form, team_rosters = load_main_files(year)
    rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists = load_player_files(year)

    results_old = pd.read_csv(f'datasets/{year}/cpl-{year}-results_old.csv')
    stats_old = pd.read_csv(f'datasets/{year}/cpl-{year}-stats_old.csv')

    # home side
    home = request.form['home']
    teams = home.split('-')

    q1 = cpl_main.get_long_name(teams[0],team_ref)
    home_team_info = team_ref[team_ref['team'] == q1]
    home_colour = home_team_info.iloc[0][4]
    home_crest = home_team_info.iloc[0][5]

    game_info = schedule[schedule['home'] == q1]
    game = game_info.iloc[0]['game']

    # away side
    q2 = cpl_main.get_long_name(teams[1],team_ref)
    away_team_info = team_ref[team_ref['team'] == q2]
    away_colour = away_team_info.iloc[0][4]
    away_crest = away_team_info.iloc[0][5]

    home_win = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['home_p'].values[0]
    draw = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['draw_p'].values[0]
    away_win = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['away_p'].values[0]
    home_form = game_form[q1]
    away_form = game_form[q2]

    home_roster = cpl_main.best_roster(q1,results,results_old,stats,stats_old,stats_seed,player_info,rated_forwards)
    away_roster = cpl_main.best_roster(q2,results,results_old,stats,stats_old,stats_seed,player_info,rated_forwards)
    home_score = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['hs'].values[0]
    away_score = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['as'].values[0]

    results_brief = cpl_main.get_results_brief(results,team_ref)
    results_brief_old = cpl_main.get_results_brief(results_old,team_ref)
    results_brief = pd.concat([results_brief,results_brief_old])
    compare = cpl_main.get_team_comparison(results_brief,q1,q2)
    q1_r = cpl_main.get_match_history(compare,q1)
    q2_r = cpl_main.get_match_history(compare,q2)

    team1, team2, team3, team4, team5, team6, team7, team8 = cpl_main.get_team_files(schedule,team_ref)

    print(home_win,away_win,draw)
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

    home_sum = home_roster['overall'].sum()
    away_sum = away_roster['overall'].sum()
    #print(home_sum,away_sum)

    return render_template('cpl-es-comparison2.html',home_table = home_roster.head(11), away_table = away_roster.head(11), home_win = home_win,
    home_history = q1_r, away_history = q2_r,
    home_team = q1, away_team = q2, away_win = away_win, draw = draw, home_form = home_form, away_form = away_form, schedule = schedule, year = year,
    home_crest = home_crest, home_colour = home_colour, away_crest = away_crest, away_colour = away_colour, headline = headline, home_score = home_score, away_score = away_score,
    team1 = team1, team2 = team2, team3 = team3, team4 = team4, team5 = team5, team6 = team6, team7 = team7, team8 = team8, other_year = other_year,
    group1 = group1, group2 = group2, group3 = group3, group4 = group4,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/teams')
def teams():
    global year
    year = '2020'
    page ='/teams-'
    roster = '/roster'
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    columns = team_ref.columns
    return render_template('cpl-es-teams.html',columns = columns,
    html_table = team_ref, year = year, page = page, roster = roster,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/teams-', methods=['POST'])
def teamsY():
    global year
    year = request.form['year']
    page ='/teams-'
    roster = '/roster'
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    columns = team_ref.columns
    return render_template('cpl-es-teams.html',columns = columns,
    html_table = team_ref, year = year, page = page, roster = roster,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/radar')
def radar():
    page = '/radar-'
    team_standings = pd.read_csv(f'datasets/{year}/cpl-{year}-standings_current.csv')
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
    return render_template('cpl-es-radar.html',columns = columns, html_table = team_ref,
    stats = team_standings,page = page, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/radar-', methods=['POST'])
def radar_19():
    year = request.form['year']
    page = '/radar-'
    if year == '2019':
        team_standings = pd.read_csv(f'datasets/{year}/cpl-{year}-standings.csv')
    else:
        team_standings = pd.read_csv(f'datasets/{year}/cpl-{year}-standings_current.csv')
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
    return render_template('cpl-es-radar.html',columns = columns, html_table = team_ref,
    stats = team_standings,page = page, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/roster', methods=['POST'])
def roster():
    stats = pd.read_csv(f'datasets/{year}/cpl-{year}-stats.csv')
    stats_seed = pd.read_csv(f'datasets/{year}/cpl-{year}-stats-seed.csv')
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')
    rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists = load_player_files(year)
    team = request.form['team']
    roster_team_info = team_ref[team_ref['team'] == team]
    roster_colour = roster_team_info.iloc[0][4]
    roster = cpl_main.get_roster_overall(team,stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info)
    roster2 = player_info[player_info['team'] == team]
    if roster.shape[0] < roster2.shape[0]:
        missing_players = roster2[~roster2['display'].isin(roster.name.unique())][['image','number','position','overall','flag','link','display']]
        a = [x.partition(' ')[0] for x in missing_players['display']]
        b = [x.partition(' ')[1:][1] for x in missing_players['display']]
        missing_players.insert(1,'first',a)
        missing_players.insert(2,'last',b)
        missing_players['name'] = missing_players['display']
        missing_players.pop('display')
        roster = pd.concat([roster,missing_players])
    names = []
    for i in range(roster.shape[0]):
        name = roster.iloc[i]['name']
        replace = player_info[player_info['display'] == name]['name'].values[0]
        names.append(replace)
    roster['name'] = names


    crest = roster_team_info.iloc[0][5]
    coach = roster_team_info[['coach','country','image','w','l','d']]
    if roster.empty:
        roster = player_info[player_info['team'] == team]
        roster = roster[['image','name','number','position','overall','flag']]
        first = [stats_seed[stats_seed['name'] == name]['first'].values[0] for name in roster['name']]
        last = [stats_seed[stats_seed['name'] == name]['last'].values[0] for name in roster['name']]
        roster.pop('name')
        roster.insert(1,'first',first)
        roster.insert(2,'last',last)
    print(roster.columns)

    return render_template('cpl-es-roster.html',team_name = team, coach = coach,
    html_table = roster, team_colour = roster_colour, year = year, crest = crest,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/player', methods=['POST'])
def player():
    stats = pd.read_csv(f'datasets/{year}/cpl-{year}-stats.csv')
    stats_seed = pd.read_csv(f'datasets/{year}/cpl-{year}-stats-seed.csv')
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')

    name = request.form['name']

    player = cpl_main.get_player_card(name,stats,stats_seed,player_info)
    team = player['team'].values[0]
    roster_team_info = team_ref[team_ref['team'] == team]
    roster_colour = roster_team_info.iloc[0][4]
    crest = roster_team_info.iloc[0][5]

    try:
        player_name = player_info[player_info['display'] == player['name'].values[0]]['display'].values[0]
    except:
        player_name = player_info[player_info['name'] == player['name'].values[0]]['display'].values[0]
    try:
        nationality = player_info[player_info['display'] == player['name'].values[0]]['nationality'].values[0]
    except:
        nationality = player_info[player_info['name'] == player['name'].values[0]]['nationality'].values[0]
    try:
        if year != '2019':
            graph_image = player_info[player_info['display'] == player_name]['graph'].values[0]
        else:
            graph_image = ''
    except:
        if year != '2019':
            graph_image = player_info[player_info['name'] == player_name]['graph'].values[0]
        else:
            graph_image = ''
    try:
        if year != '2019':
            radar_image = player_info[player_info['display'] == player_name]['radar'].values[0]
        else:
            radar_image = ''
    except:
        if year != '2019':
            radar_image = player_info[player_info['name'] == player_name]['radar'].values[0]
        else:
            radar_image = ''


    return render_template('cpl-es-player.html', name = player_name, graph = graph_image, radar = radar_image, nationality = nationality,
    team_name = team, html_table = player, team_colour = roster_colour, year = year, crest = crest,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/goals')
def goals():
    year = '2020'
    page = '/goals-'
    rated_goalscorers = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_goalscorers.csv')
    rated_g10 = rated_goalscorers.head(10)
    rated_assists = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_assists.csv')
    rated_a10 = rated_assists.head(10)
    columns_g = rated_goalscorers.columns
    columns_a = rated_assists.columns
    return render_template('cpl-es-goals.html',columns_g = columns_g, columns_a = columns_a,
    html_table = rated_g10, assists_table = rated_a10, year = year, page = page,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/goals-', methods=['POST'])
def goalsY():
    year = request.form['year']
    page = '/goals-'
    rated_goalscorers = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_goalscorers.csv')
    rated_g10 = rated_goalscorers.head(10)
    rated_assists = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_assists.csv')
    rated_a10 = rated_assists.head(10)
    columns_g = rated_goalscorers.columns
    columns_a = rated_assists.columns
    return render_template('cpl-es-goals.html',columns_g = columns_g, columns_a = columns_a,
    html_table = rated_g10, assists_table = rated_a10, year = year, page = page,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/forwards')
def forwards():
    year = '2020'
    position = 'Forwards'
    page = '/forwards-'
    rated_forwards = pd.read_csv(f'datasets/{year}/cpl-{year}-forwards.csv')
    columns = rated_forwards.columns
    return render_template('cpl-es-forwards.html',
    position = position, page = page,
    columns = columns,html_table = rated_forwards, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/forwards-', methods=['POST'])
def forwards_19():
    year = request.form['year']
    position = 'Forwards'
    page = '/forwards-'
    rated_forwards = pd.read_csv(f'datasets/{year}/cpl-{year}-forwards.csv')
    columns = rated_forwards.columns
    return render_template('cpl-es-forwards.html',
    position = position, page = page,
    columns = columns,html_table = rated_forwards, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/forwardsP90')
def forwards_90():
    year = '2020'
    position = 'Forwards per 90'
    page = '/forwardsP90-'
    rated_forwards = pd.read_csv(f'datasets/{year}/cpl-{year}-forwards-p90.csv')
    columns = rated_forwards.columns
    return render_template('cpl-es-forwards.html',
    position = position, page = page,
    columns = columns,html_table = rated_forwards, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/forwardsP90-', methods=['POST'])
def forwards_19_90():
    year = request.form['year']
    position = 'Forwards per 90'
    page = '/forwardsP90-'
    rated_forwards = pd.read_csv(f'datasets/{year}/cpl-{year}-forwards-p90.csv')
    columns = rated_forwards.columns
    return render_template('cpl-es-forwards.html',
    position = position, page = page,
    columns = columns,html_table = rated_forwards, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/midfielders')
def midfielders():
    year = '2020'
    position = 'Midfielders'
    page = '/midfielders-'
    rated_midfielders = pd.read_csv(f'datasets/{year}/cpl-{year}-midfielders.csv')
    columns = rated_midfielders.columns
    return render_template('cpl-es-midfielders.html',
    position = position, page = page,
    columns = columns,html_table = rated_midfielders, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/midfielders-', methods=['POST'])
def midfielders_19():
    year = request.form['year']
    position = 'Midfielders'
    page = '/midfielders-'
    rated_midfielders = pd.read_csv(f'datasets/{year}/cpl-{year}-midfielders.csv')
    columns = rated_midfielders.columns
    return render_template('cpl-es-midfielders.html',
    position = position, page = page,
    columns = columns,html_table = rated_midfielders, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/midfieldersP90')
def midfielders_90():
    year = '2020'
    position = 'Midfielders per 90'
    page = '/midfieldersP90-'
    rated_midfielders = pd.read_csv(f'datasets/{year}/cpl-{year}-midfielders-p90.csv')
    columns = rated_midfielders.columns
    return render_template('cpl-es-midfielders.html',
    position = position, page = page,
    columns = columns,html_table = rated_midfielders, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/midfieldersP90-', methods=['POST'])
def midfielders_19_90():
    year = request.form['year']
    position = 'Midfielders per 90'
    page = '/midfieldersP90-'
    rated_midfielders = pd.read_csv(f'datasets/{year}/cpl-{year}-midfielders-p90.csv')
    columns = rated_midfielders.columns
    return render_template('cpl-es-midfielders.html',
    position = position, page = page,
    columns = columns,html_table = rated_midfielders, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/defenders')
def defenders():
    year = '2020'
    position = 'Defenders'
    page = '/defenders-'
    rated_defenders = pd.read_csv(f'datasets/{year}/cpl-{year}-defenders.csv')
    columns = rated_defenders.columns
    return render_template('cpl-es-defenders.html',
    position = position, page = page,
    columns = columns,html_table = rated_defenders, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/defenders-', methods=['POST'])
def defenders_19():
    year = request.form['year']
    position = 'Defenders'
    page = '/defenders-'
    rated_defenders = pd.read_csv(f'datasets/{year}/cpl-{year}-defenders.csv')
    columns = rated_defenders.columns
    return render_template('cpl-es-defenders.html',
    position = position, page = page,
    columns = columns,html_table = rated_defenders, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/defendersP90')
def defenders_90():
    year = '2020'
    page = '/defendersP90-'
    rated_defenders = pd.read_csv(f'datasets/{year}/cpl-{year}-defenders-p90.csv')
    columns = rated_defenders.columns
    return render_template('cpl-es-defenders.html',
    position = position, page = page,
    columns = columns,html_table = rated_defenders, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/defendersP90-', methods=['POST'])
def defenders_19_90():
    year = request.form['year']
    position = 'Defenders per 90'
    page = '/defendersP90-'
    rated_defenders = pd.read_csv(f'datasets/{year}/cpl-{year}-defenders-p90.csv')
    columns = rated_defenders.columns
    return render_template('cpl-es-defenders.html',
    position = position, page = page,
    columns = columns,html_table = rated_defenders, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/keepers')
def keepers():
    year = '2020'
    position = 'Goal Keepers'
    page = '/keepers-'
    rated_keepers = pd.read_csv(f'datasets/{year}/cpl-{year}-keepers.csv')
    columns = rated_keepers.columns
    return render_template('cpl-es-keepers.html',
    position = position, page = page,
    columns = columns,html_table = rated_keepers, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/keepers-', methods=['POST'])
def keepers_19():
    year = request.form['year']
    position = 'Goal Keepers'
    page = '/keepers-'
    rated_keepers = pd.read_csv(f'datasets/{year}/cpl-{year}-keepers.csv')
    columns = rated_keepers.columns
    return render_template('cpl-es-keepers.html',
    position = position, page = page,
    columns = columns,html_table = rated_keepers, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/keepersP90')
def keepers_90():
    year = '2020'
    position = 'Goal Keepers per 90'
    page = '/keepersP90-'
    rated_keepers = pd.read_csv(f'datasets/{year}/cpl-{year}-keepers-p90.csv')
    columns = rated_keepers.columns
    return render_template('cpl-es-keepers.html',
    position = position, page = page,
    columns = columns,html_table = rated_keepers, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/keepersP90-', methods=['POST'])
def keepers_19_90():
    year = request.form['year']
    position = 'Goal Keepers per 90'
    page = '/keepersP90-'
    rated_keepers = pd.read_csv(f'datasets/{year}/cpl-{year}-keepers-p90.csv')
    columns = rated_keepers.columns
    return render_template('cpl-es-keepers.html',
    position = position, page = page,
    columns = columns,html_table = rated_keepers, year = year,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/discipline')
def discipline():
    year = '2020'
    page = '/discipline-'
    rated_offenders = pd.read_csv(f'datasets/{year}/cpl-{year}-discipline.csv')
    columns = rated_offenders.columns
    return render_template('cpl-es-discipline.html',columns = columns,
    html_table = rated_offenders, year = year, page = page,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/discipline-', methods=['POST'])
def discipline_19():
    year = request.form['year']
    page = '/discipline-'
    rated_offenders = pd.read_csv(f'datasets/{year}/cpl-{year}-discipline.csv')
    columns = rated_offenders.columns
    return render_template('cpl-es-discipline.html',columns = columns,
    html_table = rated_offenders, year = year, page = page,
    day = day, weekday = weekday, month = month, theme = theme)

@canples.route('/hell')
def hello():
    return 'Welcome to HELL world!'

if __name__ == "__main__":
    canples.run()
