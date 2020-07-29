from flask import Flask
from flask import Blueprint, flash, g, redirect, render_template, request, url_for
import cpl_main
from datetime import date
today = date.today()
current_year = date.today().strftime('%Y')

import numpy as np
import pandas as pd

import pickle
classifier = 'models/cpl_roster_classifier.sav'
cpl_classifier_model = pickle.load(open(classifier, 'rb'))
regressor = 'models/cpl_score_regressor.sav'
cpl_score_model = pickle.load(open(regressor, 'rb'))

canples = Flask(__name__)

def get_string(data):
    data = str(data*100)
    data = data[0:4]
    return data

def load_main_files(year):
    results = pd.read_csv(f'datasets/{year}/cpl-{year}-results.csv')
    stats = pd.read_csv(f'datasets/{year}/cpl-{year}-stats.csv')
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')
    current_teams = team_ref['team']
    results_old = results[results['hr'] != 'E'].copy()
    results_diff = pd.concat([results, results_old]).drop_duplicates(keep=False)
    schedule = cpl_main.get_schedule(results_diff) # from results create the schedule dataset
    team_stats = pd.read_csv(f'datasets/{year}/cpl-{year}-team_stats.csv')
    colours = team_ref['colour']
    results_brief = pd.read_csv(f'datasets/{year}/cpl-{year}-results_brief.csv')
    matches_predictions = pd.read_csv(f'datasets/{year}/cpl-{year}-match_predictions.csv')
    game_form = pd.read_csv(f'datasets/{year}/cpl-{year}-game_form.csv')
    team_rosters = pd.read_csv(f'datasets/{year}/cpl-{year}-team_rosters.csv')

    return results, stats, team_ref, player_info, results_old, results_diff, schedule, stats, team_stats, results_brief, matches_predictions, game_form, team_rosters

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


@canples.route('/index')
def index():
    na = 'TBD'

    year = '2020'
    other_year = '2019'

    results, stats, team_ref, player_info, results_old, results_diff, schedule, stats, team_stats, results_brief, matches_predictions, game_form, team_rosters = load_main_files(year)
    rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists = load_player_files(year)

    standings = pd.read_csv(f'datasets/{year}/cpl-{year}-standings.csv')
    standings_old = cpl_main.get_standings(results_old,1,team_ref)
    compare_standings = cpl_main.compare_standings(standings,standings_old,team_ref)

    top_team = standings.iloc[0]['team']
    top_team_info = team_ref[team_ref['team'] == top_team]
    first_colour = top_team_info.iloc[0][4]
    first_crest = top_team_info.iloc[0][5]
    top_mover = compare_standings.iloc[0]['team']
    top_crest = team_ref[team_ref['team'] == top_mover]
    top_crest = top_crest.iloc[0][5]
    top_dropper = compare_standings.iloc[-1]['team']
    bot_crest = team_ref[team_ref['team'] == top_dropper]
    bot_crest = bot_crest.iloc[0][5]

    game_week, goals, big_win, top_result, low_result, other_result = cpl_main.get_weeks_results(results[results['s'] <= 1],standings,team_ref)

    top_forward = rated_forwards.loc[0]
    top_midfielder = rated_midfielders.loc[0]
    top_defender = rated_defenders.loc[0]
    top_scorer = rated_goalscorers.loc[0]
    top_assist = rated_assists.loc[0]
    top_keeper = rated_keepers.loc[0]
    top_offender = rated_offenders.loc[0]

    if results.iloc[0]['hr'] == 'E':
        top_team, top_mover, top_dropper, first_crest, top_crest, bot_crest, first_colour = na, na, na, 'CPL-Crest-White.png', 'oneSoccer_nav.png', 'canNat_icon.png', 'w3-indigo'

    champs = year + ' Champions TBD'

    return render_template('cpl-es-index.html',top_mover = top_mover, top_dropper = top_dropper,
    goals = goals, big_win = big_win, top_result = top_result, low_result = low_result, other_result = other_result,
    top_team = top_team, top_keeper = top_keeper,top_forward = top_forward,
    top_midfielder = top_midfielder, top_defender = top_defender,
    top_scorer = top_scorer, top_assist = top_assist, top_offender = top_offender,
    first_crest = first_crest, first_colour = first_colour, top_crest = top_crest, bot_crest = bot_crest,
    today = today, year = year, other_year = other_year, champs = champs)

@canples.route('/dashboard')
def index_19():
    na = 'NA'

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
    top_mover = compare_standings.iloc[0]['team']
    top_crest = team_ref[team_ref['team'] == top_mover].iloc[0][5]
    top_dropper = compare_standings.iloc[-1]['team']
    bot_crest = team_ref[team_ref['team'] == top_dropper].iloc[0][5]

    game_week, goals, big_win, top_result, low_result, other_result = cpl_main.get_weeks_results(results[results['s'] <= 1],standings,team_ref)

    top_forward = rated_forwards.loc[0]
    top_midfielder = rated_midfielders.loc[0]
    top_defender = rated_defenders.loc[0]
    top_scorer = rated_goalscorers.loc[0]
    top_assist = rated_assists.loc[0]
    top_keeper = rated_keepers.sort_values(by=['cs'],ascending=False)
    top_keeper = top_keeper.reset_index().loc[0]
    top_offender = rated_offenders.loc[0]

    champs = 'Forge FC 2019 Champions'

    return render_template('/2019/cpl-es-index.html',top_mover = top_mover, top_dropper = top_dropper,
    goals = goals, big_win = big_win, top_result = top_result, low_result = low_result, other_result = other_result,
    top_team = top_team, top_keeper = top_keeper,top_forward = top_forward,
    top_midfielder = top_midfielder, top_defender = top_defender,
    top_scorer = top_scorer, top_assist = top_assist, top_offender = top_offender,
    first_crest = first_crest, first_colour = first_colour, top_crest = top_crest, bot_crest = bot_crest,
    today = today, year = year, other_year = other_year, champs = champs)

@canples.route('/standings')
def standings():
    year = '2020'
    other_year = '2019'
    standings = pd.read_csv(f'datasets/{year}/cpl-{year}-standings.csv')
    columns = standings.columns
    return render_template('cpl-es-standings.html',columns = columns, standings_table = standings, year = year, other_year = other_year)

@canples.route('/standings-2019')
def standings_19():
    year = '2019'
    other_year = '2020'
    standings = pd.read_csv(f'datasets/{year}/cpl-{year}-standings.csv')
    columns = standings.columns
    return render_template('2019/cpl-es-standings.html',columns = columns, standings_table = standings, year = year, other_year = other_year)

@canples.route('/best11')
def eleven():
    year = '2020'
    other_year = '2019'
    best_eleven = pd.read_csv(f'datasets/{year}/cpl-{year}-best_eleven.csv')
    return render_template('cpl-es-best_eleven.html',html_table = best_eleven, year = year, other_year = other_year)

@canples.route('/best11-2019')
def eleven_19():
    year = '2019'
    other_year = '2020'
    best_eleven = pd.read_csv(f'datasets/{year}/cpl-{year}-best_eleven.csv')
    return render_template('2019/cpl-es-best_eleven.html',html_table = best_eleven, year = year, other_year = other_year)

@canples.route('/power-rankings')
def power():
    year = '2020'
    other_year = '2019'
    power = pd.read_csv(f'datasets/{year}/cpl-{year}-power_rankings.csv')
    return render_template('cpl-es-power.html',html_table = power, year = year, other_year = other_year)

@canples.route('/versus')
def comparison1():
    year = '2020'
    other_year = '2019'
    headline = 'This Weeks Matches'
    results, stats, team_ref, player_info, results_old, results_diff, schedule, stats, team_stats, results_brief, matches_predictions, game_form, team_rosters  = load_main_files(year)
    rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists = load_player_files(year)

    # home side
    q1 = matches_predictions.iloc[0]['home']
    home_team_info = team_ref[team_ref['team'] == q1]
    home_colour = home_team_info.iloc[0][4]
    home_crest = home_team_info.iloc[0][5]

    game_info = schedule[schedule['home'] == q1]
    game = game_info.iloc[0]['game']

    # away side
    q2 = game_info.iloc[0]['away']
    away_team_info = team_ref[team_ref['team'] == q2]
    away_colour = away_team_info.iloc[0][4]
    away_crest = away_team_info.iloc[0][5]

    home_win = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['home_p'].values[0]
    draw = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['draw_p'].values[0]
    away_win = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['away_p'].values[0]
    home_form = pd.DataFrame(game_form[q1])
    away_form = pd.DataFrame(game_form[q2])
    home_roster = team_rosters[team_rosters['team'] == q1][['name','number','position','overall']][0:11]
    away_roster = team_rosters[team_rosters['team'] == q2][['name','number','position','overall']][0:11]
    home_score = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['hs'].values[0]
    away_score = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['as'].values[0]


    team1, team2, team3, team4, team5, team6, team7, team8 = cpl_main.get_team_files(schedule,team_ref)

    if (home_win and away_win) < draw:
        home_win, away_win = draw, draw

    home_win = get_string(home_win)
    away_win = get_string(away_win)
    draw = get_string(draw)

    group1 = team1 + '-' + team2
    group2 = team3 + '-' + team4
    group3 = team5 + '-' + team6
    group4 = team7 + '-' + team8

    return render_template('cpl-es-comparison.html',home_table = home_roster, away_table = away_roster, home_win = home_win,
    home_team = q1, away_team = q2, away_win = away_win, draw = draw, home_form = home_form, away_form = away_form, schedule = schedule, year = year,
    home_crest = home_crest, home_colour = home_colour, away_crest = away_crest, away_colour = away_colour, headline = headline, home_score = home_score, away_score = away_score,
    team1 = team1, team2 = team2, team3 = team3, team4 = team4, team5 = team5, team6 = team6, team7 = team7, team8 = team8, other_year = other_year,
    group1 = group1, group2 = group2, group3 = group3, group4 = group4)

@canples.route('/versus-', methods=['POST'])
def comparison2():
    year = '2020'
    other_year = '2019'
    headline = 'This Weeks Matches'
    results, stats, team_ref, player_info, results_old, results_diff, schedule, stats, team_stats, results_brief, matches_predictions, game_form, team_rosters = load_main_files(year)
    rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists = load_player_files(year)

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
    home_form = pd.DataFrame(game_form[q1])
    away_form = pd.DataFrame(game_form[q2])
    home_roster = team_rosters[team_rosters['team'] == q1][['name','number','position','overall']][0:11]
    away_roster = team_rosters[team_rosters['team'] == q2][['name','number','position','overall']][0:11]
    home_score = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['hs'].values[0]
    away_score = matches_predictions[(matches_predictions['home'] == q1) & (matches_predictions['away'] == q2)]['as'].values[0]

    team1, team2, team3, team4, team5, team6, team7, team8 = cpl_main.get_team_files(schedule,team_ref)

    if (home_win and away_win) < draw:
        home_win, away_win = draw, draw

    home_win = get_string(home_win)
    away_win = get_string(away_win)
    draw = get_string(draw)

    group1 = team1 + '-' + team2
    group2 = team3 + '-' + team4
    group3 = team5 + '-' + team6
    group4 = team7 + '-' + team8

    return render_template('cpl-es-comparison2.html',home_table = home_roster, away_table = away_roster, home_win = home_win,
    home_team = q1, away_team = q2, away_win = away_win, draw = draw, home_form = home_form, away_form = away_form, schedule = schedule, year = year,
    home_crest = home_crest, home_colour = home_colour, away_crest = away_crest, away_colour = away_colour, headline = headline, home_score = home_score, away_score = away_score,
    team1 = team1, team2 = team2, team3 = team3, team4 = team4, team5 = team5, team6 = team6, team7 = team7, team8 = team8, other_year = other_year,
    group1 = group1, group2 = group2, group3 = group3, group4 = group4)

@canples.route('/teams')
def teams():
    year = '2020'
    other_year = '2019'
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    columns = team_ref.columns
    return render_template('cpl-es-teams.html',columns = columns, html_table = team_ref, year = year, other_year = other_year)

@canples.route('/teams-2019')
def teams_19():
    year = '2019'
    other_year = '2020'
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    columns = team_ref.columns
    return render_template('2019/cpl-es-teams.html',columns = columns, html_table = team_ref, year = year, other_year = other_year)

@canples.route('/radar')
def radar():
    year = '2020'
    other_year = '2019'
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    columns = team_ref.columns
    return render_template('cpl-es-radar.html',columns = columns, html_table = team_ref, year = year, other_year = other_year)

@canples.route('/radar-2019')
def radar_19():
    year = '2019'
    other_year = '2020'
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    columns = team_ref.columns
    return render_template('2019/cpl-es-radar.html',columns = columns, html_table = team_ref, year = year, other_year = other_year)

@canples.route('/roster', methods=['POST'])
def roster():
    year = '2020'
    other_year = '2019'
    stats = pd.read_csv(f'datasets/{year}/cpl-{year}-stats.csv')
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')
    rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists = load_player_files(year)
    team = request.form['team']
    roster_team_info = team_ref[team_ref['team'] == team]
    roster_colour = roster_team_info.iloc[0][4]
    roster = cpl_main.get_roster_overall(team,stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info)
    crest = roster_team_info.iloc[0][5]
    coach = roster_team_info[['coach','country','image','w','l','d']]
    return render_template('cpl-es-roster.html',team_name = team, coach = coach, html_table = roster, team_colour = roster_colour, year = year, crest = crest, other_year = other_year)

@canples.route('/roster-2019', methods=['POST'])
def roster_19():
    year = '2019'
    other_year = '2020'
    stats = pd.read_csv(f'datasets/{year}/cpl-{year}-stats.csv')
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')
    rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists = load_player_files(year)
    team = request.form['team']
    roster_team_info = team_ref[team_ref['team'] == team]
    roster_colour = roster_team_info.iloc[0][4]
    roster = cpl_main.get_roster_overall(team,stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info)
    crest = roster_team_info.iloc[0][5]
    coach = roster_team_info[['coach','country','image','w','l','d']]
    return render_template('2019/cpl-es-roster.html',team_name = team, coach = coach, html_table = roster, team_colour = roster_colour, year = year, crest = crest, other_year = other_year)

@canples.route('/player', methods=['POST'])
def player():
    year = '2020'
    other_year = '2019'
    stats = pd.read_csv(f'datasets/{year}/cpl-{year}-stats.csv')
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')

    name = request.form['name']
    print('VIEWING \n',name,'\n')

    player = cpl_main.get_player_card(name,stats,player_info)
    print('VIEWING \n',player,'\n')
    team = player['team'].values[0]
    roster_team_info = team_ref[team_ref['team'] == team]
    roster_colour = roster_team_info.iloc[0][4]
    crest = roster_team_info.iloc[0][5]
    return render_template('cpl-es-player.html', name = player['name'].values[0], team_name = team, html_table = player, team_colour = roster_colour, year = year, crest = crest, other_year = other_year)

@canples.route('/player-2019', methods=['POST'])
def player_19():
    year = '2019'
    other_year = '2020'
    stats = pd.read_csv(f'datasets/{year}/cpl-{year}-stats.csv')
    team_ref = pd.read_csv('datasets/teams.csv')
    team_ref = team_ref[team_ref['year'] == int(year)]
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')

    name = request.form['name']

    player = cpl_main.get_player_card(name,stats,player_info)
    team = player['team'].values[0]
    roster_team_info = team_ref[team_ref['team'] == team]
    roster_colour = roster_team_info.iloc[0][4]
    crest = roster_team_info.iloc[0][5]
    return render_template('2019/cpl-es-player.html', name = player['name'].values[0],team_name = team, html_table = player, team_colour = roster_colour, year = year, crest = crest, other_year = other_year)

@canples.route('/goals')
def goals():
    year = '2020'
    other_year = '2019'
    rated_goalscorers = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_goalscorers.csv')
    rated_g10 = rated_goalscorers.head(10)
    rated_assists = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_assists.csv')
    rated_a10 = rated_assists.head(10)
    columns_g = rated_goalscorers.columns
    columns_a = rated_assists.columns
    return render_template('cpl-es-goals.html',columns_g = columns_g, columns_a = columns_a,html_table = rated_g10, assists_table = rated_a10, year = year, other_year = other_year)

@canples.route('/goals-2019')
def goals_19():
    year = '2019'
    other_year = '2020'
    rated_goalscorers = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_goalscorers.csv')
    rated_g10 = rated_goalscorers.head(10)
    rated_assists = pd.read_csv(f'datasets/{year}/cpl-{year}-rated_assists.csv')
    rated_a10 = rated_assists.head(10)
    columns_g = rated_goalscorers.columns
    columns_a = rated_assists.columns
    return render_template('2019/cpl-es-goals.html',columns_g = columns_g, columns_a = columns_a,html_table = rated_g10, assists_table = rated_a10, year = year, other_year = other_year)

@canples.route('/forwards')
def forwards():
    year = '2020'
    other_year = '2019'
    rated_forwards = pd.read_csv(f'datasets/{year}/cpl-{year}-forwards.csv')
    columns = rated_forwards.columns
    return render_template('cpl-es-forwards.html',columns = columns,html_table = rated_forwards, year = year, other_year = other_year)

@canples.route('/forwards-2019')
def forwards_19():
    year = '2019'
    other_year = '2020'
    rated_forwards = pd.read_csv(f'datasets/{year}/cpl-{year}-forwards.csv')
    columns = rated_forwards.columns
    return render_template('2019/cpl-es-forwards.html',columns = columns,html_table = rated_forwards, year = year, other_year = other_year)

@canples.route('/midfielders')
def midfielders():
    year = '2020'
    other_year = '2019'
    rated_midfielders = pd.read_csv(f'datasets/{year}/cpl-{year}-midfielders.csv')
    columns = rated_midfielders.columns
    return render_template('cpl-es-midfielders.html',columns = columns,html_table = rated_midfielders, year = year, other_year = other_year)

@canples.route('/midfielders-2019')
def midfielders_19():
    year = '2019'
    other_year = '2020'
    rated_midfielders = pd.read_csv(f'datasets/{year}/cpl-{year}-midfielders.csv')
    columns = rated_midfielders.columns
    return render_template('2019/cpl-es-midfielders.html',columns = columns,html_table = rated_midfielders, year = year, other_year = other_year)

@canples.route('/defenders')
def defenders():
    year = '2020'
    other_year = '2019'
    rated_defenders = pd.read_csv(f'datasets/{year}/cpl-{year}-defenders.csv')
    columns = rated_defenders.columns
    return render_template('cpl-es-defenders.html',columns = columns,html_table = rated_defenders, year = year, other_year = other_year)

@canples.route('/defenders-2019')
def defenders_19():
    year = '2019'
    other_year = '2020'
    rated_defenders = pd.read_csv(f'datasets/{year}/cpl-{year}-defenders.csv')
    columns = rated_defenders.columns
    return render_template('2019/cpl-es-defenders.html',columns = columns,html_table = rated_defenders, year = year, other_year = other_year)

@canples.route('/keepers')
def keepers():
    year = '2020'
    other_year = '2019'
    rated_keepers = pd.read_csv(f'datasets/{year}/cpl-{year}-keepers.csv')
    columns = rated_keepers.columns
    return render_template('cpl-es-keepers.html',columns = columns,html_table = rated_keepers, year = year, other_year = other_year)

@canples.route('/keepers-2019')
def keepers_19():
    year = '2019'
    other_year = '2020'
    rated_keepers = pd.read_csv(f'datasets/{year}/cpl-{year}-keepers.csv')
    columns = rated_keepers.columns
    return render_template('2019/cpl-es-keepers.html',columns = columns,html_table = rated_keepers, year = year, other_year = other_year)

@canples.route('/discipline')
def discipline():
    year = '2020'
    other_year = '2019'
    rated_offenders = pd.read_csv(f'datasets/{year}/cpl-{year}-discipline.csv')
    columns = rated_offenders.columns
    return render_template('cpl-es-discipline.html',columns = columns,html_table = rated_offenders, year = year, other_year = other_year)

@canples.route('/discipline-2019')
def discipline_19():
    year = '2019'
    other_year = '2020'
    rated_offenders = pd.read_csv(f'datasets/{year}/cpl-{year}-discipline.csv')
    columns = rated_offenders.columns
    return render_template('2019/cpl-es-discipline.html',columns = columns,html_table = rated_offenders, year = year, other_year = other_year)

@canples.route('/hell')
def hello():
    return 'Welcome to HELL world!'

if __name__ == "__main__":
    canples.run()
