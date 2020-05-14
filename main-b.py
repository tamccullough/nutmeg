from flask import Flask
from flask import Blueprint, flash, g, redirect, render_template, request, url_for
import cpl_main
from datetime import date
today = date.today()
current_year = date.today().strftime('%Y')

import numpy as np
import pandas as pd

import pickle
filename = 'models/cpl_roster_classifier.sav'
cpl_classifier_model = pickle.load(open(filename, 'rb'))

canples = Flask(__name__)

year = '2019'
def load_main_files(year):
    results = pd.read_csv(f'datasets/{year}/cpl-{year}-results.csv')
    stats = pd.read_csv(f'datasets/{year}/cpl-{year}-stats.csv')
    team_ref = pd.read_csv('datasets/teams.csv')
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')
    current_teams = team_ref['team']
    if year == '2019':
        team_ref = team_ref[1:]
        results_old = results[:-7].copy()
    else:
        results_old = results[results['hr'] != 'E'].copy()
    results_diff = pd.concat([results, results_old]).drop_duplicates(keep=False)
    schedule = cpl_main.get_schedule(results_diff) # from results create the schedule dataset
    team_stats = pd.read_csv(f'datasets/{year}/cpl-{year}-team_stats.csv')
    colours = team_ref['colour']
    results_brief = pd.read_csv(f'datasets/{year}/cpl-{year}-results_brief.csv')

    return results, stats, team_ref, player_info, results_old, results_diff, schedule, stats, team_stats, results_brief

results, stats, team_ref, player_info, results_old, results_diff, schedule, stats, team_stats, results_brief = load_main_files(year)

def load_player_files(year):
    # get all rated player information based on position and calculate and overall score for the individual player
    rated_forwards = pd.read_csv(f'datasets/{year}/cpl-{year}-forwards.csv')
    rated_midfielders = pd.read_csv(f'datasets/{year}/cpl-{year}-midfielders.csv')
    rated_defenders = pd.read_csv(f'datasets/{year}/cpl-{year}-defenders.csv')
    rated_goalscorers = cpl_main.top_tracked(team_stats,'goals')
    rated_keepers = pd.read_csv(f'datasets/{year}/cpl-{year}-keepers.csv')
    rated_offenders = pd.read_csv(f'datasets/{year}/cpl-{year}-discipline.csv')
    rated_assists = cpl_main.top_tracked(team_stats,'assists')
    return rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists

rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers, rated_assists = load_player_files(year)

def get_files(stats,game,q1,q2):
    game_h = cpl_main.get_home_away_comparison(stats,game,q1)
    game_a = cpl_main.get_home_away_comparison(stats,game,q2)
    compare = cpl_main.get_team_comparison(results_brief,q1,q2)
    t1_x, t1_y = cpl_main.get_NB_data(compare,q1)
    t2_x, t2_y = cpl_main.get_NB_data(compare,q2)
    home_win, draw, away_win = cpl_main.get_match_prediction(q1,q2,t1_x,t1_y,t2_x,t2_y)
    home_form = cpl_main.get_five_game_form(results,q1)
    away_form = cpl_main.get_five_game_form(results,q2)
    home_roster = cpl_main.get_compare_roster(results,q1,team_stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers)
    away_roster = cpl_main.get_compare_roster(results,q2,team_stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers)
    q1_roster = cpl_main.get_overall_roster(home_roster)
    q2_roster = cpl_main.get_overall_roster(away_roster)
    home_win, away_win, draw = cpl_main.get_final_game_prediction(cpl_classifier_model,q1_roster,q2_roster,home_win,away_win,draw)
    return home_win, draw, away_win, home_form, away_form, home_roster, away_roster

@canples.route('/index')
def index():
    na = 'NA'

    if year == '2020':
        other_year = '2019'
    else:
        other_year = '2020'

    results, stats, team_ref, player_info, results_old, results_diff, schedule, stats, team_stats, results_brief = load_main_files(year)
    rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers,rated_assists = load_player_files(year)

    standings = pd.read_csv(f'datasets/{year}/cpl-{year}-standings.csv')
    standings_old = cpl_main.get_standings(results_old,1,team_ref)
    compare_standings = cpl_main.compare_standings(standings,standings_old,team_ref)

    if year == '2019':
        top_team = 'Forge FC'
    else:
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

    game_week, goals, big_win, top_result, low_result = cpl_main.get_weeks_results(results[results['s'] <= 1],standings,team_ref)

    top_forward = rated_forwards.loc[0]
    top_midfielder = rated_midfielders.loc[0]
    top_defender = rated_defenders.loc[0]
    top_scorer = rated_goalscorers.loc[0]
    top_assist = rated_assists.loc[0]
    top_keeper = rated_keepers.loc[0]
    top_offender = rated_offenders.loc[0]

    if results.iloc[0]['hr'] == 'E':
        top_team, top_mover, top_dropper = na, na, na
    if year == '2019':
        champs = 'Forge FC 2019 Champions'
    else:
        champs = 'Undetermined for ' + year

    return render_template('cpl-es-index.html',top_mover = top_mover, top_dropper = top_dropper,
    goals = goals, big_win = big_win, top_result = top_result, low_result = low_result,
    top_team = top_team, top_keeper = top_keeper,top_forward = top_forward,
    top_midfielder = top_midfielder, top_defender = top_defender,
    top_scorer = top_scorer, top_assist = top_assist, top_offender = top_offender,
    first_crest = first_crest, first_colour = first_colour, top_crest = top_crest, bot_crest = bot_crest,
    today = today, year = year, other_year = other_year, champs = champs)

@canples.route('/index2', methods=['POST'])
def index2():
    '''global year
    global results, schedule, results_old
    global stats, team_ref, current_teams, team_stats, results_brief, colours
    global rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers'''
    na = 'NA'

    year_switch = request.form['year_switch']
    if year_switch != year:
        year = year_switch

    results, stats, team_ref, player_info, results_old, results_diff, schedule, stats, team_stats, results_brief = load_main_files(year)
    rated_forwards, rated_midfielders, rated_defenders, rated_keepers, rated_offenders, rated_goalscorers,rated_assists = load_player_files(year)

    if year == '2020':
        other_year = '2019'
    else:
        other_year = '2020'

    standings = pd.read_csv(f'datasets/{year}/cpl-{year}-standings.csv')
    standings_old = cpl_main.get_standings(results_old,1,team_ref)
    compare_standings = cpl_main.compare_standings(standings,standings_old,team_ref)

    top_mover = compare_standings.iloc[0]['team']
    top_dropper = compare_standings.iloc[-1]['team']
    top_team = standings.iloc[0]['team']

    game_week, goals, big_win, top_result, low_result = cpl_main.get_weeks_results(results[results['s'] <= 1],standings,team_ref)

    top_forward = rated_forwards.loc[0]
    top_midfielder = rated_midfielders.loc[0]
    top_defender = rated_defenders.loc[0]
    top_scorer = rated_goalscorers.loc[0]
    top_assist = rated_assists.loc[0]
    top_keeper = rated_keepers.loc[0]
    top_offender = rated_offenders.loc[0]

    if results.iloc[0]['hr'] == 'E':
        top_team, top_mover, top_dropper = na, na, na
    if year == '2019':
        top_team = 'Forge FC 2019 Champs'

    return render_template('cpl-es-index2.html',top_mover = top_mover, top_dropper = top_dropper,
    goals = goals, big_win = big_win, top_result = top_result, low_result = low_result,
    top_team = top_team, top_keeper = top_keeper,top_forward = top_forward,
    top_midfielder = top_midfielder, top_defender = top_defender,
    top_scorer = top_scorer, top_assist = top_assist, top_offender = top_offender,today = today, year = year, other_year = other_year)

@canples.route('/standings')
def standings():
    standings = pd.read_csv(f'datasets/{year}/cpl-{year}-standings.csv')
    return render_template('cpl-es-standings.html',standings_table = standings, year = year)

@canples.route('/comparison1')
def comparison1():
    if year == '2019':
        headline = 'Final Matches of the 2019 Regular Season'
    else:
        headline = 'This Weeks Matches'
    # home side
    q1 = schedule.iloc[0]['home']
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

    home_win, draw, away_win, home_form, away_form, home_roster, away_roster = get_files(stats,game,q1,q2)
    team1 = cpl_main.get_shortest_name(schedule.iloc[0]['home'],team_ref)
    team2 = cpl_main.get_shortest_name(schedule.iloc[0]['away'],team_ref)
    team3 = cpl_main.get_shortest_name(schedule.iloc[1]['home'],team_ref)
    team4 = cpl_main.get_shortest_name(schedule.iloc[1]['away'],team_ref)
    team5 = cpl_main.get_shortest_name(schedule.iloc[2]['home'],team_ref)
    team6 = cpl_main.get_shortest_name(schedule.iloc[2]['away'],team_ref)
    team7 = cpl_main.get_shortest_name(schedule.iloc[3]['home'],team_ref)
    team8 = cpl_main.get_shortest_name(schedule.iloc[3]['away'],team_ref)
    return render_template('cpl-es-comparison.html',home_table = home_roster, away_table = away_roster, home_win = home_win,
    home_team = q1, away_team = q2, away_win = away_win, home_form = home_form, away_form = away_form, schedule = schedule, year = year,
    home_crest = home_crest, home_colour = home_colour, away_crest = away_crest, away_colour = away_colour, headline = headline,
    team1 = team1, team2 = team2, team3 = team3, team4 = team4, team5 = team5, team6 = team6, team7 = team7, team8 = team8)

@canples.route('/comparison2', methods=['POST'])
def comparison2():
    if year == '2019':
        headline = 'Final Matches of the 2019 Regular Season'
    else:
        headline = 'This Weeks Matches'
    # home side
    home = request.form['home']

    q1 = cpl_main.get_long_name(home,team_ref)
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

    home_win, draw, away_win, home_form, away_form, home_roster, away_roster = get_files(stats,game,q1,q2)
    team1 = cpl_main.get_shortest_name(schedule.iloc[0]['home'],team_ref)
    team2 = cpl_main.get_shortest_name(schedule.iloc[0]['away'],team_ref)
    team3 = cpl_main.get_shortest_name(schedule.iloc[1]['home'],team_ref)
    team4 = cpl_main.get_shortest_name(schedule.iloc[1]['away'],team_ref)
    team5 = cpl_main.get_shortest_name(schedule.iloc[2]['home'],team_ref)
    team6 = cpl_main.get_shortest_name(schedule.iloc[2]['away'],team_ref)
    team7 = cpl_main.get_shortest_name(schedule.iloc[3]['home'],team_ref)
    team8 = cpl_main.get_shortest_name(schedule.iloc[3]['away'],team_ref)
    return render_template('cpl-es-comparison2.html',home_table = home_roster, away_table = away_roster, home_win = home_win,
    home_team = q1, away_team = q2, away_win = away_win, home_form = home_form, away_form = away_form, schedule = schedule, year = year,
    home_crest = home_crest, home_colour = home_colour, away_crest = away_crest, away_colour = away_colour, headline = headline,
    team1 = team1, team2 = team2, team3 = team3, team4 = team4, team5 = team5, team6 = team6, team7 = team7, team8 = team8)

@canples.route('/teams')
def teams():
    return render_template('cpl-es-teams.html',html_table = team_ref, year = year)

@canples.route('/roster', methods=['POST'])
def roster():
    team = request.form['team']
    roster_team_info = team_ref[team_ref['team'] == team]
    roster_colour = roster_team_info.iloc[0][4]
    player_info = pd.read_csv(f'datasets/{year}/player-{year}-info.csv')
    roster = cpl_main.get_roster_overall(team,stats,team_ref,rated_forwards,rated_midfielders,rated_defenders,rated_keepers,player_info)
    crest = roster_team_info.iloc[0][5]
    return render_template('cpl-es-roster.html',team_name = team, html_table = roster, team_colour = roster_colour, year = year, crest = crest)

@canples.route('/goals')
def goals():
    rated_goalscorers = cpl_main.top_tracked(team_stats,'goals')
    rated_g10 = rated_goalscorers.head(10)
    rated_g10 = rated_g10[['rank','team','name','position','goals']]
    rated_assists = cpl_main.top_tracked(team_stats,'assists')
    rated_a10 = rated_assists.head(10)
    rated_a10 = rated_a10[['rank','team','name','position','assists']]
    return render_template('cpl-es-goals.html',html_table = rated_g10, assists_table = rated_a10, year = year)

@canples.route('/forwards')
def forwards():
    rated_forwards = pd.read_csv(f'datasets/{year}/cpl-{year}-forwards.csv')
    return render_template('cpl-es-forwards.html',html_table = rated_forwards, year = year)

@canples.route('/midfielders')
def midfielders():
    rated_midfielders = pd.read_csv(f'datasets/{year}/cpl-{year}-midfielders.csv')
    return render_template('cpl-es-midfielders.html',html_table = rated_midfielders, year = year)

@canples.route('/defenders')
def defenders():
    rated_defenders = pd.read_csv(f'datasets/{year}/cpl-{year}-defenders.csv')
    return render_template('cpl-es-defenders.html',html_table = rated_defenders, year = year)

@canples.route('/keepers')
def keepers():
    rated_keepers = pd.read_csv(f'datasets/{year}/cpl-{year}-keepers.csv')
    return render_template('cpl-es-keepers.html',html_table = rated_keepers, year = year)

@canples.route('/discipline')
def discipline():
    rated_offenders = pd.read_csv(f'datasets/{year}/cpl-{year}-discipline.csv')
    return render_template('cpl-es-discipline.html',html_table = rated_offenders, year = year)

@canples.route('/hell')
def hello():
    return 'Welcome to HELL world!'

if __name__ == "__main__":
    canples.run()