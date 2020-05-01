from flask import Flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from canples import cpl_main

from datetime import date

today = date.today()

current_year = date.today().strftime('%Y')

import numpy as np
import pandas as pd

canples = Flask(__name__)

year = str(2019)

results = pd.read_csv(f'canples/datasets/{year}/cpl-{year}-results.csv')
stats = pd.read_csv(f'canples/datasets/{year}/cpl-{year}-stats.csv')

if year == '2019':
    teams_short = ['CFC', 'FCE', 'FFC', 'HFX', 'PFC', 'VFC', 'Y9']
    colours = ['w3-2019-fiesta', 'w3-2019-princess-blue', 'w3-2019-turmeric', 'w3-vivid-blue', 'w3-vivid-reddish-purple', 'w3-2019-biking-red', 'w3-vivid-yellowish-green']
else:
    teams_short = ['AO','CFC', 'FCE', 'FFC', 'HFX', 'PFC', 'VFC', 'Y9']
    colours = ['w3-vivid-red','w3-2019-fiesta', 'w3-2019-princess-blue', 'w3-2019-turmeric', 'w3-vivid-blue', 'w3-vivid-reddish-purple', 'w3-2019-biking-red', 'w3-vivid-yellowish-green']

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

results_old = results.loc[0:93].copy() # this is a temporary solution

team_stats = cpl_main.get_stats_all(stats)

rated_forwards = cpl_main.top_forwards(team_stats)
rated_midfielders = cpl_main.top_midfielders(team_stats)
rated_defenders = cpl_main.top_defenders(team_stats)
rated_goalscorers = cpl_main.top_goalscorers(team_stats)
rated_keepers = cpl_main.top_keepers(team_stats)
rated_offenders = cpl_main.top_offenders(team_stats)

schedule = cpl_main.get_schedule(results)

results_brief = cpl_main.get_results_brief(results)

def get_files(stats,game,q1,q2):
    game_h = cpl_main.get_home_away_comparison(stats,game,q1)
    game_a = cpl_main.get_home_away_comparison(stats,game,q2)
    compare = cpl_main.get_team_comparison(results_brief,q1,q2)
    t1_x, t1_y = cpl_main.get_NB_data(compare,q1)
    t2_x, t2_y = cpl_main.get_NB_data(compare,q2)
    home_win, draw, away_win = cpl_main.get_match_prediction(q1,q2,t1_x,t1_y,t2_x,t2_y)
    home_form = cpl_main.get_five_game_form(results,q1)
    away_form = cpl_main.get_five_game_form(results,q2)
    home_roster = cpl_main.get_compare_roster(game_h)
    away_roster = cpl_main.get_compare_roster(game_a)
    return home_win, draw, away_win, home_form, away_form, home_roster, away_roster

@canples.route('/index')
def index():
    standings = cpl_main.get_standings(results,1)
    standings_old = cpl_main.get_standings(results_old,1)
    compare_standings = cpl_main.compare_standings(standings,standings_old)

    top_mover = compare_standings.iloc[0]['team']
    top_dropper = compare_standings.iloc[-1]['team']
    top_team = standings.iloc[0]['team']

    game_week, goals, big_win, top_result, low_result = cpl_main.get_weeks_results(results[results['s'] <= 1],standings)

    assists = cpl_main.top_assists(team_stats)
    top_forward = rated_forwards.loc[0]
    top_midfielder = rated_midfielders.loc[0]
    top_defender = rated_defenders.loc[0]
    top_scorer = rated_goalscorers.loc[0]
    top_assist = assists.loc[0]
    top_keeper = rated_keepers.loc[0]
    top_offender = rated_offenders.loc[0]
    return render_template('cpl-es-index.html',top_mover = top_mover, top_dropper = top_dropper,
    goals = goals, big_win = big_win, top_result = top_result, low_result = low_result,
    top_team = top_team, top_keeper = top_keeper,top_forward = top_forward,
    top_midfielder = top_midfielder, top_defender = top_defender,
    top_scorer = top_scorer, top_assist = top_assist, top_offender = top_offender,today = today)

@canples.route('/standings')
def standings():
    standings = cpl_main.get_standings(results,1)
    return render_template('cpl-es-standings.html',standings_table = standings)

@canples.route('/comparison1')
def comparison1():
    # home side
    q1 = schedule.iloc[0]['home']
    game = schedule.iloc[0]['game']
    # away side
    q2 = schedule.iloc[0]['away']
    home_win, draw, away_win, home_form, away_form, home_roster, away_roster = get_files(stats,game,q1,q2)
    return render_template('cpl-es-comparison.html',home_table = home_roster, away_table = away_roster, home_win = home_win,
    away_win = away_win, home_form = home_form, away_form = away_form, schedule = schedule)

@canples.route('/comparison2', methods=['POST'])
def comparison2():
    # home side
    home = request.form['home']
    q1 = home#cpl_main.get_long_name(home)
    game_info = schedule[schedule['home'] == q1]
    game = game_info.iloc[0]['game']
    # away side
    q2 = game_info.iloc[0]['away']
    home_win, draw, away_win, home_form, away_form, home_roster, away_roster = get_files(stats,game,q1,q2)
    home_win = round(home_win,1)
    away_win = round(away_win,1)
    return render_template('cpl-es-comparison2.html',home_table = home_roster, away_table = away_roster, home_win = home_win,
    home_team = q1, away_team = q2, away_win = away_win, home_form = home_form, away_form = away_form, schedule = schedule)

@canples.route('/teams')
def teams():
    return render_template('cpl-es-teams.html')

@canples.route('/roster', methods=['POST'])
def roster():
    team = request.form['team']
    colour = team_ref[team_ref['team'] == team].copy()
    colour = colour.iloc[0]['colour']
    roster = cpl_main.get_stats_all(stats)
    roster = roster[roster['team'] == team]
    roster = roster[['team','name','number','position']]
    return render_template('cpl-es-roster.html',html_table = roster, team_colour = colour)

@canples.route('/goals')
def goals():
    rated_goalscorers = cpl_main.top_goalscorers(team_stats)
    return render_template('cpl-es-goals.html',html_table = rated_goalscorers)

@canples.route('/forwards')
def forwards():
    rated_forwards = cpl_main.top_forwards(team_stats)
    return render_template('cpl-es-forwards.html',html_table = rated_forwards)

@canples.route('/midfielders')
def midfielders():
    rated_midfielders = cpl_main.top_midfielders(team_stats)
    return render_template('cpl-es-midfielders.html',html_table = rated_midfielders)

@canples.route('/defenders')
def defenders():
    rated_defenders = cpl_main.top_defenders(team_stats)
    return render_template('cpl-es-defenders.html',html_table = rated_defenders)

@canples.route('/keepers')
def keepers():
    rated_keepers = cpl_main.top_keepers(team_stats)
    return render_template('cpl-es-keepers.html',html_table = rated_keepers)

@canples.route('/discipline')
def discipline():
    rated_offenders = cpl_main.top_offenders(team_stats)
    return render_template('cpl-es-discipline.html',html_table = rated_offenders)

@canples.route('/hell')
def hello():
    return 'Welcome to HELL world!'

if __name__ == "__main__":
    canples.run()
