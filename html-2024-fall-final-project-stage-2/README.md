# HTML 2024 Fall Final Project : MLB Prediction 

* The code for data fetching and data preprocessing are referenced from [arjun-prabhakar/mlb_outcomes](https://github.com/arjun-prabhakar/mlb_outcomes)
* You are prohibited to acquire additional data from any sources including the source above.

## Columns Overview & Introduction

### Game Information

| **Column**             | **Description**                                                |
|------------------------|----------------------------------------------------------------|
| home_team_abbr         | Abbreviation of the home team.                                 |
| away_team_abbr         | Abbreviation of the away team.                                 |
| date                   | Date when the game was played.                                 |
| is_night_game          | Boolean indicating if the game was played at night.            |
| home_team_win          | Boolean indicating if the home team won the game.              |
| season                 | The season in which the game was played.                       |

---

### Team and Pitcher Rest

| **Column**             | **Description**                                                |
|------------------------|----------------------------------------------------------------|
| home_team_rest         | Number of days of rest for the home team before the game.      |
| away_team_rest         | Number of days of rest for the away team before the game.      |
| home_pitcher_rest      | Number of days of rest for the home team's starting pitcher.   |
| away_pitcher_rest      | Number of days of rest for the away team's starting pitcher.   |

---

### Pitcher Information

| **Column**             | **Description**                                                |
|------------------------|----------------------------------------------------------------|
| home_pitcher           | Name of the home team's starting pitcher.                      |
| away_pitcher           | Name of the away team's starting pitcher.                      |

---

### Recent Performance (10-Game Rolling Average)

| **Column**                                   | **Description**                                                 |
|----------------------------------------------|---------------------------------------------------------------- |
| home_batting_batting_avg_10RA                | Home team's batting average over the last 10 games.             |
| home_batting_onbase_perc_10RA                | Home team's on-base percentage over the last 10 games.          |
| home_batting_onbase_plus_slugging_10RA       | Home team's on-base plus slugging percentage over last 10 games.|
| home_batting_leverage_index_avg_10RA         | Home team's average leverage index over the last 10 games.      |
| home_batting_RBI_10RA                        | Home team's runs batted in over the last 10 games.              |
| home_pitching_earned_run_avg_10RA            | Home team's earned run average over the last 10 games.          |
| home_pitching_SO_batters_faced_10RA          | Home team's strikeouts over the last 10 games.                  |
| home_pitching_H_batters_faced_10RA           | Home team's hits allowed over the last 10 games.                |
| home_pitching_BB_batters_faced_10RA          | Home team's walks allowed over the last 10 games.               |
| home_pitcher_earned_run_avg_10RA             | Earned run average over the last 10 games for the home starting pitcher. |
| home_pitcher_SO_batters_faced_10RA           | Strikeouts by the home starting pitcher over the last 10 games.          |
| home_pitcher_H_batters_faced_10RA            | Hits allowed by the home starting pitcher over the last 10 games.        |
| home_pitcher_BB_batters_faced_10RA           | Walks allowed by the home starting pitcher over the last 10 games.       |

* The corresponding information for away team comes with `away_` prefix 
* The detailed explanation of the statistics can be found on any of the baseball-reference box score page. For example [Shohei Ohtani's 50-50 game](https://www.baseball-reference.com/boxes/MIA/MIA202409190.shtml)

---

### Team Seasonal Statistics

| **Column**                | **Description**                                                |
|---------------------------|----------------------------------------------------------------|
| home_team_season           | Unique identifier for the team, season combination            |
| home_team_errors_mean      | Mean of errors committed by the home team.                    |
| home_team_spread_mean      | Mean of point spread for the home team.                       |
| home_team_wins_mean        | Mean number of wins for the home team.                        |

* spread = points scored - points allowed
* The corresponding information for away team comes with `away_` prefix 
* Standard deviation and skewness of the above data come in `_std` and `_skew` suffix resppectively

---

### Team Seasonal Batting 

| **Column**                                    | **Description**                                                 |
|-----------------------------------------------|-----------------------------------------------------------------|
| home_batting_batting_avg_mean                 | Mean batting average for the home team.                         |
| home_batting_onbase_perc_mean                 | Mean on-base percentage for the home team.                      |
| home_batting_onbase_plus_slugging_mean        | Mean on-base plus slugging percentage for the home team.        |
| home_batting_leverage_index_avg_mean          | Mean leverage index for the home team.                          |
| home_batting_wpa_bat_mean                     | Mean Win Probability Added (WPA) for batting for the home team. |
| home_batting_RBI_mean                         | Mean runs batted in for the home team.                          |

* The corresponding information for away team comes with `away_` prefix 
* Standard deviation and skewness of the above data come in `_std` and `_skew` suffix resppectively
* The detailed explanation of the statistics can be found on any of the baseball-reference box score page. For example [Shohei Ohtani's 50-50 game](https://www.baseball-reference.com/boxes/MIA/MIA202409190.shtml)

---

### Team Seasonal Pitching 

| **Column**                            | **Description**                                               |
|---------------------------------------|---------------------------------------------------------------|
| home_pitching_earned_run_avg_mean     | Mean earned run average (ERA) for the home team.              |
| home_pitching_SO_batters_faced_mean   | Mean strikeouts for the home team pitchers.                   |
| home_pitching_H_batters_faced_mean    | Mean hits allowed by the home team pitchers.                  |
| home_pitching_BB_batters_faced_mean   | Mean walks allowed by the home team pitchers.                 |
| home_pitching_leverage_index_avg_mean | Mean leverage index for the home team pitchers.               |
| home_pitching_wpa_def_mean            | Mean Win Probability Added (WPA) for defense by home team.    |

* The corresponding information for away team comes with `away_` prefix 
* Standard deviation and skewness of the above data come in `_std` and `_skew` suffix resppectively
* The detailed explanation of the statistics can be found on any of the baseball-reference box score page. For example [Shohei Ohtani's 50-50 game](https://www.baseball-reference.com/boxes/MIA/MIA202409190.shtml)

---


### Pitcher Seasonal  Performance 

| **Column**                            | **Description**                                                   |
|---------------------------------------|-------------------------------------------------------------------|
| home_pitcher_earned_run_avg_mean      | Mean earned run average (ERA) for the home starting pitcher.           |
| home_pitcher_SO_batters_faced_mean    | Mean strikeouts for the home starting pitcher.                         |
| home_pitcher_H_batters_faced_mean     | Mean hits allowed by the home starting pitcher.                        |
| home_pitcher_BB_batters_faced_mean    | Mean walks allowed by the home starting pitcher.                       |
| home_pitcher_leverage_index_avg_mean  | Mean leverage index for the home starting pitcher.                    |
| home_pitcher_wpa_def_mean             | Mean Win Probability Added (WPA) for defense by home starting pitcher. |

* The corresponding information for away team comes with `away_` prefix 
* Standard deviation and skewness of the above data come in `_std` and `_skew` suffix resppectively
* The detailed explanation of the statistics can be found on any of the baseball-reference box score page. For example [Shohei Ohtani's 50-50 game](https://www.baseball-reference.com/boxes/MIA/MIA202409190.shtml)
