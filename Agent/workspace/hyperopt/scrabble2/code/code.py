#!/usr/bin/env python
# coding: utf-8

# ! pip install textstat

import os
import textstat
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.metrics import mean_squared_error

import category_encoders as ce

import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping, Dataset

# %matplotlib inline
# import seaborn as sns
# from matplotlib import pyplot as plt

# plt.style.use("seaborn-whitegrid")
# plt.rc("figure", autolayout=True)
# plt.rc(
#     "axes",
#     labelweight="bold",
#     labelsize="large",
#     titleweight="bold",
#     titlesize=14,
#     titlepad=10,
# )

"""Set the directory for the data"""

ROOT_DIR = "../input/scrabble-player-rating"

# # 1. Load Data
# 
import warnings

warnings.filterwarnings("ignore")

FILE_PATH = "../data/"
# FILE_PATH= "./workspace/hyperopt/scrabble/data/"

TARGET = "NObeyesdad"
submission_path = "ori_submission.csv"
n_splits = 9
RANDOM_SEED = 73

train = pd.read_csv(os.path.join(FILE_PATH, "train.csv"))
test = pd.read_csv(os.path.join(FILE_PATH, "test.csv"))
turns = pd.read_csv(os.path.join(FILE_PATH, "turns.csv"))
games = pd.read_csv(os.path.join(FILE_PATH, "games.csv"))


# # 2. Import and Preprocess Data

def create_turn_features(df):
    """
    Function based on function from :
    https://www.kaggle.com/code/hasanbasriakcay/scrabble-eda-fe-modeling
    """

    df["rack_len"] = df["rack"].str.len()
    df["rack_len_less_than_7"] = df["rack_len"].apply(lambda x: x < 7)
    df["move_len"] = df["move"].str.len()
    df["move"].fillna("None", inplace=True)
    df["difficult_word"] = df["move"].apply(textstat.difficult_words)
    rare_letters = ["Z", "Q", "J", "X", "K", "V", "Y", "W", "G"]
    df["difficult_letters"] = df["move"].apply(lambda x: len([letter for letter in x if letter in rare_letters]))
    df["points_per_letter"] = df["points"] / df["move_len"]

    df["turn_type"].fillna("None", inplace=True)
    turn_type_unique = df["turn_type"].unique()
    df = pd.get_dummies(df, columns=["turn_type"])
    dummy_features = [f"turn_type_{value}" for value in turn_type_unique]

    char_map = {
        "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
        "I": 9, "J": 10, "K": 11, "L": 12, "M": 13, "N": 14, "O": 15,
    }
    df["y"] = df["location"].str.extract("(\d+)")[0].values
    df["y"].fillna("0", inplace=True)
    df["y"] = df["y"].astype(int)

    df["x"] = df["location"].str.extract("([A-Z])")[0].values
    df["x"].replace(char_map, inplace=True)
    df["x"].fillna("0", inplace=True)
    df["x"] = df["x"].astype(int)

    df["direction_of_play"] = df["location"].apply(lambda x: 1 if str(x)[0].isdigit() else 0)

    df["curr_board_pieces_used"] = df["move"].apply(lambda x: str(x).count(".") + sum(int(c.islower()) for c in str(x)))

    agg_func_counts = {feature: "sum" for feature in dummy_features}
    turns_grouped_counts = df.groupby(["game_id", "nickname"], as_index=False).agg(agg_func_counts)

    agg_func_stats = {
        "points": ["mean", "max"],
        "move_len": ["mean", "max"],
        "difficult_word": ["mean", "sum"],
        "difficult_letters": ["mean", "sum"],
        "points_per_letter": "mean",
        "curr_board_pieces_used": "mean",
        "direction_of_play": "mean",
        "rack_len_less_than_7": "sum",
        "turn_number": "count"
    }

    # Only take those turns where a play is made
    turns_grouped_stats = df[df["turn_type_Play"] == 1].groupby(["game_id", "nickname"], as_index=False).agg(
        agg_func_stats)
    turns_grouped_stats.columns = ["_".join(a) if a[0] not in ["game_id", "nickname"] else a[0] for a in
                                   turns_grouped_stats.columns.to_flat_index()]
    turns_grouped = turns_grouped_counts.merge(turns_grouped_stats, how="outer", on=["game_id", "nickname"])
    # Fill in games where no play is ever done (about 46 of them)
    turns_grouped.fillna(value=0, inplace=True)

    return turns_grouped


"""Wrapper function to read in, encode and impute missing values for the data"""


def load_data(bot_names=["BetterBot", "STEEBot", "HastyBot"], cat_features=[]):
    train = pd.read_csv(os.path.join(FILE_PATH, "train.csv"))
    test = pd.read_csv(os.path.join(FILE_PATH, "test.csv"))
    turns = pd.read_csv(os.path.join(FILE_PATH, "turns.csv"))
    games = pd.read_csv(os.path.join(FILE_PATH, "games.csv"))

    # Merge the splits so we can process them together
    df = pd.concat([train, test])

    # Preprocessing

    # Add in turn features
    turns_fe_df = create_turn_features(turns)
    df = df.merge(turns_fe_df, how="left", on=["game_id", "nickname"])

    # Create the bot matrix
    bot_turns_columns = [i for i in turns_fe_df.columns.tolist() if i not in ["game_id", "nickname"]]
    bot_df = df[["game_id", "nickname", "score", "rating"] + bot_turns_columns].copy()
    bot_df["bot_name"] = bot_df["nickname"].apply(lambda x: x if x in bot_names else np.nan)
    bot_df = bot_df[["game_id", "bot_name", "score", "rating"] + bot_turns_columns].dropna(subset=["bot_name"])
    bot_df.columns = ["game_id", "bot_name", "bot_score", "bot_rating"] + ["bot_" + i for i in bot_turns_columns]

    # Bring all the data together
    df = df[~df["nickname"].isin(bot_names)]  # take out the bots
    df = df.merge(bot_df, on="game_id")  # add in bot information
    df = df.merge(games, on="game_id")  # add in game information
    df["created_at"] = pd.to_datetime(df["created_at"])  # convert to datetime
    df["first"] = df["first"].apply(lambda x: "bot" if x in bot_names else "player")

    # Specify categorical variables
    for name in cat_features:
        df[name] = df[name].astype("category")
        # Add a None category for missing values
        if "None" not in df[name].cat.categories:
            df[name].cat.add_categories("None")

    # Reform splits
    train = df[df["game_id"].isin(train["game_id"])].set_index("game_id")
    test = df[df["game_id"].isin(test["game_id"])].set_index("game_id")
    return train, test


"""Now, load in the data"""

train, test = load_data(
    cat_features=["nickname", "bot_name", "time_control_name", "first", "game_end_reason", "winner", "lexicon",
                  "rating_mode"])


def score_dataset(X, y,
                  model=LGBMRegressor(n_estimators=1000, verbose=-1, random_state=42)
                  ):
    X = X.copy()
    groups = X.pop("nickname")

    scores = cross_validate(
        model, X, y, cv=GroupKFold(), groups=groups, n_jobs=-1, scoring="neg_root_mean_squared_error",
        return_train_score=True
    )

    return {"Training": -1 * np.mean(scores["train_score"]), "Validation": -1 * np.mean(scores["test_score"])}


base_features = ["nickname", "score", "turn_type_Play", "turn_type_End", "turn_type_Exchange", "turn_type_Pass",
                 "turn_type_Timeout",
                 "turn_type_Challenge", "turn_type_Six-Zero Rule", "turn_type_None", "points_mean", "points_max",
                 "move_len_mean", "move_len_max",
                 "difficult_word_mean", "difficult_word_sum", "difficult_letters_mean", "difficult_letters_sum",
                 "points_per_letter_mean",
                 "curr_board_pieces_used_mean", "direction_of_play_mean", "rack_len_less_than_7_sum",
                 "turn_number_count", "bot_name",
                 "bot_score", "bot_rating", "bot_turn_type_Play", "bot_turn_type_End", "bot_turn_type_Exchange",
                 "bot_turn_type_Pass", "bot_turn_type_Timeout",
                 "bot_turn_type_Challenge", "bot_turn_type_Six-Zero Rule", "bot_turn_type_None", "bot_points_mean",
                 "bot_points_max",
                 "bot_move_len_mean", "bot_move_len_max", "bot_difficult_word_mean", "bot_difficult_word_sum",
                 "bot_difficult_letters_mean",
                 "bot_difficult_letters_sum", "bot_points_per_letter_mean", "bot_curr_board_pieces_used_mean",
                 "bot_direction_of_play_mean",
                 "bot_rack_len_less_than_7_sum", "bot_turn_number_count", "first", "time_control_name",
                 "game_end_reason", "winner",
                 "lexicon", "initial_time_seconds", "increment_seconds", "rating_mode", "max_overtime_minutes",
                 "game_duration_seconds"]

X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
y = train["rating"]

score_dataset(X, y)


# # 3. Feature Engineering
# ## Find out the helpful feature

# ## 3.a Creating cumulative player statistics
# 
# We"ll try overall player statistics and break down statistics by various game types

def create_cumm_player_features_overall(df):
    """
    Get the running average of player scores and win ratio over the course of all of their games up to the current rating
    """

    df = df[["nickname", "created_at", "score", "winner", "game_duration_seconds"]]

    # sort by the times of the games so that we aggregate over time in the ensuing steps
    df = df.sort_values(by="created_at")

    # Initialize our new variables with 0"s
    df["cumm_avg_player_score"] = np.zeros(len(df))
    df["cumm_max_player_score"] = np.zeros(len(df))
    df["cumm_min_player_score"] = np.zeros(len(df))
    df["cumm_total_player_score"] = np.zeros(len(df))
    df["cumm_player_wins"] = np.zeros(len(df))
    df["cumm_avg_player_win_ratio"] = np.zeros(len(df))
    df["cumm_avg_game_duration_seconds"] = np.zeros(len(df))

    for nickname in df["nickname"].unique():
        """
        Create the running averages of the player game features. Very important note with these, I am shifting the averages up by one ([:-1]) and
        adding in a starting zero. this is because "expanding" takes into account the current value, and we do not actually know the current
        values of the game prior to it being played. We must remember that the rating for each game is the rating *prior* to that game being played.
        """
        df.loc[df["nickname"] == nickname, "cumm_avg_player_score"] = np.append(0, df[df["nickname"] == nickname][
                                                                                       "score"].expanding(
            min_periods=1).mean().values[:-1])
        df.loc[df["nickname"] == nickname, "cumm_max_player_score"] = np.append(0, df[df["nickname"] == nickname][
                                                                                       "score"].expanding(
            min_periods=1).max().values[:-1])
        df.loc[df["nickname"] == nickname, "cumm_min_player_score"] = np.append(0, df[df["nickname"] == nickname][
                                                                                       "score"].expanding(
            min_periods=1).min().values[:-1])

        df.loc[df["nickname"] == nickname, "cumm_player_wins"] = np.append(0, df[df["nickname"] == nickname][
                                                                                  "winner"].expanding(
            min_periods=1).sum().values[:-1])

        df.loc[df["nickname"] == nickname, "cumm_avg_player_win_ratio"] = \
            df[df["nickname"] == nickname]["cumm_player_wins"] / np.append(0, df[df["nickname"] == nickname][
                                                                                  "winner"].expanding(
                min_periods=1).count().values[:-1])

        df.loc[df["nickname"] == nickname, "cumm_avg_game_duration_seconds"] = \
            np.append(0, df[df["nickname"] == nickname]["game_duration_seconds"].expanding(min_periods=2).mean().values[
                         :-1])

    # fill in any missing values with 0
    df[["cumm_avg_player_score", "cumm_player_wins", "cumm_avg_player_win_ratio", "cumm_avg_game_duration_seconds",
        "cumm_max_player_score", "cumm_min_player_score"]] \
        = df[
        ["cumm_avg_player_score", "cumm_player_wins", "cumm_avg_player_win_ratio", "cumm_avg_game_duration_seconds",
         "cumm_max_player_score", "cumm_min_player_score"]].fillna(0)

    # resort the data by the the index (i.e. game number)
    df = df.sort_index()

    return df[["cumm_avg_player_score", "cumm_max_player_score", "cumm_min_player_score", "cumm_player_wins",
               "cumm_avg_player_win_ratio", "cumm_avg_game_duration_seconds"]]


# 單項特徵測試

# """
X = train["nickname"].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cumm_player_features_overall(train.copy()))
y = train["rating"]
score_dataset(X, y)
# """


# 與基本feature一起測試

# """
X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cumm_player_features_overall(train.copy()))
y = train["rating"]
score_dataset(X, y)


# """

def create_cumm_player_features_lexicon(df):
    """
    Get the running average of player scores and win ratio over the course of all of their games up to the current rating, broken down by lexicon
    """

    df = df[["nickname", "created_at", "score", "winner", "lexicon", "game_duration_seconds"]]

    # sort by the times of the games so that we aggregate over time in the ensuing steps
    df = df.sort_values(by="created_at")

    # Initialize our new variables with 0"s
    for lexicon in df["lexicon"].unique():
        df["cumm_avg_player_score_" + str(lexicon)] = np.zeros(len(df))
        df["cumm_player_wins_" + str(lexicon)] = np.zeros(len(df))
        df["cumm_avg_player_win_ratio_" + str(lexicon)] = np.zeros(len(df))
        df["cumm_avg_game_duration_seconds_" + str(lexicon)] = np.zeros(len(df))

    for nickname in df["nickname"].unique():
        for lexicon in df["lexicon"].unique():
            """
            Create the running averages of the player game features, by lexicon. Very important note with these, I am shifting the averages up by one ([:-1]) and
            adding in a starting zero. this is because "expanding" takes into account the current value, and we do not actually know the current
            values of the game prior to it being played. We must remember that the rating for each game is the rating *prior* to that game being played.
            """

            df.loc[(df["nickname"] == nickname) & (df["lexicon"] == lexicon), "cumm_avg_player_score_" + str(lexicon)] = \
                np.append(0, df[(df["nickname"] == nickname) & (df["lexicon"] == lexicon)]["score"].expanding(
                    min_periods=1).mean().values[:-1])

            df.loc[(df["nickname"] == nickname) & (df["lexicon"] == lexicon), "cumm_player_wins_" + str(lexicon)] = \
                np.append(0, df[(df["nickname"] == nickname) & (df["lexicon"] == lexicon)]["winner"].expanding(
                    min_periods=1).sum().values[:-1])

            df.loc[(df["nickname"] == nickname) & (df["lexicon"] == lexicon), "cumm_avg_player_win_ratio_" + str(
                lexicon)] = \
                df[(df["nickname"] == nickname) & (df["lexicon"] == lexicon)][
                    "cumm_player_wins_" + str(lexicon)] / np.append(0, df[(df["nickname"] == nickname) & (
                            df["lexicon"] == lexicon)]["winner"].expanding(min_periods=1).count().values[:-1])

            df.loc[(df["nickname"] == nickname) & (df["lexicon"] == lexicon), "cumm_avg_game_duration_seconds_" + str(
                lexicon)] = \
                np.append(0, df[(df["nickname"] == nickname) & (df["lexicon"] == lexicon)][
                                 "game_duration_seconds"].expanding(min_periods=1).mean().values[:-1])

    # fill in any missing values with 0
    for lexicon in df["lexicon"].unique():
        df[["cumm_avg_player_score_" + str(lexicon), "cumm_player_wins_" + str(lexicon),
            "cumm_avg_player_win_ratio_" + str(lexicon), "cumm_avg_game_duration_seconds_" + str(lexicon)]] = \
            df[["cumm_avg_player_score_" + str(lexicon), "cumm_player_wins_" + str(lexicon),
                "cumm_avg_player_win_ratio_" + str(lexicon), "cumm_avg_game_duration_seconds_" + str(lexicon)]].fillna(
                0)

    # resort the data by the the index (i.e. game number)
    df = df.sort_index()

    return df[df.columns.difference(["nickname", "created_at", "score", "winner", "lexicon", "game_duration_seconds"])]


# 單項特徵測試

# """
X = train["nickname"].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cumm_player_features_lexicon(train.copy()))
y = train["rating"]
score_dataset(X, y)
# """

# 與基本feature一起測試

# """
X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cumm_player_features_lexicon(train.copy()))
y = train["rating"]
score_dataset(X, y)


# """

def create_cumm_player_features_by_game_types(df):
    """
    Get the running average of player scores and win ratio over the course of all of their games up to the current rating, broken down by rating mode
    """

    df = df[["nickname", "created_at", "score", "winner", "rating_mode", "lexicon", "game_duration_seconds"]].copy()

    df = df.sort_values(by="created_at")

    for rating_mode in df["rating_mode"].unique():
        for lexicon in df["lexicon"].unique():
            df["cumm_avg_player_score_" + str(rating_mode) + "_" + str(lexicon)] = np.zeros(len(df))
            df["cumm_player_wins_" + str(rating_mode) + "_" + str(lexicon)] = np.zeros(len(df))
            df["cumm_avg_player_win_ratio_" + str(rating_mode) + "_" + str(lexicon)] = np.zeros(len(df))
            df["cumm_avg_game_duration_seconds_" + str(rating_mode) + "_" + str(lexicon)] = np.zeros(len(df))

    for nickname in df["nickname"].unique():
        for rating_mode in df["rating_mode"].unique():
            for lexicon in df["lexicon"].unique():
                df.loc[(df["nickname"] == nickname) &
                       (df["lexicon"] == lexicon) &
                       (df["rating_mode"] == rating_mode),
                       "cumm_avg_player_score_" + str(rating_mode) + "_" + str(lexicon)] = \
                    np.append(0, df[(df["nickname"] == nickname) &
                                    (df["lexicon"] == lexicon) &
                                    (df["rating_mode"] == rating_mode)]["score"].expanding(min_periods=1).mean().values[
                                 :-1])

                df.loc[(df["nickname"] == nickname) &
                       (df["lexicon"] == lexicon) &
                       (df["rating_mode"] == rating_mode),
                       "cumm_player_wins_" + str(rating_mode) + "_" + str(lexicon)] = \
                    np.append(0, df[(df["nickname"] == nickname) &
                                    (df["lexicon"] == lexicon) &
                                    (df["rating_mode"] == rating_mode)]["winner"].expanding(min_periods=1).sum().values[
                                 :-1])

                df.loc[(df["nickname"] == nickname) &
                       (df["lexicon"] == lexicon) &
                       (df["rating_mode"] == rating_mode),
                       "cumm_avg_player_win_ratio_" + str(rating_mode) + "_" + str(lexicon)] = \
                    df[(df["nickname"] == nickname) &
                       (df["lexicon"] == lexicon) &
                       (df["rating_mode"] == rating_mode)][
                        "cumm_player_wins_" + str(rating_mode) + "_" + str(lexicon)] / np.append(0, df[(df[
                                                                                                            "nickname"] == nickname) &
                                                                                                       (df[
                                                                                                            "lexicon"] == lexicon) &
                                                                                                       (df[
                                                                                                            "rating_mode"] == rating_mode)][
                                                                                                        "winner"].expanding(
                        min_periods=1).count().values[:-1])

                df.loc[(df["nickname"] == nickname) &
                       (df["lexicon"] == lexicon) &
                       (df["rating_mode"] == rating_mode),
                       "cumm_avg_game_duration_seconds_" + str(rating_mode) + "_" + str(lexicon)] = \
                    np.append(0, df[(df["nickname"] == nickname) &
                                    (df["lexicon"] == lexicon) &
                                    (df["rating_mode"] == rating_mode)]["game_duration_seconds"].expanding(
                        min_periods=1).mean().values[:-1])

    for nickname in df["nickname"].unique():
        for rating_mode in df["rating_mode"].unique():
            for lexicon in df["lexicon"].unique():
                df[["cumm_avg_player_score_" + str(rating_mode) + "_" + str(lexicon),
                    "cumm_player_wins_" + str(rating_mode) + "_" + str(lexicon),
                    "cumm_avg_player_win_ratio_" + str(rating_mode) + "_" + str(lexicon),
                    "cumm_avg_game_duration_seconds_" + str(rating_mode) + "_" + str(lexicon)]] = \
                    df[["cumm_avg_player_score_" + str(rating_mode) + "_" + str(lexicon),
                        "cumm_player_wins_" + str(rating_mode) + "_" + str(lexicon),
                        "cumm_avg_player_win_ratio_" + str(rating_mode) + "_" + str(lexicon),
                        "cumm_avg_game_duration_seconds_" + str(rating_mode) + "_" + str(lexicon)]].fillna(0)

    df = df.sort_index()

    return df[df.columns.difference(
        ["nickname", "created_at", "score", "winner", "rating_mode", "lexicon", "game_duration_seconds"])]


# 單項特徵測試

# """
cummulative_features = create_cumm_player_features_by_game_types(train.copy())
X = train["nickname"].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(cummulative_features)
y = train["rating"]
score_dataset(X, y)
# """

# 與基本feature一起測試

# """
X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(cummulative_features)
y = train["rating"]
score_dataset(X, y)


# """

# ## 3.b Creating cumulative game types by player
# 
# We"ll try adding in some statistics on the types of games being played by each player

def create_cumm_player_game_features(df):
    """
    Get the cummulative counts of bots, rating_modes, and lexicons by each player up to the current game
    """

    df = df[["nickname", "created_at", "bot_name", "rating_mode", "lexicon", "game_end_reason"]]

    encoder = ce.OneHotEncoder(cols=["bot_name", "rating_mode", "lexicon", "game_end_reason"], use_cat_names=True)
    df = df.join(encoder.fit_transform(df[["bot_name", "rating_mode", "lexicon", "game_end_reason"]]))

    df = df.sort_values(by="created_at")

    for feature_name in encoder.get_feature_names():
        df["cumm_" + str(feature_name) + "_counts"] = np.zeros(len(df))

    for nickname in df["nickname"].unique():
        for feature_name in encoder.get_feature_names():
            """
            Create the running counts of the types of games by player. Very important note with these, I am shifting the averages up by one ([:-1]) and
            adding in a starting zero. this is because "expanding" takes into account the current value, and we do not actually know the current
            values of the game prior to it being played. We must remember that the rating for each game is the rating *prior* to that game being played.
            """
            df.loc[df["nickname"] == nickname, "cumm_" + str(feature_name) + "_counts"] = \
                np.append(0, df[df["nickname"] == nickname][feature_name].expanding(min_periods=1).sum().values[:-1])

    for feature_name in encoder.get_feature_names():
        df["cumm_" + str(feature_name) + "_counts"] = df["cumm_" + str(feature_name) + "_counts"].fillna(0)

    df = df.sort_index()

    return df[df.columns.difference(["nickname", "created_at", "bot_name", "rating_mode", "lexicon",
                                     "game_end_reason"] + encoder.get_feature_names())]


# 單項特徵測試

# """
X = train["nickname"].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cumm_player_game_features(train.copy()))
y = train["rating"]
score_dataset(X, y)
# """

# 與基本feature一起測試

# """
X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cumm_player_game_features(train.copy()))
y = train["rating"]
score_dataset(X, y)


# """

# ## 3.c Creating cummulative bot statistics against each player
# 
# We"ll try adding in the cummulative bot features, by player, for each of the games

def create_cumm_bot_features(df):
    """
    Get the running average of bot ratings and scores, broken down by bot, for each player for each of their gammes
    """

    df = df[["nickname", "created_at", "bot_name", "bot_score", "bot_rating", "winner"]]
    df["score_rating_ratio"] = df["bot_score"] / df["bot_rating"]

    df = df.sort_values(by="created_at")

    for bot_name in df["bot_name"].unique():
        df["cumm_avg_bot_score_" + str(bot_name)] = np.zeros(len(df))
        df["cumm_avg_bot_rating_" + str(bot_name)] = np.zeros(len(df))
        df["cumm_avg_bot_wins_" + str(bot_name)] = np.zeros(len(df))
        df["cumm_avg_bot_win_ratio_" + str(bot_name)] = np.zeros(len(df))
        df["cumm_avg_bot_score_rating_ratio_" + str(bot_name)] = np.zeros(len(df))

    for nickname in df["nickname"].unique():
        for bot_name in df["bot_name"].unique():
            """
            Create the running averages of bot performances, by player, and by bot. Very important note with these, I am shifting the averages up by one ([:-1]) and
            adding in a starting zero. this is because "expanding" takes into account the current value, and we do not actually know the current
            values of the game prior to it being played. We must remember that the rating for each game is the rating *prior* to that game being played.
            This, however, does not apply to the bot rating, which we do know before the game is played (ratings are known before the game is played!)
            """
            df.loc[(df["nickname"] == nickname) & (df["bot_name"] == bot_name), "cumm_avg_bot_score_" + str(bot_name)] = \
                np.append(0, df[(df["nickname"] == nickname) & (df["bot_name"] == bot_name)]["bot_score"].expanding(
                    min_periods=1).mean().values[:-1])

            df.loc[
                (df["nickname"] == nickname) & (df["bot_name"] == bot_name), "cumm_avg_bot_rating_" + str(bot_name)] = \
                df[(df["nickname"] == nickname) & (df["bot_name"] == bot_name)]["bot_rating"].expanding(
                    min_periods=1).mean().values

            df.loc[(df["nickname"] == nickname) & (df["bot_name"] == bot_name), "cumm_avg_bot_wins_" + str(bot_name)] = \
                np.append(0, df[(df["nickname"] == nickname) & (df["bot_name"] == bot_name)]["winner"].expanding(
                    min_periods=1).apply(lambda x: np.sum(x == 0)).values[:-1])

            df.loc[(df["nickname"] == nickname) & (df["bot_name"] == bot_name), "cumm_avg_bot_win_ratio_" + str(
                bot_name)] = \
                df.loc[(df["nickname"] == nickname) & (df["bot_name"] == bot_name), "cumm_avg_bot_wins_" + str(
                    bot_name)] / np.append(0, df[(df["nickname"] == nickname) & (df["bot_name"] == bot_name)][
                                                  "winner"].expanding(min_periods=1).count().values[:-1])

            df.loc[
                (df["nickname"] == nickname) & (df["bot_name"] == bot_name), "cumm_avg_bot_score_rating_ratio_" + str(
                    bot_name)] = \
                np.append(0, df[(df["nickname"] == nickname) & (df["bot_name"] == bot_name)][
                                 "score_rating_ratio"].expanding(min_periods=1).mean().values[:-1])

    for bot_name in df["bot_name"].unique():
        df[["cumm_avg_bot_score_" + str(bot_name), "cumm_avg_bot_rating_" + str(bot_name),
            "cumm_avg_bot_wins_" + str(bot_name), "cumm_avg_bot_win_ratio_" + str(bot_name),
            "cumm_avg_bot_score_rating_ratio_" + str(bot_name)]] = \
            df[["cumm_avg_bot_score_" + str(bot_name), "cumm_avg_bot_rating_" + str(bot_name),
                "cumm_avg_bot_wins_" + str(bot_name), "cumm_avg_bot_win_ratio_" + str(bot_name),
                "cumm_avg_bot_score_rating_ratio_" + str(bot_name)]].fillna(0)

    df = df.sort_index()

    return df[df.columns.difference(
        ["nickname", "created_at", "bot_name", "bot_score", "bot_rating", "winner", "score_rating_ratio"])]


# 單項特徵測試

# """
X = train["nickname"].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cumm_bot_features(train.copy()))
y = train["rating"]
score_dataset(X, y)
# """

# 與基本feature一起測試

# """
X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cumm_bot_features(train.copy()))
y = train["rating"]
score_dataset(X, y)


# """

# ## 3.d Creating Cummulative Features from the Turn-based Features
# 
# We"ll try adding in the cummulative bot features, by player, for each of the games

def create_cumm_turns_features(df):
    turn_features = ["turn_type_Play", "turn_type_End",
                     "turn_type_Exchange", "turn_type_Pass", "turn_type_Timeout",
                     "turn_type_Challenge", "turn_type_Six-Zero Rule", "turn_type_None",
                     "points_mean", "points_max", "move_len_mean", "move_len_max",
                     "difficult_word_mean", "difficult_word_sum", "difficult_letters_mean",
                     "difficult_letters_sum", "points_per_letter_mean",
                     "curr_board_pieces_used_mean", "direction_of_play_mean",
                     "rack_len_less_than_7_sum", "turn_number_count"]

    # Create some features looking at the difference in performance between player and bot
    df["play_counts_diff"] = df["turn_type_Play"] - df["bot_turn_type_Play"]
    df["avg_points_diff"] = df["points_mean"] - df["bot_points_mean"]
    df["avg_move_len_diff"] = df["move_len_mean"] - df["bot_move_len_mean"]
    df["avg_points_per_letter_diff"] = df["points_per_letter_mean"] - df["bot_points_per_letter_mean"]
    df["difficult_words_count_diff"] = df["difficult_word_sum"] - df["bot_difficult_word_sum"]
    df["difficult_letters_count_diff"] = df["difficult_letters_sum"] - df["bot_difficult_letters_sum"]

    df = df[["nickname", "created_at", "play_counts_diff", "avg_points_diff", "avg_move_len_diff",
             "avg_points_per_letter_diff", "difficult_words_count_diff",
             "difficult_letters_count_diff"] + turn_features]

    df = df.sort_values(by="created_at")

    for nickname in df["nickname"].unique():
        for feature_name in turn_features:
            """
            Very important note with these, I am shifting the averages up by one ([:-1]) and
            adding in a starting zero. this is because "expanding" takes into account the current value, and we do not actually know the current
            values of the game prior to it being played. We must remember that the rating for each game is the rating *prior* to that game being played.
            """
            df.loc[df["nickname"] == nickname, "cumm_" + str(feature_name) + "_average"] = \
                np.append(0, df[df["nickname"] == nickname][feature_name].expanding(min_periods=1).mean().values[:-1])

    for feature_name in turn_features:
        df["cumm_" + str(feature_name) + "_average"] = df["cumm_" + str(feature_name) + "_average"].fillna(0)

    df = df.sort_index()

    return df[df.columns.difference(["nickname", "created_at"] + turn_features)]


# 單項特徵測試

# """
X = train["nickname"].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cumm_turns_features(train.copy()))
y = train["rating"]
score_dataset(X, y)
# """

# 與基本feature一起測試

# """
X = train[base_features].copy()
X = ce.OrdinalEncoder().fit_transform(X)
X = X.join(create_cumm_turns_features(train.copy()))
y = train["rating"]
score_dataset(X, y)


# """

# # 3.e Add in results from last game played
# 
# Add in the game results from the last game played by a player

def create_previous_game_features(df, features_to_include=["rating_mode", "lexicon", "bot_name", "time_control_name",
                                                           "score"]):
    """
    Add in the base features from the last game
    """
    df = df.sort_values(by="created_at")
    time_diff = df.groupby("nickname")["created_at"].shift(periods=0) - df.groupby("nickname")["created_at"].shift(
        periods=1)
    df = df.groupby("nickname")[features_to_include].shift(periods=1)
    df = df.add_suffix("_prev_game")
    df["time_between_games"] = time_diff.dt.total_seconds().fillna(0)
    df = df.fillna(value={"score": 0})
    df = df.sort_index()

    return df


# 測試加入前一局數據

# """
X = train[base_features].copy()
X = X.join(create_previous_game_features(train.copy()))
X = ce.OrdinalEncoder().fit_transform(X)
score_dataset(X, y)


# """

# # 特徵工程結束

# # 4. Finalize Features for Final Model
# 
# 將有效的feature都在此加入

def create_features(df, df_test=None):
    X_raw = df.copy()
    y = df["rating"].copy()

    if df_test is not None:
        X_test = df_test.copy()
        X_raw = pd.concat([X_raw, X_test])

    # Add in engineered features
    X = X_raw[base_features].copy()
    X = ce.OrdinalEncoder().fit_transform(X)

    # 加入各種擷取出的feature
    X = X.join(create_cumm_player_features_by_game_types(X_raw.copy()))
    X = X.join(create_cumm_bot_features(X_raw.copy()))
    X = X.join(create_cumm_turns_features(X_raw.copy()))

    # Reform splits
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)

    if df_test is not None:
        return X, X_test
    else:
        return X


X, X_test = create_features(train, test)
y = train["rating"].copy()

score_dataset(X, y)

# That"s the ticket - this looks like a decent featurization of the data 🙌

# # 5. Hyperparameter Tuning

# # 6. Fit final model and make predictions
# 
# We are going to fit more than one model and just use averaging to ensemble between them.

test_preds = []
train_preds = []
groups = X.pop("nickname")  # remove the player nicknames from the train set and make them groups for CV
test_groups = X_test.pop("nickname")  # remove the player nicknames from the test set
params = {}
lgb_params = {
    "objective": "regression",
    "verbose": -1,
    "n_estimators": 50000,
    **params
}
lgb_mode = LGBMRegressor(**lgb_params)
# for repeat in range(1):
#     skf = GroupKFold(n_splits=5)
#     for fold_idx, (train_index, valid_index) in enumerate(skf.split(X, y, groups=groups)):
#         X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
#         y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

#         lgb_train = lgb.Dataset(X_train, y_train)
#         lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
#         lgb_params = {
#             "objective": "regression",
#             "verbose": -1,
#             "n_estimators": 50000,
#             **params
#         }
#         model = lgb.train(lgb_params, lgb_train, valid_sets=lgb_eval, callbacks=[lgb.early_stopping(300)])

#         y_pred = model.predict(X_valid)
#         score = mean_squared_error(y_valid, y_pred, squared=False)
#         print("Fold {} MSE Score: {}".format(fold_idx, score))
#         print("----------------------")
#         test_preds.append( model.predict(X_test))
#         train_preds.append( model.predict(X))

# # Use average for ensembling of the labels

# final_test_preds = np.mean(test_preds, axis=0)
# final_train_preds = np.mean(train_preds, axis=0)

# """Take a look at the distribution of the produced ratings versus the given ratings"""

# # fig, axs = plt.subplots(2, 2, sharey=True, figsize=(20,8))
# # sns.distplot(train["rating"], ax=axs[0,0])
# # axs[0,0].set_title("Distribution of Train Ratings")
# # sns.distplot(final_train_preds , ax=axs[0,1])
# # axs[0,1].set_title("Distribution of Predicted Ratings on Train")
# # sns.distplot(final_test_preds , ax=axs[1,0])
# # axs[1,0].set_title("Distribution of Predicted Ratings on Test")

# # Create the submission
# test["rating"] = final_test_preds
# submission = test["rating"]


# submission.to_csv(FILE_PATH+submission_path, index=False)

# score=final_train_preds
