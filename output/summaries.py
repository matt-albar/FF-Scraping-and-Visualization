import pandas as pd

def build_finals_summary(df):
    champions = df[df["PlayoffRank"] == 1][
        ["Year", "ManagerName", "RegularSeasonRecord", "PointsFor", "PointsAgainst"]
    ].rename(
        columns={
            "ManagerName": "ChampionManager",
            "RegularSeasonRecord": "ChampionRegularSeasonRecord",
            "PointsFor": "ChampionPointsFor",
            "PointsAgainst": "ChampionPointsAgainst",
        }
    )

    first_losers = df[df["PlayoffRank"] == 2][
        ["Year", "ManagerName", "RegularSeasonRecord", "PointsFor", "PointsAgainst"]
    ].rename(
        columns={
            "ManagerName": "FirstLoserManager",
            "RegularSeasonRecord": "FirstLoserRegularSeasonRecord",
            "PointsFor": "FirstLoserPointsFor",
            "PointsAgainst": "FirstLoserPointsAgainst",
        }
    )

    return (
        pd.merge(champions, first_losers, on="Year", how="inner")
        .sort_values("Year")
        .reset_index(drop=True)
    )

def seasons_played_summary(df):
    return df.groupby("ManagerName")["Year"].nunique().rename("SeasonsPlayed")

def build_top3_summary(df):
    seasons_played = seasons_played_summary(df)
    top3_df = df[df["PlayoffRank"].between(1, 3, inclusive="both")]
    top3_counts = top3_df["ManagerName"].value_counts().rename("Top3Finishes")
    return (pd.concat([seasons_played, top3_counts], axis=1).fillna(0).astype(int))

def build_last_place_summary(df):
    _max_rank = df.groupby("Year")["PlayoffRank"].transform("max")
    last_place_df = df[df["PlayoffRank"] == _max_rank]
    last_place_counts = last_place_df["ManagerName"].value_counts().rename("LastPlaceFinishes")
    seasons_played = seasons_played_summary(df)
    return (pd.concat([seasons_played, last_place_counts], axis=1).fillna(0).astype(int))

def build_bottom3_summary(df):
    _max_rank = df.groupby("Year")["PlayoffRank"].transform("max")
    cutoff = (_max_rank - 2).clip(lower=1)
    bottom3_df = df[df["PlayoffRank"] >= cutoff]
    bottom3_counts = bottom3_df["ManagerName"].value_counts().rename("Bottom3Finishes")
    seasons_played = seasons_played_summary(df)
    return (pd.concat([seasons_played, bottom3_counts], axis=1).fillna(0).astype(int))
