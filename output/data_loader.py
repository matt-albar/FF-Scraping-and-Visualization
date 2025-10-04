import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "512953-history-standings")
H2H_DIR = os.path.join(os.path.dirname(__file__), "512953-history-teamgamecenter")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Robust numeric coercion:
    - remove thousands separators
    - strip any non-numeric chars (keeps digits, dot, minus)
    - convert to numeric with NaN on failure
    """
    return pd.to_numeric(
        series.astype(str)
              .str.replace(",", "", regex=False)
              .str.replace(r"[^0-9.\-]", "", regex=True)
              .str.strip(),
        errors="coerce"
    )


def load_history_data(data_dir=DATA_DIR):
    all_data = []

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            year = file.replace(".csv", "")
            file_path = os.path.join(data_dir, file)

            try:
                df = pd.read_csv(file_path, dtype=str)

                # Remove any existing Year col to avoid duplicates, then insert ours
                if "Year" in df.columns:
                    df = df.drop(columns=["Year"])
                df.insert(0, "Year", int(year))

                # Normalize "Record" -> "RegularSeasonRecord" and strip Excel's phantom year
                if "Record" in df.columns:
                    df["Record"] = (
                        df["Record"]
                        .astype(str)
                        .str.replace("/", "-", regex=False)   # 10/3/2000 -> 10-3-2000
                        .str.replace(r"-20\d{2}$", "", regex=True)  # remove trailing -2000 etc.
                    )
                    df = df.rename(columns={"Record": "RegularSeasonRecord"})

                # Reorder: Year, TeamName, PlayoffRank, then everything else
                cols = list(df.columns)
                if "TeamName" in cols and "PlayoffRank" in cols:
                    col_order = ["Year", "TeamName", "PlayoffRank"] + [
                        c for c in cols if c not in ["Year", "TeamName", "PlayoffRank"]
                    ]
                else:
                    col_order = ["Year"] + [c for c in cols if c != "Year"]

                df = df[col_order]
                all_data.append(df)
                print(f"Loaded standings {file}")

            except Exception as e:
                print(f"Error loading {file}: {e}")

    if not all_data:
        raise FileNotFoundError("No standings CSV files found.")

    # Combine
    full_df = pd.concat(all_data, ignore_index=True)

    # --- Coerce important numeric columns ---
    # PointsFor / PointsAgainst may include thousands separators -> strip then convert
    for col in ["PointsFor", "PointsAgainst"]:
        if col in full_df.columns:
            full_df[col] = _coerce_numeric(full_df[col]).fillna(0.0)

    # Other numeric-ish fields we commonly use
    for col in ["PlayoffRank", "RegularSeasonRank", "DraftPosition", "Moves", "Trades"]:
        if col in full_df.columns:
            full_df[col] = _coerce_numeric(full_df[col]).fillna(0)

    # Sort by Year, then PlayoffRank if present
    if "PlayoffRank" in full_df.columns:
        full_df = full_df.sort_values(by=["Year", "PlayoffRank"], ascending=[True, True])

    return full_df


def load_h2h_data(base_dir=H2H_DIR):
    all_games = []

    for year in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year)
        if not os.path.isdir(year_path):
            continue

        for file in os.listdir(year_path):
            if file.endswith(".csv"):
                week = file.replace(".csv", "")
                file_path = os.path.join(year_path, file)

                try:
                    df = pd.read_csv(file_path, dtype=str)

                    # Expect owner/opponent names + both totals
                    if {"Owner", "Opponent", "Total", "Opponent Total"}.issubset(df.columns):
                        temp = df[["Owner", "Opponent", "Total", "Opponent Total"]].copy()
                        temp["Year"] = int(year)
                        temp["Week"] = int(week)

                        # Scores to numeric
                        temp["Total"] = _coerce_numeric(temp["Total"]).fillna(0.0)
                        temp["Opponent Total"] = _coerce_numeric(temp["Opponent Total"]).fillna(0.0)

                        all_games.append(temp)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    if not all_games:
        raise FileNotFoundError("No head-to-head matchup files found.")

    return pd.concat(all_games, ignore_index=True)


def load_current_managers(csv_path=os.path.join(os.path.dirname(__file__), "managers_current.csv")):
    """
    Load the list of current managers from managers_current.csv.
    Expects a column 'ManagerName' in the CSV.
    """
    try:
        managers = pd.read_csv(csv_path)
        if "ManagerName" not in managers.columns:
            raise ValueError("CSV must contain a 'ManagerName' column")
        return managers["ManagerName"].dropna().unique().tolist()
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. Using all managers.")
        return None
