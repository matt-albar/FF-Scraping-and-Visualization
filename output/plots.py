import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from summaries import (
    build_finals_summary,
    build_top3_summary,
    build_last_place_summary,
    build_bottom3_summary,
    seasons_played_summary
)

# ---------------------------
# Helper for labeled bar charts
# ---------------------------
def _bar_chart(df, value_cols, title, ylabel, colors):
    """
    Generic bar chart function.
    Expects df with index=ManagerName and cols=value_cols.
    """
    # Add seasons played to labels
    df["Label"] = df.index + " (" + df["SeasonsPlayed"].astype(str) + ")"
    plot_df = df.set_index("Label")[value_cols]

    ax = plot_df.sort_values(value_cols[0], ascending=False).plot(
        kind="bar", figsize=(12,6), edgecolor="black", color=colors
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Manager (Seasons Played)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    if len(value_cols) > 1:
        plt.legend(title="Result")
    else:
        plt.legend().remove()

    # Force integer ticks only
    ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))

    # Value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', fontsize=9)

    plt.tight_layout()
    plt.show()


# ---------------------------
# Specific Plots
# ---------------------------


def plot_team_names_over_time(df, current_managers=None):
    """
    Timeline of team names by season and manager.
    Each contiguous run of the same team name is a colored bar with one label.
    Requires columns: Year (int), ManagerName, TeamName.
    """

    # Keep only the columns we need and clean
    use = df[["Year", "ManagerName", "TeamName"]].copy()
    use = use.dropna(subset=["ManagerName", "Year"])
    use["Year"] = use["Year"].astype(int)
    use["TeamName"] = use["TeamName"].fillna("Unknown").astype(str).str.strip()

    # Filter to current managers (optional)
    if current_managers is not None:
        cm_norm = [m.strip().lower() for m in current_managers]
        use = use[use["ManagerName"].str.strip().str.lower().isin(cm_norm)]

    if use.empty:
        print("⚠️ No rows to plot (check filters/columns).")
        return

    # We want one row per manager per year (latest if duplicates)
    use = (use.sort_values(["ManagerName", "Year"])
               .drop_duplicates(subset=["ManagerName", "Year"], keep="last"))

    managers = use["ManagerName"].dropna().unique().tolist()
    years = sorted(use["Year"].unique().tolist())
    y_positions = {m: i for i, m in enumerate(managers)}

    # Color mapping: same team name → same color (stable hash)
    palette = plt.get_cmap("tab20").colors
    def color_for(name):
        return palette[hash(name) % len(palette)]

    fig_h = max(6, 0.6 * len(managers))
    fig, ax = plt.subplots(figsize=(18, fig_h))

    # Draw bars per contiguous block of identical name
    for m in managers:
        md = use[use["ManagerName"] == m].sort_values("Year")
        rows = md[["Year", "TeamName"]].values.tolist()

        # Build contiguous blocks (same name & consecutive years)
        blocks = []
        if rows:
            s_year, e_year, cur_name = rows[0][0], rows[0][0], rows[0][1]
            prev_year = rows[0][0]
            for yr, nm in rows[1:]:
                if nm == cur_name and yr == prev_year + 1:
                    e_year = yr
                else:
                    blocks.append((s_year, e_year, cur_name))
                    s_year, e_year, cur_name = yr, yr, nm
                prev_year = yr
            blocks.append((s_year, e_year, cur_name))

        # Plot blocks as horizontal bars
        y = y_positions[m]
        for s, e, nm in blocks:
            left = s - 0.5
            width = (e - s + 1)
            ax.barh(y=y, width=width, left=left, height=0.55,
                    color=color_for(nm), edgecolor="white", linewidth=0.6)
            # Label once per block, centered
            ax.text((s + e) / 2, y, nm,
                    ha="center", va="center", fontsize=8,
                    color="black", bbox=dict(boxstyle="round,pad=0.2",
                                             facecolor="white", alpha=0.7,
                                             edgecolor="none"))

    # Axes formatting
    ax.set_yticks(range(len(managers)))
    ax.set_yticklabels(managers)
    ax.set_xticks(years)
    ax.set_xlim(min(years) - 0.6, max(years) + 0.6)
    ax.set_xlabel("Season")
    ax.set_ylabel("Manager")
    ax.set_title("Team Names by Season and Manager")
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_champ_vs_loser_lollipop(df, current_managers=None):
    """
    Lollipop chart comparing Championships, 1st Loser finishes, and Last Place finishes by Manager.
    Each manager has 3 separate lollipops side by side.
    """

    df["PlayoffRank"] = pd.to_numeric(df["PlayoffRank"], errors="coerce")

    # Find last place rank per year
    last_place_flags = []
    for year, group in df.groupby("Year"):
        last_rank = group["PlayoffRank"].max()
        last_place_flags.extend(group["ManagerName"][group["PlayoffRank"] == last_rank].tolist())

    # Counts
    champ_counts = df[df["PlayoffRank"] == 1].groupby("ManagerName").size()
    loser_counts = df[df["PlayoffRank"] == 2].groupby("ManagerName").size()
    last_counts = pd.Series(last_place_flags).value_counts()
    seasons_played = df.groupby("ManagerName")["Year"].nunique()

    # Build summary
    summary = (
        pd.DataFrame({
            "Championships": champ_counts,
            "FirstLosers": loser_counts,
            "LastPlace": last_counts,
            "SeasonsPlayed": seasons_played
        })
        .fillna(0)
        .astype(int)
    )

    # Labels
    summary["Label"] = summary.index + " (" + summary["SeasonsPlayed"].astype(str) + ")"

    # Sorting with custom rules
    summary = (
        summary.sort_values(
            by=["Championships", "FirstLosers", "LastPlace", "SeasonsPlayed"],
            ascending=[False, False, True, True]  # custom order
        )
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(summary))

    width = 0.25  # spacing between categories

    # Championships
    ax.vlines(x - width, 0, summary["Championships"], color="blue", alpha=0.7)
    ax.scatter(x - width, summary["Championships"], color="blue", s=100, label="Championships")

    # First Losers
    ax.vlines(x, 0, summary["FirstLosers"], color="orange", alpha=0.7)
    ax.scatter(x, summary["FirstLosers"], color="orange", s=100, label="1st Losers")

    # Last Place
    ax.vlines(x + width, 0, summary["LastPlace"], color="red", alpha=0.7)
    ax.scatter(x + width, summary["LastPlace"], color="red", s=100, label="Last Place")

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Label"], rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Comparison of Championships, 1st Loser, and Last Place Finishes")

    # Y axis: integer only
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Legend at top center
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)

    plt.tight_layout()
    plt.show()


def line_playoff_ranks(df, current_managers=None):
    """
    Plot each manager's playoff finish (PlayoffRank) over time.
    """
    plot_df = df[["Year", "ManagerName", "PlayoffRank"]].dropna().copy()
    plot_df["Year"] = plot_df["Year"].astype(int)
    plot_df["PlayoffRank"] = plot_df["PlayoffRank"].astype(int)

    if current_managers is not None:
        cm_norm = [m.strip().lower() for m in current_managers]
        plot_df = plot_df[plot_df["ManagerName"].str.strip().str.lower().isin(cm_norm)]

    if plot_df.empty:
        print("⚠️ No data available for the specified managers.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    for manager, group in plot_df.groupby("ManagerName"):
        group = group.sort_values("Year")
        ax.plot(group["Year"], group["PlayoffRank"], marker="o")
        first_year, first_rank = group["Year"].iloc[0], group["PlayoffRank"].iloc[0]
        ax.text(first_year - 0.2, first_rank, manager,
                ha="right", va="center", fontsize=9)

    ax.invert_yaxis()

    league_sizes = df.groupby("Year")["PlayoffRank"].max()
    years = sorted(df["Year"].unique())
    labels = [f"{year} ({league_sizes.loc[year]})" for year in years]
    ax.set_xticks(years)
    ax.set_xticklabels(labels, rotation=45)

    max_rank = df["PlayoffRank"].max()
    ax.set_yticks(range(1, max_rank + 1))

    ax.set_title("Current Manager Finish Each Year", fontsize=16)
    ax.set_xlabel("Year (league size in parentheses)", fontsize=12)
    ax.set_ylabel("Playoff Rank (lower is better)", fontsize=12)

    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    plt.show()


def plot_head_to_head_wins(h2h_df, title="Head-to-Head Wins Matrix", show_top10=True):
    """
    Plot a heatmap of head-to-head wins (raw counts) between managers,
    and optionally show a table of the top 10 most successful matchups.

    Parameters:
    - h2h_df: DataFrame with columns ['Owner','Opponent','Total','Opponent Total']
    - title: chart title
    - show_top10: whether to print the top 10 matchups table
    """

    df = h2h_df.copy()

    # Determine winner and loser
    df["Winner"] = df.apply(
        lambda row: row["Owner"] if row["Total"] > row["Opponent Total"] else row["Opponent"],
        axis=1
    )
    df["Loser"] = df.apply(
        lambda row: row["Opponent"] if row["Total"] > row["Opponent Total"] else row["Owner"],
        axis=1
    )

    # Count wins (Winner vs Loser)
    win_counts = df.groupby(["Winner", "Loser"]).size().unstack(fill_value=0)

    # Ensure matrix is square (all managers in both axes)
    managers = sorted(set(df["Owner"]).union(df["Opponent"]))
    win_counts = win_counts.reindex(index=managers, columns=managers, fill_value=0)

    # Plot heatmap of wins
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        win_counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Wins"},
        linewidths=0.5,
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Opponent")
    plt.ylabel("Manager")
    plt.tight_layout()
    plt.show()

    if show_top10:
        # Flatten matrix for top matchups
        flat = (
            win_counts.stack()
            .reset_index()
            .rename(columns={"Winner": "Manager", "Loser": "Opponent", 0: "Wins"})
        )
        flat = flat[flat["Manager"] != flat["Opponent"]]  # drop self-vs-self

        # Add losses (opponent's wins against manager)
        flat["Losses"] = flat.apply(
            lambda row: win_counts.loc[row["Opponent"], row["Manager"]], axis=1
        )

        # Top 10 by Wins
        top10 = flat.sort_values("Wins", ascending=False).head(10)

        print("🏆 Top 10 Head-to-Head Matchups (Most Wins vs One Opponent):")
        display(
            top10.reset_index(drop=True)
                 .reset_index()
                 .rename(columns={"index": "Rank"})[["Rank", "Manager", "Opponent", "Wins", "Losses"]]
                 .assign(Rank=lambda d: d["Rank"] + 1)  # make rank start at 1
        )


def plot_draft_position_podiums(df):
    """
    Three heatmaps showing how many times each draft position produced:
      - Champion (1st)
      - First Loser (2nd)
      - Third Place (3rd)
    Draft positions are ordered 1..N across all panels, with vertical
    orange separators drawn BETWEEN the subplots.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = df[["DraftPosition", "PlayoffRank"]].dropna().copy()
    data["DraftPosition"] = pd.to_numeric(data["DraftPosition"], errors="coerce").astype("Int64")
    data["PlayoffRank"]   = pd.to_numeric(data["PlayoffRank"],   errors="coerce").astype("Int64")
    data = data.dropna().astype(int)

    max_slot = int(data["DraftPosition"].max())
    slot_order = list(range(1, max_slot + 1))

    titles = {
        1: "Champion's Draft Position",
        2: "First Loser Draft Position",
        3: "Third Place Draft Position",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, (rank, title) in enumerate(titles.items()):
        subset = data[data["PlayoffRank"] == rank]
        counts = (subset["DraftPosition"]
                  .value_counts()
                  .reindex(slot_order, fill_value=0)
                  .astype(int))

        heat_df = pd.DataFrame([counts.values], columns=slot_order)

        ax = axes[i]
        sns.heatmap(
            heat_df,
            annot=True, fmt="d", cmap="Blues", cbar=False,
            linewidths=0.5, linecolor="gray", ax=ax
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Draft Position")
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_xticklabels(slot_order)

    # --- draw separators AFTER layout so positions are correct ---
    plt.tight_layout()
    fig.canvas.draw()  # ensure positions are updated

    pos = [ax.get_position(fig) for ax in axes]
    y0 = min(p.y0 for p in pos)
    y1 = max(p.y1 for p in pos)
    x_sep1 = (pos[0].x1 + pos[1].x0) / 2.0
    x_sep2 = (pos[1].x1 + pos[2].x0) / 2.0

    for x in (x_sep1, x_sep2):
        fig.add_artist(Line2D([x, x], [y0, y1],
                              transform=fig.transFigure,
                              color="orange", linewidth=3))

    plt.show()



def plot_alltime_points_for_vs_against(df, current_managers=None):
    """
    Grouped bar chart of all-time points for and against, per manager.
    Includes number of seasons played in parentheses next to each manager.
    """
    df_all = df.copy()

    # Filter only current managers if provided
    if current_managers is not None:
        cm_norm = [m.strip().lower() for m in current_managers]
        df_all = df_all[df_all["ManagerName"].str.strip().str.lower().isin(cm_norm)]

    # Ensure numeric
    df_all["PointsFor"] = pd.to_numeric(df_all["PointsFor"], errors="coerce")
    df_all["PointsAgainst"] = pd.to_numeric(df_all["PointsAgainst"], errors="coerce")

    # Aggregate
    df_summary = df_all.groupby("ManagerName").agg(
        PointsFor=("PointsFor", "sum"),
        PointsAgainst=("PointsAgainst", "sum"),
        SeasonsPlayed=("Year", "nunique")
    ).reset_index()

    # Add label with seasons played
    df_summary["Label"] = (
        df_summary["ManagerName"] + " (" + df_summary["SeasonsPlayed"].astype(str) + ")"
    )

    # Sort by Points For (descending)
    df_summary = df_summary.sort_values("PointsFor", ascending=False)

    # Plot grouped bar chart
    ax = df_summary.set_index("Label")[["PointsFor", "PointsAgainst"]].plot(
        kind="bar",
        figsize=(14, 7),
        edgecolor="black",
        color=["steelblue", "salmon"]
    )

    plt.title("All-Time Points For vs Points Against (with Seasons Played)", fontsize=16)
    plt.xlabel("Manager (Seasons Played)", fontsize=12)
    plt.ylabel("Total Points", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Points")

    # Add labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=9, label_type="edge")

    plt.tight_layout()
    plt.show()

def plot_avg_points_per_season(df, current_managers=None):
    """
    Bar chart of average points for and against per season, per manager.
    """
    df_all = df.copy()

    # Filter only current managers if provided
    if current_managers is not None:
        cm_norm = [m.strip().lower() for m in current_managers]
        df_all = df_all[df_all["ManagerName"].str.strip().str.lower().isin(cm_norm)]

    # Ensure numeric
    df_all["PointsFor"] = pd.to_numeric(df_all["PointsFor"], errors="coerce")
    df_all["PointsAgainst"] = pd.to_numeric(df_all["PointsAgainst"], errors="coerce")

    # Aggregate totals + seasons
    df_summary = df_all.groupby("ManagerName").agg(
        PointsFor=("PointsFor", "sum"),
        PointsAgainst=("PointsAgainst", "sum"),
        SeasonsPlayed=("Year", "nunique")
    ).reset_index()

    # Compute averages
    df_summary["AvgPointsFor"] = df_summary["PointsFor"] / df_summary["SeasonsPlayed"]
    df_summary["AvgPointsAgainst"] = df_summary["PointsAgainst"] / df_summary["SeasonsPlayed"]

    # Add label with seasons played
    df_summary["Label"] = (
        df_summary["ManagerName"] + " (" + df_summary["SeasonsPlayed"].astype(str) + ")"
    )

    # Sort by AvgPointsFor
    df_summary = df_summary.sort_values("AvgPointsFor", ascending=False)

    # Plot grouped bar chart
    ax = df_summary.set_index("Label")[["AvgPointsFor", "AvgPointsAgainst"]].plot(
        kind="bar",
        figsize=(14, 7),
        edgecolor="black",
        color=["mediumseagreen", "indianred"]
    )

    plt.title("Average Points Per Season (with Seasons Played)", fontsize=16)
    plt.xlabel("Manager (Seasons Played)", fontsize=12)
    plt.ylabel("Avg Points Per Season", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Points")

    # Add labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=9, label_type="edge")

    plt.tight_layout()
    plt.show()



def plot_points_for_vs_against(df, year):
    """
    Plot side-by-side pie charts of Points For and Points Against for a given year.
    - Champion slice in Points For is outlined in red and labeled 'Champion'
    - Last place slice in Points Against is outlined in red and labeled 'Last Place'
    - Adds playoff rank labels under percentages
    - Shows draft position under each manager's name
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    df_year = df[df["Year"] == year].copy()
    if df_year.empty:
        print(f"No data available for year {year}")
        return

    # Aggregate just in case of duplicate rows
    df_year = df_year.groupby(["ManagerName"], as_index=False).agg({
        "PointsFor": "sum",
        "PointsAgainst": "sum",
        "PlayoffRank": "min",
        "DraftPosition": "min"   # <-- assumes DraftPosition exists
    })

    points_for = df_year.set_index("ManagerName")["PointsFor"].astype(float)
    points_against = df_year.set_index("ManagerName")["PointsAgainst"].astype(float)
    ranks = df_year.set_index("ManagerName")["PlayoffRank"].astype(int)
    draft_positions = df_year.set_index("ManagerName")["DraftPosition"].astype(int)

    champion = ranks.idxmin()
    last_place = ranks.idxmax()

    managers = ranks.sort_values().index.tolist()

    explode_for = [0.1 if m == champion else 0 for m in managers]
    explode_against = [0.1 if m == last_place else 0 for m in managers]
    colors = sns.color_palette("tab20", len(managers))

    def rank_label(rank):
        if rank == 1:
            return "Champion"
        elif rank == 2:
            return "1st Loser"
        else:
            return f"{rank}th"

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Points For
    wedges, texts, autotexts = axes[0].pie(
        [float(points_for[m]) for m in managers],
        labels=[f"{m}\nDraft Position {draft_positions[m]}\n{int(points_for[m])}" for m in managers],
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        explode=explode_for,
        colors=colors,
        wedgeprops={"edgecolor": "white"},
        textprops={"fontsize": 9}
    )
    for j, auto in enumerate(autotexts):
        m = managers[j]
        rank_val = int(ranks.loc[m])
        auto.set_text(f"{auto.get_text()}\n{rank_label(rank_val)}")
        auto.set_color("white")
        if m == champion:
            wedges[j].set_edgecolor("red")
            wedges[j].set_linewidth(2.5)
    axes[0].set_title(f"Points For – {year}", fontsize=14)

    # Points Against
    wedges, texts, autotexts = axes[1].pie(
        [float(points_against[m]) for m in managers],
        labels=[f"{m}\nDraft Position {draft_positions[m]}\n{int(points_against[m])}" for m in managers],
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        explode=explode_against,
        colors=colors,
        wedgeprops={"edgecolor": "white"},
        textprops={"fontsize": 9}
    )
    for j, auto in enumerate(autotexts):
        m = managers[j]
        rank_val = int(ranks.loc[m])
        auto.set_text(f"{auto.get_text()}\n{rank_label(rank_val)}")
        auto.set_color("white")
        if m == last_place:
            wedges[j].set_edgecolor("red")
            wedges[j].set_linewidth(2.5)
    axes[1].set_title(f"Points Against – {year}", fontsize=14)

    plt.suptitle(f"Points For vs Against – {year}", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_final4_vs_bottom4(df, current_managers=None):
    """
    Grouped bar chart comparing Final Four vs Bottom Four finishes by manager.
    Sorted by: Final Four DESC, Bottom Four ASC, Seasons Played ASC.
    """

    df["PlayoffRank"] = pd.to_numeric(df["PlayoffRank"], errors="coerce")

    # Final Four finishes
    final4_counts = df[df["PlayoffRank"] <= 4].groupby("ManagerName").size()

    # Bottom Four finishes (dynamic per year)
    bottom4_flags = []
    for year, group in df.groupby("Year"):
        max_rank = group["PlayoffRank"].max()
        bottom4_flags.extend(
            group["ManagerName"][group["PlayoffRank"] >= max_rank - 3].tolist()
        )
    bottom4_counts = pd.Series(bottom4_flags).value_counts()

    # Seasons played
    seasons_played = df.groupby("ManagerName")["Year"].nunique()

    # Combine into summary DataFrame
    summary = (
        pd.DataFrame({
            "FinalFour": final4_counts,
            "BottomFour": bottom4_counts,
            "SeasonsPlayed": seasons_played
        })
        .fillna(0)
        .astype(int)
    )

    summary["Label"] = summary.index + " (" + summary["SeasonsPlayed"].astype(str) + ")"

    # --- Sorting ---
    summary = (
        summary.sort_values(
            by=["FinalFour", "BottomFour", "SeasonsPlayed"],
            ascending=[False, True, True]
        )
        .reset_index(drop=True)
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(summary))
    width = 0.35

    ax.bar(x - width/2, summary["FinalFour"], width, label="Final Four", color="green")
    ax.bar(x + width/2, summary["BottomFour"], width, label="Bottom Four", color="purple")

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Label"], rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Final Four vs. Bottom Four Finishes by Manager")

    # Force integer y-axis ticks
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Legend at top center
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)

    plt.tight_layout()
    plt.show()



def table_lifetime_wins_losses(h2h_df, current_managers=None):
    """
    Build a lifetime win-loss summary table for each manager.
    Uses head-to-head matchup data (h2h_df).
    """
    df_wl = h2h_df.copy()

    # Normalize names
    if current_managers is not None:
        cm_norm = [cm.strip().lower() for cm in current_managers]
        df_wl = df_wl[df_wl["Owner"].str.strip().str.lower().isin(cm_norm)]

    # Derive win/loss column
    df_wl["Win"] = (df_wl["Total"] > df_wl["Opponent Total"]).astype(int)
    df_wl["Loss"] = (df_wl["Total"] < df_wl["Opponent Total"]).astype(int)

    # Aggregate wins/losses
    summary = (
        df_wl.groupby("Owner")
        .agg({"Win": "sum", "Loss": "sum", "Year": "nunique"})
        .rename(columns={"Year": "SeasonsPlayed"})
        .reset_index()
    )

    # Add Win % column
    summary["WinPct"] = (summary["Win"] / (summary["Win"] + summary["Loss"])) * 100

    # Sort: Wins desc, Losses asc, Seasons asc
    summary = summary.sort_values(
        by=["Win", "Loss", "SeasonsPlayed"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    # --- Plot table ---
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(summary)))
    ax.axis("off")

    # Create the table
    table = ax.table(
        cellText=summary.round(2).values,
        colLabels=summary.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    # Bold header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(11)
            cell.set_text_props(weight="bold")

    # --- Place title directly above the table ---
    # Get table bounding box
    fig.canvas.draw()
    bbox = table.get_window_extent(fig.canvas.get_renderer())
    bbox = bbox.transformed(fig.transFigure.inverted())

    # Add text just above the table
    fig.text(
        0.5, bbox.y1 + 0.02,  # a bit above the table
        "Lifetime Wins and Losses by Manager",
        ha="center", va="bottom", fontsize=14, weight="bold"
    )

    plt.show()

def export_charts_to_pdf(filename, chart_functions):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    with PdfPages(filename) as pdf:
        for func, args, kwargs in chart_functions:
            plt.close("all")  # avoid overlapping

            # Run plotting function
            result = func(*args, **kwargs)

            # If function returned a figure, use it. Otherwise, grab current active fig.
            if isinstance(result, plt.Figure):
                fig = result
            else:
                fig = plt.gcf()

            # Save the figure to PDF
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Metadata
        d = pdf.infodict()
        d["Title"] = "Fantasy Football League Report"
        d["Author"] = "NFL Fantasy Visualization Tool"

































