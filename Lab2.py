from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def main():
    print("===== LAB 2: World Energy Data Analysis =====")

    # -------------------------------------------------------------------------
    # Step 0: load the CSV 
    # -------------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "WorldEnergy.csv"
    if not data_path.exists():
        raise FileNotFoundError("Dataset file not found: " + str(data_path))

    df = pd.read_csv(data_path)
    print("\nStep 0 - Dataset loaded")
    print("Raw dataset shape:", df.shape)

    # Pick one emissions column for the later scatter plot (use the first name that exists)
    emissions_col = None
    if "co2" in df.columns:
        emissions_col = "co2"
    elif "co2_emissions" in df.columns:
        emissions_col = "co2_emissions"
    elif "co2_per_capita" in df.columns:
        emissions_col = "co2_per_capita"
    elif "greenhouse_gas_emissions" in df.columns:
        emissions_col = "greenhouse_gas_emissions"

    # -------------------------------------------------------------------------
    # Step 1: keep only China and Malaysia, year >= 2000, and the columns we need
    # -------------------------------------------------------------------------
    selected_countries = ["China", "Malaysia"]

    required_columns = [
        "country",
        "year",
        "population",
        "gdp",
        "primary_energy_consumption",
        "energy_cons_change_pct",
        "energy_per_capita",
        "renewables_consumption",
        "fossil_fuel_consumption",
    ]
    if emissions_col is not None:
        required_columns.append(emissions_col)

    for col in required_columns:
        if col not in df.columns:
            raise ValueError("CSV is missing this column: " + col)

    row_mask = (df["country"].isin(selected_countries)) & (df["year"] >= 2000)
    analysis_df = df.loc[row_mask, required_columns].copy()

    print("\nStep 1 - Selected data")
    print("Selected data shape:", analysis_df.shape)
    print("Countries included:", analysis_df["country"].unique().tolist())
    print("Year range:", int(analysis_df["year"].min()), "-", int(analysis_df["year"].max()))

    # -------------------------------------------------------------------------
    # Step 2: turn text into numbers, sort, remove duplicate rows, fill gaps
    # -------------------------------------------------------------------------
    for col in required_columns:
        if col not in ("country", "year"):
            analysis_df[col] = pd.to_numeric(analysis_df[col], errors="coerce")

    analysis_df = analysis_df.sort_values(["country", "year"])
    analysis_df = analysis_df.drop_duplicates(subset=["country", "year"])
    analysis_df = analysis_df.reset_index(drop=True)

    numeric_cols = [c for c in required_columns if c not in ("country", "year")]
    analysis_df[numeric_cols] = (
        analysis_df.groupby("country")[numeric_cols]
        .apply(lambda g: g.interpolate(limit_direction="both").ffill().bfill())
        .reset_index(level=0, drop=True)
    )
    analysis_df = analysis_df.dropna(subset=numeric_cols).reset_index(drop=True)

    cleaned_data_path = base_dir / "lab2_cleaned_data.csv"
    analysis_df.to_csv(cleaned_data_path, index=False)

    print("\nStep 2 - Data cleaning")
    print("Cleaned data shape:", analysis_df.shape)
    print("Missing values after cleaning:")
    print(analysis_df.isna().sum())

    # -------------------------------------------------------------------------
    # Step 3: IQR rule for outliers, then scatter population vs GDP
    # -------------------------------------------------------------------------
    q1_pop, q3_pop = analysis_df["population"].quantile([0.25, 0.75])
    q1_gdp, q3_gdp = analysis_df["gdp"].quantile([0.25, 0.75])
    iqr_pop = q3_pop - q1_pop
    iqr_gdp = q3_gdp - q1_gdp

    pop_outlier = (analysis_df["population"] < q1_pop - 1.5 * iqr_pop) | (
        analysis_df["population"] > q3_pop + 1.5 * iqr_pop
    )
    gdp_outlier = (analysis_df["gdp"] < q1_gdp - 1.5 * iqr_gdp) | (
        analysis_df["gdp"] > q3_gdp + 1.5 * iqr_gdp
    )
    analysis_df["is_outlier"] = pop_outlier | gdp_outlier

    fig, ax = plt.subplots(figsize=(9, 6))
    for country in selected_countries:
        subset = analysis_df[analysis_df["country"] == country]
        normal = subset[~subset["is_outlier"]]
        outlier = subset[subset["is_outlier"]]

        ax.scatter(normal["population"], normal["gdp"], s=35, alpha=0.75, label=country + " normal")
        if len(outlier) > 0:
            ax.scatter(
                outlier["population"],
                outlier["gdp"],
                s=90,
                marker="x",
                color="red",
                label=country + " outlier",
            )

    ax.set_title("Step 3: Population vs GDP (Outlier Check)")
    ax.set_xlabel("Population")
    ax.set_ylabel("GDP")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    outlier_scatter_path = base_dir / "lab2_step3_outlier_scatter.png"
    plt.savefig(outlier_scatter_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nStep 3 - Population/GDP outlier check")
    print("Outlier points:", int(analysis_df["is_outlier"].sum()))

    # -------------------------------------------------------------------------
    # Step 4A: three scatter plots (correlation style)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    scatter_pairs = [
        ("gdp", "primary_energy_consumption", "Primary Energy Consumption vs GDP"),
        ("gdp", "energy_per_capita", "Energy Per Capita vs GDP"),
        ("population", "primary_energy_consumption", "Primary Energy Consumption vs Population"),
    ]

    for ax, (x_col, y_col, title) in zip(axes, scatter_pairs):
        for country in selected_countries:
            subset = analysis_df[analysis_df["country"] == country]
            ax.scatter(subset[x_col], subset[y_col], s=30, alpha=0.8, label=country)
        ax.set_title(title)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_col.replace("_", " ").title())
        ax.grid(alpha=0.25)
        ax.legend()

    plt.tight_layout()
    step4_scatter_path = base_dir / "lab2_step4_correlation_scatter.png"
    plt.savefig(step4_scatter_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Step 4B: line charts over time (2 rows, 2 columns)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    time_metrics = [
        ("population", "Population Over Time"),
        ("gdp", "GDP Over Time"),
        ("primary_energy_consumption", "Primary Energy Consumption Over Time"),
        ("energy_per_capita", "Energy Per Capita Over Time"),
    ]

    for ax, (metric, title) in zip(axes.flatten(), time_metrics):
        for country in selected_countries:
            subset = analysis_df[analysis_df["country"] == country]
            ax.plot(subset["year"], subset[metric], marker="o", linewidth=1.6, label=country)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(alpha=0.25)
        ax.legend()

    plt.tight_layout()
    time_line_path = base_dir / "lab2_step4_time_lines.png"
    plt.savefig(time_line_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Step 4C: fossil vs renewables as % of primary energy (extra charts)
    # Share = (that source / primary energy) * 100
    # -------------------------------------------------------------------------
    primary = analysis_df["primary_energy_consumption"].replace(0, np.nan)
    analysis_df["fossil_share_pct"] = (analysis_df["fossil_fuel_consumption"] / primary * 100.0).clip(
        lower=0.0
    )
    analysis_df["renew_share_pct"] = (analysis_df["renewables_consumption"] / primary * 100.0).clip(
        lower=0.0
    )
    analysis_df["other_share_pct"] = (
        100.0 - analysis_df["fossil_share_pct"] - analysis_df["renew_share_pct"]
    ).clip(lower=0.0)

    # Bar chart: for each year, four bars (China fossil, China renew, Malaysia fossil, Malaysia renew)
    years_list = sorted(analysis_df["year"].unique())
    bar_width = 0.2
    # Small shifts left/right so the four bars sit next to each other
    off0 = -1.5 * bar_width
    off1 = -0.5 * bar_width
    off2 = 0.5 * bar_width
    off3 = 1.5 * bar_width

    fig, ax = plt.subplots(figsize=(14, 6))
    for yr in years_list:
        for country, x_shift_pair, dark in [
            ("China", (off0, off1), True),
            ("Malaysia", (off2, off3), False),
        ]:
            one_row = analysis_df[(analysis_df["country"] == country) & (analysis_df["year"] == yr)]
            if len(one_row) == 0:
                continue
            f_pct = float(one_row["fossil_share_pct"].iloc[0])
            r_pct = float(one_row["renew_share_pct"].iloc[0])
            fossil_color = "#4a4a4a" if dark else "#7a7a7a"
            renew_color = "#2ca02c" if dark else "#98df8a"
            ax.bar(yr + x_shift_pair[0], f_pct, width=bar_width, color=fossil_color, edgecolor="white")
            ax.bar(yr + x_shift_pair[1], r_pct, width=bar_width, color=renew_color, edgecolor="white")

    legend_elements = [
        Patch(facecolor="#4a4a4a", label="China — Fossil fuels"),
        Patch(facecolor="#2ca02c", label="China — Renewables"),
        Patch(facecolor="#7a7a7a", label="Malaysia — Fossil fuels"),
        Patch(facecolor="#98df8a", label="Malaysia — Renewables"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_title("Step 4: Fossil vs Renewables share of primary energy (bar chart)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of primary energy (%)")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fossil_renew_bar_path = base_dir / "lab2_step4_fossil_renew_bar.png"
    plt.savefig(fossil_renew_bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Stacked area: one panel per country
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors_stack = ("#4a4a4a", "#2ca02c", "#c7c7c7")
    labels_stack = ("Fossil fuels", "Renewables", "Other")

    for ax, country in zip(axes, selected_countries):
        sub = analysis_df[analysis_df["country"] == country].sort_values("year")
        fossil_vals = sub["fossil_share_pct"].values
        renew_vals = sub["renew_share_pct"].values
        other_vals = sub["other_share_pct"].values
        ax.stackplot(sub["year"], fossil_vals, renew_vals, other_vals, colors=colors_stack, alpha=0.9)
        ax.set_title(country)
        ax.set_xlabel("Year")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Share of primary energy (%)")
    fig.suptitle("Step 4: Fossil vs Renewables vs Other (stacked area)", y=1.02)
    legend_patches = [
        Patch(facecolor=colors_stack[0], edgecolor="white", label=labels_stack[0]),
        Patch(facecolor=colors_stack[1], edgecolor="white", label=labels_stack[1]),
        Patch(facecolor=colors_stack[2], edgecolor="white", label=labels_stack[2]),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout()
    fossil_renew_area_path = base_dir / "lab2_step4_fossil_renew_stacked_area.png"
    plt.savefig(fossil_renew_area_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nStep 4 - Correlation, time trends, and fossil/renewables charts done")

    # -------------------------------------------------------------------------
    # Step 5: linear fit y = slope * year + intercept, predict 2024–2028
    # np.polyfit returns [slope, intercept] for a straight line
    # -------------------------------------------------------------------------
    forecast_target = "primary_energy_consumption"
    forecast_years = [2024, 2025, 2026, 2027, 2028]
    forecast_rows = []
    # Store slope and intercept for each country (simple dictionary)
    slope_by_country = {}
    intercept_by_country = {}

    for country in selected_countries:
        country_data = analysis_df[analysis_df["country"] == country][["year", forecast_target]]
        country_data = country_data.sort_values("year").reset_index(drop=True)
        if len(country_data) < 2:
            continue

        x_years = country_data["year"].values.astype(float)
        y_energy = country_data[forecast_target].values.astype(float)
        slope, intercept = np.polyfit(x_years, y_energy, 1)
        slope_by_country[country] = float(slope)
        intercept_by_country[country] = float(intercept)

        for fy in forecast_years:
            pred = slope * fy + intercept
            forecast_rows.append(
                {
                    "country": country,
                    "year": fy,
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "predicted_" + forecast_target: float(pred),
                }
            )

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_csv_path = base_dir / "lab2_step5_forecast_primary_energy.csv"
    forecast_df.to_csv(forecast_csv_path, index=False)

    print("\nStep 5 - Linear regression and forecasts for 2024–2028")
    if len(forecast_df) > 0:
        print(forecast_df.to_string(index=False))
    for country in selected_countries:
        if country in slope_by_country:
            print(
                "  "
                + country
                + ": slope="
                + str(slope_by_country[country])
                + ", intercept="
                + str(intercept_by_country[country]),
            )

    # One plot: scatter (history), line (trend), crosses (forecast years)
    fig, ax = plt.subplots(figsize=(10, 6))
    year_min = int(analysis_df["year"].min())
    year_max_plot = max(int(analysis_df["year"].max()), 2028)
    x_line = np.arange(year_min, year_max_plot + 1, 0.5)

    for country in selected_countries:
        if country not in slope_by_country:
            continue
        slope = slope_by_country[country]
        intercept = intercept_by_country[country]

        hist = analysis_df[analysis_df["country"] == country][["year", forecast_target]]
        ax.scatter(hist["year"], hist[forecast_target], s=40, alpha=0.85, label=country + " (historical)")

        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, linewidth=2.0, alpha=0.9, label=country + " trendline")

        pred = forecast_df[forecast_df["country"] == country]
        if len(pred) > 0:
            ax.scatter(
                pred["year"],
                pred["predicted_" + forecast_target],
                s=55,
                marker="x",
                linewidths=1.5,
                label=country + " forecast 2024–2028",
            )

    ax.set_title("Step 5: Primary energy — scatter, trendline, and forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Primary energy consumption")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    forecast_plot_path = base_dir / "lab2_forecast_primary_energy.png"
    plt.savefig(forecast_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Scatter: same energy on x, emissions on y (compare the two countries)
    co2_scatter_path = None
    if emissions_col is not None:
        fig, ax = plt.subplots(figsize=(9, 6))
        for country in selected_countries:
            sub = analysis_df[analysis_df["country"] == country]
            ax.scatter(sub[forecast_target], sub[emissions_col], s=45, alpha=0.8, label=country)

        if emissions_col in ("co2", "co2_emissions"):
            y_label = "CO2 emissions (total)"
        elif emissions_col == "co2_per_capita":
            y_label = "CO2 emissions per capita"
        elif emissions_col == "greenhouse_gas_emissions":
            y_label = "Greenhouse gas emissions"
        else:
            y_label = emissions_col.replace("_", " ").title()

        ax.set_title("Step 5: Emissions vs primary energy (compare countries)")
        ax.set_xlabel("Primary energy consumption")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        co2_scatter_path = base_dir / "lab2_step5_energy_vs_co2_scatter.png"
        plt.savefig(co2_scatter_path, dpi=300, bbox_inches="tight")
        plt.close()

        if emissions_col == "greenhouse_gas_emissions":
            print(
                "\nStep 5 - Note: scatter uses greenhouse_gas_emissions "
                "because co2 columns are not in this file.",
            )
    else:
        print("\nStep 5 - No emissions scatter (no co2 / co2_per_capita / greenhouse_gas_emissions column).")

    print("\nGenerated output files:")
    output_files = [
        outlier_scatter_path,
        step4_scatter_path,
        time_line_path,
        fossil_renew_bar_path,
        fossil_renew_area_path,
        forecast_plot_path,
        cleaned_data_path,
        forecast_csv_path,
    ]
    if co2_scatter_path is not None:
        output_files.append(co2_scatter_path)
    for file_path in output_files:
        status = "OK" if file_path.exists() else "MISSING"
        print("- " + file_path.name + ": " + status)


if __name__ == "__main__":
    main()
