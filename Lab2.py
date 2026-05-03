from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    print("===== LAB 2: World Energy Data Analysis =====")

    # Step 0: Load dataset.
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "WorldEnergy.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    df = pd.read_csv(data_path)
    print("\nStep 0 - Dataset loaded")
    print("Raw dataset shape:", df.shape)

    # Step 1: Select required scope and variables.
    selected_countries = ["China", "Malaysia"]
    required_columns = [
        "country",
        "year",
        "population",
        "gdp",
        "primary_energy_consumption",
        "energy_cons_change_pct",
        "energy_per_capita",
    ]
    analysis_df = df.loc[
        (df["country"].isin(selected_countries)) & (df["year"] >= 2000),
        required_columns,
    ].copy()
    print("\nStep 1 - Selected data")
    print("Selected data shape:", analysis_df.shape)
    print("Countries included:", analysis_df["country"].unique().tolist())
    print("Year range:", int(analysis_df["year"].min()), "-", int(analysis_df["year"].max()))

    # Step 2: Feature selection and data cleaning.
    for col in required_columns:
        if col not in {"country", "year"}:
            analysis_df[col] = pd.to_numeric(analysis_df[col], errors="coerce")

    analysis_df = (
        analysis_df.sort_values(["country", "year"])
        .drop_duplicates(subset=["country", "year"])
        .reset_index(drop=True)
    )

    # Fill missing values for numeric features inside each country.
    numeric_feature_cols = [c for c in required_columns if c not in {"country", "year"}]
    analysis_df[numeric_feature_cols] = (
        analysis_df.groupby("country")[numeric_feature_cols]
        .apply(lambda group: group.interpolate(limit_direction="both").ffill().bfill())
        .reset_index(level=0, drop=True)
    )
    analysis_df = analysis_df.dropna(subset=numeric_feature_cols).reset_index(drop=True)

    cleaned_data_path = base_dir / "lab2_cleaned_data.csv"
    analysis_df.to_csv(cleaned_data_path, index=False)

    print("\nStep 2 - Data cleaning")
    print("Cleaned data shape:", analysis_df.shape)
    print("Missing values after cleaning:")
    print(analysis_df.isna().sum())

    # Step 3: Outlier scatter (population vs GDP) for data reliability check.
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

        ax.scatter(normal["population"], normal["gdp"], s=35, alpha=0.75, label=f"{country} normal")
        if not outlier.empty:
            ax.scatter(
                outlier["population"],
                outlier["gdp"],
                s=90,
                marker="x",
                color="red",
                label=f"{country} outlier",
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

    # Step 4A: Correlation scatter charts.
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

    # Step 4B: Time trend charts (2x2 layout).
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

    print("\nStep 4 - Correlation and time trend charts completed")

    # Step 5: Forecast next five years (simple average yearly change).
    forecast_target = "primary_energy_consumption"
    forecast_rows = []

    for country in selected_countries:
        country_data = (
            analysis_df[analysis_df["country"] == country][["year", forecast_target]]
            .sort_values("year")
            .reset_index(drop=True)
        )
        if len(country_data) < 2:
            continue

        # Basic method: use average annual change from historical data.
        yearly_change = country_data[forecast_target].diff().dropna()
        avg_change = yearly_change.mean()
        last_year = int(country_data["year"].iloc[-1])
        last_value = float(country_data[forecast_target].iloc[-1])

        for step in range(1, 6):
            future_year = last_year + step
            predicted_value = last_value + avg_change * step
            forecast_rows.append(
                {
                    "country": country,
                    "year": future_year,
                    f"predicted_{forecast_target}": float(predicted_value),
                }
            )

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_csv_path = base_dir / "lab2_step5_forecast_primary_energy.csv"
    forecast_df.to_csv(forecast_csv_path, index=False)

    print("\nStep 5 - Forecasting for the next five years")
    print(forecast_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    for country in selected_countries:
        hist = analysis_df[analysis_df["country"] == country][["year", forecast_target]]
        pred = forecast_df[forecast_df["country"] == country][
            ["year", f"predicted_{forecast_target}"]
        ]
        ax.plot(
            hist["year"],
            hist[forecast_target],
            marker="o",
            label=f"{country} historical",
        )
        ax.plot(
            pred["year"],
            pred[f"predicted_{forecast_target}"],
            marker="x",
            linestyle="--",
            label=f"{country} forecast",
        )

    ax.set_title("Historical and Forecasted Primary Energy Consumption")
    ax.set_xlabel("Year")
    ax.set_ylabel("Primary Energy Consumption")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    forecast_plot_path = base_dir / "lab2_forecast_primary_energy.png"
    plt.savefig(forecast_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nGenerated output files:")
    output_files = [
        outlier_scatter_path,
        step4_scatter_path,
        time_line_path,
        forecast_plot_path,
        cleaned_data_path,
        forecast_csv_path,
    ]
    for file_path in output_files:
        status = "OK" if file_path.exists() else "MISSING"
        print(f"- {file_path.name}: {status}")


if __name__ == "__main__":
    main()