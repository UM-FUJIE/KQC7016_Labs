import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from itertools import combinations

# 1. Load data
print("Loading WorldEnergy dataset...")
base_dir = Path(__file__).resolve().parent
candidate_files = sorted(base_dir.glob("WorldEnergy*.csv"))
if not candidate_files:
    raise FileNotFoundError("WorldEnergy dataset file was not found in the script directory.")
data_file = candidate_files[0]
print(f"Using data file: {data_file.name}")
df = pd.read_csv(data_file)

# 2. Step 1: Data selection
# Select China, Malaysia, and Germany from year 2010 onward.
target_countries = ["China", "Malaysia", "Germany"]
df_lab3 = df[(df["country"].isin(target_countries)) & (df["year"] >= 2010)].copy()

# 3. Step 2: Feature selection and cleaning
# Core metric: renewables share of electricity.
df_clean = df_lab3[["country", "year", "renewables_share_elec"]].dropna()

print(f"Data cleaning completed with {len(df_clean)} valid records.")

# 4. Step 3: Run ANOVA test
china_data = df_clean[df_clean["country"] == "China"]["renewables_share_elec"]
malaysia_data = df_clean[df_clean["country"] == "Malaysia"]["renewables_share_elec"]
germany_data = df_clean[df_clean["country"] == "Germany"]["renewables_share_elec"]

# One way ANOVA
f_stat, p_value = stats.f_oneway(china_data, malaysia_data, germany_data)

print("\n" + "="*30)
print("        ANOVA Results")
print("="*30)
print(f"F-Statistic: {f_stat:.4f}")
print(f"P-Value:     {p_value:.4e}")

# 5. Step 4: Comparative analysis
alpha = 0.05
if p_value < alpha:
    print("\nConclusion: At alpha 0.05, reject the null hypothesis.")
    print("There is a statistically significant difference among the three countries.")
else:
    print("\nConclusion: At alpha 0.05, fail to reject the null hypothesis.")
    print("No statistically significant difference was found among the three countries.")

# Difference analysis: group stats, mean gaps, pairwise tests
group_stats = (
    df_clean.groupby("country")["renewables_share_elec"]
    .agg(["mean", "std", "count"])
    .sort_values("mean", ascending=False)
)
print("\n" + "=" * 30)
print("   Group Statistics and Gaps")
print("=" * 30)
print(group_stats.round(3).to_string())

print("\nPairwise mean differences:")
for c1, c2 in combinations(group_stats.index.tolist(), 2):
    mean_diff = group_stats.loc[c1, "mean"] - group_stats.loc[c2, "mean"]
    print(f"{c1} - {c2}: {mean_diff:.3f}")

print("\nPairwise Welch t-test with Bonferroni correction:")
pair_count = 3
for c1, c2 in combinations(target_countries, 2):
    g1 = df_clean[df_clean["country"] == c1]["renewables_share_elec"]
    g2 = df_clean[df_clean["country"] == c2]["renewables_share_elec"]
    t_stat, p_pair = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
    p_adjusted = min(p_pair * pair_count, 1.0)
    pooled_sd = ((g1.std(ddof=1) ** 2 + g2.std(ddof=1) ** 2) / 2) ** 0.5
    cohen_d = (g1.mean() - g2.mean()) / pooled_sd if pooled_sd > 0 else float("nan")
    print(
        f"{c1} vs {c2} -> t={t_stat:.3f}, raw p={p_pair:.4e}, "
        f"adj p={p_adjusted:.4e}, Cohen's d={cohen_d:.3f}"
    )

# 6. Step 5: Visualization
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Boxplot plus points for group comparison
sns.boxplot(
    data=df_clean,
    x="country",
    y="renewables_share_elec",
    hue="country",
    palette="Set2",
    legend=False,
    ax=axes[0],
)
sns.stripplot(
    data=df_clean, x="country", y="renewables_share_elec", color=".3", alpha=0.5, ax=axes[0]
)
axes[0].set_title("ANOVA Comparison from 2010 onward")
axes[0].set_ylabel("Renewables Share Percent")
axes[0].set_xlabel("Country")

# Plot 2: Yearly trend
sns.lineplot(
    data=df_clean,
    x="year",
    y="renewables_share_elec",
    hue="country",
    marker="o",
    ax=axes[1],
)
axes[1].set_title("Yearly Trend from 2010 onward")
axes[1].set_ylabel("Renewables Share Percent")
axes[1].set_xlabel("Year")
axes[1].legend(title="Country")

plt.tight_layout()
output_plot = base_dir / "anova_renewables_analysis.png"
plt.savefig(output_plot, dpi=300, bbox_inches="tight")
print(f"\nFigure saved: {output_plot.name}")
plt.show()