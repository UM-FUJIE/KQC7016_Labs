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

# 5. Step 4: Statistical assumptions + comparative summary
alpha = 0.05

# --- Statistical assumptions (for interpreting ANOVA / t-tests) ---
print("\n" + "=" * 30)
print("   Statistical Assumptions")
print("=" * 30)

# Normality test: Shapiro-Wilk on each country's renewables share
print("\nNormality test (Shapiro-Wilk), H0: data is normal")
for country in target_countries:
    x = df_clean[df_clean["country"] == country]["renewables_share_elec"]
    w_stat, p_norm = stats.shapiro(x)
    flag = "reject normality" if p_norm < alpha else "OK for normality"
    print(f"  {country}: W={w_stat:.4f}, p={p_norm:.4e} ({flag})")

# Homogeneity of variance: Levene across the three groups
lev_stat, p_levene = stats.levene(china_data, malaysia_data, germany_data)
lev_flag = "reject equal variances" if p_levene < alpha else "OK for equal variances"
print("\nHomogeneity of variance (Levene), H0: variances are equal")
print(f"  Levene statistic={lev_stat:.4f}, p={p_levene:.4e} ({lev_flag})")
print("Note: If assumptions are weak, Welch t-test (unequal variance) is still robust.")

# ANOVA conclusion
if p_value < alpha:
    print("\nConclusion (ANOVA): At alpha 0.05, reject the null hypothesis.")
    print("There is a statistically significant difference among the three countries.")
else:
    print("\nConclusion (ANOVA): At alpha 0.05, fail to reject the null hypothesis.")
    print("No statistically significant difference was found among the three countries.")

# Group stats and mean gaps
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

# 6. Step 5: Bonferroni pairwise t-tests (table)
# Three comparisons -> multiply each raw p by 3, cap at 1.0
print("\n" + "=" * 30)
print("   Pairwise t-tests (Bonferroni)")
print("=" * 30)
print("Welch t-test (equal_var=False); Bonferroni: p_adj = min(p_raw * 3, 1)")

# Pair labels: Chinese + short English (readable if console is not UTF-8)
pair_labels = [
    ("China", "Germany", "China vs Germany (China vs Germany)"),
    ("China", "Malaysia", "China vs Malaysia (China vs Malaysia)"),
    ("Germany", "Malaysia", "Germany vs Malaysia (Germany vs Malaysia)"),
]
n_pairs = len(pair_labels)
pair_rows = []
for c1, c2, pair_cn in pair_labels:
    g1 = df_clean[df_clean["country"] == c1]["renewables_share_elec"]
    g2 = df_clean[df_clean["country"] == c2]["renewables_share_elec"]
    _t_stat, p_raw = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
    p_bonf = min(p_raw * n_pairs, 1.0)
    # Cohen's d (independent samples, pooled SD)
    n1, n2 = len(g1), len(g2)
    v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled_sd = (((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) ** 0.5
    if pooled_sd > 0:
        cohen_d = (g1.mean() - g2.mean()) / pooled_sd
    else:
        cohen_d = float("nan")
    pair_rows.append(
        {
            "Pair": pair_cn,
            "p_raw": p_raw,
            "p_Bonferroni": p_bonf,
            "Cohens_d": cohen_d,
        }
    )

pairwise_df = pd.DataFrame(pair_rows)
# Nice display: scientific p-values, d rounded
display_df = pairwise_df.copy()
display_df["p_raw"] = display_df["p_raw"].map(lambda x: f"{x:.4e}")
display_df["p_Bonferroni"] = display_df["p_Bonferroni"].map(lambda x: f"{x:.4e}")
display_df["Cohens_d"] = display_df["Cohens_d"].map(lambda x: f"{x:.4f}")
print("\n", display_df.to_string(index=False))

# 7. Step 6: Visualization
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