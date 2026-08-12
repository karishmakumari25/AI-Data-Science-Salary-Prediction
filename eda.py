import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("dataset/ai_ds_job_salaries_2026.csv")



plt.figure(figsize=(8,5))
sns.histplot(df["salary_usd"], bins=50, kde=True)
plt.title("Salary Distribution")
plt.xlabel("Salary (USD)")
plt.ylabel("Count")
plt.savefig("images/salary_distribution.png")
plt.show()



plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="years_experience",
    y="salary_usd"
)
plt.title("Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (USD)")
plt.savefig("images/experience_vs_salary.png")
plt.show()



top_jobs = (
    df.groupby("job_title")["salary_usd"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,6))
top_jobs.plot(kind="bar")
plt.title("Top 10 Highest Paying Jobs")
plt.xlabel("Job Title")
plt.ylabel("Average Salary (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/top_paying_jobs.png")
plt.show()



plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    y="salary_usd"
)
plt.title("Salary Box Plot")
plt.ylabel("Salary (USD)")
plt.savefig("images/salary_boxplot.png")
plt.show()



plt.figure(figsize=(12,8))

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.show()


print("All EDA graphs created successfully ✅")