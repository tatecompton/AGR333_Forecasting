import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# Load and prepare data
wasde_df = pd.read_csv("WASDE.csv")
wasde_df["SUR"] = wasde_df["end_stocks"] / wasde_df["total_use"]

# 1. Scatter plot with linear regression
sns.regplot(x="SUR", y="corn_price", data=wasde_df)
plt.title("Corn Price vs SUR")
plt.xlabel("Stock-to-Use Ratio")
plt.ylabel("Corn Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("lab14_scatter.png")
plt.clf()

# 2. Linear regression
linear_model = smf.ols("corn_price ~ SUR", data=wasde_df).fit()
print("Linear Regression Summary:")
print(linear_model.summary())

# 3. Non-linear regression with 1/SUR
nonlinear_model = smf.ols("corn_price ~ I(1/SUR)", data=wasde_df).fit()
print("Non-Linear Regression Summary:")
print(nonlinear_model.summary())

# 4. Regression by time period
wasde_df["period"] = wasde_df["year"].apply(lambda x: "1973-2005" if x <= 2005 else "2006-2019")
sns.lmplot(x="SUR", y="corn_price", hue="period", data=wasde_df)
plt.title("Corn Price vs SUR by Period")
plt.xlabel("Stock-to-Use Ratio")
plt.ylabel("Corn Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("lab14_periods.png")
plt.clf()

# 5. Interaction model
wasde_df["P2006"] = (wasde_df["year"] > 2005).astype(int)
interaction_model = smf.ols("corn_price ~ SUR + P2006 + SUR:P2006", data=wasde_df).fit()
print("Interaction Model Summary:")
print(interaction_model.summary())