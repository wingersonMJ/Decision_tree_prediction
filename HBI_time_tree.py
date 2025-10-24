
import numpy as np
import pandas as pd
from tableone import TableOne

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

import seaborn as sns
import matplotlib.pyplot as plt

path = "..\\Data\\HBI_time.xlsx"
df = pd.read_excel(path)
df.shape # 558

df.dropna(subset='PPCS', inplace=True, axis=0)
df.shape # 558

df.dropna(subset=["HBI_total", "time_since_injury"], inplace=True, axis=0)
df.shape # 323

df = df[df['HBI_total'] > 0]
df.shape # 276

df['PPCS'].head(20)
df['PPCS'].value_counts()
df.columns

# Create cog and som vars
cognitive = ['hbi_attention', 'hbi_distract', 'hbi_concentrate',
       'hbi_remembering', 'hbi_directions', 'hbi_daydream', 'hbi_confused',
       'hbi_forgetthings', 'hbi_finishthings', 'hbi_figurethings',
       'hbi_learnthings']

somatic = ['hbi_headache', 'hbi_dizzy', 'hbi_roomspin',
       'hbi_faint', 'hbi_blurry', 'hbi_double', 'hbi_sick', 'hbi_tired',
       'hbi_tiredeasily']

df['HBI_cognitive'] = df[cognitive].sum(axis=1)
df['HBI_somatic'] = df[somatic].sum(axis=1)


##########################
# Table one
##########################
table_columns = {
    'variables': 
        ['time_since_injury', 'sex1f', 'age', 'loc', 'conc_hx', 'time_sx', 'PPCS', 
         'add_adhd', 'ld_dyslexia', 'anxiety', 'depression', 'migraines', 'exercise_since_injury', 
         'headaches', 'headache_severity', 'current_sleep_problems', 'history_sleep_problems', 
         'HBI_total', 'HBI_cognitive', 'HBI_somatic'],
    'categorical': 
        ['sex1f', 'loc', 'conc_hx', 'PPCS', 'add_adhd', 'ld_dyslexia', 'anxiety', 'depression', 
         'migraines', 'exercise_since_injury', 'headaches', 'current_sleep_problems', 
         'history_sleep_problems'],
    'continuous': 
        ['time_since_injury', 'age', 'time_sx', 'headache_severity', 'HBI_total', 'HBI_cognitive', 'HBI_somatic']
}

overview = TableOne(
    data=df,
    columns=table_columns['variables'],
    categorical=table_columns['categorical'],
    continuous=table_columns['continuous'],
    decimals=2,
)
print(overview.tabulate(tablefmt='github'))

print(f"Range time_since_injury: {df['time_since_injury'].min()} - {df['time_since_injury'].max()}")
print(f"Range HBI_total: {df['HBI_total'].min()} - {df['HBI_total'].max()}")
print(f"Range HBI_cognitive: {df['HBI_cognitive'].min()} - {df['HBI_cognitive'].max()}")
print(f"Range HBI_somatic: {df['HBI_somatic'].min()} - {df['HBI_somatic'].max()}")

print(f"Range time_sx: {df['time_sx'].min()} - {df['time_sx'].max()}")
print(f"Mean time_sx: {df['time_sx'].mean()} | Median: {df['time_sx'].median()}")



#############################
# Log regression models
############################
base_model_log = smf.logit(formula="PPCS ~ HBI_total + time_since_injury", data = df).fit()
base_model_log.summary()
np.exp(base_model_log.params)
np.exp(base_model_log.conf_int())

int_model_log = smf.logit(formula="PPCS ~ HBI_total + time_since_injury + HBI_total:time_since_injury", data = df).fit()
int_model_log.summary()
np.exp(int_model_log.params)
np.exp(int_model_log.conf_int())

sens_model_log = smf.logit(formula="PPCS ~ HBI_total + time_since_injury + HBI_total:time_since_injury + age + sex1f + conc_hx", data = df).fit()
sens_model_log.summary()
np.exp(sens_model_log.params)
np.exp(sens_model_log.conf_int())

# Likelihood ratio test for logistic model comparisons
def likelihood_ratio_test(full, reduced):

    log_likelihood_diff = 2 * (full.llf - reduced.llf)
    degrees_freedom = full.df_model - reduced.df_model
    p_value = chi2.sf(log_likelihood_diff, degrees_freedom)

    print(f"LR stat:            {log_likelihood_diff:.4f}")
    print(f"Degrees of Freedom: {degrees_freedom}")
    print(f"p-value:            {p_value:.4f}")

likelihood_ratio_test(int_model_log, base_model_log)
likelihood_ratio_test(sens_model_log, int_model_log)
likelihood_ratio_test(sens_model_log, base_model_log)


################################
# Cross-validation accuracy
cv_value = 60
################################
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer, recall_score

sensitivity_scorer = make_scorer(recall_score, pos_label=1)
specificity_scorer = make_scorer(recall_score, pos_label=0)

y = df['PPCS']
X = df[['HBI_total', 'time_since_injury']]
X_int = X.copy()
X_int['HBI_total:time_since_injury'] = df['HBI_total'] * df['time_since_injury']

# Re-fit base model
base_log_skl = LogisticRegression(solver='lbfgs')
base_log_skl.fit(X,y)
base_log_skl.coef_
base_log_skl.intercept_

cv_base_log = cross_validate(base_log_skl, X, y, cv=cv_value, scoring={'accuracy' : 'accuracy', 'roc_auc': 'roc_auc', 'sensitivity': sensitivity_scorer, 'specificity': specificity_scorer})
print(f"Base Mean Accuracy: {cv_base_log['test_accuracy'].mean()}")
print(f"Base Mean auc:      {cv_base_log['test_roc_auc'].mean()}")
print(f"Base Mean Sensitivity:  {cv_base_log['test_sensitivity'].mean():.4f}")
print(f"Base Mean Specificity:  {cv_base_log['test_specificity'].mean():.4f}")

# Re-fit interaction model
X_int = X.copy()
X_int['HBI_total:time_since_injury'] = df['HBI_total'] * df['time_since_injury']
int_log_skl = LogisticRegression(solver='lbfgs')
int_log_skl.fit(X_int,y)
int_log_skl.coef_
int_log_skl.intercept_

cv_int_log = cross_validate(int_log_skl, X_int, y, cv=cv_value, scoring={'accuracy' : 'accuracy', 'roc_auc': 'roc_auc', 'sensitivity': sensitivity_scorer, 'specificity': specificity_scorer})
print(f"Int Mean Accuracy: {cv_int_log['test_accuracy'].mean()}")
print(f"Int Mean auc:      {cv_int_log['test_roc_auc'].mean()}")
print(f"Int Mean Sensitivity:  {cv_int_log['test_sensitivity'].mean():.4f}")
print(f"Int Mean Specificity:  {cv_int_log['test_specificity'].mean():.4f}")


# Re-fit sensitivity model
temp_df = df.copy()
temp_df = temp_df.dropna(subset=['HBI_total', 'time_since_injury', 'age', 'sex1f', 'conc_hx'])
X_sens = temp_df[['HBI_total', 'time_since_injury', 'age', 'sex1f', 'conc_hx']]
X_sens['HBI_total:time_since_injury'] = X_sens['HBI_total'] * X_sens['time_since_injury']
y_sens = temp_df['PPCS']
sens_log_skl = LogisticRegression(solver='lbfgs')
sens_log_skl.fit(X_sens,y_sens)
sens_log_skl.coef_
sens_log_skl.intercept_

cv_sens_log = cross_validate(sens_log_skl, X_sens, y_sens, cv=cv_value, scoring={'accuracy' : 'accuracy', 'roc_auc': 'roc_auc', 'sensitivity': sensitivity_scorer, 'specificity': specificity_scorer})
print(f"Sens Mean Accuracy: {cv_sens_log['test_accuracy'].mean()}")
print(f"Sens Mean auc:      {cv_sens_log['test_roc_auc'].mean()}")
print(f"Sens Mean Sensitivity:  {cv_sens_log['test_sensitivity'].mean():.4f}")
print(f"Sens Mean Specificity:  {cv_sens_log['test_specificity'].mean():.4f}")



###########################
# Tree
############################
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

tree_cols = ['HBI_total', 'time_since_injury']

X_tree = df[tree_cols]
y_tree = df['PPCS']

# Tree
tree_log = DecisionTreeClassifier(random_state=42, criterion='gini', min_samples_leaf=5)

# Cost-complexity pruning
    # Error + alpha*tree complexity
path = tree_log.cost_complexity_pruning_path(X_tree, y_tree)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Show alpha vs impurity
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.savefig("\\figs\\alpha_v_impurity")
plt.show()

# Show tree nodes and depth vs alpha
tree_building = []
for ccp_alpha in ccp_alphas:
    trees = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    trees.fit(X_tree, y_tree)
    tree_building.append(trees)

node_counts = [trees.tree_.node_count for trees in tree_building]
depth = [trees.tree_.max_depth for trees in tree_building]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()
plt.savefig("\\figs\\alpha_v_tree_size")
plt.show()

# Show tree alpha and cv score
train_scores = [trees.score(X_tree, y_tree) for trees in tree_building]
cv_scores = []
for tree in tree_building:
    scores = cross_validate(tree, X_tree, y_tree, cv=cv_value, scoring='accuracy')
    cv_scores.append(scores['test_score'].mean())

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("classification accuracy")
ax.set_title("Accuracy vs alpha for training and validation sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, cv_scores, marker="o", label="validation", drawstyle="steps-post")
ax.legend()
plt.savefig("\\figs\\Figure_1")
plt.show()

best_idx  = np.argmax(cv_scores)
print(f"Best accuracy: {max(cv_scores)}")
best_alpha = ccp_alphas[best_idx]
print(f"Best alpha:    {best_alpha}")


###########################
# Final tree w/ post-pruning

final_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha, criterion='gini', min_samples_leaf=5)
final_tree = final_tree.fit(X_tree, y_tree)
final_tree.feature_names_in_
final_tree.feature_importances_

tree_cv = cross_validate(final_tree, X_tree, y_tree, cv=cv_value, scoring={'accuracy' : 'accuracy', 'roc_auc': 'roc_auc', 'sensitivity': sensitivity_scorer, 'specificity': specificity_scorer})
print(f"Final Tree Mean Accuracy: {tree_cv['test_accuracy'].mean()}")
print(f"Final Tree Mean auc:      {tree_cv['test_roc_auc'].mean()}")
print(f"Final Tree Mean Sensitivity:  {tree_cv['test_sensitivity'].mean():.4f}")
print(f"Final Tree Mean Specificity:  {tree_cv['test_specificity'].mean():.4f}")

# Visualize
plt.figure(figsize=(12, 9))
plot_tree(final_tree, filled=True, feature_names=X_tree.columns, class_names=True, proportion=False)
plt.savefig("\\figs\\Final_tree")
plt.show()




#############################
# Deeper tree

deep_tree = DecisionTreeClassifier(random_state=42)
deep_tree = deep_tree.fit(X_tree, y_tree)

tree_cv_deep = cross_validate(deep_tree, X_tree, y_tree, cv=cv_value, scoring=['accuracy', 'roc_auc'])
print(f"Deep Tree Mean Accuracy: {tree_cv_deep['test_accuracy'].mean()}")
print(f"Deep Tree Mean auc:      {tree_cv_deep['test_roc_auc'].mean()}")

# Visualize
plt.figure(figsize=(12, 9))
plot_tree(deep_tree, filled=True, feature_names=X_tree.columns, class_names=True, proportion=False)
plt.show()





#######################
# Decision boundary plot
from sklearn.preprocessing import LabelEncoder
from mlxtend.plotting import plot_decision_regions
import matplotlib.ticker as mticker


def decision_boundary_plot(X, y, X_train, y_train, clf, feature_indexes, save_path, title=None):
    # Encode labels if needed
    if y.dtype != 'int':
        y = pd.Series(LabelEncoder().fit_transform(y))
        y_train = pd.Series(LabelEncoder().fit_transform(y_train))

    # Extract 2 selected features
    X_selected = X.iloc[:, feature_indexes].values
    X_train_selected = X_train.iloc[:, feature_indexes].values

    # Fit the classifier
    clf.fit(X_train_selected, y_train.values)

    # Plot decision regions
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X=X_selected, y=y.values, clf=clf, legend=2)
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y)}"))
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ["No PPCS", "PPCS"], loc="upper left")
    plt.xlabel("Time since injury (days)")
    plt.ylabel("HBI score")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

feature_indexes = [1,0]

decision_boundary_plot(X_tree, y_tree, X_tree, y_tree, final_tree, feature_indexes, title="Decision Boundary", save_path="\\figs\\decision_boundary")

decision_boundary_plot(X_tree, y_tree, X_tree, y_tree, final_tree, feature_indexes, title="Decision Boundary", save_path="\\figs\\decision_boundary_2")


decision_boundary_plot(X_tree, y_tree, X_tree, y_tree, deep_tree, feature_indexes, title="Decision Boundary of Deep Tree", save_path="\\figs\\decision_boundary_deep")


decision_boundary_plot(X_tree, y_tree, X_tree, y_tree, int_log_skl, feature_indexes, title="Decision Boundary of Int model", save_path="\\figs\\decision_boundary_Linear")


decision_boundary_plot(X_tree, y_tree, X_tree, y_tree, base_log_skl, feature_indexes, title="Decision Boundary of base model")


decision_boundary_plot(X_sens, y_sens, X_sens, y_sens, sens_log_skl, feature_indexes, title="Decision Boundary of sens model")




##################
# Bootstapping to get mean/median and 95% CI's for risk ratios
##################
plt.figure(figsize=(12, 9))
plot_tree(final_tree, filled=True, feature_names=X_tree.columns, class_names=True, proportion=False)
plt.show()

B = 1000 # number of bootstraps 
rng = np.random.default_rng(42)

# terminal leaves
leaf_ids_full = final_tree.apply(X_tree.values)
leaves = np.sort(np.unique(leaf_ids_full))

rows = []
n = len(y_tree)
y_arr = np.asarray(y_tree).astype(int)

for b in range(B):
    # generate random integers
    idx = rng.integers(0, n, n)\
    # use those integers to pull random x and y samples
    Xb = X_tree.values[idx]
    yb = y_arr[idx]

    # Apply tree to those random samples
    leaf_b = final_tree.apply(Xb)

    # Get overall rate of psac in random sample
    p_overall = yb.mean()
    # psac rate in each leaf
    df_b = pd.DataFrame({"leaf": leaf_b, "y": yb})
    # get summary info: % with psac, total samples, number with psac
    leaf_summary = (
        df_b.groupby("leaf", as_index=False)
            .agg(
                pct_pos=("y", "mean"),
                n_leaf=("y", "count"),
                pos_leaf=("y", "sum"),
            )
    )

    # risk ratio in each leaf vs overall risk
    leaf_summary['RR_vs_overall'] = (
        leaf_summary['pct_pos'] / p_overall if p_overall > 0 else np.nan
    )

    leaf_summary = (
        leaf_summary.set_index("leaf")
                    .reindex(leaves)
                    .reset_index()
    )

    # add col to identify which bootstrap iteration it is
    leaf_summary['bootstrap'] = b

    # adds to empy rows variable
    rows.append(leaf_summary[["bootstrap","leaf","pct_pos","RR_vs_overall"]])

# concat all the rows into one long table
boot_long = pd.concat(rows, ignore_index=True)

# get per-leaf summary
from scipy.stats import norm
def normal_ci(x, alpha=0.05):
    m = np.nanmean(x) # mean
    se = np.nanstd(x, ddof=1) # std error
    z = norm.ppf(1 - alpha/2)   # 1.96 for 95% CI
    return m, m - z*se, m + z*se
# apply it line-by-line
summary_normal = (
    boot_long.groupby("leaf")
    .agg(
        pct_pos_ci=("pct_pos", lambda x: normal_ci(x)),
        RR_ci=("RR_vs_overall", lambda x: normal_ci(x)),
    )
)

summary_normal = pd.concat(
    [summary_normal.drop(["pct_pos_ci","RR_ci"], axis=1),
     summary_normal["pct_pos_ci"].apply(pd.Series).rename(columns={0:"pct_mean",1:"pct_lo",2:"pct_hi"}),
     summary_normal["RR_ci"].apply(pd.Series).rename(columns={0:"RR_mean",1:"RR_lo",2:"RR_hi"})],
    axis=1
)

print(summary_normal)


##############
# plot per leaf
for leaf in boot_long["leaf"].unique():
    df_leaf = boot_long[boot_long["leaf"] == leaf]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Leaf {leaf} (n bootstraps={len(df_leaf)})")

    # hist of pct with psac
    axes[0].hist(df_leaf["pct_pos"].dropna(), bins=20, color="skyblue", edgecolor="black")
    axes[0].set_title("pct_pos")
    axes[0].set_xlabel("Proportion positive")
    axes[0].set_ylabel("Frequency")

    # hist of risk ratio vs overall
    axes[1].hist(df_leaf["RR_vs_overall"].dropna(), bins=20, color="salmon", edgecolor="black")
    axes[1].set_title("RR vs overall")
    axes[1].set_xlabel("Risk ratio")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()