#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:07:32 2019

@author: lennert
"""

#%%
# LJ19062019
# import statements

import numpy as np # for numerical methods, linear algebra, etc
import pandas as pd # for data wrangling
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for statistical data visualization
import statsmodels.formula.api as smf # for logistic regression
import statsmodels.api as sm # for statistical models
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# AIF360 imports
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset


# Set seed for reproducibility
np.random.seed(1995)

# NB: first 1000 lines of code were writting in Spyder 3.3.6
# Spyder 4.0.0 supports code folding, so I updated Spyder on (04/11/2019)
# If any problems may arise as a result of this update, consider reverting
# to the 3.3.6 release.

#%%

##### FUNCTIONS
def time_diff_to_float(from_date, to_date):
    """Convert date/time stamp strings into datetime types and return
    difference between dates in days (float)"""
    out = (pd.to_datetime(to_date) -  pd.to_datetime(from_date)) / pd._libs.tslibs.timedeltas.Timedelta(days = 1)
    return out

def categorical_var_dist(dataframe, variable_name):
    """Calculates the distribution of occurrences of a specified 
    categorical variable"""
    counts = dataframe[variable_name].value_counts()
    total = dataframe[variable_name].value_counts().sum()
    out = round((counts / total) * 100, 2)
    return out

def fix_column_names(dataframe):
    """Make column names of a dataframe usable for any function by making all
    letters lower case, trimming and replacing all spaces, hyphens and brackets
    with underscores"""
    
    dataframe.columns = dataframe.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('-', '_').str.replace('___', '_').str.replace('__', '_')
    
    return dataframe

def run_logit_model(dependent, independent, dataframe, intercept = True):
    """..."""
    
    # Fix names of variables and dataframe
    dependent = dependent.strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace('___', '_').replace('__', '_')
    for string in independent:
        string = string.strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace('___', '_').replace('__', '_')
    dataframe = fix_column_names(dataframe)
    
    # Create dummy variables for categorical variables and remove reference
    # category from dataframe (i.e., the most represented category)
        
    return True



def write_to_tex_table(table):
    TABLE_LOCATION = '/Users/lennertjansen/Documents/Studie/Econometrie/master/thesis/scripts/recidivism/mytable.tex'
    
    with open(TABLE_LOCATION, 'w') as tf:
        tf.write(table.to_latex())

#def create_plot(plot_type = 'hist', data, x_var, y_var, x_lab, y_lab, title):
#    return True


#%%
# =============================================================================
# GLOBALS
# =============================================================================
        
# Figure directory
        
FIGURE_DIR ='/Users/lennertjansen/Documents/Studie/Econometrie/master/thesis/scripts/figures/recidivism/'


# Figure dimensions (A4 Dimensions)
FIG_DIM = (11.7, 8.27)

#%%
# =============================================================================
# DATA PREPARATION & EXPLORATORY DATA ANALYSIS
# =============================================================================
# Data import
# General recidivism data set (compas scores and two year recidivism)
raw_df = pd.read_csv("/Users/lennertjansen/Documents/Studie/Econometrie/master/thesis/scripts/data/compas/compas-scores-two-years.csv")
cross_table_sex_race_general_raw = pd.crosstab(raw_df.sex, raw_df.race, margins = True)


# Violent recidivism dataset
raw_violent_df = pd.read_csv("/Users/lennertjansen/Documents/Studie/Econometrie/master/thesis/scripts/data/compas/compas-scores-two-years-violent.csv")
cross_table_sex_race_violent_raw = pd.crosstab(raw_violent_df.sex, raw_violent_df.race, margins = True)

# Get dimensions
size = raw_df.shape

# Summary of basic columnwise statistics (like summary() in R)
summary = raw_df.describe()

# take a look at the first few rows
raw_df.head()

# test some simple plots
raw_df['age'].plot()
raw_df['age'].hist()

raw_df['race'].apply(len)

# Correlations and patter sclots
corr_mat = raw_df.corr()

corr_heatmap = sns.heatmap(corr_mat,
            xticklabels = corr_mat.columns,
            yticklabels = corr_mat.columns)
# IDEA: make correlation maps / heatmaps of all numerical variables



# Get column names as dataframe to convert to latex table
raw_df_colnames = raw_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('-', '_').str.replace('___', '_').str.replace('__', '_')

#type(raw_df_colnames.values)
variable_descript_df = pd.DataFrame(raw_df_colnames.values,
                                    columns = ['Variable name'])
#print(variable_descript_df.to_latex())

##### DATA PREPARATION
# Methodology replication from original paper
# Based on: https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb

# Select columns of interest
# general recidivism
df = raw_df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 
             'priors_count', 'days_b_screening_arrest', 'decile_score',
             'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

# violent recidivism
df_violent = raw_violent_df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 
             'priors_count', 'days_b_screening_arrest', 'decile_score',
             'is_recid', 'two_year_recid', 'two_year_recid.1', 'c_jail_in', 'c_jail_out']]

# Filter out certain nonsensical values
# general recidivism
df = df[(df['days_b_screening_arrest'] <= 30) 
         & (df['days_b_screening_arrest'] >= -30) & (df['is_recid'] != -1) &
         (df['c_charge_degree'] != 'O') & (df['score_text'] != 'N/A')]

# violent recidivism
df_violent = df_violent[(df_violent['days_b_screening_arrest'] <= 30) 
         & (df_violent['days_b_screening_arrest'] >= -30) & (df_violent['is_recid'] != -1) &
         (df_violent['c_charge_degree'] != 'O') & (df_violent['score_text'] != 'N/A')]

df_violent.shape

# Add column with days spent in jail
# NOTE: remove negative values in this new column
df['length_of_stay'] = time_diff_to_float(df['c_jail_in'], df['c_jail_out'])

# Check for missing values
raw_df.isna().sum() 
df.isna().sum()

nrow = df.shape[0]
ncol = df.shape[1]

#%%

# =============================================================================
# DEMOGRAPHIC BREAKDOWN GANG
# =============================================================================
# Crosstables to LaTeX tables
# General
cross_table_general_sex_race = pd.crosstab(df.sex, df.race, margins = True)
write_to_tex_table(cross_table_general_sex_race)

# Violent
cross_table_violent_sex_race = pd.crosstab(df_violent.sex, df_violent.race, margins = True)
write_to_tex_table(cross_table_violent_sex_race)

# Corresponding Pie Charts
# Pie chart for general recidivism
# race
labels = list(cross_table_general_sex_race.columns)
del labels[-1] #delete last term from labels ("All")
#sizes = cross_table_general_sex_race.loc[["All"], labels].values
sizes = [3175,   31, 2103,  509,   11,  343]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'purple', 'orange']

# gender
sizes_gender = [1175, 4997]
labels_gender = ['Female', 'Male']
colors_gender = ['#c2c2f0','#ffb3e6']

# Plot
# Outer donuts                
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)

# Inner donut
plt.pie(sizes_gender,colors=colors_gender,radius=0.75,startangle=90)
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
#plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()
#plt.figure.savefig(FIGURE_DIR + 'pie_demo_breakdown_general.png')


# Corresponding Pie Charts
# Pie chart for violent recidivism
# race
labels_violent = list(cross_table_violent_sex_race.columns)
del labels_violent[-1] #delete last term from labels ("All")
#sizes_violent = cross_table_violent_sex_race.loc[["All"], labels].values
sizes_violent = [1918,   26, 1459,  355,    7,  255]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'purple', 'orange']

# gender
sizes_violent_gender = [841, 3179]
labels_gender = ['Female', 'Male']
colors_gender = ['#c2c2f0','#ffb3e6']

# Plot
# Outer donuts                
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
patches, texts = plt.pie(sizes_violent, colors=colors, shadow=True, startangle=90)

# Inner donut
plt.pie(sizes_violent_gender,colors=colors_gender,radius=0.75,startangle=90)
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.legend(patches, labels, loc="best", prop={'size': 14.5})
plt.axis('equal')
plt.tight_layout()
plt.show()

#%%
# AGE DISTRIBUTIONS

#Color palette Set2 separate colors
Set2_green_rgb = (0.4, 0.7607843137254902, 0.6470588235294118)
Set2_orange_rgb = (0.9882352941176471, 0.5529411764705883, 0.3843137254901961)

# GENERAL

# General: total
sns.set(style="darkgrid")
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
hist_gen_age_total = sns.countplot(x = "age",
                                  data = df,
                                  color = Set2_green_rgb,
              linewidth = 0.5, alpha = 0.8)
hist_gen_age_total.set_title("Total age distibution for general recidivism",
                           size = 20)
hist_gen_age_total.set_ylabel("Count", size = 15)
hist_gen_age_total.set_xlabel("Age", size = 15)
# Decrease tick-frequency. Only show every 10th tick
for ind, label in enumerate(hist_gen_age_total.get_xticklabels()):
    if ind % 10 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.tight_layout()
hist_gen_age_total.figure.savefig(FIGURE_DIR + 'hist_gen_age_total.png')


# General: Caucasian vs African-American
sns.set(style="darkgrid")
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
hist_gen_age_race = sns.countplot(x = "age",
                                  data = df[(df.race == 'African-American') | (df.race == 'Caucasian')],
                                  hue = "race", palette = "Set2",
              linewidth = 0.5, alpha = 0.9)
hist_gen_age_race.set_title("Age distribution comparison by race for general recidivism",
                           size = 20)
hist_gen_age_race.set_ylabel("Count", size = 15)
hist_gen_age_race.set_xlabel("Age", size = 15)
# Decrease tick-frequency. Only show every 10th tick
for ind, label in enumerate(hist_gen_age_race.get_xticklabels()):
    if ind % 10 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.tight_layout()
hist_gen_age_race.figure.savefig(FIGURE_DIR + 'hist_gen_age_race.png')


# General: Male vs Female
sns.set(style="darkgrid")
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
hist_gen_age_sex = sns.countplot(x = "age", data = df, hue = "sex", palette = "Set2",
              linewidth = 0.5, alpha = 0.9)
hist_gen_age_sex.set_title("Age distribution comparison by gender for general recidivism",
                           size = 20)
hist_gen_age_sex.set_ylabel("Count", size = 15)
hist_gen_age_sex.set_xlabel("Age", size = 15)
# Decrease tick-frequency. Only show every 10th tick
for ind, label in enumerate(hist_gen_age_sex.get_xticklabels()):
    if ind % 10 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.tight_layout()
hist_gen_age_sex.figure.savefig(FIGURE_DIR + 'hist_gen_age_sex.png')


# VIOLENT
# Violent: total
sns.set(style="darkgrid")
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
hist_vio_age_total = sns.countplot(x = "age",
                                  data = df_violent,
                                  color = Set2_orange_rgb,
              linewidth = 0.5, alpha = 0.8)
hist_vio_age_total.set_title("Total age distibution for violent recidivism",
                           size = 20)
hist_vio_age_total.set_ylabel("Count", size = 15)
hist_vio_age_total.set_xlabel("Age", size = 15)
# Decrease tick-frequency. Only show every 10th tick
for ind, label in enumerate(hist_vio_age_total.get_xticklabels()):
    if ind % 10 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.tight_layout()
hist_vio_age_total.figure.savefig(FIGURE_DIR + 'hist_vio_age_total.png')

# Violent: Caucasian vs African-American
sns.set(style="darkgrid")
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
hist_vio_age_race = sns.countplot(x = "age",
                                  data = df_violent[(df_violent.race == 'African-American') | (df_violent.race == 'Caucasian')],
                                  hue = "race", palette = "Set2",
              linewidth = 0.5, alpha = 0.9)
hist_vio_age_race.set_title("Age distribution comparison by race for violent recidivism",
                           size = 20)
hist_vio_age_race.set_ylabel("Count", size = 15)
hist_vio_age_race.set_xlabel("Age", size = 15)
# Decrease tick-frequency. Only show every 10th tick
for ind, label in enumerate(hist_vio_age_race.get_xticklabels()):
    if ind % 10 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.tight_layout()
hist_vio_age_race.figure.savefig(FIGURE_DIR + 'hist_vio_age_race.png')


# Violent: Male vs Female
sns.set(style="darkgrid")
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
hist_vio_age_sex = sns.countplot(x = "age", data = df_violent, hue = "sex", palette = "Set2",
              linewidth = 0.5, alpha = 0.9)
hist_vio_age_sex.set_title("Age distribution comparison by gender for violent recidivism",
                           size = 20)
hist_vio_age_sex.set_ylabel("Count", size = 15)
hist_vio_age_sex.set_xlabel("Age", size = 15)
# Decrease tick-frequency. Only show every 10th tick
for ind, label in enumerate(hist_vio_age_sex.get_xticklabels()):
    if ind % 10 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.tight_layout()
hist_vio_age_sex.figure.savefig(FIGURE_DIR + 'hist_vio_age_sex.png')

#%%
# =============================================================================
# OTHER PLOTS: e.g., DECILE SCORE DISTRIBUTION COMPARISONS
# =============================================================================

# Longer lengths of stay are slightly correlated with higher COMPAS scores
corr_coeff_length_stay_vs_decile_score = np.corrcoef(df['length_of_stay'], df['decile_score'])[1][0]

# 
# Tables of Age category breakdowns (in percentages)
df['age_cat'].value_counts()
categorical_var_dist(df, 'age_cat')

# Race
df['race'].value_counts()
categorical_var_dist(df, 'race')

# Gender
df['sex'].value_counts()
categorical_var_dist(df, 'sex')

# Risk score category
df['score_text'].value_counts()
categorical_var_dist(df, 'score_text')

# Percentage of defendants that recidivate in two years
recid_percentage = round((df[(df['two_year_recid'] == 1)].shape[0] / nrow) * 100, 2)
#print(recid_percentage, "%")

cross_table_sex_race = pd.crosstab(df.sex, df.race, margins = True)
#cross_table_sex_race.to_latex(escape = False)

write_to_tex_table(cross_table_sex_race)

df['decile_score'][(df['race'] == 'African-American')].hist(bins = 10)
df['decile_score'][(df['race'] == 'Caucasian')].hist(bins = 10)

# Decile scores for black defendants
#plt.subplot(1, 2, 1)
plt.hist(df['decile_score'][(df['race'] == 'African-American')],
         color = 'lightgreen', rwidth = 0.9)
plt.title('African-American defendant\'s decile scores')
plt.xlabel('Recidivism decile score')
plt.ylabel('Count')
plt.ylim(0, 650)
#plt.xticks(np.arange(1, 10 + 1, 1.0))
plt.xticks(df['decile_score'].unique())
plt.grid(True)
plt.tight_layout()

# Decile scores for white defendants
#plt.subplot(1, 2, 2)
plt.hist(df['decile_score'][(df['race'] == 'Caucasian')],
         color = 'lightgreen', rwidth = 0.9)
plt.title('Caucasian defendant\'s decile scores')
plt.xlabel('Recidivism decile score')
plt.ylabel('Count')
plt.ylim(0, 650)
#plt.xticks(np.arange(1, 10 + 1, 1.0))
plt.xticks(df['decile_score'].unique())
plt.grid(True)
plt.tight_layout()

# Decile scores for Asian defendants
#plt.subplot(1, 2, 2)
plt.hist(df['decile_score'][(df['race'] == 'Asian')],
         color = 'lightgreen', rwidth = 0.9)
plt.title('Distribution of decile scores')
plt.xlabel('Recidivism decile score')
plt.ylabel('Count')
plt.ylim(0, 20)
#plt.xticks(np.arange(1, 10 + 1, 1.0))
plt.xticks(df['decile_score'].unique())
plt.grid(True)
plt.tight_layout()

cross_table_score_race = pd.crosstab(df.decile_score, df.race, margins = True)

sns.countplot(x = "race", data = df, palette = 'hls')
# Histogram of decile scores among african-americans
sns.set(style="darkgrid")
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
hist_decile_black = sns.countplot(x = "decile_score", data = df[(df.race == 'African-American')],
              palette = 'Greens')
plt.ylim(0, 650)
hist_decile_black.set_title("Distribution of decile scores among black defendants")
hist_decile_black.set_ylabel("Count")
hist_decile_black.set_xlabel("Decile score")
hist_decile_black.figure.savefig(FIGURE_DIR + 'hist_decile_black.png')

# Histogram of decile scores among caucasians
sns.set(style="darkgrid")
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
hist_decile_white = sns.countplot(x = "decile_score", data = df[(df.race == 'Caucasian')],
              palette = 'Greens')
plt.ylim(0, 650)
hist_decile_white.set_title("Distribution of decile scores among white defendants")
hist_decile_white.set_ylabel("Count")
hist_decile_white.set_xlabel("Decile score")
hist_decile_white.figure.savefig(FIGURE_DIR + 'hist_decile_white.png')

# Pairplots (scatterplots and linegraphs) of comparisons between variables
# grouped by race
sns.set(style="darkgrid")
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
pairplot_race = sns.pairplot(data = df, hue = "race")
#pairplot_race.set(title = "Pairwise comparisons of variables by race")
pairplot_race.fig.savefig(FIGURE_DIR + 'pairplot_race.png')

sns.pairplot(data = df, hue = "sex")

# Histogram of decile scores among Male defendants
sns.set(style="darkgrid")
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
hist_decile_male = sns.countplot(x = "decile_score", data = df[(df.sex == 'Male')],
              palette = 'Reds')
plt.ylim(0, 1200)
hist_decile_male.set_title("Distribution of decile scores among male defendants")
hist_decile_male.set_ylabel("Count")
hist_decile_male.set_xlabel("Decile score")
hist_decile_male.figure.savefig(FIGURE_DIR + 'hist_decile_male.png')

# Histogram of decile scores among Female defendants
sns.set(style="darkgrid")
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
hist_decile_female = sns.countplot(x = "decile_score", data = df[(df.sex == 'Female')],
              palette = 'Reds')
plt.ylim(0, 1200)
hist_decile_female.set_title("Distribution of decile scores among female defendants")
hist_decile_female.set_ylabel("Count")
hist_decile_female.set_xlabel("Decile score")
hist_decile_female.figure.savefig(FIGURE_DIR + 'hist_decile_female.png')

#%%
# =============================================================================
# LOGISTIC REGRESSIONS
# =============================================================================
# =============================================================================
# MODEL 1
# =============================================================================
# Logistic regression model for comparing variables effect 
# on risk scores (low (1-4) vs Med-High) i.e., LOW vs NOT-LOW
# Main question: is there a significant difference in COMPAS scores between
# races?

# create dummy variable columns for every category of each categorical var
df_logit = pd.get_dummies(df, columns = ['c_charge_degree', 'age_cat', 
                                         'race', 'sex'])

# Create dummy outcome variable where Low or 0 corresponds to a score: 1-4
# and High or 1 corresponds to a risk score of 5-10
df_logit['score_cat'] = np.where(df_logit['score_text'] == 'Low', 0,
        df_logit['score_text'])
df_logit['score_cat'] = np.where(df_logit['score_cat'] != 0,
        1, df_logit['score_cat'])
df_logit['score_cat'] = df_logit['score_cat'].astype('int64')

# Fix column names (see function description)
df_logit = fix_column_names(df_logit)

# Drop reference categories
df_logit = df_logit.drop(['age_cat_25_45', 'race_caucasian', 'sex_male',
                          'c_charge_degree_f'], axis = 1)

logit_model =smf.Logit(df_logit['score_cat'], sm.add_constant(df_logit[['sex_female',
          'age_cat_greater_than_45', 'age_cat_less_than_25',
          'race_african_american', 'race_asian', 'race_hispanic',
          'race_native_american', 'race_other', 'priors_count',
          'c_charge_degree_m']].astype('float')))

# NB: in the ProPublica report's methodology the variable "two_year_recid" is
# included in the logit model's explanatory variables, where the task is to
# predict a defendant's COMPAS score category (high or low). 
# I'm wondering whether this is a correct model specification, as ...

logit_results = logit_model.fit()

logit_results.summary()
logit_results.params
logit_results.predict
dir(logit_results.summary())

# Print model summary as latex table
print(logit_results.summary().as_latex())

write_to_tex_table(logit_results.summary())

# print model summary as LaTeX table (WIP)
logit_results.summary().as_latex_tabular()
# alternatively
for table in logit_results.summary().tables:
    print(table.as_latex_tabular())

conf_mat_test = pd.crosstab(df_logit.two_year_recid,
                            df_logit.score_cat, margins = True)
# =============================================================================
# MODEL 2
# =============================================================================
# Logistic regression model for comparing variables effect 
# recidivism label
# Main question: is there a significant difference between recidivism
# predictions between races?
logit_model2 =smf.Logit(df_logit['two_year_recid'], sm.add_constant(df_logit[['sex_female',
          'age_cat_greater_than_45', 'age_cat_less_than_25',
          'race_african_american', 'race_asian', 'race_hispanic',
          'race_native_american', 'race_other', 'priors_count',
          'c_charge_degree_m']].astype('float')))

logit_results2 = logit_model2.fit()
logit_results2.summary()

print(logit_results2.summary().as_latex())

# TODO: add all other races together to form one group or just remove them from
# the dataset

# =============================================================================
# Logit Regression Model 3:(violent_recid)Score category as dependent variable
# =============================================================================
# create dummy variable columns for every category of each categorical var
df_violent_logit = pd.get_dummies(df_violent, columns = ['c_charge_degree', 'age_cat', 
                                         'race', 'sex'])

# Create dummy outcome variable where Low or 0 corresponds to a score: 1-4
# and High or 1 corresponds to a risk score of 5-10
df_violent_logit['score_cat'] = np.where(df_violent_logit['score_text'] == 'Low', 0,
        df_violent_logit['score_text'])
df_violent_logit['score_cat'] = np.where(df_violent_logit['score_cat'] != 0,
        1, df_violent_logit['score_cat'])
df_violent_logit['score_cat'] = df_violent_logit['score_cat'].astype('int64')

# Fix column names (see function description)
df_violent_logit = fix_column_names(df_violent_logit)

# Drop reference categories
df_violent_logit = df_violent_logit.drop(['age_cat_25_45', 'race_caucasian', 'sex_male',
                          'c_charge_degree_f'], axis = 1)
    

logit_violent_model =smf.Logit(df_violent_logit['score_cat'],
                               sm.add_constant(df_violent_logit[['sex_female',
          'age_cat_greater_than_45', 'age_cat_less_than_25',
          'race_african_american', 'race_asian', 'race_hispanic',
          'race_native_american', 'race_other', 'priors_count',
          'c_charge_degree_m']].astype('float')))


logit_violent_results = logit_violent_model.fit()



print(logit_violent_results.summary().as_latex())

# =============================================================================
# Logit model 4: violent model 2: recidivism prediction
# =============================================================================
logit_violent_model2 =smf.Logit(df_violent_logit['two_year_recid'],
                                sm.add_constant(df_violent_logit[['sex_female',
          'age_cat_greater_than_45', 'age_cat_less_than_25',
          'race_african_american', 'race_asian', 'race_hispanic',
          'race_native_american', 'race_other', 'priors_count',
          'c_charge_degree_m']].astype('float')))

logit_violent_results2 = logit_violent_model2.fit()
logit_violent_results2.summary()

dir(logit_violent_results2)
logit_violent_results2.summary2()

print(logit_violent_results2.summary().as_latex())


# =============================================================================
# STOCHASTIC JUNGLE (RANDOM FOREST)
# =============================================================================
# Create feature mat X and target vec y for random forest model
# Drop string variables and variables that (implicitly) define the target var
X_rf = df_logit.drop(['score_cat', 'c_jail_in', 'c_jail_out',
                      'score_text', 'decile_score',
                      'two_year_recid', 'is_recid'], axis = 1)
y_rf = df_logit['score_cat']

# (randomly) split data into training (70%) and test set (30%)
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf,
                                                                y_rf,
                                                                test_size = 0.3)

# Create random forest classifier model
rf_model = RandomForestClassifier(max_depth = 2, random_state = 0,
                                n_estimators = 100)

# Fit model on training set
rf_model.fit(X_rf_train, y_rf_train)

# Make predictions using test set
y_rf_pred = rf_model.predict(X_rf_test)

# Compute prediction accuracy of model
print("Accuracy:",metrics.accuracy_score(y_rf_test, y_rf_pred))

# Find the relative importance of features, scaled such that they sum to 1
feature_imp = pd.Series(rf_model.feature_importances_,
                        index = X_rf.columns).sort_values(ascending=False)
feature_imp

# Creating a bar plot
sns.set(style="darkgrid")
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
barplot_rf_feature_imp = sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Feature importance: predicting risk score category")
plt.legend()

barplot_rf_feature_imp.figure.savefig(FIGURE_DIR +'barplot_rf_feature_imp.png')

# Find features with relative importance lower than 1%
feature_imp[np.where(feature_imp < 0.01)[0]]

unimp_features = feature_imp[np.where(feature_imp < 0.01)[0]].index


# =============================================================================
# RANDOM FOREST 2: Predicting actual recidivism
# =============================================================================
X_rf2 = df_logit.drop(['score_cat', 'c_jail_in', 'c_jail_out',
                      'score_text', 'decile_score',
                      'two_year_recid', 'is_recid'], axis = 1)
y_rf2 = df_logit['two_year_recid']

X_rf2_train, X_rf2_test, y_rf2_train, y_rf2_test = train_test_split(X_rf2,
                                                                    y_rf2,
                                                                    test_size = 0.3)

rf_model2 = RandomForestClassifier(max_depth = 2,
                                   random_state = 0, n_estimators = 100)

rf_model2.fit(X_rf2_train, y_rf2_train)

y_rf2_pred = rf_model2.predict(X_rf2_test)

# Compute prediction accuracy of model
print("Accuracy:",metrics.accuracy_score(y_rf2_test, y_rf2_pred))

# Find the relative importance of features, scaled such that they sum to 1
feature_imp2 = pd.Series(rf_model2.feature_importances_,
                        index = X_rf2.columns).sort_values(ascending=False)
feature_imp2

# Creating a bar plot
sns.set(style="darkgrid")
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
barplot_rf2_feature_imp = sns.barplot(x=feature_imp2, y=feature_imp2.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Feature importance: predicting recidivism")
plt.legend()

barplot_rf2_feature_imp.figure.savefig(FIGURE_DIR +'barplot_rf2_feature_imp.png')

# =============================================================================
# CDF OF RISK SCORE PER RACE
# =============================================================================
# Analogous to CDF part of: https://fairmlbook.org/code/creditscore.html

df_by_group = pd.DataFrame(columns = ['race', 'count'])
df_by_group['race'] = df.race.unique()
df_by_group['count'] = df.race.value_counts()[df_by_group['race']].values
df_by_group = df_by_group.set_index('race')

# TODO change name from dummies_test to something more logical
dummies_test = pd.get_dummies(df, columns = ['decile_score'])

# TODO: DELETE
# =============================================================================
# df_race_decile = dummies_test[['decile_score_1',
#                               'decile_score_2',
#                               'decile_score_3',
#                               'decile_score_4',
#                               'decile_score_5',
#                               'decile_score_6',
#                               'decile_score_7',
#                               'decile_score_8',
#                               'decile_score_9',
#                               'decile_score_10']].groupby(['race']).sum()
# =============================================================================

df_race_decile = dummies_test.groupby(['race']).sum() 
df_race_decile = df_race_decile[['decile_score_1','decile_score_2',
                                 'decile_score_3','decile_score_4',
                                 'decile_score_5','decile_score_6',
                                 'decile_score_7','decile_score_8',
                                 'decile_score_9','decile_score_10']]
df_by_race = df_by_group.merge(df_race_decile, left_on = None,
                                    right_on = None, left_index = True,
                                    right_index = True)
# TODO DELETE
# =============================================================================
# # Plot CDF
# plt.plot(np.cumsum(df_by_race.loc['African-American',
#                :][1 : len(df_by_race.loc['African-American', :])] / df_by_race.loc['African-American', :][0]), 'r--')
# 
# plt.plot(np.cumsum(df_by_race.loc['Caucasian',
#                :][1 : len(df_by_race.loc['Caucasian', :])] / df_by_race.loc['Caucasian', :][0]))
# 
# plt.plot(np.cumsum(df_by_race.loc['Asian',
#                :][1 : len(df_by_race.loc['Asian', :])] / df_by_race.loc['Asian', :][0]))
# plt.xticks(range(11))
# plt.legend()
# plt.tight_layout()
# 
# np.cumsum(df_by_race.loc['African-American',
#                :][1 : len(df_by_race.loc['African-American', :])] / df_by_race.loc['African-American', :][0])
# 
# 
# 
# len(np.cumsum(df_by_race.loc['African-American', :][1 : len(df_by_race.loc['African-American', :])] / df_by_race.loc['African-American', :][0]))
# np.cumsum(df_by_race.loc['African-American', :][1 : len(df_by_race.loc['African-American', :])] / df_by_race.loc['African-American', :][0]).values
# 
# =============================================================================
# Create appropriate CDF dataframe for sns line plot argument style
df_race_cdf = pd.DataFrame({'x' : range(11)})

for race_string in df_by_race.index:
    
    dec_score_vec = df_by_race.loc[race_string, :]
    
    score_vec_len = len(dec_score_vec)
    
    dec_score_total = dec_score_vec[0]
    
    cumsum_vec = np.cumsum(dec_score_vec[1 : score_vec_len] / dec_score_total).values
    
    df_race_cdf[race_string] = np.insert(cumsum_vec, 0, 0)
    

# Create and save the plot
sns.set(style="darkgrid")
plt.figure(figsize = (FIG_DIM[0], FIG_DIM[1]))
cdf_by_race_plot = sns.lineplot(data = df_race_cdf)
cdf_by_race_plot.set_title("CDF of decile score distribution per race")
cdf_by_race_plot.set_ylabel("Probability")
cdf_by_race_plot.set_xlabel("Decile score")
cdf_by_race_plot.figure.savefig(FIGURE_DIR + 'cdf_by_race_plot.png')

# =============================================================================
# RECREATION OF PLOTS MENTIONED BY FELLER ET AL. (2016)
# =============================================================================
# Select subset of dataframe containing only black or white defendants
df_bw = df[(df['race'] == 'African-American') | (df['race'] == 'Caucasian')]

# Create dataframe for plot
df_bw_calib_plot = pd.DataFrame({'decile_score' : range(1, 11)})

# =============================================================================
# df_bw[(df.decile_score == 10) & (df.race == 'Caucasian')][['two_year_recid']].sum(axis = 0)
# 
# df_bw.where((df.race == 'African-American') & (df.decile_score == 5))[['two_year_recid']].sum(axis = 0)
# =============================================================================
index = 1
for i in ['African-American', 'Caucasian']:
    # Set filter condition for race
    #filter_race = df_bw.race == i
    
    # Create empty column in plot's dataframe for race-specific recidivism rate
    df_bw_calib_plot['recid_rate_' + i] = np.nan
    
    df_bw_calib_plot = fix_column_names(df_bw_calib_plot)
    
    col_name = df_bw_calib_plot.columns[index]
    index = index + 1
    
    for j in range(1, 11):
        # Set filter condition for score
        #filter_score = df_bw.decile_score == j
        #i = 'African-American'
        #j = 7
        # Subset dataframe for race and decile score
        #sub_df_bw = df_bw.where((df.race == i) & (df.decile_score == j))
        sub_df_bw = df_bw[(df_bw.race == i) & (df_bw.decile_score == j)]
        row_total = sub_df_bw.shape[0]
        row_recid = sub_df_bw[['two_year_recid']].sum(axis = 0)
        
        value = row_recid / row_total
        df_bw_calib_plot.loc[j - 1, col_name] = value.values
        
        

sns.distplot(df_bw_calib_plot['recid_rate_african_american'], x = df_bw_calib_plot.decile_score, hist = True, kde = False, label='Black')
sns.distplot(df_bw_calib_plot['recid_rate_caucasian'], x = df_bw_calib_plot.decile_score, hist = True, kde = False, label='White')
        
df_bw_calib_plot['recid_rate_african_american'].hist(bins = 10)

df_bw_calib_plot.rename(columns = {'decile_score' : 'x'})

sns.lineplot(data = df_bw_calib_plot['recid_rate_african_american'], label = 'Black')
sns.lineplot(data = df_bw_calib_plot['recid_rate_caucasian'], label = 'White')
plt.ylim(0, 1)

sns.lineplot(data = df_bw_calib_plot['recid_rate_male'], label = 'Male')
sns.lineplot(data = df_bw_calib_plot['recid_rate_female'], label = 'Female')
plt.ylim(0, 1)

sns.distplot(df_bw_calib_plot['recid_rate_african_american'], kde = False)
sns.distplot(df_bw_calib_plot['recid_rate_caucasian'], kde = False)

# =============================================================================
# Histogram
# =============================================================================
plt.hist(df_bw_calib_plot['recid_rate_african_american'], 10)

plt.hist(df['decile_score'][(df['race'] == 'African-American') & (df['two_year_recid'] == 1)] ,
         color = 'lightgreen', rwidth = 0.9)
plt.hist(df['decile_score'][(df['race'] == 'Caucasian') & (df['two_year_recid'] == 1)],
         color = 'lightgreen', rwidth = 0.9)
plt.title('African-American defendant\'s decile scores')
plt.xlabel('Recidivism decile score')
plt.ylabel('Count')
plt.ylim(0, 650)
#plt.xticks(np.arange(1, 10 + 1, 1.0))
plt.xticks(df['decile_score'].unique())
plt.grid(True)
plt.tight_layout()


plt.bar(x = df_bw_calib_plot.decile_score,
        height = df_bw_calib_plot.recid_rate_african_american)
plt.bar(df_bw_calib_plot.decile_score,
        df_bw_calib_plot.recid_rate_caucasian)
plt.bar(df_bw_calib_plot.decile_score, df_bw_calib_plot.recid_rate_male)
plt.bar(df_bw_calib_plot.decile_score, df_bw_calib_plot.recid_rate_female)
# =============================================================================
# Recreation of Feller et al's plot but for gender bias
# =============================================================================
index = 3
for i in ['Female', 'Male']:
    # Set filter condition for race
    #filter_race = df_bw.race == i
    
    # Create empty column in plot's dataframe for gender-specific recidivism rate
    df_bw_calib_plot['recid_rate_' + i] = np.nan
    
    df_bw_calib_plot = fix_column_names(df_bw_calib_plot)
    
    col_name = df_bw_calib_plot.columns[index]
    index = index + 1
    
    for j in range(1, 11):
        # Set filter condition for score
        #filter_score = df_bw.decile_score == j
        #i = 'African-American'
        #j = 7
        # Subset dataframe for race and decile score
        #sub_df_bw = df_bw.where((df.race == i) & (df.decile_score == j))
        sub_df = df[(df.sex == i) & (df.decile_score == j)]
        row_total = sub_df.shape[0]
        row_recid = sub_df[['two_year_recid']].sum(axis = 0)
        
        value = row_recid / row_total
        df_bw_calib_plot.loc[j - 1, col_name] = value.values

# =============================================================================
# CLASSIFICATION: FAIRNESS METRICS, BMA's, etc
# =============================================================================
# Copy dataframe for section-specific manipulations (think of a better sol.)
df_fm = df

# Take subset of dataframe containing only black OR white defendants
df_fm = df_fm[(df_fm.race == 'African-American') |  (df_fm.race == 'Caucasian')]

# Create dummy outcome variable where Low or 0 corresponds to a score: 1-4
# and High or 1 corresponds to a risk score of 5-10
# NB: this is similar to R's ifelse function
df_fm['score_cat'] = np.where(df_fm['score_text'] == 'Low', 0, 1)

# TO DELETE
# =============================================================================
# df_fm['score_cat'] = np.where(df_fm['score_text'] == 'Low', 0,
#         df_fm['score_text'])
# df_fm['score_cat'] = np.where(df_fm['score_cat'] != 0,
#         1, df_fm['score_cat'])
# =============================================================================

# Convert protected attributes to numerical values
# Note the following convention: 1 = undesirable / unprivileged
# and 0 denotes desirable / privileged
df_fm['race'] = np.where(df_fm['race'] == 'African-American', 1, 0)
df_fm['sex'] = np.where(df_fm['sex'] == 'Female', 1, 0)

# Drop non-numerical variables
df_fm = df_fm.select_dtypes(['number'])

# Attribute / label mapping
label_map = {1.0 : 'Did recid', 0.0 : 'Did not recid'}

protected_attribute_maps = [{1.0 : 'Female', 0.0 : 'Male'},
                            {1.0 : 'African-American', 0.0 : 'Caucasian'}]

df_fm = df_fm[['two_year_recid','race', 'score_cat']]
df_fm_true = df_fm[['race', 'two_year_recid']]
df_fm_true.rename(columns = {'two_year_recid' : 'labels'}, inplace = True)
df_fm_pred = df_fm_true.copy()
df_fm_pred.labels = df_fm[['score_cat']]

# Creating the binary label dataset with ground truth column and race
bld_true = BinaryLabelDataset(favorable_label = 0,
                              unfavorable_label = 1,
                              df = df_fm_true,
                              label_names = ['labels'],
                              protected_attribute_names = ['race'])

# Creating the binary label dataset with predicted outcome column and race
bld_pred = BinaryLabelDataset(favorable_label = 0,
                              unfavorable_label = 1,
                              df = df_fm_pred, 
                              label_names = ['labels'],
                              protected_attribute_names = ['race'])

# Mapping the groups defined by the sensitive attribute
unprivileged_map = [{'race' : 1}]
privileged_map = [{'race' : 0}]

# Compute the classification metrics for the black vs white comparison
fairness_metrics = ClassificationMetric(dataset = bld_true,
                     classified_dataset = bld_pred,
                     unprivileged_groups = unprivileged_map,
                     privileged_groups = privileged_map)

# Confusion matrices for total pop, whites, blacks
conf_mat_total = fairness_metrics.binary_confusion_matrix(privileged = None)
conf_mat_white = fairness_metrics.binary_confusion_matrix(privileged = True)
conf_mat_black = fairness_metrics.binary_confusion_matrix(privileged = False)

for i, j in conf_mat_total.items():
    print('The number of ' + i + 's is ' + str(j))

type(conf_mat_total)
conf_mat_total.keys()
conf_mat_total.values()
conf_mat_keys_ordered = ['TN', 'FP', 'FN', 'TP']
conf_mat_values_ordered = []

for i in range(len(conf_mat_keys_ordered)):
    print(i)
    conf_mat_values_ordered[i] = conf_mat_total.get(conf_mat_keys_ordered[i])




# Convert the confusion matrices to dataframes
df_cm_total = pd.DataFrame(np.asarray(conf_mat_values_ordered).reshape(2,2), 
                           columns = ['low_risk', 'high_risk'], 
                           index = ['did_not_recid', 'did_recid'])

# Add column of sum per row
df_cm_total['Total'] = df_cm_total.sum(axis = 1)

# Add row of sum per column
df_cm_total.loc['Total'] = df_cm_total.sum(axis = 0)

# Convert confusion matrix cells to rates
df_cm_rates_total = df_cm_total.copy()
df_cm_rates_total.iloc[0,[0,1]] = df_cm_rates_total.iloc[0,[0,1]] / df_cm_rates_total.Total[0]
df_cm_rates_total.iloc[1, [0, 1]] = df_cm_rates_total.iloc[1, [0, 1]] / df_cm_rates_total.Total[1]

df_cm_rates_total.iloc[2, [0, 1]] = df_cm_rates_total.iloc[2, [0, 1]] / df_cm_rates_total.Total[2]
df_cm_rates_total.iloc[:, 2] = df_cm_rates_total.iloc[:, 2] / df_cm_rates_total.loc['Total'][2]

def confusion_matrix_to_dataframe(cm_dict, rates = False):
    """Takes a confusion matrix dictionary as argument and returns a 2 by 2 
    confusion matrix dataframe"""
    
    # Desired order of error rate keys
    cm_dict_keys_ordered = ['TN', 'FP', 'FN', 'TP']
    
    # Empty variable in which the ordered values will be stored
    cm_dict_values_ordered = []

    # Iteratively fill empty array with ordered values of error rates
    for i in range(len(cm_dict_keys_ordered)):
        print(i)
        cm_dict_values_ordered[i] = cm_dict.get(cm_dict_keys_ordered[i])
        
    # Convert the confusion matrices to dataframes
    cm_df = pd.DataFrame(np.asarray(cm_dict_values_ordered).reshape(2,2), 
                           columns = ['low_risk', 'high_risk'], 
                           index = ['did_not_recid', 'did_recid'])
    
    # Add column of sum per row
    cm_df['Total'] = cm_df.sum(axis = 1)

    # Add row of sum per column
    cm_df.loc['Total'] = cm_df.sum(axis = 0)
    
    return cm_df

df_test = confusion_matrix_to_dataframe(cm_dict = conf_mat_total)
        
# =============================================================================
# FAIRNESS METRICS
# =============================================================================
# Statistical Parity difference (SPD)
spd_pre_race = fairness_metrics.statistical_parity_difference()

# Disparate Impact Ratio
dir_pre_race = fairness_metrics.disparate_impact()

# Average Odds Difference and Average absolute odds difference
aod_pre_race = fairness_metrics.average_odds_difference()
aaod_pre_race = fairness_metrics.average_abs_odds_difference()

# Equal Opportunity Difference aka true positive rate difference
eod_pre_race = fairness_metrics.equal_opportunity_difference()


# Generealized entropy index with various alpha's
fairness_metrics.between_all_groups_generalized_entropy_index(alpha = 2)


ClassificationMetric(dataset = bld_true, classified_dataset = bld_pred,
                     unprivileged_groups = None,
                     privileged_groups = None).false_positive_rate()

df_fm.head()

# TO DELETE
# =============================================================================
# bld_pred.align_datasets
# bld_true.temporarily_ignore('score_cat')
# =============================================================================
# TO DELETE: NOT YET NECESSARY
# =============================================================================
# # Convert dataframe to structured dataset
# sd = StructuredDataset(df = df_fm,
#                        label_names = ['two_year_recid', 'score_cat'],
#                        protected_attribute_names = ['race', 'sex'])
#                        #unprivileged_protected_attributes = ['African-American', 'Female'],
#                        #privileged_protected_attributes = ['Caucasian', 'Male'])
# =============================================================================
# Create a subset of the dataframe with only the bare necessities
# (ground truth, prediction, race)