# -*- coding: utf-8 -*-
"""
Econ 406 Module

function file

I will investigate some factors correlated to the happiness level including
life-expectancy, GDP per capita, gender equality, etc. I will use the dataset
to know possible related factors so we can know what are some actions
the government or individual can take to raise the social well being.
"""

import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# A function to import data and clean data
def import_and_clean_data():
    """
    import the dataset, clean the data by drop NA values, and use the cleaned
    dataset for other functions to preditc and model

    Returns
    -------
    the modified dataset

    """
    dataset = pd.read_csv(r"C:\Users\22209\Desktop\final project\2020.csv")
    dataset = dataset.dropna()
    return dataset

# A function to do data visualization
def data_visualization():
    """
    plot the data of each variables to the happiness level
    as scatterplot to show the correlations. see whether any obvious
    relationships exist between the variables and the happiness level.

    Returns
    -------
    None.

    """
    # scatter plots of five different factors
    dataset = import_and_clean_data()
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax1.scatter(dataset['Logged GDP per capita'], dataset['Ladder score'])
    ax2.scatter(dataset['Healthy life expectancy'], dataset['Ladder score'])
    ax3.scatter(dataset['Freedom to make life choices'],
                dataset['Ladder score'])
    ax4.scatter(dataset['Generosity'], dataset['Ladder score'])
    ax5.scatter(dataset['Perceptions of corruption'], dataset['Ladder score'])

    # heatmap of five different factors
    dataset = dataset[['Logged GDP per capita', 'Social support',
                       'Healthy life expectancy',
                       'Freedom to make life choices',
                       'Perceptions of corruption', 'Generosity',
                       'Ladder score']].corr()
    sns.heatmap(dataset, annot=True, fmt='.2f')
    plt.title("Correlation map of related factors")

    # generating pair plot for the dataset
    sns.pairplot(dataset)

# A function to run the model
def generate_stats():
    """
    use the cleaned dataset to generate descriptive stats.

    Returns
    -------
    description of the dataset

    """
    dataset = import_and_clean_data()
    return dataset.describe()

# A function generate some descriptive statistics from your data
def run_model():
    """
    choose OLS or Logistic regression model based on the regression line
    plotted above. write the formula for the model with happiness on the
    left side and the other variables on the right side. Rename the
    columns I want to use such as life-expectancy, GDP per capita,
    gender equality, etc, to predict the happiness level.


    Returns
    -------
    None.

    """
    dataset = import_and_clean_data()
    sns.lmplot(x='Logged GDP per capita', y='Ladder score',
               data=dataset)
    sns.lmplot(x='Healthy life expectancy', y='Ladder score',
               data=dataset)
    sns.lmplot(x='Freedom to make life choices', y='Ladder score',
               data=dataset)
    sns.lmplot(x='Generosity', y='Ladder score',
               data=dataset)
    sns.lmplot(x='Perceptions of corruption', y='Ladder score',
               data=dataset)

    dataset = dataset[['Ladder score',
                       'Logged GDP per capita',
                       'Healthy life expectancy',
                       'Freedom to make life choices',
                       'Generosity',
                       'Perceptions of corruption'
                       ]]
    dataset = dataset.rename(columns={
        'Ladder score': 'Happiness',
        'Logged GDP per capita': 'gdp_cap',
        'Healthy life expectancy':'life_exp',
        'Freedom to make life choices': 'free',
        'Generosity': 'gener',
        'Perceptions of corruption': 'corruption'
        })

    mod = smf.ols(formula=
                  'Happiness~life_exp+free+gdp_cap+gener+corruption',
                  data=dataset)
    res = mod.fit()
    print(res.summary())
