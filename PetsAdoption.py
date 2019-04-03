# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 07:16:26 2019
Pets Adoption
@author: ahmed
"""
# libraries
import csv
import pandas as pd
import os
import gc
import time
import eli5
import json
import random
import warnings
warnings.filterwarnings("ignore")
import datetime


import numpy as np
import scipy as sp 
import seaborn as sns 
import lightgbm as lgb
import xgboost as xgb
import plotly.offline as py
import plotly.tools as tls
import plotly.plotly as ppt
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import matplotlib as matplotlib

from numpy import *
from os import path
from math import sqrt
from PIL import Image
plt.style.use('ggplot')
from matplotlib import rc
from pandas import DataFrame
from functools import partial
from tqdm import tqdm_notebook 
#ppt.init_notebook_mode(connected=True)
from collections import Counter
from wordcloud import WordCloud
from IPython.display import display 
from catboost import CatBoostClassifier
from nltk.tokenize import TweetTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 100)
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
# Reading Data    
breeds = pd.read_csv('breed_labels.csv', delimiter=',')
colors = pd.read_csv('color_labels.csv', delimiter=',')
states = pd.read_csv('state_labels.csv', delimiter=',')
train = pd.read_csv('train.csv', delimiter=',')
test = pd.read_csv('test.csv', delimiter=',')
sub = pd.read_csv('sample_submission.csv', delimiter=',')
train['dataset_type'] = 'train'
test['dataset_type'] = 'test'
all_data = pd.concat([train, test],sort=True)

# Define a new variable (# of words in the description)
no_of_words = train.Description.str.count(' ')+1
no_of_words.fillna(0, inplace=True)
no_of_words = no_of_words.astype(np.int64)
train['no_of_words'] = no_of_words

# Organize pets with names and without
word_list = train.Name
train.loc[train.Name == 'No Name Yet', 'Name']='No Name'
train.loc[train.Name == NaN, 'Name']='No Name'
train.Name=train['Name'].fillna('No Name')

# Adoption speed classes counts
train['AdoptionSpeed'].value_counts().sort_index().plot('barh', color='teal');
plt.title('Adoption speed classes counts');

# Count Names
list1=train.Name[train.Gender==1]
counts = Counter(list1)
print(counts)

# Adoption speed classes rates
plt.figure(figsize=(14, 6));
g = sns.countplot(x='AdoptionSpeed', data=all_data.loc[all_data['dataset_type'] == 'train'])
plt.title('Adoption speed classes rates');
ax=g.axes #annotate axis = seaborn axis
for p in ax.patches:
    ax.annotate(f"{p.get_height() / train.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
         textcoords='offset points') 

# Number of cats and dogs in train and test data
all_data['Type'] = all_data['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')
plt.figure(figsize=(10, 6));
sns.countplot(x='dataset_type', data=all_data, hue='Type');
plt.title('Number of cats and dogs in train and test data');

# Reading the Faster Adopted Dogs' Names
list1=train.Name[train.Gender==1][train.AdoptionSpeed==0]
counts = Counter(list1)
print(counts.most_common(10))

# First 10 names of Dogs
word1 = [word1 for word1,cnt in counts.most_common(10)]
word1

# Reading the Faster Adopted Cats' Names
list2=train.Name[train.Gender==2]
counts = Counter(list2)
print(counts)

# First 10 names of Cats
word2 = [word2 for word2,cnt in counts.most_common(10)]
word2

# Adding "Best Names" variables to the data set
words = (word1+word2)
train['bestname']=0
bestname=train['bestname']
for i in range(1,len(words)):
    train.loc[train.Name == words[i], 'bestname']=1
    train.loc[train.Name != words[i], 'bestname']=0
    train.fillna(0, inplace=True)
train.drop('Description', axis=1).head()

# Cats' Names Figure
file = open("Cat_Names.txt","w") 
nam = str(train.Name[train.Gender==2])
file.write(nam)
file.close()

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# Read the whole text.

text = open(path.join(d, 'Cat_Names.txt')).read()

# read the mask / color image taken from
# http://jirkavinse.deviantart.com/art/quot-Real-Life-quot-Alice-282261010
alice_coloring = np.array(Image.open(path.join(d, "Cat_image.jpg")))
stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=2000, mask=alice_coloring,
               stopwords=stopwords, max_font_size=130, random_state=42, contour_width=1, contour_color='steelblue')
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(alice_coloring)

# show
fig, axes = plt.subplots(3, 1)
axes[0].imshow(wc, interpolation="bilinear")
# recolor wordcloud and show
# we could also give color_func=image_colors directly in the constructor
axes[1].imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
axes[2].imshow(alice_coloring, cmap=plt.cm.gray, interpolation="bilinear")
for ax in axes:
    ax.set_axis_off()
matplotlib.rcParams['figure.figsize'] = [130, 50]
plt.show()

# Dogs' Names Figure
file = open("Dog_Names.txt","w") 
nam = str(train.Name[train.Gender==1])
file.write(nam)
file.close()

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# Read the whole text.

text = open(path.join(d, 'Dog_Names.txt')).read()

# read the mask / color image taken from
# http://jirkavinse.deviantart.com/art/quot-Real-Life-quot-Alice-282261010
alice_coloring = np.array(Image.open(path.join(d, "Dog_image.jpg")))
stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=2000, mask=alice_coloring,
               stopwords=stopwords, max_font_size=130, random_state=42, contour_width=1, contour_color='steelblue')
# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(alice_coloring)

# show
fig, axes = plt.subplots(3, 1)
axes[0].imshow(wc, interpolation="bilinear")
# recolor wordcloud and show
# we could also give color_func=image_colors directly in the constructor
axes[1].imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
axes[2].imshow(alice_coloring, cmap=plt.cm.gray, interpolation="bilinear")
for ax in axes:
    ax.set_axis_off()
matplotlib.rcParams['figure.figsize'] = [130, 50]
plt.show()

# Most popular pet names and AdoptionSpeed
print('Most popular pet names and AdoptionSpeed')
for n in train['Name'].value_counts().index[:5]:
    print(n)
    print(train.loc[train['Name'] == n, 'AdoptionSpeed'].value_counts().sort_index())
    print('')

# Calculate the no names percentage
train['Name'] = train['Name'].fillna('Unnamed')
test['Name'] = test['Name'].fillna('Unnamed')
all_data['Name'] = all_data['Name'].fillna('Unnamed')

train['No_name'] = 0
train.loc[train['Name'] == 'Unnamed', 'No_name'] = 1
test['No_name'] = 0
test.loc[test['Name'] == 'Unnamed', 'No_name'] = 1
all_data['No_name'] = 0
all_data.loc[all_data['Name'] == 'Unnamed', 'No_name'] = 1

print(f"Rate of unnamed pets in train data: {train['No_name'].sum() * 100 / train['No_name'].shape[0]:.4f}%.")
print(f"Rate of unnamed pets in test data: {test['No_name'].sum() * 100 / test['No_name'].shape[0]:.4f}%.")

# Adoption Speed by Name and No Name
plt.figure(figsize=(18, 8));
make_count_plot(df=all_data.loc[all_data['dataset_type'] == 'train'], x='No_name', title='and having a name')

# Adoption Speed by Pet Type 
plt.figure(figsize=(18, 8));
make_count_plot(df=all_data.loc[all_data['dataset_type'] == 'train'], x='Type', title='by pet Type')

# Pets Age
fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.title('Distribution of pets age');
train['Age'].plot('hist', label='train');
test['Age'].plot('hist', label='test');
plt.legend();

plt.subplot(1, 2, 2)
plt.title('Distribution of pets age (log)');
np.log1p(train['Age']).plot('hist', label='train');
np.log1p(test['Age']).plot('hist', label='test');
plt.legend();

plt.figure(figsize=(10, 6));
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and age');

# Gender
g = sns.catplot(x="Gender", hue="AdoptionSpeed", col="dataset_type",data=train, kind="count",height=4, aspect=.7);


sns.factorplot('Type', col='Gender', data=all_data, kind='count', hue='dataset_type');
plt.subplots_adjust(top=0.8)
plt.suptitle('Count of cats and dogs in train and test set by gender');

train['Age'].value_counts().head(10)

plt.figure(figsize=(10, 6));
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and age');

data = []
for a in range(5):
    df = train.loc[train['AdoptionSpeed'] == a]

    data.append(go.Scatter(
        x = df['Age'].value_counts().sort_index().index,
        y = df['Age'].value_counts().sort_index().values,
        name = str(a)
    ))
layout = go.Layout(dict(title = "AdoptionSpeed trends by Age",
                  xaxis = dict(title = 'Age (months)'),
                  yaxis = dict(title = 'Counts'),
                  )
                  )
py.iplot(dict(data=data, layout=layout), filename='basic-line')
in['Pure_breed'] = 0
train.loc[train['Breed2'] == 0, 'Pure_breed'] = 1
test['Pure_breed'] = 0
test.loc[test['Breed2'] == 0, 'Pure_breed'] = 1
all_data['Pure_breed'] = 0
all_data.loc[all_data['Breed2'] == 0, 'Pure_breed'] = 1

print(f"Rate of pure breed pets in train data: {train['Pure_breed'].sum() * 100 / train['Pure_breed'].shape[0]:.4f}%.")
print(f"Rate of pure breed pets in test data: {test['Pure_breed'].sum() * 100 / test['Pure_breed'].shape[0]:.4f}%.")

def plot_four_graphs(col='', main_title='', dataset_title=''):
    plt.figure(figsize=(20, 12));
    plt.subplot(2, 2, 1)
    make_count_plot(df=train, x=col, title=f'and {main_title}')

    plt.subplot(2, 2, 2)
    sns.countplot(x='dataset_type', data=all_data, hue=col);
    plt.title(dataset_title);

    plt.subplot(2, 2, 3)
    make_count_plot(df=train.loc[train['Type'] == 1], x=col, title=f'and {main_title} for dogs')

    plt.subplot(2, 2, 4)
    make_count_plot(df=train.loc[train['Type'] == 2], x=col, title=f'and {main_title} for cats')
    
plot_four_graphs(col='Pure_breed', main_title='having pure breed', dataset_title='Number of pets by pure/not-pure breed in train and test data')

# Breeda
breeds_dict = {k: v for k, v in zip(breeds['BreedID'], breeds['BreedName'])}
train['Breed1_name'] = train['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
train['Breed2_name'] = train['Breed2'].apply(lambda x: '_'.join(breeds_dict[x]) if x in breeds_dict else '-')

test['Breed1_name'] = test['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
test['Breed2_name'] = test['Breed2'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else '-')

all_data['Breed1_name'] = all_data['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
all_data['Breed2_name'] = all_data['Breed2'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else '-')
fig, ax = plt.subplots(figsize = (20, 18))
plt.subplot(2, 2, 1)
text_cat1 = ' '.join(all_data.loc[all_data['Type'] == 'Cat', 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_cat1)
plt.imshow(wordcloud)
plt.title('Top cat breed1')
plt.axis("off")

plt.subplot(2, 2, 2)
text_dog1 = ' '.join(all_data.loc[all_data['Type'] == 'Dog', 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_dog1)
plt.imshow(wordcloud)
plt.title('Top dog breed1')
plt.axis("off")

plt.subplot(2, 2, 3)
text_cat2 = ' '.join(all_data.loc[all_data['Type'] == 'Cat', 'Breed2_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_cat2)
plt.imshow(wordcloud)
plt.title('Top cat breed1')
plt.axis("off")

plt.subplot(2, 2, 4)
text_dog2 = ' '.join(all_data.loc[all_data['Type'] == 'Dog', 'Breed2_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_dog2)
plt.imshow(wordcloud)
plt.title('Top dog breed2')
plt.axis("off")
plt.show()

# Types
colors_dict = {k: v for k, v in zip(colors['ColorID'], colors['ColorName'])}
train['Color1_name'] = train['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
train['Color2_name'] = train['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
train['Color3_name'] = train['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

test['Color1_name'] = test['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
test['Color2_name'] = test['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
test['Color3_name'] = test['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

all_data['Color1_name'] = all_data['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
all_data['Color2_name'] = all_data['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
all_data['Color3_name'] = all_data['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

def make_factor_plot(df, x, col, title, main_count=main_count, hue=None, ann=True, col_wrap=4):

    if hue:
        g = sns.factorplot(col, col=x, data=df, kind='count', col_wrap=col_wrap, hue=hue);
    else:
        g = sns.factorplot(col, col=x, data=df, kind='count', col_wrap=col_wrap);
    plt.subplots_adjust(top=0.9);
    plt.suptitle(title);
    ax = g.axes
    plot_dict = prepare_plot_dict(df, x, main_count)
    if ann:
        for a in ax:
            for p in a.patches:
                text = f"{plot_dict[p.get_height()]:.0f}%" if plot_dict[p.get_height()] < 0 else f"+{plot_dict[p.get_height()]:.0f}%"
                a.annotate(text, (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='green' if plot_dict[p.get_height()] > 0 else 'red', rotation=0, xytext=(0, 10),
                     textcoords='offset points')  
sns.factorplot('dataset_type', col='Type', data=all_data, kind='count', hue='Color1_name', palette=['Black', 'Brown', '#FFFDD0', 'Gray', 'Gold', 'White', 'Yellow']);
plt.subplots_adjust(top=0.8)
plt.suptitle('Counts of pets in datasets by main color');

make_factor_plot(df=train, x='Color1_name', col='AdoptionSpeed', title='Counts of pets by main color and Adoption Speed')


train['full_color'] = (train['Color1_name'] + '__' + train['Color2_name'] + '__' + train['Color3_name']).str.replace('__', '')
test['full_color'] = (test['Color1_name'] + '__' + test['Color2_name'] + '__' + test['Color3_name']).str.replace('__', '')
all_data['full_color'] = (all_data['Color1_name'] + '__' + all_data['Color2_name'] + '__' + all_data['Color3_name']).str.replace('__', '')

make_factor_plot(df=train.loc[train['full_color'].isin(list(train['full_color'].value_counts().index)[:12])], x='full_color', col='AdoptionSpeed', title='Counts of pets by color and Adoption Speed')

# 
gender_dict = {1: 'Male', 2: 'Female', 3: 'Mixed'}
for i in all_data['Type'].unique():
    for j in all_data['Gender'].unique():
        df = all_data.loc[(all_data['Type'] == i) & (all_data['Gender'] == j)]
        top_colors = list(df['full_color'].value_counts().index)[:5]
        j = gender_dict[j]
        print(f"Most popular colors of {j} {i}s: {' '.join(top_colors)}")
        
plot_four_graphs(col='MaturitySize', main_title='MaturitySize', dataset_title='Number of pets by MaturitySize in train and test data')


#
make_factor_plot(df=all_data, x='MaturitySize', col='Type', title='Count of cats and dogs in train and test set by MaturitySize', hue='dataset_type', ann=False)

#
plot_four_graphs(col='FurLength', main_title='FurLength', dataset_title='Number of pets by FurLength in train and test data')

#
fig, ax = plt.subplots(figsize = (20, 18))
plt.subplot(2, 2, 1)
text_cat1 = ' '.join(all_data.loc[(all_data['FurLength'] == 1) & (all_data['Type'] == 'Cat'), 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text_cat1)
plt.imshow(wordcloud)
plt.title('Top cat breed1 with short fur')
plt.axis("off")

plt.subplot(2, 2, 2)
text_dog1 = ' '.join(all_data.loc[(all_data['FurLength'] == 1) & (all_data['Type'] == 'Dog'), 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text_dog1)
plt.imshow(wordcloud)
plt.title('Top dog breed1 with short fur')
plt.axis("off")

plt.subplot(2, 2, 3)
text_cat2 = ' '.join(all_data.loc[(all_data['FurLength'] == 2) & (all_data['Type'] == 'Cat'), 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text_cat2)
plt.imshow(wordcloud)
plt.title('Top cat breed1 with medium fur')
plt.axis("off")
plt.subplot(2, 2, 4)
text_dog2 = ' '.join(all_data.loc[(all_data['FurLength'] == 2) & (all_data['Type'] == 'Dog'), 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text_dog2)
plt.imshow(wordcloud)
plt.title('Top dog breed2 with medium fur')
plt.axis("off")
plt.show()

#
plt.figure(figsize=(20, 12));
plt.subplot(2, 2, 1)
make_count_plot(df=train, x='Vaccinated', title='Vaccinated')
plt.xticks([0, 1, 2], ['Yes', 'No', 'Not sure']);
plt.title('AdoptionSpeed and Vaccinated');

plt.subplot(2, 2, 2)
make_count_plot(df=train, x='Dewormed', title='Dewormed')
plt.xticks([0, 1, 2], ['Yes', 'No', 'Not sure']);
plt.title('AdoptionSpeed and Dewormed');

plt.subplot(2, 2, 3)
make_count_plot(df=train, x='Sterilized', title='Sterilized')
plt.xticks([0, 1, 2], ['Yes', 'No', 'Not sure']);
plt.title('AdoptionSpeed and Sterilized');

plt.subplot(2, 2, 4)
make_count_plot(df=train, x='Health', title='Health')
plt.xticks([0, 1, 2], ['Healthy', 'Minor Injury', 'Serious Injury']);
plt.title('AdoptionSpeed and Health');

plt.suptitle('Adoption Speed and health conditions');

#
train['health'] = train['Vaccinated'].astype(str) + '_' + train['Dewormed'].astype(str) + '_' + train['Sterilized'].astype(str) + '_' + train['Health'].astype(str)
test['health'] = test['Vaccinated'].astype(str) + '_' + test['Dewormed'].astype(str) + '_' + test['Sterilized'].astype(str) + '_' + test['Health'].astype(str)


make_factor_plot(df=train.loc[train['health'].isin(list(train.health.value_counts().index[:5]))], x='health', col='AdoptionSpeed', title='Counts of pets by main health conditions and Adoption Speed')

#
plt.figure(figsize=(20, 16))
plt.subplot(3, 2, 1)
sns.violinplot(x="AdoptionSpeed", y="Age", data=train);
plt.title('Age distribution by Age');
plt.subplot(3, 2, 3)
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Vaccinated", data=train);
plt.title('Age distribution by Age and Vaccinated');
plt.subplot(3, 2, 4)
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Dewormed", data=train);
plt.title('Age distribution by Age and Dewormed');
plt.subplot(3, 2, 5)
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Sterilized", data=train);
plt.title('Age distribution by Age and Sterilized');
plt.subplot(3, 2, 6)
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Health", data=train);
plt.title('Age distribution by Age and Health');

train.loc[train['Quantity'] > 11][['Name', 'Description', 'Quantity', 'AdoptionSpeed']].head(10)

#
train['Quantity_short'] = train['Quantity'].apply(lambda x: x if x <= 5 else 6)
test['Quantity_short'] = test['Quantity'].apply(lambda x: x if x <= 5 else 6)
all_data['Quantity_short'] = all_data['Quantity'].apply(lambda x: x if x <= 5 else 6)
plot_four_graphs(col='Quantity_short', main_title='Quantity_short', dataset_title='Number of pets by Quantity_short in train and test data')

#
train['Free'] = train['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
test['Free'] = test['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
all_data['Free'] = all_data['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
plot_four_graphs(col='Free', main_title='Free', dataset_title='Number of pets by Free in train and test data')

#
all_data.sort_values('Fee', ascending=False)[['Name', 'Description', 'Fee', 'AdoptionSpeed', 'dataset_type']].head(10)

#
plt.figure(figsize=(16, 6));
plt.subplot(1, 2, 1)
plt.hist(train.loc[train['Fee'] < 400, 'Fee']);
plt.title('Distribution of fees lower than 400');

plt.subplot(1, 2, 2)
sns.violinplot(x="AdoptionSpeed", y="Fee", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and Fee');

#
plt.figure(figsize=(16, 10));
sns.scatterplot(x="Fee", y="Quantity", hue="Type",data=all_data);
plt.title('Quantity of pets and Fee');

#
states_dict = {k: v for k, v in zip(states['StateID'], states['StateName'])}
train['State_name'] = train['State'].apply(lambda x: '_'.join(states_dict[x].split()) if x in states_dict else 'Unknown')
test['State_name'] = test['State'].apply(lambda x: '_'.join(states_dict[x].split()) if x in states_dict else 'Unknown')
all_data['State_name'] = all_data['State'].apply(lambda x: '_'.join(states_dict[x].split()) if x in states_dict else 'Unknown')


all_data['State_name'].value_counts(normalize=True).head()


#
plt.figure(figsize=(13, 6));
train['AdoptionSpeed'].value_counts().sort_index().plot('bar', color=['lime', 'limegreen', 'forestgreen', 'green', 'darkgreen']);
plt.title('Speed Classes Counts');

#
# set height of bar
h= np.zeros((2, 5))
for i in range(0, 2):
    for j in range(0, 5):
        h[i][j]=len(train.Type[train.Type==i+1][train.AdoptionSpeed==j])
#print(h)

# Set position of bar on X axis
k= np.zeros((2, 5))
for i in range(0, 2):
    for j in range(0, 5):
        k[i][j]=j+0.45*i
#print(k) 

# Make the plot
plt.figure(figsize=(13, 7));
plt.bar(k[0], h[0], color='indianred', width=0.35, edgecolor='white', label='Cat')
plt.bar(k[1], h[1], color='teal', width=0.35, edgecolor='white', label='Dog')

# Add xticks on the middle of the group bars
plt.xlabel('Adoption Speed', fontweight='bold')
plt.xticks([0.25, 1.25, 2.25, 3.25, 4.25], ['0', '1', '2', '3', '4'])
plt.title('Number of Cats and Dogs in Adoption Speed');
 
# Create legend & Show graphic
plt.legend()
plt.show()

#
plt.figure(figsize=(13, 6));
train.Age.value_counts().sort_index()[0:70].plot('bar');
plt.title('Age Counts (Month)');

#
# Data
r = [0,1,2,3,4]
male = [0,0,0,0,0]
female = [0,0,0,0,0]
for i in r: 
    male[i]  = len(train.Gender[train.Gender==1][train.AdoptionSpeed==i]);
    female[i]= len(train.Gender[train.Gender==2][train.AdoptionSpeed==i])
raw_data = {'male': male,'female': female}
df = pd.DataFrame(raw_data)
 
# From raw value to percentage
totals = [i+j for i,j in zip(df['male'], df['female'])]
male = [i / j * 100 for i,j in zip(df['male'], totals)]
female = [i / j * 100 for i,j in zip(df['female'], totals)]
 
# plot
plt.figure(figsize=(13, 6));
names = ('0','1','2','3','4')

# Create orange Bars
plt.bar(r, male, color='#f9bc86', edgecolor='white', width=0.70, label="Male")

# Create blue Bars
plt.bar(r, female, bottom=[i for i in (male)], color='#a3acff', edgecolor='white', width=0.70, label="Female")

# Add a legend
plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

# Custom x axis
plt.xticks(r, names)
plt.xlabel("Adoption Speed")
 
# Show graphic
plt.show()

#
r = [0,1,2,3,4]
h = [0,1,2,3,4,5,6,7]
Speed = r
Color1 = [[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6,7]]

for i in r: 
    Speed[i] = len(train.AdoptionSpeed[train.AdoptionSpeed==i]);
    for j in h:
        Color1[i][j] = len(train.Color1[train.Color1==j][train.AdoptionSpeed==i]);
#        Color2[i] = len(train.Color2[train.Color2==j][train.AdoptionSpeed==i]);
#        Color3[i] = len(train.Color3[train.Color3==j][train.AdoptionSpeed==i]);

    
# Make data: I have 3 groups and 7 subgroups
group_names=['0', '1', '2', '3', '4']
group_size = Speed
subgroup_names=['C.1.0', 'C.1.1', 'C.1.2', 'C.1.3', 'C.1.4', 'C.1.5', 'C.1.6', 'C.1.7',
                'C.1.0', 'C.1.1', 'C.1.2', 'C.1.3', 'C.1.4', 'C.1.5', 'C.1.6', 'C.1.7',
                'C.1.0', 'C.1.1', 'C.1.2', 'C.1.3', 'C.1.4', 'C.1.5', 'C.1.6', 'C.1.7',
                'C.1.0', 'C.1.1', 'C.1.2', 'C.1.3', 'C.1.4', 'C.1.5', 'C.1.6', 'C.1.7',
                'C.1.0', 'C.1.1', 'C.1.2', 'C.1.3', 'C.1.4', 'C.1.5', 'C.1.6', 'C.1.7',]
subgroup_size = Color1[0] + Color1[1] + Color1[2] + Color1[3] + Color1[4] 
 
# Create colors
a, b, c, d, e=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Oranges, plt.cm.Purples]
 
# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(group_size, radius=1.7-0.5, labels=group_names, labeldistance=0.5, 
                  colors=[a(0.7), b(0.7), c(0.7), d(0.7), e(0.7)] )
plt.setp( mypie, width=0.8, edgecolor='white')
 
# Second Ring (Inside)
mypie2, _ = ax.pie(subgroup_size, radius=1.7,
                   labels=subgroup_names, labeldistance=1.2,
                   colors=[a(0.0), a(0.1), a(0.2), a(0.3), a(0.4), a(0.5), a(0.6), a(0.7),
                           b(0.0), b(0.1), b(0.2), b(0.3), b(0.4), b(0.5), b(0.6), b(0.7),
                           c(0.0), c(0.1), c(0.2), c(0.3), c(0.4), c(0.5), c(0.6), c(0.7),
                           d(0.0), d(0.1), d(0.2), d(0.3), d(0.4), d(0.5), d(0.6), d(0.7),
                           e(0.0), e(0.1), e(0.2), e(0.3), e(0.4), e(0.5), e(0.6), e(0.7)])
plt.setp( mypie2, width=0.9, edgecolor='white')
plt.margins(0,0)
 
# show it
plt.show()

################
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

import pandas as pd
feature_imp = pd.Series(clf.feature_importances_,index=testdata.columns.values[0:21]).sort_values(ascending=False)
feature_imp

#
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

