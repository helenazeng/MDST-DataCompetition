#This file takes the test and train csv and outputs a newtest and newtrain csv with new columns
#It transforms all the non comment columns into numerical

import pandas as pd
import numpy as np
import json

def testbook(x):
    if x=='What textbook?':
        return 1
    elif x=='Barely cracked it open':
        return 2
    elif x=='Essential to passing':
        return 5
    elif x=="It's a must have":
        return 4
    elif x=='You need it sometimes':
        return 3
    else:
        return np.nan

def interest(x):
    if x=='Meh':
        return 1
    elif x=='Low':
        return 2
    elif x=='Sorta interested':
        return 3
    elif x=='Really into it':
        return 4
    elif x=="It's my life":
        return 5
    else:
        return np.nan

def grade(x):
    letter=['F','D-','D','D+','C-', 'C','C+','B-','B','B+','A-', 'A', 'A+']
    if x in letter:
        return letter.index(x)+1
    else:
        return np.nan

def toIndicate(x,value):
    return (x==value)*1

def toIndicatelist(x,value):
    for j in x.split(','):
        phrase=j.replace('[','').replace(']','').replace('"','').strip()
        if value==phrase:
            return 1
    return 0


def formatthedata(train):
    column='date'
    train[column+'y']=pd.to_datetime(train[column],format ='%m/%d/%Y').map(lambda x: (x.year-2000))
    train[column+'m']=pd.to_datetime(train[column],format ='%m/%d/%Y').map(lambda x: (x.month))
    train[column+'d']=pd.to_datetime(train[column],format ='%m/%d/%Y').map(lambda x: (x.day))
    train[column+'w']=pd.to_datetime(train[column],format ='%m/%d/%Y').map(lambda x: (x.weekday()))
    train.drop(column, axis=1, inplace=True)

    column='forcredit'
    for value in ['Yes', 'No']:
        train[column+value]=train[column].map(lambda x: toIndicate(x,value))
    train.drop(column, axis=1, inplace=True)

    column='attendance'
    for value in ['Mandatory', 'Not Mandatory']:
        train[column+value]=train[column].map(lambda x: toIndicate(x,value))
    train.drop(column, axis=1, inplace=True)

    column='textbookuse'
    train[column]=train[column].map(testbook)

    column='interest'
    train[column]=train[column].map(interest)

    column='grade'
    gradeind=['Not sure yet','WD','Audit/No Grade','INC','Rather not say','P']
    for value in gradeind:
        train[column+value]=train[column].map(lambda x: toIndicate(x,value))
    train[column]=train[column].map(grade)

    with open('tags.txt', 'r') as f:
        tags=json.load(f)
    column='tags'
    for value in tags:
        train[column+value]=train[column].map(lambda x: toIndicatelist(x,value))
    train.drop(column, axis=1, inplace=True)

    with open('depts.txt', 'r') as f:
        depts=json.load(f)
    column='dept'
    for value in depts:
        train[column+value]=train[column].map(lambda x: toIndicate(x,value))
    train.drop(column, axis=1, inplace=True)

    column='online'
    value='online'
    train[column]=train[column].map(lambda x: toIndicate(x,value))


train=pd.read_csv("train.csv",header=0)
test=pd.read_csv("test.csv",header=0)

formatthedata(train)
formatthedata(test)

train.to_csv('newtrain.csv',index=False)
test.to_csv('newtest.csv',index=False)
