# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 18:25:04 2019

@author: Sahil Nathani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('KidnappingVictims.csv')

states = df['STATE/UT'].unique()
reasons = df['Pupose'].unique()
#=['For Adoption' 'For Begging' 'For Camel racing' 'For Illicit intercourse'
# 'For marriage' 'For Prostitution' 'For Ransom' 'For Revenge' 'For Sale'
#'For Selling body parts' 'For Slavery' 'For unlawful activity' 'Others'
#'Total']

andhra_2001 = df[(df['STATE/UT']=='Andhra Pradesh') & (df['YEAR']==2001)]
#total_cases.append(df_andhra_2001[(df_andhra_2001['Pupose']=='Total')]['Grand Total'])
andhra_2002 = df[(df['STATE/UT']=='Andhra Pradesh') & (df['YEAR']==2002)]
andhra_2003 = df[(df['STATE/UT']=='Andhra Pradesh') & (df['YEAR']==2003)]
andhra_2004 = df[(df['STATE/UT']=='Andhra Pradesh') & (df['YEAR']==2004)]
andhra_2005 = df[(df['STATE/UT']=='Andhra Pradesh') & (df['YEAR']==2005)]
andhra_2006 = df[(df['STATE/UT']=='Andhra Pradesh') & (df['YEAR']==2006)]
andhra_2007 = df[(df['STATE/UT']=='Andhra Pradesh') & (df['YEAR']==2007)]
andhra_2008 = df[(df['STATE/UT']=='Andhra Pradesh') & (df['YEAR']==2008)]
andhra_2009 = df[(df['STATE/UT']=='Andhra Pradesh') & (df['YEAR']==2009)]
andhra_2010 = df[(df['STATE/UT']=='Andhra Pradesh') & (df['YEAR']==2010)]
andhra_2011 = df[(df['STATE/UT']=='Andhra Pradesh') & (df['YEAR']==2011)]
andhra_2012 = df[(df['STATE/UT']=='Andhra Pradesh') & (df['YEAR']==2012)]

arun_2001 = df[(df['STATE/UT']=='Arunachal Pradesh') & (df['YEAR']==2001)]
arun_2002 = df[(df['STATE/UT']=='Arunachal Pradesh') & (df['YEAR']==2002)]
arun_2003 = df[(df['STATE/UT']=='Arunachal Pradesh') & (df['YEAR']==2003)]
arun_2004 = df[(df['STATE/UT']=='Arunachal Pradesh') & (df['YEAR']==2004)]
arun_2005 = df[(df['STATE/UT']=='Arunachal Pradesh') & (df['YEAR']==2005)]
arun_2006 = df[(df['STATE/UT']=='Arunachal Pradesh') & (df['YEAR']==2006)]
arun_2007 = df[(df['STATE/UT']=='Arunachal Pradesh') & (df['YEAR']==2007)]
arun_2008 = df[(df['STATE/UT']=='Arunachal Pradesh') & (df['YEAR']==2008)]
arun_2009 = df[(df['STATE/UT']=='Arunachal Pradesh') & (df['YEAR']==2009)]
arun_2010 = df[(df['STATE/UT']=='Arunachal Pradesh') & (df['YEAR']==2010)]
arun_2011 = df[(df['STATE/UT']=='Arunachal Pradesh') & (df['YEAR']==2011)]
arun_2012 = df[(df['STATE/UT']=='Arunachal Pradesh') & (df['YEAR']==2012)]

assam2001 = df[(df['STATE/UT']=='Assam') & (df['YEAR']==2001)]
assam2002 = df[(df['STATE/UT']=='Assam') & (df['YEAR']==2002)]
assam2003 = df[(df['STATE/UT']=='Assam') & (df['YEAR']==2003)]
assam2004 = df[(df['STATE/UT']=='Assam') & (df['YEAR']==2004)]
assam2005 = df[(df['STATE/UT']=='Assam') & (df['YEAR']==2005)]
assam2006 = df[(df['STATE/UT']=='Assam') & (df['YEAR']==2006)]
assam2007 = df[(df['STATE/UT']=='Assam') & (df['YEAR']==2007)]
assam2008 = df[(df['STATE/UT']=='Assam') & (df['YEAR']==2008)]
assam2009 = df[(df['STATE/UT']=='Assam') & (df['YEAR']==2009)]
assam2010 = df[(df['STATE/UT']=='Assam') & (df['YEAR']==2010)]
assam2011 = df[(df['STATE/UT']=='Assam') & (df['YEAR']==2011)]
assam2012 = df[(df['STATE/UT']=='Assam') & (df['YEAR']==2012)]

bihar2001 = df[(df['STATE/UT']=='Bihar') & (df['YEAR']==2001)]
bihar2002 = df[(df['STATE/UT']=='Bihar') & (df['YEAR']==2002)]
bihar2003 = df[(df['STATE/UT']=='Bihar') & (df['YEAR']==2003)]
bihar2004 = df[(df['STATE/UT']=='Bihar') & (df['YEAR']==2004)]
bihar2005 = df[(df['STATE/UT']=='Bihar') & (df['YEAR']==2005)]
bihar2006 = df[(df['STATE/UT']=='Bihar') & (df['YEAR']==2006)]
bihar2007 = df[(df['STATE/UT']=='Bihar') & (df['YEAR']==2007)]
bihar2008 = df[(df['STATE/UT']=='Bihar') & (df['YEAR']==2008)]
bihar2009 = df[(df['STATE/UT']=='Bihar') & (df['YEAR']==2009)]
bihar2010 = df[(df['STATE/UT']=='Bihar') & (df['YEAR']==2010)]
bihar2011 = df[(df['STATE/UT']=='Bihar') & (df['YEAR']==2011)]
bihar2012 = df[(df['STATE/UT']=='Bihar') & (df['YEAR']==2012)]

print(bihar2012['Grand Total'])