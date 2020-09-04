# -*- coding: utf-8 -*-

# Salaries in Pandas

import pandas as pd
sal = pd.read_csv('Salaries.csv')
head = sal.head()
info = sal.info()
average_base_pay = sal['BasePay'].mean()
max_overtime_pay = sal['OvertimePay'].max()
joseph_job_title = sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle']
joseph_job_benefits = sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']
ford = sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]['EmployeeName']
ford2 = sal.loc[sal['TotalPayBenefits'].idxmax()]

# lowest paid person
lowest = sal.iloc[sal['TotalPayBenefits'].argmin()]

#getting the average base pay
average_base_pay = sal.groupby('Year').mean()['BasePay']

# how many unique job titles
unique_job_titles = sal['JobTitle'].nunique()

# % most common jobs
five_most_common_jobs = sal['JobTitle'].value_counts().head(5)

# How many jobs were represented by only one person
one_person_jobs = sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1

# How many people have the word chief in their job title
def chief_string(title):
    if 'chief' in title.lower().split():
        return True
    else:
        return False
    
chief_in_string = sal['JobTitle'].apply(lambda x: chief_string(x))
sum_of_chief_in_string = sum(chief_in_string)