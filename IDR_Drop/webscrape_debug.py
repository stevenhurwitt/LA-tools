from bs4 import BeautifulSoup
import datetime
import pandas as pd
import numpy as np
import IDRdrop
import EPOwebscrape
import emailscrape
import json
import os

def open_json():
    
    last_days = datetime.datetime.today() - datetime.timedelta(3)  
    filename = 'email_bodies_05_24_2019.json'
    with open(filename, 'r') as f:
        emails = json.load(f)
        emails = json.loads(emails)
            
    test = pd.DataFrame.from_dict(emails, orient = 'index')
    return test

test = open_json()
jcp = test[test.user == 'jcp-engie']

user = jcp.iloc[0]['user']
pw = jcp.iloc[0]['pw']
ngrid = False
acct_list = jcp.iloc[0]['accts']

browser, url = EPOwebscrape.logon(user, pw, ngrid)

AIDs = []

soup = BeautifulSoup(browser.page_source, features = 'html5lib')
table = soup.find('tbody', {'role' : 'rowgroup'})

if type(acct_list) == str:
        acct_list = acct_list.split(" ")


#get EPO AID value for every account
for accts in acct_list:
    results = EPOwebscrape.big_match(accts, table)
    AIDs.append(results)

    #make list of AIDs, split into list of lists of 5
final = []

for aid_list in AIDs:
    for aid in aid_list:
        final.append(aid)

try:
    n = 5
    final2 = [final[i * n:(i + 1) * n] for i in range((len(final) + n - 1) // n )]  

    for elem in final2:
        EPOwebscrape.export_data(elem, browser)

except:
    n = 2
    final2 = [final[i * n:(i + 1) * n] for i in range((len(final) + n - 1) // n )]  

    for elem in final2:
        EPOwebscrape.export_data(elem, browser)
