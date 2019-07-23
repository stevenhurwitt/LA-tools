#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'IDR_Drop\\Notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # NEPOOL IDR Drop
#%% [markdown]
# Implements class to batch tasks used for IDR drops.

#%%
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import IDRdrop
import EPOwebscrape
import json
import os


#%%
with open('bodies.json', 'r') as f:
    bodies = json.load(f)

bodies = json.loads(bodies)

#print(bodies.keys())
for item in bodies.values():
    print(item['accts'])


#%%
'SUEZ' in 'SUEZ_HIST'

#%% [markdown]
# ## Download files from EPO portal

#%%
#browser, url = logon('somer-nengie', 'rb4171', False)

browser, url = EPOwebscrape.logon('SUEZ_HIST', '3824', True)

accts_to_find = ['0040677012', '0021483009']
#accts_to_find = ['0040677012', '0021483009', '0067880031', '0074816023'] #enter accts as list
#accts_to_find = ['28905980018', '26121881010']
AIDs = []

soup = BeautifulSoup(browser.page_source)
table = soup.find('tbody', {'role' : 'rowgroup'})

#get EPO AID value for every account
for accts in accts_to_find:
    results = EPOwebscrape.big_match(accts, table)
    AIDs.append(results)

#make list of AIDs, split into list of lists of 5
final = []
AIDs

for aid_list in AIDs:
    for aid in aid_list:
        final.append(aid)

n = 5
final2 = [final[i * n:(i + 1) * n] for i in range((len(final) + n - 1) // n )]  
final2

for elem in final2:
    EPOwebscrape.export_data(elem, browser)

#%% [markdown]
# ## Show downloaded files from EPO portal
#%% [markdown]
# Here *filepath* is a directory containing downloaded EPO files. Code will print 20 most recent files.

#%%
#filepath = os.getcwd()
readpath = 'C:\\Users\\wb5888\\Downloads'

myfiles = IDRdrop.show_dir(readpath, 20)
print(myfiles)

#%% [markdown]
# Choose files to split into Raw IDR files.

#%%
index = [0]

splitfiles = list(myfiles.files[index])
print('files to split: ')
print(splitfiles)

#%% [markdown]
# ## Batch process downloaded EPO files into Raw IDRs

#%%
readpath = 'C:\\Users\\wb5888\\Downloads'
writepath = 'C:\\Users\\wb5888\\Documents\\Raw IDR Data'
utility = 'CLP'
error_log = []

for file in splitfiles:
    try:
        os.chdir(readpath)
        filedf = pd.read_csv(file, sep = ",", header = 0)
    
        IDRdrop.raw_split(filedf, readpath, writepath)
        print('success, file: ', file)
        
    except:
        error_log = error_log.append(file)
        print('error, file: ', file)

#%% [markdown]
# ## Show Raw IDR files based on utility

#%%
utility = "MECO"

#%% [markdown]
# Here *rawpath* is directory containing Raw IDRs - 25 most recent will be shown.

#%%
rawpath = 'C:\\Users\\wb5888\\Documents\\Raw IDR Data\\NEPOOL\\' + utility

rawfiles = IDRdrop.show_dir(rawpath, 50)
print(rawfiles)

#%% [markdown]
# Choose Raw IDRs to filter into IDR files.

#%%
index = [36]

processfiles = list(rawfiles.files[:5])
print('files to processed: ')
print(processfiles)

#%% [markdown]
# ## Batch filter Raw IDR into IDR files to be dropped

#%%
readpath = rawpath
writepath = 'C:\\Users\\wb5888\\Documents\\IDR Data\\NEPOOL\\' + utility
error_log = []

for dropfile in processfiles:
    try:
        IDRdrop.data_drop(dropfile, rawpath, writepath)
        print('success, file: ', dropfile)
    
    except:
        error_log = error_log.append(dropfile)
        print("error, file: ", dropfile)


#%%
error_log


#%%
dir(EPOwebscrape)


#%%



#%%



