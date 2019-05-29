from bs4 import BeautifulSoup
import datetime
import pandas as pd
import numpy as np
import IDRdrop
import EPOwebscrape
#import emailscrape
import json
import os

def main():

    #output_dict, warnings, filename = emailscrape.get_emails()
    filename = 'bodies.json'

    try:
        emails = output_dict
        
    except:
        
        with open(filename, 'r') as f:
            emails = json.load(f)
            emails = json.loads(emails)

    for item in emails.values():
        
        username = item['user']
        pwd = item['pw']
        accts_to_find = item['accts']

        NGrid = ('SUEZ' in username)

        scrape_data(username, pwd, NGrid, accts_to_find)
        split_data()
        filter_data()

def scrape_data(user_, pw_, ngrid, acct_list):

    browser, url = EPOwebscrape.logon(user_, pw_, ngrid)

    AIDs = []

    soup = BeautifulSoup(browser.page_source, features = 'html5lib')
    table = soup.find('tbody', {'role' : 'rowgroup'})

    #get EPO AID value for every account
    for accts in acct_list:
        results = EPOwebscrape.big_match(accts, table)
        AIDs.append(results)

    #make list of AIDs, split into list of lists of 5
    final = []

    for aid_list in AIDs:
        for aid in aid_list:
            final.append(aid)

    n = 5
    final2 = [final[i * n:(i + 1) * n] for i in range((len(final) + n - 1) // n )]  

    for elem in final2:
        EPOwebscrape.export_data(elem, browser)
        

def split_data():
    
    day_before = datetime.datetime.now() - datetime.timedelta(days = 3)

    #readpath = 'C:\\Users\wb5888\Downloads'
    readpath = '/Users/stevenhurwitt/Downloads'

    myfiles = IDRdrop.show_dir(readpath, 20)

    rec_files = myfiles[myfiles.time > day_before]
    csv_ind = [file.split('.')[1] == 'csv' for file in rec_files.files]
    rec_files_csv = rec_files[csv_ind]
    index = rec_files_csv.index.values.astype(int)

    # Choose files to split into Raw IDR files.

    splitfiles = list(myfiles.files[index])
    print('files to split: ')
    print(splitfiles)


    #Batch process downloaded EPO files into Raw IDRs

    #writepath = 'C:\\Users\\wb5888\\Documents\\Python Code\\IDR_Drop\\Raw IDR Data'
    writepath = '/Volumes/USB30FD/Python Code/IDR_Drop/Raw IDR Data'
    errorlog = []

    for file in splitfiles:
        try:
            os.chdir(readpath)
            filedf = pd.read_csv(file, sep = ",", header = 0)
    
            IDRdrop.raw_split(filedf, readpath, writepath)
            print('success, file: ', file)
        
        except:
            errorlog.append(file)
            print('error, file: ', file)
    return(errorlog)


def filter_data():
    #Show Raw IDR files based on utility

    #rawpath = 'C:\\Users\\wb5888\\Documents\\Python Code\\IDR_Drop\\Raw IDR Data'
    rawpath = '/Volumes/USB30FD/Python Code/IDR_Drop/Raw IDR Data'

    day_before = datetime.datetime.now() - datetime.timedelta(days = 3)
    rawfiles = IDRdrop.show_dir(rawpath, 50)
    rec_raw = rawfiles[rawfiles.time > day_before]
    csv_ind = [file.split('.')[1] == 'csv' for file in rec_raw.files]
    rec_raw_csv = rec_raw[csv_ind]
    index_raw = rec_raw_csv.index.values.astype(int)

    #Choose Raw IDRs to filter into IDR files.

    processfiles = list(rawfiles.files[index_raw])
    print('files to process: ')
    print(processfiles)


    #Batch filter Raw IDR into IDR files to be dropped

    #writepath = 'C:\\Users\\wb5888\\Documents\\IDR Data\\NEPOOL\\'
    writepath = '/Volumes/USB30FD/Python Code/IDR_Drop/IDR Data/NEPOOL'
    error_log2 = []

    for dropfile in processfiles:
        try:
            IDRdrop.data_drop(dropfile, rawpath, writepath)
            print('success, file: ', dropfile)
    
        except:
            error_log2.append(dropfile)
            print("error, file: ", dropfile)


    return(error_log2)


main()

