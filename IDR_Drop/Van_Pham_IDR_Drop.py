from bs4 import BeautifulSoup
import datetime
import pandas as pd
import numpy as np
import IDRdrop
import EPOwebscrape
import emailscrape
import json
import os

def main():

    base_path = os.getcwd()

    #get emails, subset into recent (last three days)
    output_dict, filename = emailscrape.get_emails()

    last_days = datetime.datetime.today() - datetime.timedelta(3)    

    try:
        test = pd.DataFrame.from_dict(output_dict, orient = 'index')
        if type(test.date[0]) == str:
            test.date = pd.to_datetime(test.date)
            
        sub = test[test.date > last_days]
        
        accts_success = [len(accts) > 0 for accts in sub.accts]
        accts_fail = [not val for val in accts_success]
        
        good = sub[accts_success].reset_index(drop = True)
        
        if len(accts_fail) > 0:
            bad = sub[accts_fail].reset_index(drop = True)
            mail_error = 'EMAIL_SCRAPE_ERROR.csv'
            os.chdir(base_path + '\\Logs')
            bad.to_csv(mail_error, header = True, index = False)
            os.chdir(base_path)
        
    except:
        
        with open(filename, 'r') as f:
            emails = json.load(f)
            emails = json.loads(emails)
            
        test = pd.DataFrame.from_dict(emails, orient = 'index')
        if type(test.date[0]) == str:
            test.date = pd.to_datetime(test.date)
            
        sub = test[test.date > last_days]
        
        accts_success = [len(accts) > 0 for accts in sub.accts]
        accts_fail = [not val for val in accts_success]
        
        good = sub[accts_success].reset_index()
        if len(accts_fail) > 0:
            bad = sub[accts_fail].reset_index()
            mail_error = 'EMAIL_SCRAPE_ERROR.csv'

            os.chdir(base_path + '\\Logs')
            bad.to_csv(mail_error, header = True, index = False)
            os.chdir(base_path)
        
    email_error = []
    
    for item in good.index.values:
        
        username = good.user[item]
        pwd = good.pw[item]

        if type(good.accts[item]) == 'str':
            accts_to_find = good.accts[item].split(" ")

        else:
            accts_to_find = good.accts[item]

        NGrid = ('SUEZ' in username)

        try:
            scrape_data(username, pwd, NGrid, accts_to_find)

        except:
            print('error downloading for ', username)
            email_error.append(username)

    good['web_error'] = [(name in email_error) for name in good.user]
    print('download failed for, ', len(email_error), 'of', len(good.web_error), 'accounts - check log.') 


    #for EPO files downloaded, split into sep raw idr files
    try:
        good['raw_parse_success'] = split_data()

    except:
        print('split data into Raw IDR error')

    try:
        filter_data()

    except:
        print('filter Raw IDR to IDR error')

    os.chdir(base_path + '\\Logs')
    main_log = 'IDR_DROP_SUCCESS.csv'
    
    if main_log in os.listdir(os.getcwd()):
        main_log = main_log.split('.')[0] + '_2.csv'

    good.to_csv(main_log, header = True, index = False)
    os.chdir(base_path)



### Helper Functions ###   

def scrape_data(user_, pw_, ngrid, acct_list):

    browser, url = EPOwebscrape.logon(user_, pw_, ngrid)

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

def split_data():
    
    day_before = datetime.datetime.now() - datetime.timedelta(days = 3)

    readpath = 'C:\\Users\wb5888\Downloads'
    #readpath = '/Users/stevenhurwitt/Downloads'

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

    basepath = 'C:\\Users\\wb5888\\Documents\\Python_Code\\IDR_Drop\\'
    writepath = 'C:\\Users\\wb5888\\Documents\\Python_Code\\IDR_Drop\\Raw IDR Data'
    #writepath = '/Volumes/USB30FD/Python Code/IDR_Drop/Raw IDR Data'
    split_fail = []
    

    for file in splitfiles:
        overall_acct_success = []
        
        try:
            os.chdir(readpath)
            filedf = pd.read_csv(file, sep = ",", header = 0)
    
            per_acct_success = IDRdrop.raw_split(filedf, readpath, writepath)
            overall_acct_success.append(per_acct_success)

            print('success, file: ', file)
        
        except:
            split_fail.append(file)
            print('error, file: ', file)

    log_file_name = 'SPLIT_FILE_ERROR_' + datetime.datetime.strftime(datetime.datetime.now(), '%m_%d_%Y') + '.csv'
    if log_file_name in os.listdir(os.getcwd()):
        log_file_name = log_file_name.split('.')[0] + '_2.csv'
    

    pd.DataFrame(split_fail).to_csv(log_file_name, header = False, index = False)
            
    return(overall_acct_success)


def filter_data():
    #Show Raw IDR files based on utility

    rawpath = 'C:\\Users\\wb5888\\Documents\\Python_Code\\IDR_Drop\\Raw IDR Data'
    #rawpath = '/Volumes/USB30FD/Python Code/IDR_Drop/Raw IDR Data'

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

    writepath = 'C:\\Users\\wb5888\\Documents\\Python_Code\\IDR_Drop\\IDR Data\\NEPOOL\\'
    #writepath = '/Volumes/USB30FD/Python Code/IDR_Drop/IDR Data/NEPOOL'
    error_log2 = []

    for dropfile in processfiles:
        try:
            IDRdrop.data_drop(dropfile, rawpath, writepath)
            print('success, file: ', dropfile)
    
        except:
            error_log2.append(dropfile)
            print("error, file: ", dropfile)

    os.chdir(base_path + '\\Logs')
    log_file_idr_name = 'RAW_TO_IDR_FILE_ERROR_' + datetime.datetime.strftime(datetime.datetime.now(), '%m_%d_%Y') + '.csv'

    if log_file_idr_name in os.listdir(os.getcwd()):
        log_file_idr_name = log_file_idr_name.split('.')[0] + '_2.csv'

    pd.DataFrame(error_log2).to_csv(log_file_idr_name, header = False, index = False)
    os.chdir(base_path)

if __name__ == "__main__":
    main()

