from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cx_Oracle
import os

#function to show top n files in directory sorted by date modified
#returns pandas dataframe
def show_dir(path, n):

    os.chdir(path)
    files = os.listdir(path)

    time = []
    for file in files:
        sys_time = round(os.path.getmtime(path + "/" + file))
        time.append(datetime.fromtimestamp(sys_time))

    time = np.array(time)
    lab = ['files']

    filedf = pd.DataFrame(files, columns = lab)

    filedf['time'] = time
    filedf = filedf.sort_values(by = 'time', axis = 0, ascending = False).reset_index(drop = True)

    print("files found in dir: ", path)
    return(filedf.head(n))

#function to take file from EPO in readdir of meters in utility
#splits a downloaded csv from EPO into raw meter IDR files
def raw_split(filedf, readdir, writedir):

    account = filedf.Account.unique()
    fail = []
    
    util = str(acct_from_LDC(account[0])).split('_')[0]
    writedir = "\\".join([writedir, util])
    
    print(util)
    print(writedir)
    
    os.chdir(writedir)
    print('found ' + str(len(account)) + ' accounts.')

    for name in account:
        sub = filedf.loc[filedf.Account == name,:].reset_index(drop = True)

        acct_id = acct_from_LDC(name)
        write_name = str(acct_id) + "_IDR_RAW.csv"
        
        if write_name not in os.listdir(writedir):
            
            try:
                sub.to_csv(write_name, sep = ",", header = True, index = False)
                print(write_name, ' written.')
                fail.append(True)

            except:
                print('error writing ', write_name)
                fail.append(False)

    return(fail)
    
#function to check for blanks and merge usage if there is less than 70% blanks
def filemerge(df1, df2):

    df1 = df1[(df1.isnull().sum(axis = 1)/df1.shape[1] < .7)]
    df2 = df2[(df2.isnull().sum(axis = 1)/df2.shape[1] < .7)]
    
    first_date1 = df1.loc[:,'Date'].iloc[0] 
    first_date2 = df2.loc[:,'Date'].iloc[0]

    fd1 = datetime.strptime(first_date1, '%m/%d/%Y')
    fd2 = datetime.strptime(first_date2, '%m/%d/%Y')
    
    if fd1 < fd2:
        new_dat = pd.concat([df1, df2], ignore_index = True)
        date_count = new_dat.groupby('Date').agg('count').sum(axis = 1)
        print('spot check output file at date ', date_count.idxmax())
    
    elif fd1 > fd2:
        new_dat = pd.concat([df2, df1], ignore_index = True)
        date_count = new_dat.groupby('Date').agg('count').sum(axis = 1)
        print('spot check output file at date ', date_count.idxmax())
        
    return new_dat

#function to check for nonzeros b/t lower & upper bound before writing to csv
#if below LB, don't write. if above UB, write.
def mindthegap(df, filename, LB, UB):
        
        hours = df.columns.values[4:] #get columns w/ hours
        p_nzero = (df.loc[:,hours] != 0).sum().sum()/len(hours) / df.shape[0] #calc % of nonzeros
    
        if p_nzero < LB:
            pass
            #print(round(1 - p_nzero, 4), "percent zeros, ", filename, " not saved.")
    
        if p_nzero > LB:
            #print(round(1 - p_nzero, 4), "percent zeros, ", filename, " saved.")
            df.to_csv(filename, sep = ",", header = True, index = False)

def acct_from_LDC(acct):
    
    uid = 'tesi_interface'
    pwd = 'peint88'

    ip = '172.25.152.125'
    port = '1700'
    service_name = 'tppe.mytna.com'
    dsn = cx_Oracle.makedsn(ip, port, service_name=service_name)
   
    con = cx_Oracle.connect(user = uid, password = pwd, dsn = dsn)
    cur = con.cursor()
    query = "select distinct B.AccountID from pwrline.acctservicehist D, pwrline.account B  where B.name like '%" + str(acct) + "%' and D.marketcode = 'NEPOOL'"
    cur.execute(query)
    for result in cur:
        acct_id = result[0].split('NEPOOL_')[1]
    return(acct_id)

#function to turn raw IDR into cleaned IDR file
def data_drop(rawfile, readpath, writepath):

        os.chdir(readpath)

        #print('reading file...')
        try:
            raw = pd.read_csv(rawfile, sep = ",", header = 0)
        
        except:
            raw = pd.read_excel(rawfile, header = 0, sheet_name = 0)

        #group by units to filter kWh
        combos = dict(list(raw.groupby('Units')))
        rel_channels = combos['kWh']

        #group by channels into unique
        rel_channels.groupby('Channel')
        uniq_channels = pd.unique(rel_channels['Channel'])

        utility = rawfile.split("_")[0]
        #utility = "CLP"
        writepath = writepath + str(utility)
        
        os.chdir(writepath)

        if len(uniq_channels) > 1:
            ch_data = pd.DataFrame([])
        
            clean_file1 = rawfile.replace("_RAW", "")
            clean_file2 = rawfile.replace("RAW", "3")
            
            #clean_file1 = rawfile.replace("CityofStamford", "CLP")
            #clean_file2 = rawfile.replace("CityofStamford", "CLP_3")
            
            #clean_file1 = clean_file1.replace(".xlsx", ".csv")
            #clean_file2 = clean_file2.replace(".xlsx", ".csv")


            for channel in uniq_channels:
                if(len(channel) == 1) and (int(channel) == 4):
                    channel = '3'
                    channel = channel + ' kWh'
                
                ch_info = np.array([channel.split(' ')[:2]])
            
                ch_data = ch_data.append(pd.DataFrame(data = ch_info, columns = ['ch_name', 'ch_num'], index = [0]), ignore_index = True)            
           

            if len(ch_data.loc[ch_data.ch_num == "1"]) > 1:
                ch1_tmp1 = raw.loc[raw.Channel == uniq_channels[0],:]
                ch1_tmp2 = raw.loc[raw.Channel == uniq_channels[2],:]

                #check for blanks
                channel1 = filemerge(ch1_tmp1, ch1_tmp2)
                mindthegap(channel1, clean_file1, .4, .7)
            
            else:
                channel1 = raw.loc[raw.Channel == uniq_channels[0],:]
                mindthegap(channel1, clean_file1, .4, .7)
            
            #check channel 3's
            #if more than one, subset and gap check
            if len(ch_data.loc[ch_data.ch_num == "3"]) > 1:
                #print("more than one channel 3")
                ch3_tmp1 = raw.loc[raw.Channel == uniq_channels[1],:]
                ch3_tmp2 = raw.loc[raw.Channel == uniq_channels[3],:]
            
                mindthegap(ch3_tmp1, clean_file2, .4, .7)
                mindthegap(ch3_tmp2, clean_file2, .4, .7)
            
            else:
                channel3 = raw.loc[raw.Channel == uniq_channels[1],:]
                mindthegap(channel3, clean_file2, .4, .7)
        
        
        elif len(uniq_channels) == 1:
            clean_data = raw.loc[raw.Units == 'kWh',:]
            clean_file = rawfile.replace("_RAW", "")
            #clean_file = rawfile.replace("CityofStamford", "CLP")
            #clean_file = clean_file.replace(".xlsx", ".csv")
            #print("writing single channel data...")
            mindthegap(clean_data, clean_file, .4, .7)
