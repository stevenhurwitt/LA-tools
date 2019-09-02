from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
def raw_split(filedf, readdir, writedir, utility):

    account = filedf.Account.unique()
    fail = []
    os.chdir(writedir)
    print('found ' + str(len(account)) + ' accounts.')

    for name in account:
        sub = filedf.loc[filedf.Account == name,:].reset_index(drop = True)
        
        ldc = str(name).split(' ')[0]
        acct_id = '_'.join([ldc, utility])
        
        write_name = ''.join([acct_id, "_IDR_RAW.csv"])
        
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
        print('spot check output file at date {}.'.format(date_count.idxmax()))
    
    elif fd1 > fd2:
        new_dat = pd.concat([df2, df1], ignore_index = True)
        date_count = new_dat.groupby('Date').agg('count').sum(axis = 1)
        print('spot check output file at date: {}.'.format(date_count.idxmax()))
        
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

    

#function to turn raw IDR into cleaned IDR file
def data_drop(rawfile, readpath, writepath):

        os.chdir(readpath)

        #print('reading file...')
        raw = pd.read_csv(rawfile, sep = ",", header = 0)

        #group by units to filter kWh
        combos = dict(list(raw.groupby('Units')))
        rel_channels = combos['kWh']

        #group by channels into unique
        rel_channels.groupby('Channel')
        uniq_channels = pd.unique(rel_channels['Channel'])

        full_file_name = rawfile.split("_")
        for item in full_file_name:
            try:
                int(item)
                pass
            except:
                utility = item
                break
          
        
        os.chdir(writepath)

        if len(uniq_channels) > 1:
            ch_data = pd.DataFrame([])
        
            clean_file1 = rawfile.replace("_RAW", "")
            clean_file2 = rawfile.replace("RAW", "3")

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
            #print("writing single channel data...")
            mindthegap(clean_data, clean_file, .4, .7)

def hor_to_vert(file):
    
    f = pd.read_csv(file)
    hrs = f.columns[4:]
    new_f = f.melt(id_vars = 'Date', value_vars = hrs)
    dt_ind = new_f['Date'] + ' ' + new_f['variable']
    new_f.drop(['Date', 'variable'], axis = 1, inplace = True)
    
    new_dt = [a.replace('24:00', '00:00') for a in dt_ind]
    new_f.set_index(pd.to_datetime(new_dt), drop = True, inplace = True)
    new_f.sort_index(axis = 0, inplace = True)
    
    og = file.split('.')
    new_name = ''.join([og[0], '_vert', '.csv'])
    new_f.to_csv(new_name, header = None)
    print('transformed and wrote {} to {}.'.format(file, new_name))
