
import numpy as np
import pandas as pd
import datetime as dt
from pandas.io.json import json_normalize
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
from collections import deque
import json
import cx_Oracle
import math
import os


def cp_query(pr_num, rev_num):
    
    selectstr1 = "(select distinct F.name as CustomerName, F.Customerid, B.name as LDC_Account, B.Accountid,"
    selectstr1 = "".join([selectstr1, "D.uidaccount, D.marketcode, A.Contractid, A.Revision "])
    selectstr1 = "".join([selectstr1, "from pwrline.account B, pwrline.lscmcontract A, pwrline.lscmcontractitem C, "])
    selectstr1 = "".join([selectstr1, "pwrline.acctservicehist D, pwrline.customer F "])
    selectstr1 = "".join([selectstr1, "where C.uidcontract=A.uidcontract and C.uidaccount=B.uidaccount and B.uidaccount=D.uidaccount "])
    selectstr1 = "".join([selectstr1, "and B.uidcustomer=F.uidcustomer and A.contractid='", pr_num, "' and A.revision=", rev_num, ") A "])

    s2str = "select distinct A.*, B.starttime, B.stoptime, B.overridecode as Tag_Type, B.val as Tag, B.strval as SOURCE_TYPE, B.lstime as Timestamp "
    s2str = "".join([s2str, "from pwrline.acctoverridehist B, ", selectstr1])
    s2str = "".join([s2str, "where A.uidaccount=B.uidaccount and (A.marketcode='PJM' OR  A.marketcode='NEPOOL' OR A.marketcode= 'NYISO' OR A.marketcode= 'MISO') "])
    s2str = "".join([s2str, "and (B.overridecode ='TRANSMISSION_TAG_OVRD' OR B.overridecode='CAPACITY_TAG_OVRD') "])
    s2str = "".join([s2str, "order by A.customername, B.overridecode, A.accountid, B.starttime"])

    return(s2str)


def OracleAPI(query):
    
    uid = 'tesi_interface'
    pwd = 'peint88'

    ip = '172.25.152.125'
    port = '1700'
    service_name = 'tppe.mytna.com'
    dsn = cx_Oracle.makedsn(ip, port, service_name=service_name)
    
    result_list = []
    con = cx_Oracle.connect(user = uid, password = pwd, dsn = dsn)
    cur = con.cursor()
    cur.execute(query)
    
    for result in cur:
        result_list.append(result)
        
    return(result_list)


def checkPRdates(data, PR_rev):
    
    cap = [a.split('_')[0] == 'CAPACITY' for a in data.TagType]
    trans = [a.split('_')[0] == 'TRANSMISSION' for a in data.TagType]
    caps = data[cap]
    trans_data = data[trans]
    
    output = []
    
    if np.sum(cap) > 0:
        cap_error_df = cap_check(caps, PR_rev)
        output.append(cap_error_df)
    
    if np.sum(trans) > 0:
        trans_error_df = trans_check(trans_data, PR_rev)
        output.append(trans_error_df)
  
    return(output)
    
    
def cap_check(data, PR_rev):
    
    ### make dataframe with errors:
    ## missing start, missing stop, bad date, E string, zero tag, tag gap
    pr_num = PR_rev.split('_')[0]
    rev_num = PR_rev.split('_')[1]

    query = "select starttime, stoptime from pwrline.lscmcontract where contractid='" + pr_num + "' and revision=" + rev_num

    PRstart, PRstop = OracleAPI(query)[0]

    missing_start_cap = []
    missing_stop_cap = []
    est_cap = []
    zero_cap = []
    date_gap_cap = []
    date_error_cap = []
    

    for acct in np.unique(data.LDC_Account):
    
        if min(data.StartTime[data.LDC_Account == acct]) > PRstart:
            missing_start_cap.append(('missing start', acct))
    
        if max(data.StopTime[data.LDC_Account == acct]) < PRstop:
            missing_stop_cap.append(('missing stop', acct))
        
        start_checks = [starts.month == 6 and starts.day == 1 for starts in data.StartTime[data.LDC_Account == acct]]
        
        stop_checks = [stops.month == 5 and stops.day == 31 for stops in data.StopTime[data.LDC_Account == acct]]
        
        if (False in start_checks) or (False in stop_checks):
            date_error_cap.append(('bad date', acct))
         
        est_tag = [string == 'E' for string in data.SourceType[data.LDC_Account == acct]]
        
        if (True in est_tag):
            est_cap.append(('estimated tag', acct))
            
        zero_tag = [val == 0 for val in data.Tag[data.LDC_Account == acct]]
        
        if (True in zero_tag):
            zero_cap.append(('zero tag', acct))
            
        diffs = []
        td = data.StopTime[data.LDC_Account == acct].diff()
        td[td.isnull()] = dt.timedelta(days = 365)
        yr_diff = [round(a / dt.timedelta(days = 365)) for a in td]
        
        m = len(yr_diff)
        last_diff = yr_diff.pop()
        if (last_diff != 10) or (len(yr_diff) != (m - 1)):
            date_gap_cap.append(('tag gap', acct))
            
        
        date_error_cap = missing_start_cap + missing_stop_cap + est_cap + zero_cap
        
        e = dict()
        [e [t [0]].append(t [1]) if t [0] in list(e.keys()) 
             else e.update({t [0]: [t [1]]}) for t in date_error_cap]
        
        cap_error_df = pd.DataFrame.from_dict(e, orient = 'index')
        p = len(cap_error_df.columns)
        cap_error_df.columns = ['meter']*p
        cap_error_df = cap_error_df.T
            
    
    if cap_error_df.empty:
        print("cap tags cover PR start & end dates for all accts in", PR_rev)
    
    else:
        print(cap_error_df)
        
    return(cap_error_df)
              

def trans_check(data, PR_rev):
    
    ### make dataframe with errors:
    ## missing start, missing stop, bad date, E string, zero tag, tag gap
    pr_num = PR_rev.split('_')[0]
    rev_num = PR_rev.split('_')[1]

    query = "select starttime, stoptime from pwrline.lscmcontract where contractid='" + pr_num + "' and revision=" + rev_num

    PRstart, PRstop = OracleAPI(query)[0]

    missing_start_trans = []
    missing_stop_trans = []
    est_trans = []
    zero_trans = []
    date_gap_trans = []
    date_error_trans = []
    

    for acct in np.unique(data.LDC_Account):
    
        if min(data.StartTime[data.LDC_Account == acct]) > PRstart:
            missing_start_trans.append(('missing start', acct))
    
        if max(data.StopTime[data.LDC_Account == acct]) < PRstop:
            missing_stop_trans.append(('missing stop', acct))
        
        start_checks = [starts.month == 1 and starts.day == 1 for starts in data.StartTime[data.LDC_Account == acct]]
        
        stop_checks = [stops.month == 12 and stops.day == 31 for stops in data.StopTime[data.LDC_Account == acct]]
        
        if (False in start_checks) or (False in stop_checks):
            date_error_trans.append(('bad date', acct))
         
        est_tag = [string == 'E' for string in data.SourceType[data.LDC_Account == acct]]
        
        if (True in est_tag):
            est_trans.append(('estimated tag', acct))
            
        zero_tag = [val == 0 for val in data.Tag[data.LDC_Account == acct]]
        
        if (True in zero_tag):
            zero_trans.append(('zero tag', acct))
            
        diffs = []
        td = data.StopTime[data.LDC_Account == acct].diff()
        td[td.isnull()] = dt.timedelta(days = 365)
        yr_diff = [round(a / dt.timedelta(days = 365)) for a in td]
        
        m = len(yr_diff)
        last_diff = yr_diff.pop()
        if (last_diff != 10) or (len(yr_diff) != (m - 1)):
            date_gap_trans.append(('tag gap', acct))
            
        
        date_error_trans = missing_start_trans + missing_stop_trans + est_trans + zero_trans
        
        e = dict()
        [e [t [0]].append(t [1]) if t [0] in list(e.keys()) 
             else e.update({t [0]: [t [1]]}) for t in date_error_cap]
        
        trans_error_df = pd.DataFrame.from_dict(e, orient = 'index')
        p = len(trans_error_df.columns)
        trans_error_df.columns = ['meter']*p
        trans_error_df = trans_error_df.T
    
    if trans_error_df.empty:
        print("trans tags cover PR start & end dates for all accts in", PR_rev)
    
    else:
        print(trans_error_df)
        
    return(trans_error_df)
              

def get_report(PR_rev):
    
    pr_num = PR_rev.split('_')[0]
    rev_num = PR_rev.split('_')[1]
    
    query = cp_query(pr_num, rev_num)
    output = OracleAPI(query)

    captag = pd.DataFrame.from_records(output)
    captag.columns = ['CustomerName', 'CustomerID', 'LDC_Account', 'AccountID', 'UIDaccount', 'MarketCode', 'ContractID', 'Revision', 'StartTime', 'StopTime', 'TagType', 'Tag', 'SourceType', 'TimeStamp']

    return(captag)
    
    
def export_report(PR_rev, Write_dir):  

    capreport = get_report(PR_rev)
    os.chdir(Write_dir)
    filename = 'CP_' + PR_rev + '.csv'
    dir_files = [file.split('.')[0] for file in os.listdir(Write_dir)]
    
    if filename.split('.')[0] not in dir_files:
        
        capreport.to_csv(filename, sep = ",", header = True, index = False)
        print('saved file as', filename)

    else:
        
        overwrite = input('file exists, overwrite? (Yes or No)')
        
        if overwrite.lower() == 'yes':
            capreport.to_csv(filename, sep = ",", header = True, index = False)
            print('saved file as', filename)
        
        elif overwrite.lower() == 'no':
            print('file ', filename, 'not saved.')
            
        else:
            print('command not recognized, input "yes" or "no".')
            
            
def download_idr(pr_rev, report, write_dir):

    subfolder = "_".join([pr_rev, 'CH3'])
    idr_dir = os.path.join(write_dir, subfolder)

    try:
        os.mkdir(idr_dir)
        print('created directory: {}'.format(idr_dir))
    
    except:
        print('dir already created.')
    

    meters = np.unique(report.AccountID)
    n = len(meters)
    time = round(3.4*n, 4)

    print('downloading forecasts for {} meters...'.format(len(np.unique(report.AccountID))))
    print('expect to take {} seconds.'.format(time))
    count = 0

    for index, accts in enumerate(meters):
        idr_file = ''.join([accts, '.csv'])
    
        if idr_file not in os.listdir(idr_dir):
            try:
                pipe_import(idr_file, idr_dir)
    
            except:
                print('error w/ download, acct {}.'.format(accts))
    
        elif idr_file in os.listdir(idr_dir):
            pass
    
        count += 1
        if (count > 0 and count % 5 == 0):
            print('downloaded data for {} out of {} meters...'.format(count, n))
        
    print('download complete')
    return(meters, idr_dir)


def merge_idr(meters, idr_dir):

    print("importing and merging .csv's...")
    master_idr = pd.DataFrame()

    for accts in meters:
    
        try:
            idr_file = ''.join([accts, '.csv'])
    
            acct_idr = data_import(idr_file, idr_dir)
            acct_idr.columns = [accts]
    
            master_idr = pd.concat([master_idr, acct_idr], axis = 1)
            master_idr.fillna(0, inplace = True, axis = 1)
    
        except:
            print('import error, acct {}.'.format(accts))

        
    print('read in and merged ch 3.')
    master_idr.head()
    master_idr.fillna(0, inplace = True, axis = 1)
    
    return(master_idr)
            

def offer_summary(master_idr, report, min_cp, min_diff):

    tag_date = dt.datetime.strptime('2019-08-29 17:00:00', '%Y-%m-%d %H:%M:%S')

    act_max = pd.DataFrame(master_idr.apply(max, axis = 0))
    cp_max = master_idr.loc[master_idr.index == tag_date].reset_index(drop = True).T

    start_yrs = [yr.year == 2019 for yr in report.StartTime]
    cap = report[['AccountID', 'Tag']].loc[start_yrs].reset_index(drop = True)
    cap = cap.set_index('AccountID')

    annual_use = pd.DataFrame(.001*master_idr.apply(sum, axis = 0), columns = ['Annual_Use'])

    peak_data = pd.concat([annual_use, act_max, cp_max, cap], axis = 1).round(decimals = 3)
    peak_data.columns = ['Annual_Use_MWh', 'Act_Peak', 'CP', 'Tag']
    peak_data['Act_Tag_Diff'] = (peak_data.Act_Peak - peak_data.Tag)/peak_data.Tag*100
    peak_data['Cap_Tag_Diff'] = (peak_data.CP - peak_data.Tag)/peak_data.Tag*100
    
    print(peak_data)

    tot_vol = round(sum(peak_data.Annual_Use_MWh), 4)
    tot_peak = round(sum(peak_data.Act_Peak), 4)
    tot_CP = round(sum(peak_data.CP), 4)
    tag_tot = round(sum(peak_data.Tag), 4)

    print('PR has total usage of {} MWh.'.format(tot_vol))
    print('PR has an estimated tag total of {} kWh.'.format(tag_tot))
    print('PR has CP peak sum of {} kWh.'.format(tot_CP))
    print('PR has peak (sum(act_peak)) of {} kWh, and {} meters.'.format(tot_peak, len(peak_data.index)))
    high_cp = [p > min_cp for p in peak_data.Tag]
    big_err = [(abs(d) > min_diff) and (d != float('inf')) for d in peak_data.Act_Tag_Diff]
    
    probs = [a and b for a, b in zip(high_cp, big_err)]

    problems = peak_data[probs]
    
    if problems.empty:
        print('no cap tags included - try lowering parameters.')
        return
    
    n = len(problems.index)
    a = math.ceil(math.sqrt(n))
    b = n // a

    if (a * b < n):
        if (a < b):
            a += 1
        else:
            b += 1

    fig, axes = plt.subplots(nrows=a, ncols=b, sharex=True, sharey=False, figsize=(50,30))
    
        
    print('graphing forecasts...')
    
    if n == 1:
        ax = axes
        
        meter = problems.index[0]
        ax.set_title(meter, fontsize = 36);
        plt.rc('font', size = 28)
        meter_df = master_idr.loc[:,meter]
        rec_yr = [a < 2020 for a in meter_df.index.year]
        meter_df[rec_yr].plot(y = meter, ax = ax);
        
    elif n > 1:
        
        try:
            axes_list = [item for sublist in axes for item in sublist]
            axes_list = deque(axes_list)

            for m in problems.index:

                ax = axes_list.popleft();
                ax.set_title(m, fontsize = 36);
                plt.rc('font', size = 28)
                meter_df = master_idr.loc[:,m]
                rec_yr = [a < 2020 for a in meter_df.index.year]
                meter_df[rec_yr].plot(y = m, ax = ax);
                
        except:
            print('plot failed to iterate with params (a, b, n): {}, sorry man (needs debug)'.format((a, b, n)))
        
    
    return(problems)


def pipe_import(filename, path):

    account = filename.split('.')[0]
    ch = '3'
    final = ",".join([account, ch])
    write_path = os.path.join(path,filename)

    cmd_prmpt = ["C:\LODESTAR\Bin\intdexp", "-c", "Data Source=TPPE;User ID=tesi_interface;Password=peint88;LSProvider=ODP;",\
                 "-q", "pwrline", "-f", "C:\LODESTAR\cfg\examples\Cfg\lodestar.cfg", "-s", "01/01/2019", "-t", "12/31/2019",\
                 "-dtuse", "PARTIAL", "-d", "hi", final, "-o", write_path]

    x = Popen(cmd_prmpt, stdout = PIPE, stderr = PIPE)
    output, errors = x.communicate()
    
    
def data_import(file, path):
    
    os.chdir(path)
    data = pd.read_csv(file, header = None, index_col = 0, skiprows = 6)

    data.reset_index(drop = True, inplace = True)
    data.drop(data.columns[[1, 3]], axis = 1, inplace = True)
    data.columns = ['ch3', 'time']
    data.time = pd.to_datetime(data.time)
    data.index = data.time
    data.drop(data.columns[1], axis = 1, inplace = True)
    
    return(data)


def issue_command(command):
    process = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    return process.communicate()
    

def batch_reports(PR_rev_list, Write_dir):
    
    for pr in PR_rev_list:
    
        print(' ')
        print('generating report...')
        report = get_report(pr)
    
        print(' ')
        print('------------------------')
        print(' ')
    
        print('checking dates, PR', pr)
        checkPRdates(report, pr)
    
        print(' ')
        print('------------------------')
        print(' ')
    
        print('exporting report...')
        export_report(pr, Write_dir)
    
        print(' ')
        print('------------------------')





