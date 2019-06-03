
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import datetime as dt
import json
import cx_Oracle
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

    uid = 'tesi_interface'
    pwd = 'peint88'

    ip = '172.25.152.125'
    port = '1700'
    service_name = 'tppe.mytna.com'
    dsn = cx_Oracle.makedsn(ip, port, service_name=service_name)

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
    
    pr_num = PR_rev.split('_')[0]
    rev_num = PR_rev.split('_')[1]

    query = "select starttime, stoptime from pwrline.lscmcontract where contractid='" + pr_num + "' and revision=" + rev_num

    PRstart, PRstop = OracleAPI(query)[0]


    missing_start_cap = []
    missing_stop_cap = []
    date_error_cap = []
    date_error_cap = []
    
    missing_start_trans = []
    missing_stop_trans = []
    date_error_trans = []
    

    for acct in np.unique(data.LDC_Account):
    
        if min(data.StartTime[data.LDC_Account == acct]) > PRstart:
            print('PR start date before first Cap Tag, acct: ', acct)
            missing_start_cap.append(acct)
    
        if max(data.StopTime[data.LDC_Account == acct]) < PRstop:
            print('PR end date after latest Cap Tag, acct: ', acct)
            missing_stop_cap.append(acct)
        
        start_checks = [starts.month == 6 and starts.day == 1 for starts in data.StartTime[data.LDC_Account == acct]]
        
        stop_checks = [stops.month == 5 and stops.day == 31 for stops in data.StopTime[data.LDC_Account == acct]]
        
        if (False in start_checks) or (False in stop_checks):
            date_error_cap.append(acct)
         
            
    if len(date_error_cap) > 0:
        print('date error:', date_error_cap)

    if len(missing_start_cap) > 0:
        print("cap tags don't cover PR start for: ", missing_start_cap)
    
    if len(missing_stop_cap) > 0:
        print("cap tags don't cover PR end for: ", missing_stop_cap)

    elif len(missing_start_cap) == len(missing_stop_cap) == 0:
        print("cap tags cover PR start & end dates for all accts in", PR_rev)
        
    date_error_np = np.unique(np.array(date_error_cap))
    return(date_error_np)
    

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
            
            
def fix_my_PR(cap_report):

    sub = cap_report.loc[:,['AccountID', 'ContractID', 'Revision', 'StartTime', 'StopTime']]

    start_checks = [starts.month == 6 and starts.day == 1 for starts in sub.StartTime]
    stop_checks = [stops.month == 5 and stops.day == 31 for stops in sub.StopTime]

    sub['start_flag'] = start_checks
    sub['stop_flag'] = stop_checks

    sub.head()

    def fix_starts(data):
    
        if not data.start_flag:
            new_date = dt.datetime(year = data.StartTime.year, month = 6, day = 1)
            return(new_date)
    
        else:
            return(data.StartTime)

    def fix_stops(data):
    
        if not data.stop_flag:
            new_date = dt.datetime(year = data.StopTime.year, month = 5, day = 31)
            return(new_date)
    
        else:
            return(data.StopTime)

    prob_start = []
    prob_stop = []

    for index, row in sub.iterrows():
        up_start = fix_starts(row)
        up_stop = fix_stops(row)
    
        prob_start.append((index, up_start))
        prob_stop.append((index, up_stop))
    
    prob_start = pd.DataFrame.from_records(prob_start)
    prob_stop = pd.DataFrame.from_records(prob_stop)

    prob_start.columns = ['index', 'right start']
    prob_stop.columns = ['index', 'right stop']

    problems = pd.concat([prob_start, prob_stop], join = 'inner', axis = 1).drop(['index'], axis = 1)

    updated = sub.join(problems)
    choose = [not (a and b) for a, b in zip(updated.start_flag, updated.stop_flag)]
    final = updated[choose]
    
    return(final)
            
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





