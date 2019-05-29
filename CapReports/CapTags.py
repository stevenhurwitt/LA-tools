from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
import selenium.webdriver.support.ui as ui
from selenium.webdriver.common.keys import Keys
import selenium.webdriver as webdriver
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import json
import cx_Oracle
import os

opts = Options()
opts.headless = True
opts.add_argument('--ignore-certificate-errors')
opts.add_argument('--start-maximized')
prefs ={"profile.default_content_settings.popups": 0, "download.default_directory": "C:\\Users\\wb5888\\Python Code\\CapReport\\Downloads\\", "directory_upgrade": True}
opts.add_experimental_option("prefs", prefs)
assert opts.headless
browser = Chrome(executable_path = 'C:\\Users\\wb5888\\chromedriver.exe', options = opts)


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


    missing_start = []
    missing_stop = []
    
    

    for acct in np.unique(data.LDC_Account):
    
        if min(data.StartTime[data.LDC_Account == acct]) > PRstart:
            print('PR start date before first Cap Tag, acct: ', acct)
            missing_start.append(acct)
    
        if max(data.StopTime[data.LDC_Account == acct]) < PRstop:
            print('PR end date after latest Cap Tag, acct: ', acct)
            missing_stop.append(acct)


    if len(missing_start) > 0:
        print("cap tags don't cover PR start for: ", missing_start)
    
    if len(missing_stop) > 0:
        print("cap tags don't cover PR end for: ", missing_stop)

    elif len(missing_start) == len(missing_stop) == 0:
        print("cap tags cover PR start & end dates for all accts in", PR_rev)
    

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





