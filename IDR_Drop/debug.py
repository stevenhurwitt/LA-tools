import pandas as pd
import numpy as np
import json
import os
import datetime
from bs4 import BeautifulSoup
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
import selenium.webdriver.support.ui as ui
from selenium.webdriver.common.keys import Keys
import selenium.webdriver as webdriver
from pandas.io.json import json_normalize

##################################


# In[188]:


def acct_match(table_acct, str_acct):
    return((table_acct in str_acct) or (str_acct in table_acct))

def big_match(str_acct, table):
    
    linking = []
    for row in table.findAll('tr'):
        cells = row.findAll('td')
        account = cells[1].find(text = True)
        
        if acct_match(account, str_acct):
            cells[0].input['selected'] = 'true'
            found = cells[0].input['value']
            
            linking.append(found)

    return linking

def check_the_box(value, browser):
    checkboxes = browser.find_elements_by_xpath("//input[@type='checkbox']")

    for checkbox in checkboxes:
        if checkbox.get_attribute('value') == value:
            checkbox.click()




##################################

base_path = os.getcwd()

last_days = datetime.datetime.today() - datetime.timedelta(3) 

with open('email_bodies_05_23_2019.json', 'r') as f:
        emails = json.load(f)
        emails = json.loads(emails)
            
test = pd.DataFrame.from_dict(emails, orient = 'index')
if type(test.date[0]) == str:
            test.date = pd.to_datetime(test.date)
            
sub = test[test.date > last_days]
        
accts_success = [len(accts) > 0 for accts in sub.accts]
accts_fail = [not val for val in accts_success]
        
good = sub[accts_success].reset_index()

email_error = []

if len(accts_fail) > 0:
    bad = sub[accts_fail].reset_index()
    mail_error = 'EMAIL_SCRAPE_ERROR.csv'

    os.chdir(base_path + '/Logs')
    bad.to_csv(mail_error, header = True, index = False)
    os.chdir(base_path)

for item in good.index.values:
        
    username = good.user[item]
    pw = good.pw[item]

    if type(good.accts[item]) == str:
        accts_to_find = good.accts[item].split(" ")

    else:
        accts_to_find = good.accts[item]

    ngrid = ('SUEZ' in username)

    print('looking for', accts_to_find)

    opts = Options()
    opts.headless = True
    opts.add_argument('--ignore-certificate-errors')
    opts.add_argument('--start-maximized')

    try:
        prefs ={"profile.default_content_settings.popups": 0, "download.default_directory": "C:\\Users\\wb5888\\Python Code\\IDR_Drop\\Downloads\\", "directory_upgrade": True}

    except:
        prefs ={"profile.default_content_settings.popups": 0, "download.default_directory": "/Volumes/USB30FD/Python Code/IDR_Drop/Downloads//", "directory_upgrade": True}

    opts.add_experimental_option("prefs", prefs)
    assert opts.headless

    #setup headless browser, get ngrid url
    try:
        browser = Chrome(executable_path = 'C:\\Users\\wb5888\\chromedriver.exe', options = opts)

    except:
        browser = Chrome(executable_path = '/Users/stevenhurwitt/chromedriver', options = opts)
    
    if ngrid == True:
        url = 'https://ngrid.epo.schneider-electric.com/ngrid/cgi/eponline.exe'
        
    if ngrid == False:
        url = 'https://eversource.epo.schneider-electric.com/eversource/cgi/eponline.exe'
        
    browser.get(url)
       
    #see all elements on pg
    #ids = browser.find_elements_by_xpath('//*[@id]')

    ##Login Page
    #store username, pw, etc
    #send values to login
    #try:
    wait = ui.WebDriverWait(browser,30)
    wait.until(lambda browser: browser.find_element_by_id('userid'))
    
    user = browser.find_element_by_id('userid')
    password = browser.find_element_by_id('password')
    login = browser.find_element_by_id('contin')

    #except:
        #user = browser.find_element_by_xpath("\\div[@id='login']\\ul[@class='form row-fluid']\\li[@class='col-xs-12 col-sm-6']")
        #password = browser.find_element_by_xpath(".ul.form.row-fluid.li.col-xs-12.col-sm-6[input.id = password]")
        #login = browser.find_element_by_xpath(".ul.form.row-fluid.li.col-xs-12.col-sm-6[input.id = contin]")

    user.send_keys(username)
    password.send_keys(pw)

    #sanity check
    print('user: ', user.get_attribute('value'))
    print('password: ', password.get_attribute('value'))
    print('logging on...')
    login.click()
    browser.execute_script('''function submitlogin(event) {document.frmEPO.submit();}''' )
    wait = ui.WebDriverWait(browser,30)
    wait.until(lambda browser: browser.find_element_by_id('LastNDays'))

    ##Accounts Page
    #set recent days to be 400
    #could config to use dates....
    lastndays = browser.find_element_by_id('LastNDays')

    #browser.execute_script("arguments[0].value = '400';", recdays)
    browser.execute_script("arguments[0].value = '400'", lastndays)

    print('set to last ', lastndays.get_attribute('value'), ' days.')
    browser.execute_script("document.getElementById('LastNDays').focus();")
    
    AIDs = []

    soup = BeautifulSoup(browser.page_source, features = 'html5lib')
    table = soup.find('tbody', {'role' : 'rowgroup'})

    if type(accts_to_find) == str:
        accts_to_find = accts_to_find.split(" ")


    #get EPO AID value for every account
    for accts in accts_to_find:
        results = big_match(accts, table)
        AIDs.append(results)

    #make list of AIDs, split into list of lists of 5
    final = []

    for aid_list in AIDs:
        for aid in aid_list:
            final.append(aid)

    try:
        n = 5
        final2 = [final[i * n:(i + 1) * n] for i in range((len(final) + n - 1) // n )]  

        for item in final2:
            check_the_box(item, browser)
    
            browser.execute_script('''document.frmEPO.button.value='export'; document.frmEPO.submit();''')

            wait = ui.WebDriverWait(browser,10)
            wait.until(lambda browser: browser.find_element_by_id('userid'))

            print('disabling demand...')
            browser.execute_script('''function disabledemand() {if (document.frmEPO.demand) {
		document.frmEPO.demand.disabled=true;
		document.frmEPO.demand.checked=false;}}; disabledemand''')
            
            print('selecting hourly interval...')
            browser.execute_script('''function setintervaltype() {if (document.frmEPO.demand && document.frmEPO.intervaltype[0]) {
		if ( document.frmEPO.demand.checked == true ) {
			if ( document.frmEPO.intervaltype[1].checked == true ) {alert("Convert to Demand can only be selected with the Native Interval Length. [Un-check Convert to Demand if Hourly data is desired]");}
			document.frmEPO.intervaltype[0].checked = true;
			document.frmEPO.intervaltype[1].checked = false;
			document.frmEPO.intervaltype[0].disabled = true;
			document.frmEPO.intervaltype[1].disabled = true;}
		else {document.frmEPO.intervaltype[0].disabled = false;
			document.frmEPO.intervaltype[1].disabled = false;
			document.frmEPO.intervaltype[1].checked = true;
			document.frmEPO.intervaltype[0].checked = false;}}}; setintervaltype()''')

            print('submitting...')
            browser.execute_script('''document.frmEPO.button.value="contin"''')
            browser.execute_script('''document.frmEPO.submit();''')

            wait = ui.WebDriverWait(browser,10)
            wait.until(lambda browser: browser.find_element_by_class_name('tableContain'))

            dl_file_list = soup.findAll('div', {'class': 'tableContain'})
            dl_file = [find('a').get_attrs('href') for hyper in dl_file_list]
            print('looking for file:', dl_file)

            print('finding download link...')
            link = browser.find_element_by_partial_link_text('"Hourly Data File"')
            link.click()
            print('downloaded EPO data file.')

            browser.back()
            browser.back()

            wait = ui.WebDriverWait(browser,10)
            wait.until(lambda browser: browser.find_element_by_id('userid'))
    
            for item in final2:
                check_the_box(item, browser)


    except:
        n = 2
        final2 = [final[i * n:(i + 1) * n] for i in range((len(final) + n - 1) // n )]  

        for item in final2:
            check_the_box(item, browser)
    
            browser.execute_script('''document.frmEPO.button.value='export'; document.frmEPO.submit();''')

            wait = ui.WebDriverWait(browser,10)
            wait.until(lambda browser: browser.find_element_by_id('userid'))

            print('disabling demand')
            browser.execute_script('''function disabledemand() {if (document.frmEPO.demand) {
		document.frmEPO.demand.disabled=true;
		document.frmEPO.demand.checked=false;}}; disabledemand()''')
            
            print('selecting hourly interval...')
            browser.execute_script('''function setintervaltype() {if (document.frmEPO.demand && document.frmEPO.intervaltype[0]) {
		if ( document.frmEPO.demand.checked == true ) {
			if ( document.frmEPO.intervaltype[1].checked == true ) {alert("Convert to Demand can only be selected with the Native Interval Length. [Un-check Convert to Demand if Hourly data is desired]");}
			document.frmEPO.intervaltype[0].checked = true;
			document.frmEPO.intervaltype[1].checked = false;
			document.frmEPO.intervaltype[0].disabled = true;
			document.frmEPO.intervaltype[1].disabled = true;}
		else {document.frmEPO.intervaltype[0].disabled = false;
			document.frmEPO.intervaltype[1].disabled = false;
			document.frmEPO.intervaltype[1].checked = true;
			document.frmEPO.intervaltype[0].checked = false;}}}; setintervaltype()''')
            

            print('submitting')
            browser.execute_script('''document.frmEPO.button.value="contin"''')
            browser.execute_script('''document.frmEPO.submit();''')

            wait = ui.WebDriverWait(browser,10)
            wait.until(lambda browser: browser.find_element_by_class_name('tableContain'))

            dl_file_list = soup.findAll('div', {'class': 'tableContain'})
            dl_file = [find('a').get_attrs('href') for hyper in dl_file_list]
            print('looking for file:', dl_file)

            print('finding download link')
            link = browser.find_element_by_partial_link_text('"Hourly Data File"')
            link.click()
            print('downloaded EPO data file.')

            browser.back()
            browser.back()

            wait = ui.WebDriverWait(browser,10)
            wait.until(lambda browser: browser.find_element_by_id('userid'))
    
            for item in final2:
                check_the_box(item, browser)
