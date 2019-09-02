from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
import selenium.webdriver.support.ui as ui
from selenium.webdriver.common.keys import Keys
import selenium.webdriver as webdriver
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json
import os


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

def bodies_json(bodies):

    test = pd.DataFrame.from_dict(bodies, orient = 'index')

    if type(test.date[0]) == str:
            test.date = pd.to_datetime(test.date)

    last_days = max(test.date) - dt.timedelta(7) 

    sub = test[test.date > last_days]
        
    accts_success = [len(accts) > 0 for accts in sub.accts]
    accts_fail = [not val for val in accts_success]
        
    good = sub[accts_success].reset_index(drop = True)

    email_error = []

    if len(accts_fail) > 0:
        bad = sub[accts_fail].reset_index()
        mail_error = 'EMAIL_SCRAPE_ERROR.csv'

        bad.to_csv(mail_error, header = True, index = False)

    return(good)

### Submit Login Info

def logon(username, pw, ngrid):

    opts = Options()
    #opts.headless = True
    opts.add_argument('--ignore-certificate-errors')
    opts.add_argument('--start-maximized')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    #opts.binary_location = '/usr/bin/google-chrome'
    prefs = {
                'download.default_directory': '/Users/stevenhurwitt/la-tools-test/IDR_Drop/Downloads',
                'download.prompt_for_download': False,
                'download.directory_upgrade': True,
                'safebrowsing.enabled': False,
                'safebrowsing.disable_download_protection': True}
    #prefs ={"profile.default_content_settings.popups": 0, "download.default_directory": "/home/jupyter-engiela/la-tools-test/IDR_Drop/Downloads/", "directory_upgrade": True}
    opts.add_experimental_option("prefs", prefs)
    #assert opts.headless

    #setup headless browser, get ngrid url
    browser = Chrome(executable_path = '/Users/stevenhurwitt/chromedriver', options = opts)
    
    if ngrid == True:
        url = 'https://ngrid.epo.schneider-electric.com/ngrid/cgi/eponline.exe'
        #url = 'https:\\ngrid.epo.schneider-electric.com\\ngrid\\cgi\\eponline.exe'
        
    if ngrid == False:
        url = 'https://eversource.epo.schneider-electric.com/eversource/cgi/eponline.exe'
        #url = 'https:\\eversource.epo.schneider-electric.com\\eversource\\cgi\\eponline.exe'

    browser.get(url)
       
    #see all elements on pg
    #ids = browser.find_elements_by_xpath('//*[@id]')

    ##Login Page
    #store username, pw, etc
    #send values to login
    #try:
    wait = ui.WebDriverWait(browser,10)
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
    wait = ui.WebDriverWait(browser,10)
    wait.until(lambda browser: browser.find_element_by_id('LastNDays'))

    ##Accounts Page
    #set recent days to be 400
    #could config to use dates....
    lastndays = browser.find_element_by_id('LastNDays')

    #browser.execute_script("arguments[0].value = '400';", recdays)
    browser.execute_script("arguments[0].value = '400'", lastndays)

    print('set to last ', lastndays.get_attribute('value'), ' days.')
    browser.execute_script("document.getElementById('LastNDays').focus();")
    
    return(browser, url)

def idr_download(row, good):

    ngrid = ('SUEZ' in good.user[row])
    browser, url = logon(good.user[row], good.pw[row], ngrid)
    print('logging on...')

    accts_to_find = good.accts[row]
    print('looking for accts {}.'.format(good.accts[row]))
    AIDs = []

    soup = BeautifulSoup(browser.page_source)
    table = soup.find('tbody', {'role' : 'rowgroup'})
    
    if table:
        print('found items in portal.')
    
    else:
        print('login error.')

    #get EPO AID value for every account
    print('trying search & download...')
    for accts in accts_to_find:
        results = big_match(accts, table)
        AIDs.append(results)

    #make list of AIDs, split into list of lists of 5
    final = []
    AIDs

    for aid_list in AIDs:
        for aid in aid_list:
            final.append(aid)

    n = 4
    final2 = [final[i * n:(i + 1) * n] for i in range((len(final) + n - 1) // n )]  
    final2

    browser.implicitly_wait(2)
    for elem in final2:
        export_data(elem, browser, ngrid)
        print('exported {}.'.format(elem))

def export_data(list_of_4, browser, ngrid):
    
    if (len(list_of_4) > 1) and ngrid == False:
        for item in list_of_4:
            check_the_box(item, browser)
            
    elif ngrid == True:
        for item in list_of_4:
            check_the_box(item, browser)
    
    browser.execute_script('''document.frmEPO.button.value='export'; document.frmEPO.submit();''')

    browser.implicitly_wait(20)

    
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

    browser.implicitly_wait(20)
    link = browser.find_element_by_partial_link_text('Hourly Data File')
    link.click()
    print('downloaded EPO data file.')
    
    browser.back()
    browser.back()
    
    for item in list_of_4:
        check_the_box(item, browser)
