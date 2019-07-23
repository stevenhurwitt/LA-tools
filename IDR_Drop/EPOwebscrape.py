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


### Submit Login Info

def logon(username, pw, ngrid):

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
    
    return(browser, url)


def export_data(list_of_5, browser):
    
    for item in list_of_5:
        check_the_box(item, browser)
    
    browser.execute_script('''document.frmEPO.button.value='export'; document.frmEPO.submit();''')

    wait = ui.WebDriverWait(browser,10)
    wait.until(lambda browser: browser.find_element_by_id('userid'))

    try:
        browser.execute_script('''document.frmEPO.intervaltype[0].checked = false; document.frmEPO.intervaltype[1].checked = true;''')

        browser.execute_script('''document.frmEPO.intervaltype[0].checked = false; document.frmEPO.intervaltype[1].checked = true;''')
        browser.execute_script('''document.frmEPO.button.value="contin"''')
        browser.execute_script('''; document.frmEPO.submit();''')

        link = browser.find_element_by_link_text('Hourly Data File(60 minutes per interval)(CSV)')
        link.click()
        print('downloaded EPO data file.')
    
        browser.back()
        browser.back()
    
        for item in list_of_5:
            check_the_box(item, browser)

    except:

                print('download failed')
                browser.back()
    
                for item in list_of_5:
                    check_the_box(item, browser)
