# # Email Scrape

# Read in emails from outlook, store ones from utility EPO in dictionary, parse dictionary to get another dict of accounts, user-id, password.

import win32com.client
import datetime
import pprint
import json



# ### Functions to Turn Messages to Dictionaries

# In[41]:


def flatten(l):
    out = []
    for item in l:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            if item != '' and item != ' ':
                out.append(item)
    return out

def clean_acct(acct):
    return ''.join(acct.split(' - '))

#turn date string into datetime object
def str_to_date(datestring):
    return (datetime.datetime.strptime(str(datestring).split('+')[0],"%Y-%m-%d %H:%M:%S"))

def date_to_str(datetime_obj):
    return (datetime.datetime.strftime(datetime_obj, format = '%m/%d/%Y %H:%M:%S'))

def Merge(dict1, dict2): 
    res = {**dict1, **dict2} 
    return res 

def ba_sa_merge(a, b): 
    merged = []
    
    for k in range(0, len(a)):
        m_acct = a[k] + '_' + b[k]
        merged.append(m_acct)
    return merged


def aaron_to_dict(body):
    
    parse = body.split('\n')
    parse = [thing.splitlines() for thing in parse]

    parse2 = flatten(parse)
    strp_len = [len(item.strip(' ')) for item in parse2]

    acct_start = parse2.index('Account Number(s): ') + 1
    acct_end = strp_len.index(0)

    for item in parse2:

        split_item = item.split(': ')
        if split_item[0]  == 'User Id':
            user = ('user', split_item[1])
        
        elif split_item[0] == 'Password':
            pw = ('pw', split_item[1])
            
        elif split_item[0] == 'Customer':
            name = ('name', split_item[1])
        
        
    accts = parse2[acct_start:acct_end]
    clean_accts = [clean_acct(acct) for acct in accts]
    accts_tup = ('accts', clean_accts)
    row = [user, pw, name, accts_tup]
    return row

def admin_to_dict(body):
    
    parse = body.split('\n')
    parse = [thing.splitlines() for thing in parse]

    parse2 = flatten(parse)
    
    if (len(body) > 2500) or ("SA Number" in body):

        try:
            ba_list = [''.join(elem.split('-')) for elem in parse2[9].split(' ')]

            sa_list = [elem for elem in parse2[10].split(' ')]
            
            accts = ('accts', ba_sa_merge(ba_list, sa_list))

            for item in parse2:

                split_item = item.split(': ')
                if split_item[0]  == 'User Id':
                    user = ('user', split_item[1])
        
                elif split_item[0] == 'Password':
                    pw = ('pw', split_item[1])
            
                elif split_item[0] == 'Customer':
                    name = ('name', split_item[1])
                    
            row = [user, pw, name, accts]
                
        except:
            
            for index, item in enumerate(parse2):
                if item.lower() == 'user id':
                    special_index = index

            name = ('name', parse2[special_index + 1])
            ba_list = parse2[special_index + 2].split(' ')
            sa_list = parse2[special_index + 3].split(' ')
            user = ('user', parse2[special_index + 4])
            pw = ('pw', parse2[special_index + 6])

            ba_list = [ba.replace('-', '') for ba in ba_list]
                    
            accts = ('accts', ba_sa_merge(ba_list, sa_list))
                    
            row = [user, pw, name, accts]        
                    
                    
                    

    else:
        for item in parse2:

            split_item = item.split(': ')
        
            if split_item[0]  == 'User ID':
                user = ('user', split_item[1])
        
            elif split_item[0] == 'Password':
                pw = ('pw', split_item[1])
            
            elif split_item[0] == 'Customer':
                name = ('name', split_item[1])

            elif split_item[0] == 'Account Number':
                acct_list = [str(acct) for acct in split_item[1].split(' ')]
                accts = ('accts', acct_list)
        
        row = [user, pw, name, accts]
    
    return(row)

def ngrid_to_dict(body):
    
    parse = body.split('\n')
    parse = [thing.splitlines() for thing in parse]

    parse2 = flatten(parse)
    accts_list = []
    names_list = []
        
    for item in parse2:
        split_item = item.split(': ')
        if split_item[0] == 'Userid':
            username = split_item[1].replace(' ', '')
            user = ('user', username)
        
        if split_item[0] == 'Password':
            password = split_item[1].replace(' ', '')
            pw = ('pw', password)
        
        try:
            split_item2 = [thing.split(' ') for thing in split_item][0]
            split_item3 = list(filter(None, split_item2))
            int(split_item3[0])
            accts_list.append(split_item3[0])
            full_name = ' '.join(split_item3[1:])
            names_list.append(full_name)
        
        except:
            pass
        
    accts = ('accts', accts_list)
    name = ('name', names_list)
        
    row = [user, pw, name, accts]
    return(row)

def iter_mail(sender_func, mailbox, index):
    
    today = datetime.datetime.now()
    last_day = today - datetime.timedelta(1)
    last_day

    last_day_str = last_day.strftime("%Y-%m-%d %H:%M:%S")
    last_day_str2 = last_day_str +"+00:00'"

    print('scraping emails...')
    #start iterating through emails
    mail = mailbox.GetLast()
    i = index
    
    msg_row = sender_func(mail.Body)
    msg_row.append(('date', str_to_date(mail.ReceivedTime)))

    master = [(i, dict(msg_row))]
    
    i += 1
   
    while mail:
        
        mail = mailbox.GetPrevious()
        
        try:
            msg_row = sender_func(mail.Body)
            msg_row.append(('date', str_to_date(mail.ReceivedTime)))
            
            msg_row = dict(msg_row)

            new = (i, msg_row)
        
            master.append(new)
            
            i += 1
            
        
        except:
            pass
            
    master = dict(master)
            
    return(master, i)

    


# ### Get Filtered Messages (Sender, Date, Body)

# aaron_to_dict will gives a dictionary for each account of the structure  
#   
#     {'user': 'nhengi-ston3n',
#          'pw': 'fm54f7',
#          'name': 'STONEWALL KITCHEN LLC',
#          'accts': ['800531501']}  
#  
#  The following code will add the date email is received (if within last week) w/ structure: 
#    
#      {'user': 'nhengi-ston3n',
#          'pw': 'fm54f7',
#          'name': 'STONEWALL KITCHEN LLC',
#          'accts': ['800531501'],
#          'date': datetime.datetime(2019, 5, 16, 10, 17, 21)}
#  
#  nest these dictionaries into a scary looking dictionary:
#  
#      {0: {'user': 'nhengi-ston3n', 
#             'pw': 'fm54f7', 
#             'name': 'STONEWALL KITCHEN LLC', 
#             'accts': ['800531501'], 
#             'date': datetime.datetime(2019, 5, 16, 10, 17, 21)}, 
#       1: {'user': 'nhengi-morg1n', 
#             'pw': '6z2s5e', 
#             'name': 'MORGAN ADVANCED CERAMICS', \
#             'accts': ['800514701'], 
#             'date': datetime.datetime(2019, 4, 17, 12, 34, 30)}, 
#       2: {'user': 'nhengi-ferr1n', 
#             'pw': 'xx27b4', 
#             'name': 'FERROTEC AMERICA CORP', 
#             'accts': ['800511301'], 
#             'date': datetime.datetime(2019, 4, 17, 12, 7, 55)},
#         
#             #.....

# In[42]:


def get_emails():

    global inbox
    
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    inbox = outlook.Folders('ENGIENA-GSERNASERVICES (ENGIE North America)')
    folder = inbox.Folders("Inbox")
    messages = folder.Items

    aaron_filter = "[SenderEmailAddress] = 'aaron.downing@eversource.com'"
    admin_filter = "[SenderEmailAddress] = 'epoadmin@eversource.com'"
    ngrid_filter = "[SenderEmailAddress] = 'michael.stanton@nationalgrid.com'"

    today = datetime.datetime.now()

    aaron = messages.Restrict(aaron_filter)
    admin = messages.Restrict(admin_filter)
    ngrid = messages.Restrict(ngrid_filter)
    
    print(len(aaron), 'aaron emails,', len(admin), 'epo emails', len(ngrid), 'ngrid emails')
    
    j = 0
    error = 0
    
    try:
        scrape, j2 = iter_mail(aaron_to_dict, aaron, j)
        
    except:
        print('error parsing aaron')
        error += 1
        
    
    try:
        admin_scrape, j3 = iter_mail(admin_to_dict, admin, j2)
        master = Merge(scrape, admin_scrape)
        
    except:
        print('error parsing EPO admin')
        error += 1
        master = {}
    
    try:
        ngrid_scrape, j4 = iter_mail(ngrid_to_dict, ngrid, j3)
        master = Merge(master, ngrid_scrape)
        
    except:
        print('error parsing ngrid')
        error += 1
        if error == 3:
            quit
    

    print('error with', error, 'of 3 inboxes')
    print('found ', len(master.keys()), ' new emails: ')
    
    pretty_json = json.dumps(master, default = lambda date: date_to_str(date), sort_keys = True, indent = 4)
    lame_json = json.dumps(master, default = lambda date: date_to_str(date), sort_keys = True)

    json_name = 'email_bodies_' + date_to_str(today).split(' ')[0].replace('/', '_') + '.json'

    print('writing .json object')
    
    with open(json_name, 'w') as f:
        json.dump(lame_json, f)
    
    #print(pretty_json)
    return master, json_name
    #https://docs.microsoft.com/en-us/office/vba/api/outlook.mailitem


def main():
    output_dict, filename = get_emails()
    print('file saved as', filename)
    return(output_dict)

main()
