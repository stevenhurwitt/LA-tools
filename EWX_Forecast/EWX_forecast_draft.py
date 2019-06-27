
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import json
import pprint
import os
pp = pprint.PrettyPrinter(1)
basepath = os.getcwd()
json_files = os.path.join(basepath, 'json')
csv_files = os.path.join(basepath, 'csv')

print('working in {}'.format(basepath))


def sortdir(filepath, num):
    os.chdir(filepath)
    files = np.array(os.listdir())
    time = []
    for file in files:
        try:
            sys_time = round(os.path.getmtime(filepath + "\\" + file))
        except:
            sys_time = round(os.path.getmtime(filepath + "/" + file))
        
        time.append(dt.datetime.fromtimestamp(sys_time))

    time = np.array(time)
    lab = ['files']
    filedf = pd.DataFrame(files, columns = lab)

    filedf['time'] = time
    filedf = filedf.sort_values(by = 'time', axis = 0, ascending = False).reset_index(drop = True)

    print("files found in dir: ", filepath)
    print(filedf.head(num))
    return(filedf.head(num))

def parse_engie(payload):

    with open(payload) as raw:
        idr_engie = json.load(raw)

    trnx = idr_engie['transactioninfo']
    acct = idr_engie['account']

    print('saving data files')
    try:
        ts_sca_data = acct['timeseriesdatascalar']
        sca_payload = pd.DataFrame.from_dict(ts_sca_data).iloc[:,1:]
        sca_payload['start'] = pd.to_datetime(sca_payload.start)
        sca_payload['stop'] = pd.to_datetime(sca_payload.stop)
        sca_payload.v = [float(val) for val in sca_payload.v]
    
    except:
        sca_payload = None
    
    ts_idr_data = acct['timeseriesdataidr']
    n = len(ts_idr_data)

    ch = ts_idr_data[0]['channel']
    hb = ts_idr_data[0]['heartbeat']
    print('found', hb, 'heartbeats')
    idr_payload = pd.DataFrame.from_dict(ts_idr_data[0]['reads'])
    idr_payload.v = [float(val) for val in idr_payload.v]
    
    idr_payload.t = pd.to_datetime(idr_payload.t)
    idr_payload = idr_payload.set_index(pd.DatetimeIndex(idr_payload.t))
    idr_payload = idr_payload.drop('t', axis = 1)
           
    print('found {} reads, creating dataset.'.format(n))

        
    for i in range(1,n):
        reads = ts_idr_data[i]['reads']
        temp = pd.DataFrame.from_dict(reads)
        
        temp.v = [float(val) for val in temp.v]
    
        temp.t = pd.to_datetime(temp.t)
        temp = temp.set_index(pd.DatetimeIndex(temp.t))
        temp = temp.drop('t', axis = 1)
            
            #tempname = "_".join([filename.split('.')[0], 'year', str(i), '.csv'])
            #print('writing {}'.format(tempname))
            
            #temp.to_csv(tempname, header = True, index = False)
        idr_payload = pd.concat([idr_payload, temp], axis = 0)
        
    print(idr_payload.head())
    print('...')
    print(idr_payload.tail())
    

    print('saving meterid and cap tags')
    meterid = '_'.join([acct['market'], acct['discocode'], acct['accountnumber']])
    
    try:
        caps = acct['captag'][0]

        caps_df = pd.DataFrame.from_records(caps, index = [0]).iloc[:,2:]
        caps_df['start'] = pd.to_datetime(caps_df['start'])
        caps_df['stop'] = pd.to_datetime(caps_df['stop'])
        caps_df.v = [float(val) for val in caps_df.v]
    
    except:
        caps_df = None

    return(idr_payload, int(hb), sca_payload, caps_df, meterid)


def parse_ewx(file):
    
    with open(file) as raw:
        print("loading json...")
        data = json.load(raw) #raw json file
        
    acct = data['account'] #get account data
    ch3 = acct['timeseriesdataidr'] #dictionary of acct attributes
    n = len(ch3)
        
    reads = ch3[0]['reads']
    master_df = pd.DataFrame.from_dict(reads)
    master_df.t = pd.to_datetime(master_df.t)
        
    print('found {} reads, creating dataset.'.format(n))
        
    for i in range(1,n):
        reads = ch3[i]['reads']
        temp = pd.DataFrame.from_dict(reads)
        temp.t = pd.to_datetime(temp.t)
        master_df = pd.concat([master_df, temp]).reset_index(drop = True)
    
    print("saving to dataframe...")
    
    master_df = master_df.set_index(master_df.t)
    master_df = master_df.drop('t', axis = 1)
    
    return(master_df)

#### validation functions ####

def periodic_zero(idr, margin, threshold):
    
    tmp = idr.copy()

    tmp['d'] = [time.dayofweek for time in tmp.index]
    tmp['h'] = [time.hour for time in tmp.index]

    #bool if value less than margin
    zeroreadmask = tmp['v'] <= margin #margin = .01
    
    #group zero reads by weekday and hour
    day_hr = list(zip(tmp.index.dayofweek, tmp.index.hour))
    zero_read_group = zeroreadmask.groupby([tmp.index.dayofweek, tmp.index.hour])

    #find proportion of zero reads
    weekly_periodic_reads = pd.DataFrame(zero_read_group.sum().astype(int) / zero_read_group.count())
    
    weekly_periodic_reads.index.names = ['d', 'h']
    weekly_periodic_reads.columns = ['pz']
    
    zeros = pd.merge(tmp, weekly_periodic_reads, how = 'left', right_index = True, left_on = ['d', 'h'])
    
    low_reads = [(zero > 0 and zero < threshold) for zero in zeros.pz]
    zeros['lr'] = low_reads
    
    return(zeros)

def interval_gap_check(tmp2):
    val_diff = tmp2.v.diff().fillna(value = 0)
    time_diff = tmp2.index.to_series().diff()
    time_diff = time_diff.dt.seconds.div(3600, fill_value = 3600)

    tmp2['vd'] = val_diff
    tmp2['td'] = time_diff
    
    #check interval gaps
    gap_after_index = [(float(td) != 1) for td in time_diff]
    tmp2['gap'] = gap_after_index
    
    return(tmp2)


def variance_validation(tmp2, time_window, centered, n_sd):

    tmp2['rm'] = tmp2['v'].rolling(window = time_window, min_periods = 1, center = centered).mean()
    tmp2['mc'] = tmp2.v - tmp2.rm

    tmp2['crm'] = tmp2['mc'].rolling(window = time_window, min_periods = 10, center = centered).mean()
    tmp2['crsd'] = tmp2['mc'].rolling(window = time_window, min_periods = 10, center = centered).std()

    tmp2['var'] = (tmp2['mc'] - tmp2['crm'])/tmp2['crm']

    tmp2['spike'] = tmp2['mc'] > (tmp2['crm'] + (n_sd + 1) * tmp2['crsd'])
    tmp2['dip'] = tmp2['mc'] < (tmp2['crm'] - n_sd * tmp2['crsd'])
    
    return(tmp2)


def dst_check(tmp2):
    beg_for = dt.datetime.strptime('03/08/2018', '%m/%d/%Y')
    end_for = dt.datetime.strptime('03/14/2018', '%m/%d/%Y')
    beg_back = dt.datetime.strptime('11/01/2018', '%m/%d/%Y')
    end_back = dt.datetime.strptime('11/07/2018', '%m/%d/%Y')
    
    date_check = [(((date >= beg_for) and (date <= end_for)) or ((date >= beg_back) and (date <= end_back))) for date in tmp2.index]
    
    time_check = [diff != 1 for diff in tmp2.td]
    
    dst = [a and b for a, b in zip(date_check, time_check)]
    tmp2['dst'] = dst
    
    return(tmp2)

##fix bad time interval (15 min, etc)
def fix_interval(data):
    
    times = data.index.to_series()
    times = pd.to_datetime(times)

    minute = [int(v.minute) for v in times]
    data['min'] = minute

    date = list(zip(data.d, data.h))
    data['date_zip'] = date
    
    bad_time = data.loc[data.td != 1,:]
    good_time = data.loc[data.td == 1,:]
    
    bt_group = bad_time.groupby([bad_time.date_zip])

    final = bad_time.loc[bad_time['min'] == 0,:]
    adj_v = [4*val for val in final.v]

    final['v'] = adj_v

    final_out = pd.concat([good_time, final], axis = 0)
    final_out.sort_index(inplace = True)

    return(final_out)

#### estimation functions ####

def dst_fix(tmp2):
    for i, index in enumerate(tmp2.index):
        
        if (tmp2.dst[index] == True) and (tmp2.td[index] == 0):
            tmp2.drop(label = index, axis = 0)
            
        elif (tmp2.dst[index] == True) and (tmp2.td[index] == 2):
            add_time = index + dt.timedelta(hours = 1)
            tmp2.index.insert((i+1), add_time)
        
        return(tmp2)
    
def interp(vals, flag):
    need_interp = vals.copy()
    for j, error in enumerate(flag):
        if error:
            need_interp[j] = np.nan
    need_interp.columns = 'interp'
    return(need_interp)

def gen_year(data, num_days):
    most_recent = max(data.index)
    year_back = most_recent - dt.timedelta(days = num_days, hours = most_recent.hour)
    oldest = min(data.index)
    gap = oldest - year_back
    gap_hr = int(divmod(gap.total_seconds(), 3600)[0])
    
    year_data = data[year_back:most_recent]
    agg = year_data.groupby(['mon', 'd', 'h'])['lin'].median()
    year_forward = most_recent + dt.timedelta(days = 364, hours = 24 - most_recent.hour)
    delta = year_forward - most_recent
    delta_hr = int(divmod(delta.total_seconds(), 3600)[0])

    next_year = []
    for i in range(1, delta_hr):
        next_year.append(most_recent + dt.timedelta(hours = i))

    month = [a.month for a in next_year]
    day = [a.dayofweek for a in next_year]
    hour = [a.hour for a in next_year]

    forecast = pd.DataFrame({'t':next_year, 'mon':month, 'd':day, 'h':hour, 'date_zip':list(zip(month, day, hour))})
    forecast.set_index('t', drop = True, inplace = True)
    forecast['lin'] = agg[forecast.date_zip].reset_index(drop = True).values.tolist()
    return(forecast)

def timeshift(data, until):
    year = 0
    master = gen_year(data, 364)
    print('forecasted year {} of {} with {} reads.'.format(year, until, len(master.lin)))
    year += 1
    
    while year < until:
        if (year % 6 == 0 and year > 0):
            num_days = 371
        else:
            num_days = 364
        
        forecast = gen_year(master, num_days)
        master = pd.concat([master, forecast], axis = 0)
        print('forecasted year {} of {} with {} reads.'.format(year, until, len(forecast.lin)))
        year += 1
    
    master = master['lin']
    return(master)



def forecast_main(json_file, read, write):
    
    print('parsing data files...')
    #parse json file
    try:
        name = json_file.split('_')[1:]
        filename = '_'.join(name)
        filename = filename.replace('json', 'csv')
        print('using filename {}.'.format(filename))
    
    except:
        filename = 'forecast_ch3.csv'
        print('using filename {}.'.format(filename))
    
    if read is not None and type(json_file) == str:
        os.chdir(read)
        idr, hb, sca, caps, meter = parse_engie(json_file)
        idr = idr.loc[pd.notnull(idr.index),:]
        print('read {} from {}.'.format(filename, read))
    
    else:
        idr = json_file
    
    print('running data validations...')
    #check for nonperiodic zeros
    tmp2 = periodic_zero(idr, .01, 1)
    print('...')
    
    tmp2['mon'] = [a.month for a in tmp2.index]
 
    #get value & time differences
    tmp2 = interval_gap_check(tmp2)
    print('...')
    
    #check spikes & dips
    #time_window = int(60*24*3600/hb)
    time_windows = int(30*24)
    centered = True
    n_sd = 2

    tmp2 = variance_validation(tmp2, time_window, centered, 2)
    print('...')
    
    #check for dst (missing hour 3/8-3/14 and extra value 11/1-11/7)
    tmp2 = dst_check(tmp2)
    print('...')
    
    #fix nonhour interval reads
    tmp2 = fix_interval(tmp2)
    print('...')
    
    print('usage validated.')
    
    print('running usage estimation flags...')
    data_filter = [a or b or c or d for a, b, c, d in zip(tmp2.lr, tmp2.gap, tmp2.spike, tmp2.dip)]
    tmp2['err'] = data_filter
    
    tmp2['interp'] = interp(tmp2.v, tmp2.err)
    
    linear = tmp2.interp.interpolate(method = 'linear', axis = 0, in_place = False, limit_direction = 'forward')
    tmp2['lin'] = linear
    
    #final validated data
    final = tmp2.copy()
    
    
    if write is not None:
        print('writing validated usage file to .csv...')
        os.chdir(write)
        final.to_csv(filename, header = True, index = True)
        print('wrote {} to {}.'.format(filename, write))
        
        
    print('forecasting...')
    forecast = timeshift(final, 10)
    
    if write is not None:
        print('writing validated usage file to .csv...')
        os.chdir(write)
        final.to_csv(filename, header = True, index = True)
        print('wrote {} to {}.'.format(filename, write))
        print('writing forecasts to .csv...')
        name = filename.split('.')[0]
        ts_name = '_'.join([name, 'timeshift'])
        ts_name = '.'.join([ts_name, 'csv'])
        forecast.to_csv(ts_name)
        print('wrote {} to {}.'.format(ts_name, write))
        
    
    return(forecast)
    


# ## Run Main Forecasting 

# ### To Do

# 1. Fill in gaps in missing time values (if gap = true)
# 2. Interpolate usage values for missing time values (if gap or lr = true)
# 3. Fill usage gaps with likeday estimation
# 4. More validations :)

# In[170]:


filedf = sortdir(json_files, 10)
engie_json = filedf.files[2]
print(' ')
print('using file:', engie_json)


# In[228]:


os.chdir(json_files)
clean_idr = forecast_main(engie_json, json_files, csv_files)


# In[226]:


clean_idr.head()


# In[8]:


cols = clean_idr.columns

key = ['usage value', 'weekday', 'hour', 'periodic zeros', 'low read', 'month', 'value difference', 'time difference',  'flag for gaps (td =/= 1)', 'rolling mean', 'mean centered usage', 'centered rolling mean', 'centered rolling std dev', 'variance', 'flags spikes', 'flags dips', 'flags dst', 'flags errors (lr, pz, gap, spike, dip, dst)', 'interpolated values', 'minutes (for nonhourly data)', 'day and hour combined']

col_key = pd.DataFrame({'key':key})
col_key.set_index(cols, inplace = True, drop = True)
col_key


# In[9]:


clean_idr.columns

#final_out.to_csv('year_forecast2.csv')


# ## Test main_forecast

# In[173]:


#parse json file
os.chdir(json_files)
idr, hb, sca, caps, meter = parse_engie(engie_json)
idr = idr.loc[pd.notnull(idr.index),:]
    
#check for nonperiodic zeros
tmp2 = periodic_zero(idr, .01, 1)

tmp2['mon'] = [a.month for a in tmp2.index]

#get value & time differences
tmp2 = interval_gap_check(tmp2)

#check spikes & dips
time_window = int(60*24*3600/hb)
centered = True
n_sd = 2

tmp2 = variance_validation(tmp2, time_window, centered, 2)

#check for dst errors
tmp2 = dst_check(tmp2)

data_filter = [a or b or c or d for a, b, c, d in zip(tmp2.lr, tmp2.gap, tmp2.spike, tmp2.dip)]
tmp2['err'] = data_filter

#flag for interpolation, linear interpolate
tmp2['interp'] = interp(tmp2.v, tmp2.err)
linear = tmp2.interp.interpolate(method = 'linear', axis = 0, in_place = False, limit_direction = 'forward')
tmp2['lin'] = linear

#fix nonhour interval reads
tmp2 = fix_interval(tmp2)

final = tmp2.copy()

print('forecasting...')
forecast = timeshift(final, 10)


# In[230]:


tmp2.head()


# In[136]:


tmp2.plot(y = 'lin')


# In[61]:


most_recent = max(sca.start)
last_yr = most_recent - dt.timedelta(days = 364)

recent_sca = sca[sca.stop > last_yr]
recent_sca


# ## Timeshift

# In[176]:


data = final.copy()
num_days = 364


# In[222]:


data = final.copy()
most_recent = max(data.index)
year_back = most_recent - dt.timedelta(days = num_days, hours = most_recent.hour)

oldest = min(data.index)
gap = oldest - year_back
gap_hr = int(divmod(gap.total_seconds(), 3600)[0])

if year_back < oldest:
    
    fill_year = []
    for j in range(0, gap_hr):
        fill_year.append(year_back + dt.timedelta(hours = j))
        
    fill_year = pd.DataFrame({'t':fill_year})
    fill_year.t = pd.to_datetime(fill_year.t)
    fill_year.set_index('t', drop = True, inplace = True)
    
    fill_year['d'] = [a.dayofweek for a in fill_year.index]
    fill_year['mon'] = [a.month for a in fill_year.index]
    fill_year['h'] = [a.hour for a in fill_year.index]
    fill_year['min'] = [a.minute for a in fill_year.index]
    fill_year['date_zip'] = list(zip(fill_year.d, fill_year.h))
    
    buildout = pd.concat([fill_year, data], axis = 0)
    year_data = buildout.copy()

else: 
    
    year_data = data[year_back:most_recent]    
    
print(buildout.head())
print(buildout.tail())

agg = year_data.groupby(['mon', 'd', 'h'])['v'].median()
year_forward = most_recent + dt.timedelta(days = 364, hours = 24 - most_recent.hour)
delta = year_forward - most_recent
delta_hr = int(divmod(delta.total_seconds(), 3600)[0])

next_year = []
for i in range(1, delta_hr):
    next_year.append(most_recent + dt.timedelta(hours = i))

month = [a.month for a in next_year]
day = [a.dayofweek for a in next_year]
hour = [a.hour for a in next_year]

forecast = pd.DataFrame({'t':next_year, 'mon':month, 'd':day, 'h':hour, 'date_zip':list(zip(day, hour))})
forecast.set_index('t', drop = True, inplace = True)
forecast['v'] = agg[forecast.date_zip].reset_index(drop = True).values.tolist()


# In[143]:


test = timeshift(tmp2, 10)


# In[190]:


final.plot(y = 'rm')


# In[134]:


test.to_csv('test_forecast.csv')


# In[ ]:





# ## Build out IDR from Scalar

# In[223]:


group_means = tmp2.groupby(['d', 'h'])['v'].median()
group_crsd = tmp2.groupby(['d', 'h'])['crsd'].median()
group_crm = tmp2.groupby(['d', 'h'])['crm'].median()

fill_year['v'] = group_means[fill_year.date_zip].reset_index(drop = True).values.tolist()
fill_year['crsd'] = group_crsd[fill_year.date_zip].reset_index(drop = True).values.tolist()
fill_year['crm'] = group_crsd[fill_year.date_zip].reset_index(drop = True).values.tolist()
#use = [a * b + c for a, b, c in zip(fill_year.crm, fill_year.crsd, fill_year.v)]
#makeup_df['v'] = use                                    

tmp2_sub = tmp2[['v', 'mon']]
fill_year_sub = fill_year[['v', 'mon']]

master = pd.DataFrame(pd.concat([fill_year_sub, tmp2_sub]))


# In[248]:


most_recent = max(sca.start)
last_yr = most_recent - dt.timedelta(days = 364)

recent_sca = sca[sca.stop > last_yr]
recent_sca

crsd = tmp2.groupby(['mon'])['crsd'].mean()
crm = tmp2.groupby(['mon'])['crm'].mean()
mean = tmp2.groupby(['mon'])['v'].mean()

monthly = pd.DataFrame({'mean':mean, 'crm': crm, 'crsd':crsd})

sca_days = [(b - a).days for a, b in zip(recent_sca.start, recent_sca.stop)]
month = recent_sca.stop.dt.month - 1
month[month==0] = 12
avg_daily = [round(b/a,2) for a, b in zip(sca_days, recent_sca.v)]

scalar_stats = pd.DataFrame({'sca':recent_sca.v, 'mon':month, 'days':sca_days, 'avg_daily':avg_daily})
scalar_stats.set_index('mon', drop = True, inplace = True)

monthly_stats = pd.concat([scalar_stats, monthly], axis = 1)
monthly_stats


# In[ ]:





# ## Scalar/IDR Checks

# In[238]:


#start_date = min(recent_sca.start)
#recent_idr = master.v[master.index >= start_date]

scalar_idr_group_start = [makeup_df.index >= a for a in recent_sca.start]
scalar_idr_group_stop = [makeup_df.index <= b for b in recent_sca.stop]

monthly_idr = []
monthly_count = []

for i, a in enumerate(scalar_idr_group_start):
    b = scalar_idr_group_stop[i]
    bt = [c and d for c, d in zip(a, b)]
    monthly_use = sum(makeup_df.v[bt])
    monthly_count.append(sum(bt)/24)
    monthly_idr.append(round(monthly_use, 4))
    
daily_avg = [round(m/d, 2) for m, d in zip(monthly_idr, monthly_count)]
                       
idr_monthly_check = pd.DataFrame({'start':recent_sca.start, 'stop':recent_sca.stop,                                   'monthly':monthly_idr, 'days':monthly_count, 'daily avg':daily_avg})
print('monthly idr sums: ')
print(idr_monthly_check)
print('')
print('scalar data:')
print(recent_sca)

month = recent_sca.stop.dt.month - 1
month[month==0] = 12
sca_days = [(b - a).days for a, b in zip(recent_sca.start, recent_sca.stop)]
avg_daily = [round(b/a,2) for a, b in zip(sca_days, recenyt_sca.v)]

correction_factor = [b/a for a, b in zip(daily_avg, avg_daily)]

easy_sca = pd.DataFrame({'mon':month, 'v':recent_sca.v, 'days':sca_days, 'avg':daily_avg, 'cf':correction_factor})
#easy_sca.set_index('m', inplace = True)
easy_sca.cf[easy_sca.cf.isnull()] = 1
print(easy_sca)

master_cor = master.join(easy_sca.cf, on = 'mon')
master_cor.cf[master_cor.cf.isnull()] = 1
master_cor['v'] = [a*b for a, b in zip(master_cor.v, master_cor.cf)]


# In[ ]:





# In[56]:


master_cor.plot(y = 'v')


# In[111]:


mon_index = master_mon.index.values.tolist()
sca_index = easy_sca.index.values.tolist()
subset = [a in mon_index for a in sca_index]
easy_sca[subset]
easy_sca


# In[29]:


#recent = [yr.year < 2020 for yr in tmp2.index]
tmp2.plot(y = 'lin')


# In[19]:


tmp2[recent].plot(y = 'v', color = 'orange')


# In[18]:


end_yr = [yr.month > 10 for yr in tmp2.index]
keep = [a and b for a, b in zip(end_yr, tmp2.err)]
tmp2[keep]


# In[ ]:





# In[8]:





# In[ ]:


#for points where linear interpolation don't work, use likeday estimation
counts = pd.DataFrame(tmp2.groupby([tmp2.d, tmp2.h, tmp2.err])['v'].count())
means = pd.DataFrame(tmp2.groupby([tmp2.d, tmp2.h, tmp2.err])['v'].mean())

means.iloc[means.index.get_level_values('err') == True]
means.iloc[means.index.get_level_values('err') == False]
#pd.merge(tmp, weekly_periodic_reads, how = 'left', right_index = True, left_on = ['d', 'h'])






