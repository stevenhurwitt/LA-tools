
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import json
import math
from collections import deque
import pprint
import os


##### import functions ####

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

def read_idr(filename, header_index):
    f = pd.read_csv(filename, header = header_index)
    f.columns = ['t', 'v']
    f['v'] = pd.to_numeric(f['v'], errors = 'coerce')
    f.t = pd.to_datetime(f.t)
    f.set_index('t', inplace = True, drop = True)
    f.fillna(0, inplace = True)
    return(f)

def merge_idr(meter_data):

    master = pd.DataFrame()
    cols = []

    for meter, data in meter_data.items():
        master = pd.concat([master, data], axis = 1)
        cols.append(meter)
    
    master.columns = cols
    print(master.head())
    return(master)

def iter_plot(idr_df):
    n = len(idr_df.columns)
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
        
        meter = idr_df.columns[0]
        ax.set_title(meter, fontsize = 36);
        plt.rc('font', size = 28)
        #rec_yr = [a < 2020 for a in meter_df.index.year]
        idr_df.plot(y = meter, ax = ax);
        
    elif n > 1:
        
        axes_list = [item for sublist in axes for item in sublist]
        axes_list = deque(axes_list)

        for m in idr_df.columns:

            ax = axes_list.popleft();
            ax.set_title(m, fontsize = 36);
            plt.rc('font', size = 28)
            
            meter_df = idr_df.loc[:,m]
            #rec_yr = [a < 2020 for a in meter_df.index.year]
            meter_df.plot(y = m, ax = ax);


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
        sca_payload['v'] = pd.to_numeric(sca_payload['v'], errors = 'coerce')
    
    except:
        sca_payload = None
    
    ts_idr_data = acct['timeseriesdataidr']
    n = len(ts_idr_data)

    ch = ts_idr_data[0]['channel']
    hb = ts_idr_data[0]['heartbeat']
    print('found', hb, 'heartbeats')
    idr_payload = pd.DataFrame.from_dict(ts_idr_data[0]['reads'])
    idr_payload['v'] = pd.to_numeric(idr_payload['v'], errors = 'coerce')
    
    idr_payload.t = pd.to_datetime(idr_payload.t)
    idr_payload = idr_payload.set_index(pd.DatetimeIndex(idr_payload.t))
    idr_payload = idr_payload.drop('t', axis = 1)
           
    print('found {} reads, creating dataset.'.format(n))

        
    for i in range(1,n):
        reads = ts_idr_data[i]['reads']
        temp = pd.DataFrame.from_dict(reads)
        
        temp['v'] = pd.to_numeric(temp['v'], errors = 'coerce')
    
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
        caps_df['v'] = pd.to_numeric(caps_df['v'], coerce = 'errors')
    
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


def variance_validation(tmp2, time_window, centered, n_sd):

    tmp2['rm'] = tmp2['v'].rolling(window = time_window, min_periods = 1, center = centered).mean()
    tmp2['mc'] = tmp2.v - tmp2.rm

    tmp2['crm'] = tmp2['mc'].rolling(window = time_window, min_periods = 10, center = centered).mean()
    tmp2['crsd'] = tmp2['mc'].rolling(window = time_window, min_periods = 10, center = centered).std()

    tmp2['var'] = (tmp2['mc'] - tmp2['crm'])/tmp2['crm']

    tmp2['spike'] = tmp2['mc'] > (tmp2['crm'] + (n_sd + 1) * tmp2['crsd'])
    tmp2['dip'] = tmp2['mc'] < (tmp2['crm'] - n_sd * tmp2['crsd'])
    
    return(tmp2)


##fix bad time interval (15 min, etc)
## EWX port: interval gap check (data)
## appends columns - vd: value difference (float)
##                  td: time difference (timedelta, hours)
##                 gap: gap after index (bool)
##
#### lives inside fix_interval() function

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

## DST Check (data)
## check between DST periods for either:
## duplicate hour or missing hour.
## appends column - dst: flag for dst errors need fill (bool)
##
#### lives inside fix_interval()

def dst_check(tmp2):
    beg_for = dt.datetime.strptime('03/08/2019', '%m/%d/%Y')
    end_for = dt.datetime.strptime('03/14/2019', '%m/%d/%Y')
    beg_back = dt.datetime.strptime('10/31/2019', '%m/%d/%Y')
    end_back = dt.datetime.strptime('11/07/2019', '%m/%d/%Y')
    
    date_check = [(((date >= beg_for) and (date <= end_for)) or ((date >= beg_back) and (date <= end_back))) for date in tmp2.index]
    
    time_check = [diff != 1 for diff in tmp2.td]
    
    dst = [a and b for a, b in zip(date_check, time_check)]
    tmp2['dst'] = dst
    
    return(tmp2)

#### fix nonhourly intervals (data)
## groups data by date (zipped breakdown of y/m/d hh:mm),
## sums over minutes per y/m/d hour,
## subsets to only hours.
## merges with data, returns.
##
#### lives inside fix_interval()


def fix_nonhour(data):
    
    data['y'] = data.index.year
    data['mon'] = data.index.month
    data['d'] = data.index.day
    data['h'] = data.index.hour
    data['min'] = data.index.minute
    
    data['date'] = list(zip(data.index.year, data.index.month, data.index.day, data.index.hour, data.index.minute))
    hourly = data.groupby(data['date']).sum()
    
    real_val = hourly['v']
    real_val.reset_index(drop = True, inplace = True)

    time = data.index.to_series()
    hr_time = time[[a == 0 for a in data['min']]]
    hr_time.reset_index(drop = True, inplace = True)

    adj_forecast = pd.concat([hr_time, real_val], axis = 1)
    adj_forecast.columns = ['t', 'v']

    adj = adj_forecast[adj_forecast.t.notnull()]
    adj.set_index('t', drop = True, inplace = True)

    if (len(adj) == len(data.v)):
        data = data.join(adj, how = 'inner', on = 't', lsuffix = '_orig', rsuffix = '')
        print(data.columns)
    
    else:
        print('length mismatch - trying merge anyways (expect NAs).')
        data = pd.merge(data, adj, how = 'left', right_on = 'date', left_on = ['date'])
        
    return(data)
        

## fix time intervals (data)
## breaks dates into zipped date (y/m/d hh:mm)
## applies gap check, DST check, nonhourly fix
##
## returns dataset with cols of validation results

def fix_interval(data):
    
    data['date'] = list(zip(data.index.year, data.index.month, data.index.day, data.index.hour, data.index.minute))
    
    data = interval_gap_check(data)
    data = dst_check(data)
    data = dst_fix(data)
    #data = fix_nonhour(data)
    data.fillna(.123456789, inplace = True)
    data['na'] = [a == .123456788 for a in data.v]
        
    final_out = data.copy()
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
    agg = year_data.groupby(['mon', 'd', 'h'])['lin'].mean()
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

def gen_year_idr(data, num_days, val):
    most_recent = max(data.index)
    year_back = most_recent - dt.timedelta(days = num_days, hours = most_recent.hour)
    oldest = min(data.index)
    gap = oldest - year_back
    gap_hr = int(divmod(gap.total_seconds(), 3600)[0])
    
    year_data = data[year_back:most_recent]
    #lin = list(year_data['lin'])
    val += val
    #agg = year_data.groupby(['mon', 'd', 'h'])['lin']
    
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
    #forecast['lin'] = pd.to_numeric(lin)
    return(forecast, val)

def timeshift(data, until, scalar, vals):
    until += 1
    most_recent = max(data.index)
    year_back = most_recent - dt.timedelta(days = 364, hours = most_recent.hour)
    oldest = min(data.index)
    gap = oldest - year_back
    gap_hr = int(divmod(gap.total_seconds(), 3600)[0])
    
    year = 0
    year_data = data[year_back:most_recent]
    #year_data = year_data
    #print(year_data.head())

    if scalar:
        future = gen_year(data, 364)
        future = future
        master = pd.concat([year_data, future], axis = 0)
        print('forecasted year {} of {} with {} reads.'.format(year, until-1, len(future.lin)))
        year += 1
    
        while year < until:
            if (year % 6 == 0 and year > 0):
                num_days = 371
            else:
                num_days = 364
        
            forecast = gen_year(master, num_days)
            forecast = forecast
            master = pd.concat([master, forecast], axis = 0)
            print('forecasted year {} of {} with {} reads.'.format(year, until-1, len(forecast.lin)))
            year += 1

        master = pd.DataFrame(master['lin'])
        master.columns = ['v']
        
    else:

        future, vals = gen_year_idr(data, 364, vals)
        future = future
        master = pd.concat([year_data, future], axis = 0)
        n = len(master.index)
        vals = vals[:2*n]
        print('forecasted year {} of {} with {} reads.'.format(year, until-1, len(vals)))
        year += 1
    
        while year < until:
            if (year % 6 == 0 and year > 0):
                num_days = 371
            else:
                num_days = 364
        
            forecast, vals = gen_year_idr(master, num_days, vals)
            m = (year+2)*num_days*24
            vals = vals[:m]

            master = pd.concat([master, forecast], axis = 0)
            print('forecasted year {} of {} with {} reads.'.format(year, until-1, len(vals)))
            year += 1
        
        final_n = len(vals)
        master = master.iloc[:final_n,]
        master['v'] = vals
    
        master = pd.DataFrame(master['v'])
        master.columns = ['v']
    
    return(master)



def forecast_main(json_file, years, read, write, scalar):
    
    print('parsing data files...')
    #parse json file
    if 'json' in json_file:
        name = json_file.split('_')[1:]
        filename = '_'.join(name)
        filename = filename.replace('json', 'csv')
        print('using filename {}.'.format(filename))
        
        if read is not None and type(json_file) == str:
            os.chdir(read)
            try:
                idr, hb, sca, caps, meter = parse_engie(json_file)
            
            except:
                idr = pd.read_csv(filename)
                idr.columns = ['t', 'v']
                idr['v'] = pd.to_numeric(idr['v'])
                idr.t = pd.to_datetime(idr.t)
                idr.set_index(pd.DatetimeIndex(idr.t), inplace = True, drop = True)
                idr = idr.drop('t', axis = 1)
            
        idr.fillna(.123456789, inplace = True)
        print('read {} from {}.'.format(filename, read))
    
    else:
        os.chdir(read)
        filename = json_file
        print('using filename {}.'.format(filename))
    
        idr = pd.read_csv(filename)
        print(idr.head())
        idr.columns = ['t', 'v']
        idr['v'] = pd.to_numeric(idr['v'], errors = 'coerce')
        idr.t = pd.to_datetime(idr.t)
        idr.set_index(pd.DatetimeIndex(idr.t), inplace = True, drop = True)
        idr = idr.drop('t', axis = 1)
            
        idr.fillna(.123456789, inplace = True)
        print('read {} from {}.'.format(filename, read))
    

    
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
    time_window = int(30*24)
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
    
    tmp2['na'] = [v == .123456789 for v in tmp2['v']]
    
    print('usage validated.')
    
    print('running usage estimation flags...')
    data_filter = [a or b or c or d or e for a, b, c, d, e in zip(tmp2.lr, tmp2.gap, tmp2.spike, tmp2.dip, tmp2.na)]
    tmp2['err'] = data_filter
    
    tmp2['interp'] = interp(tmp2.v, tmp2.err)
    
    linear = tmp2.interp.interpolate(method = 'linear', axis = 0, in_place = False, limit_direction = 'forward')
    tmp2['lin'] = linear
    tmp2.lin[tmp2.lin.isnull()] = tmp2.v[tmp2.lin.isnull()]
    
    #final validated data
    final = tmp2.copy()
    val = filename.split('.')[0]
    val_file = ''.join([val, '_val.csv'])
    
    if write is not None:
        print('writing validated usage file to .csv...')
        os.chdir(write)
        final.to_csv(val_file, header = True, index = True)
        print('wrote {} to {}.'.format(filename, write))
        
        
    print('forecasting...')
    vals = list(tmp2.lin)
    forecast = timeshift(final, years, scalar, vals)
    
    if write is not None:
        print('writing forecasts to .csv...')
        name = filename.split('.')[0]
        ts_name = '_'.join([name, 'timeshift'])
        ts_name = '.'.join([ts_name, 'csv'])
        forecast.to_csv(ts_name, header = False)
        print('wrote {} to {}.'.format(ts_name, write))
        
    
    return(forecast)
    

