import pandas as pd
import requests
import json
import os
requests.packages.urllib3.disable_warnings()

base = 'C:\\Users\\wb5888\\LA-tools'
write = 'C:\\Users\\wb5888\\Documents'

os.chdir(base)

#### file search for meter data ####

#deal w bearer token 3600s timeout
meter = 'NEPOOL_MECO_5161473048'
bearer = 'ya29.Glw8B4Tq7XhbG_JtYRq4gBSVKGw-j4cLiGhFfs_N4XjopyDHNJIDZNCrAuOpQ-SuDMj-auupakYN9J8qfmJ0BQKn9nXHllulFHicawdr6ed-5Nn-y691G-nHomAG-A'

def get_user_token():
    header = {'Authorization':bearer_auth, 'X-NAMESPACE':'na.engie.com', 'accept':'application/json'}

    user = 'https://api.cloud.energyworx.com/_ah/api/ewx/v1/user/get'

    user_info = requests.get(user, headers = header, verify = False).json()
    refresh_token = user_info['refreshToken']
    #update = 'api.cloud.energyworx.com/_ah/api'
    #h = {'grant_type': 'refresh_token', }
    #print('user refresh token {}.'.format(user_info['refreshToken']))
    return(refresh_token)

def parse_ingest(data):

    ts_sca_data = data['account']['timeseriesdatascalar']
    sca_payload = pd.DataFrame.from_dict(ts_sca_data).iloc[:,1:]
    sca_payload['start'] = pd.to_datetime(sca_payload.start)
    sca_payload['stop'] = pd.to_datetime(sca_payload.stop)
    sca_payload['v'] = pd.to_numeric(sca_payload['v'], errors = 'coerce')

    ts_idr_data = data['account']['timeseriesdataidr']
    n = len(ts_idr_data)

    ch = ts_idr_data[0]['channel']
    hb = ts_idr_data[0]['heartbeat']

    idr_payload = pd.DataFrame.from_dict(ts_idr_data[0]['reads'])
    idr_payload['v'] = pd.to_numeric(idr_payload['v'], errors = 'coerce')
    
    idr_payload['t'] = pd.to_datetime(idr_payload.t)
    idr_payload = idr_payload.set_index(pd.DatetimeIndex(idr_payload.t))
    idr_payload = idr_payload.drop('t', axis = 1)
           
    print('found {} reads, creating dataset.'.format(n))
       
    for i in range(1,n):
        reads = ts_idr_data[i]['reads']
        temp = pd.DataFrame.from_dict(reads)
        
        temp['v'] = pd.to_numeric(temp['v'], errors = 'coerce')
    
        temp['t'] = pd.to_datetime(temp.t)
        temp = temp.set_index(pd.DatetimeIndex(temp.t))
        temp = temp.drop('t', axis = 1)
            
        idr_payload = pd.concat([idr_payload, temp], axis = 0)

    idr_payload = idr_payload.loc[idr_payload.v.notnull(),:] 

    return(idr_payload, sca_payload)
    

def parse_response(data):
    ch3 = data['account']['timeseriesdataidr']

    hb = ch3[0]['heartbeat']
    meterid = '_'.join([data['account']['market'], data['account']['discocode'], data['account']['accountnumber']])

    n = len(ch3)
        
    reads = ch3[0]['reads']
    master_df = pd.DataFrame.from_dict(reads)
    master_df['t'] = pd.to_datetime(master_df.t)
        
    print('found {} reads, creating dataset.'.format(n))
        
    for i in range(1,n):
        reads = ch3[i]['reads']
        temp = pd.DataFrame.from_dict(reads)
        temp['t'] = pd.to_datetime(temp.t)
        master_df = pd.concat([master_df, temp]).reset_index(drop = True)
    
    
    master_df = master_df.set_index(master_df.t)
    master_df = master_df.drop('t', axis = 1)


    #cap tags
    caps = data['account']['captag'][0]
    caps_df = pd.DataFrame.from_records(caps, index = [0])
    caps_df['start'] = pd.to_datetime(caps_df['start'])
    caps_df['stop'] = pd.to_datetime(caps_df['stop'])
    caps_df['v'] = pd.to_numeric(caps_df['v'], errors = 'coerce')

    return(master_df, caps_df)

#### function to download most recent 
#### response & ingesiton payloads
#### for a meter, given user's bearer token.

## returns idr, scalar, caps & forecast.
## forecast & caps are None if no response found.
def file_download(meter, bearer):

    #get request to filesearch API
    print('sending get request for meter file search...')
    bearer_auth = ''.join(['Bearer ', bearer])
    header = {'Authorization':bearer_auth, 'X-NAMESPACE':'na.engie.com', 'accept':'application/json'}
    url = ''.join(['https://api.cloud.energyworx.com/_ah/api/ewx/v1/storage/files/search?tags=', meter, '&limit=20'])
    #print(url, header)
    #turn to dataframe, get most recent response & ingestion blobkeys
    search_result_raw = requests.get(url, headers = header, verify = False)
    #print(search_result_raw)
    search_result = search_result_raw.json()
    #print('got json')
    search_result_df = pd.DataFrame(search_result['items'])
    print('found results (choosing most recent):')
    print(search_result_df.head())

    #get most recent
    search_result_df['lastUpdatedDatetime'] = pd.to_datetime(search_result_df['lastUpdatedDatetime'])
    response = ['response' in a for a in search_result_df.tags]
    ingestion = ['ingestion' in a for a in search_result_df.tags]

    try:
        resp_recent = max(search_result_df.lastUpdatedDatetime[response])
        resp_uri = search_result_df.uri[search_result_df.lastUpdatedDatetime == resp_recent].values[0]
        resp_blob = resp_uri.split('/')[3]
        url_resp = ''.join(['https://console.cloud.energyworx.com', resp_uri])
        response_found = True
   
    except:
        print('no response payload found.')
        response_found = False

    try:
        ingest_recent = max(search_result_df.lastUpdatedDatetime[ingestion])
        ingest_uri = search_result_df.uri[search_result_df.lastUpdatedDatetime == ingest_recent].values[0]
        ingest_blob = ingest_uri.split('/')[3]
        ingest_found = True

    except:
        print('ingestion files not found?')
        ingest_found = False

    print('sending get request for data files...')
    #get request to getfile API
    if response_found:
        url_resp = ''.join(['https://console.cloud.energyworx.com', resp_uri])
        file_response = requests.get(url_resp, headers = header, verify = False).json()
        print('parsing response payload...')
        forecast, caps = parse_response(file_response)

        filename = ''.join([meter, '_CH3.csv'])
        forecast.to_csv(filename)
        print('response parsed.')

    else:
        forecast, caps = None, None
    
    if ingest_found:
        url_ingest = ''.join(['https://console.cloud.energyworx.com', ingest_uri])
        file_ingest = requests.get(url_ingest, headers = header, verify = False).json()
        print('parsing ingestion payload...')
        idr, scalar = parse_ingest(file_ingest)
        print('ingestion parsed.')

    else:
        idr, scalar = None, None

    return(idr, scalar, caps, forecast)


####### GRAB CH 3, MULT METERS ##########

meters = ['NEPOOL_MECO_5161473048']
bearer = 'ya29.Glw8B4Tq7XhbG_JtYRq4gBSVKGw-j4cLiGhFfs_N4XjopyDHNJIDZNCrAuOpQ-SuDMj-auupakYN9J8qfmJ0BQKn9nXHllulFHicawdr6ed-5Nn-y691G-nHomAG-A'
os.chdir(write)

for m in meters:
    try:
        ch1, sca, cap, ch3 = file_download(m, bearer)
        filename = ''.join([m, '_CH3', '.csv'])
        print('writing forecast for {}.'.format(filename))
        ch3.to_csv(filename)
    
    except:
        print('file download failed for {}'.format(m))