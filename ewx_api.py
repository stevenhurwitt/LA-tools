import pandas as pd
import requests
import json
import os
os.chdir('/Users/stevenhurwitt/LA-tools')


#### file search for meter data ####

#deal w bearer token 3600s timeout
meter = 'NEPOOL_NRI_2529175006'

bearer = 'ya29.Gls8ByCYr6KTgAuawee_bnMVgabznB8UYURRnldjD341GTiw4bE7Iuetd8mZKr2ZgnX68CPBiDmXGVh3Q5qkKd1td7Ev5e2l0M14Wy_yYtKHtTi9ktbHPIBX6y8a'
bearer_auth = ''.join(['Bearer ', bearer])

#get request to filesearch API
header = {'Authorization':bearer_auth, 'X-NAMESPACE':'na.engie.com', 'accept':'application/json'}

user = 'https://api.cloud.energyworx.com/_ah/api/ewx/v1/user/get'

user_info = requests.get(user, headers = header).json()
refresh_token = user_info['refreshToken']

#update = 'api.cloud.energyworx.com/_ah/api'
#h = {'grant_type': 'refresh_token', }
#print('user refresh token {}.'.format(user_info['refreshToken']))

url = ''.join(['https://api.cloud.energyworx.com/_ah/api/ewx/v1/storage/files/search?tags=', meter, '&limit=20'])

#turn to dataframe, get most recent response & ingestion blobkeys
search_result = requests.get(url, headers = header).json()
search_result_df = pd.DataFrame(search_result['items'])

search_result_df['lastUpdatedDatetime'] = pd.to_datetime(search_result_df['lastUpdatedDatetime'])
response = ['response' in a for a in search_result_df.tags]
ingestion = ['ingestion' in a for a in search_result_df.tags]

resp_recent = max(search_result_df.lastUpdatedDatetime[response])
ingest_recent = max(search_result_df.lastUpdatedDatetime[ingestion])

resp_uri = search_result_df.uri[search_result_df.lastUpdatedDatetime == resp_recent].values[0]
ingest_uri = search_result_df.uri[search_result_df.lastUpdatedDatetime == ingest_recent].values[0]

resp_blob = resp_uri.split('/')[3]
ingest_blob = ingest_uri.split('/')[3]

#get request to getfile API
url_resp = ''.join(['https://console.cloud.energyworx.com', resp_uri])
url_ingest = ''.join(['https://console.cloud.energyworx.com', ingest_uri])

file_response = requests.get(url_resp, headers = header).json()
file_ingest = requests.get(url_ingest, headers = header).json()

#get ewx payload

ch3 = file_response['account']['timeseriesdataidr']

hb = ch3[0]['heartbeat']
meterid = '_'.join([file_response['account']['market'], file_response['account']['discocode'], file_response['account']['accountnumber']])

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

print(master_df.head())
print('...')
print(master_df.tail())

#cap tags
caps = file_response['account']['captag'][0]
caps_df = pd.DataFrame.from_records(caps, index = [0])
caps_df['start'] = pd.to_datetime(caps_df['start'])
caps_df['stop'] = pd.to_datetime(caps_df['stop'])
caps_df['v'] = pd.to_numeric(caps_df['v'], errors = 'coerce')

print(caps_df)
print('found {} hearbeats for meter {}.'.format(hb, meterid))

#get engie payload

ts_sca_data = file_ingest['account']['timeseriesdatascalar']
sca_payload = pd.DataFrame.from_dict(ts_sca_data).iloc[:,1:]
sca_payload['start'] = pd.to_datetime(sca_payload.start)
sca_payload['stop'] = pd.to_datetime(sca_payload.stop)
sca_payload['v'] = pd.to_numeric(sca_payload['v'], errors = 'coerce')

ts_idr_data = file_ingest['account']['timeseriesdataidr']
n = len(ts_idr_data)

ch = ts_idr_data[0]['channel']
hb = ts_idr_data[0]['heartbeat']
print('found {} heartbeats'.format(hb))

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
            
    idr_payload = pd.concat([idr_payload, temp], axis = 0)

idr_payload = idr_payload.loc[idr_payload.v.notnull(),:] 
#print(idr_payload.head())
#print('...')
#print(idr_payload.tail())

#print(sca_payload)
    