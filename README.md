# LA-tools

Contains Python tools for Load Analytics department - Engie, NA.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/stevenhurwitt/LA-tools/master)

## 1. Cap Tag Report 
Found in [Cap Tag Directory](CapReports/CapTagReports.ipynb) .

Creates Cap Tag Report given PR and Revision numbers (as "PR_rev").

Pulls from TPPE and outputs .csv file of cap and trans tags for all meters in a PR.
Checks start and end dates of tags with start and stop dates of PR.
Can be used to batch through multiple PR's.

**To Do** Add functionality similar to Offer Summary Main tool - check tags with summer and winter peaks.


## 2. .json to .csv Parser
Found in [this file in the main directory](Forecasts-JSONtoCSV.ipynb).

Parses EWX .json files and outputs forecasts to .csv (can be used if EWX forecasts aren't coming through to ALPS).

**To Do** Add functionality to parse payloads we send to EWX (can be used for tickets, to recover data, etc.).
Add functionality to check heartbeat (time difference) of forecasts.

## 3. NEPOOL IDR Drop
Found in [IDR_Drop Directory](IDR_Drop/)

1. [Email scrape tool](/IDR_Drop/emailscrape.py) - Parses utility emails for accounts, EPO logins and passwords.
2. [EPO webscrape tool](/IDR_Drop/EPOwebscrape.py) - Automates downloading of IDR data from EPO portal.
3. [IDR filter](/IDR_Drop/IDRdrop.py) - splits IDR data into Raw IDR files, filters raw IDR into ch. 1 and/or ch. 3 to be dropped into ALPS.
4. [Main code](/IDR_Drop/Van_Pham_IDR_Drop.py) - puts everything together, to be automated.
