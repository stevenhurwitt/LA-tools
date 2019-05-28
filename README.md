# LA-tools

Contains Python tools for Load Analytics at Engie, NA.

# Overview

## 1. Cap Tag Report

Creates Cap Tag Report given PR and Revision numbers (as "PR_rev").

Pulls from TPPE and outputs .csv file of cap and trans tags for all meters in a PR.
Checks start and end dates of tags with start and stop dates of PR.
Can be used to batch through multiple PR's.

### to do

Add functionality similar to Offer Summary Main tool - check tags with summer and winter peaks.


## 2. .json to .csv Parser

Parses EWX .json files and outputs forecasts to .csv (can be used if EWX forecasts aren't coming through to ALPS).

### to do

Add functionality to parse payloads we send to EWX (can be used for tickets, to recover data, etc.).
Add functionality to check heartbeat (time difference) of forecasts.

## 3. NEPOOL IDR Drop

1. Email scrape tool - Parses utility emails for accounts, EPO logins and passwords.
2. EPO webscrape tool - Automates downloading of IDR data from EPO portal.
3. IDR filter - splits IDR data into Raw IDR files, filters raw IDR into ch. 1 and/or ch. 3 to be dropped into ALPS.
