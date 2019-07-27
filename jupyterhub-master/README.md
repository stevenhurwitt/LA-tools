# Services for Jupyterhub

## Privacy Policy

Privacy Notice
This privacy notice discloses the privacy practices for (website address). This privacy notice applies solely to information collected by this website. It will notify you of the following:

What personally identifiable information is collected from you through the website, how it is used and with whom it may be shared.
The security procedures in place to protect the misuse of your information.

We are the sole owners of the information collected on this site. We only have access to/collect information that you voluntarily give us via email or other direct contact from you. We will not sell or rent this information to anyone.

We will use your information to respond to you, regarding the reason you contacted us. We will not share your information with any third party outside of our organization, other than as necessary to fulfill your request, e.g. to ship an order.


While we use encryption to protect sensitive information transmitted online, we also protect your information offline. Only employees who need the information to perform a specific job  are granted access to personally identifiable information. The computers/servers in which we store personally identifiable information are kept in a secure environment.

If you feel that we are not abiding by this privacy policy, you should contact us via email.

## LA-tools README

### 1. Cap Tag Report 
Found in [Cap Tag Directory](~/CapReports/CapTagReport.ipynb) .

Creates Cap Tag Report given PR and Revision numbers (as "PR_rev").

Pulls from TPPE and outputs .csv file of cap and trans tags for all meters in a PR.
Checks start and end dates of tags with start and stop dates of PR.
Can be used to batch through multiple PR's.

**To Do** Add functionality similar to Offer Summary Main tool - check tags with summer and winter peaks.


### 2. .json to .csv Parser
Found in [this file in the main directory](~/Forecasts-JSONtoCSV.ipynb).

Parses EWX .json files and outputs forecasts to .csv (can be used if EWX forecasts aren't coming through to ALPS).

**To Do** Add functionality to parse payloads we send to EWX (can be used for tickets, to recover data, etc.).
Add functionality to check heartbeat (time difference) of forecasts.

### 3. EWX Forecasting Clone
Found in [EWX Forecasting Directory](~/EWX_Forecast/)

Attempts to replicate EWX Forecasting default flow configs. Goal is to be given Engie's payload to EWX (as .json) and output a similarly forecasted ch. 3 and cap tags

**To Do** Scalar buildouts, full monthly sum checks, UI, etc. etc.

### 4. NEPOOL IDR Drop
Found in [IDR_Drop Directory](IDR_Drop/)

1. [Email scrape tool](~/IDR_Drop/emailscrape.py) - Parses utility emails for accounts, EPO logins and passwords.
2. [EPO webscrape tool](~/IDR_Drop/EPOwebscrape.py) - Automates downloading of IDR data from EPO portal.
3. [IDR filter](~/IDR_Drop/IDRdrop.py) - splits IDR data into Raw IDR files, filters raw IDR into ch. 1 and/or ch. 3 to be dropped into ALPS.
4. [Main code](~/IDR_Drop/Van_Pham_IDR_Drop.py) - puts everything together, to be automated.

**To Do** Fix bug in webscrape, or make workaround where we can drop downloaded raw (groups of ~5) to directory where **IDR filter** will parse these into IDR_Success, IDR_Error folders

## Public, a nbviewer service for jupyterhub

This code adds nbviewer-like functionality, via a JupyterHub Service, to a jupyterhub installation. 
That is, it allows users to "publish" their notebooks so that they can be seen as HTML.

Served at /services/public/username/... 

See also the Javascript nbextension for copying notebooks to a user's public_html folder:

https://github.com/Calysto/notebook-extensions

## Accounts, a more flexible way of adding accounts

Creates accounts and passwords following the xkcd password style.

Served at /services/accounts

Opens a form and textarea where each line should be in one of the following forms:

```
   username                             OR
   username@address                     OR
   Real Name, username@email       OR
   Real Name, username-email            OR
   Real Name, email@email, username
```

Can send email, and/or display the passwords. Optional field for name.
