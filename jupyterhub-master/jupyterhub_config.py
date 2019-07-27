import sys

c.JupyterHub.template_paths = ['/opt/tljh/user/share/jupyterhub/templates/']

from oauth2client.client import flow_from_clientsecrets
from oauth2client.client import OAuth2WebServerFlow

flow = flow_from_clientsecrets('/opt/tljh/client_secrets.json',
                               scope='https://www.googleapis.com/auth/',
                               redirect_uri='http://http://tljh-engie.southcentralus.cloudapp.azure.com/hub/auth_return')
