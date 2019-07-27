import sys

c.Authenticator.admin_users = set(["dblank"])

c.JupyterHub.services = [
    {
        'name': 'public',
        'url': 'http://104.210.158.231',
        'command': [sys.executable, './public_handler.py'],
    },
    {
        'name': 'accounts',
        'url': '104.210.158.231',
        'command': [sys.executable, './accounts.py'],
    },
]

c.JupyterHub.template_paths = ['/opt/tljh/user/share/jupyterhub/templates/']

from oauth2client.client import flow_from_clientsecrets
from oauth2client.client import OAuth2WebServerFlow

flow = flow_from_clientsecrets('/opt/tljh/client_secrets.json',
                               scope='https://www.googleapis.com/auth/',
                               redirect_uri='http://http://tljh-engie.southcentralus.cloudapp.azure.com/hub/auth_return')
