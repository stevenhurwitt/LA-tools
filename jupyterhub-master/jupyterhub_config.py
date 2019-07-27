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
        'url': '104.210.158.231:10102',
        'command': [sys.executable, './accounts.py'],
    },
]
