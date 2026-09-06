from fastapi.testclient import TestClient
from tridifleet.api import app


def test_admin_lab_flow(monkeypatch):
    monkeypatch.setenv('TRIDIFLEET_AUDIT_DB', ':memory:')
    with TestClient(app) as client:
        assert client.get('/api/lab/status').status_code == 401
        assert client.post('/api/auth/login',json={'username':'admin','password':'tridifleet-local'}).status_code == 200
        assert client.post('/api/lab/configure',json={'n_totems':2,'radius_km':.7,'seed':9,'detection_radius_m':6}).status_code == 200
        assert client.get('/').status_code == 200
        assert client.get('/static/app.js').status_code == 200
        assert client.post('/api/lab/creatives',json={'ad_id':'soap','name':'Sabonete','duration_seconds':10,'tags':['higiene']}).status_code == 200
        assert client.post('/api/lab/creatives',json={'ad_id':'soap','name':'Sabonete','duration_seconds':10}).status_code == 409
        assert client.post('/api/lab/start').status_code == 200
        assert client.post('/api/lab/pause',json={'paused':True}).status_code == 200
        assert client.post('/api/lab/speed',json={'speed':99}).json()['speed'] == 1
        assert client.post('/api/lab/audit/verify').json()['chain_valid']
        with client.websocket_connect('/ws/lab') as ws:
            assert ws.receive_json()['configured']
