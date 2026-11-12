"""Only loopback, the trusted home LAN and the owner's Tailscale tailnet may reach HTTP or websocket control."""
import ipaddress
from urllib.parse import urlsplit

TRUSTED=[ipaddress.ip_network(n) for n in ('192.168.0.0/24','100.64.0.0/10','fd7a:115c:a1e0::/48')]

class LocalNetworkOnly:
    def __init__(self,app):
        self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type'] in {'http','websocket'}:
            host=(scope.get('client') or ('',0))[0]
            try:
                ip=ipaddress.ip_address(host)
                allowed=ip.is_loopback or any(ip in net for net in TRUSTED)
            except ValueError:
                allowed=False
            headers=dict(scope.get('headers',[]))
            origin=headers.get(b'origin',b'').decode()
            request_host=headers.get(b'host',b'').decode()
            if origin and urlsplit(origin).netloc!=request_host:
                allowed=False
            if not allowed:
                if scope['type']=='websocket':
                    await send({'type':'websocket.close','code':1008})
                else:
                    await send({'type':'http.response.start','status':403,'headers':[]})
                    await send({'type':'http.response.body','body':b'Local network and same-origin access only'})
                return
        await self.app(scope,receive,send)
