import json
import urllib.request
from html.parser import HTMLParser
from explorer3.paths import DATA
class Assets(HTMLParser):
    def __init__(self):
        super().__init__();self.urls=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('src','href') and tag in ('script','link'):
                self.urls.append(value)
html=urllib.request.urlopen('http://127.0.0.1:8030/').read().decode()
p=Assets();p.feed(html)
for url in p.urls:
    if not url.startswith('data:'):
        assert url.startswith('/'),f'External asset: {url}'
        response=urllib.request.urlopen('http://127.0.0.1:8030'+url)
        assert response.status==200
report={'local_asset_count':len(p.urls),'cdn':False,'ui_http':200,'browser_rendering':'not_verified_no_browser_tool'}
(DATA/'validation/ui-assets.json').write_text(json.dumps(report,indent=2))
print(report)
