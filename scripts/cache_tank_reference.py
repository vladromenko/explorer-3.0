"""Cache all linked English MentorPi T1 manual pages as reference, never execute their code."""
import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit
from urllib.request import Request, urlopen

PREFIX = 'https://wiki.hiwonder.com/projects/MentorPi-T1/en/latest/docs/'
ROOT = Path.home()/'.local/share/explorer3/cache/references/mentorpi-t1-complete'


class Page(HTMLParser):
    def __init__(self, url):
        super().__init__(); self.url=url; self.links=set(); self.text=[]; self.skip=0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'): self.skip += 1
        if tag == 'a':
            href = dict(attrs).get('href', '')
            link = urldefrag(urljoin(self.url, href))[0]
            if link.startswith(PREFIX) and urlsplit(link).path.endswith('.html'):
                self.links.add(link)

    def handle_endtag(self, tag):
        if tag in ('script', 'style'): self.skip=max(0,self.skip-1)

    def handle_data(self, text):
        if not self.skip and text.strip(): self.text.append(text.strip())


def fetch(url):
    with urlopen(Request(url, headers={'User-Agent':'Explorer3 local reference archive'}),timeout=30) as response:
        data=response.read(); resolved=response.url
    page=Page(url); page.feed(data.decode('utf-8'))
    name=Path(urlsplit(url).path).name
    (ROOT/name).write_bytes(data)
    (ROOT/(name+'.txt')).write_text('\n'.join(page.text)+'\n')
    return {'url':url,'resolved_url':resolved,'file':name,'bytes':len(data),
            'sha256':hashlib.sha256(data).hexdigest()}, page.links


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    pending={PREFIX+'resources_download.html'};seen=set();sources=[];errors=[];navigation=set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while pending:
            current=sorted(pending-seen);pending=set();seen.update(current)
            futures={pool.submit(fetch,url):url for url in current}
            for future in concurrent.futures.as_completed(futures):
                url=futures[future]
                try:
                    record,links=future.result();sources.append(record);pending.update(links-seen)
                    if url==PREFIX+'resources_download.html':navigation=links|{url}
                except Exception as exc:errors.append({'url':url,'error':str(exc)})
    missing=sorted(navigation-{s['url'] for s in sources})
    manifest={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'purpose':'reference only; no commands executed',
              'scope':'all linked English T1 HTML chapters; external images/videos/downloads remain links',
              'sources':sorted(sources,key=lambda s:s['url']),'errors':errors,'missing_navigation_pages':missing}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pages':len(sources),'errors':errors,'missing_navigation_pages':missing,'manifest':str(ROOT/'manifest.json')}))
    if missing or not sources:raise SystemExit(1)


if __name__=='__main__':main()
