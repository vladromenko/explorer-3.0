"""Actual browser against an isolated application; no motor/model endpoints."""
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
import pytest


def test_preparation_and_guides_without_execution(tmp_path):
    from playwright.sync_api import sync_playwright, expect
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',8037))
    env = {**os.environ,'EXPLORER3_DATA':str(tmp_path),'PYTHONPATH':str(root)}
    env.pop('PYTEST_CURRENT_TEST',None)
    with (tmp_path/'server.log').open('w') as log:
        process = subprocess.Popen([str(root/'.venv/bin/python'),'scripts/readiness_test_server.py'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            ready = False
            deadline = time.monotonic()+45
            while not ready and time.monotonic()<deadline and process.poll() is None:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8037/datasets/guide',timeout=1):ready=True
                except (OSError,TimeoutError):time.sleep(.2)
            assert ready, (tmp_path/'server.log').read_text()
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True,args=['--no-sandbox'])
                page = browser.new_page(viewport={'width':390,'height':844})
                errors=[]
                page.on('pageerror',lambda e:errors.append(str(e)))
                for route in ('/datasets/guide','/datasets/acceptance','/training/prepare'):
                    page.goto('http://127.0.0.1:8037'+route)
                    expect(page.locator('.nicegui-content')).to_be_visible()
                    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
                expect(page.get_by_role('button',name='Запустить подготовленное обучение',exact=True)).to_be_disabled()
                assert not errors
                browser.close()
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=5)
