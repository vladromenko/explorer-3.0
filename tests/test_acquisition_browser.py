"""Finite browser acceptance of existing pages; no hardware or learned-skill claim."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def test_acquisition_pages_without_hardware(tmp_path):
    from playwright.sync_api import sync_playwright, expect
    root = Path(__file__).resolve().parents[1]
    artifact = Path.home() / '.local/share/explorer3/simlab/skill-transfer/preflight-v1/packet.json'
    if not artifact.exists():
        import pytest
        pytest.skip('Homebrain simulation donor artifact is required for artifact-view acceptance')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    env = {**os.environ, 'EXPLORER3_DATA': str(tmp_path), 'EXPLORER3_TEST_PACKET': str(artifact), 'PYTHONPATH': str(root), 'EXPLORER3_TEST_PORT': str(port)}
    env.pop('PYTEST_CURRENT_TEST', None)
    base = f'http://127.0.0.1:{port}'
    with (tmp_path / 'server.log').open('w') as log:
        process = subprocess.Popen([str(root / '.venv/bin/python'), 'scripts/acquisition_test_server.py'], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            ready = False
            end = time.monotonic() + 45
            while not ready and time.monotonic() < end and process.poll() is None:
                try:
                    with urllib.request.urlopen(base + '/api/swarm/status', timeout=1):
                        ready = True
                except (OSError, TimeoutError):
                    time.sleep(.2)
            assert ready, (tmp_path / 'server.log').read_text()
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
                page = browser.new_page(viewport={'width': 1280, 'height': 900})
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.goto(base + '/swarm')
                page.get_by_role('button', name='принеси печенье', exact=True).click()
                try:
                    expect(page.get_by_text('нужен пример захвата SO-101 с измеренной геометрией', exact=True)).to_be_visible()
                except AssertionError:
                    raise AssertionError(page.locator('body').inner_text()[-5000:])
                page.get_by_role('button', name='Отменить запрос навыка', exact=True).click()
                expect(page.get_by_text('запрос навыка отменён', exact=True)).to_be_visible()
                # Actual selection validator refuses incompatible geometry.
                page.request.post(base + '/fixture/incompatible')
                expect(page.get_by_role('button', name='Выбрать опыт', exact=False)).to_be_visible()
                page.get_by_role('button', name='Выбрать опыт', exact=False).click()
                expect(page.get_by_text('Размеры выходят за область первого переноса', exact=True)).to_be_visible()
                request = page.request.post(base + '/fixture/data').json()
                expect(page.get_by_role('link', name='Исходный опыт SO-101', exact=True)).to_be_visible()
                link = page.get_by_role('link', name='Исходный опыт SO-101', exact=True).get_attribute('href')
                packet = page.request.get(base + link).json()
                assert packet == json.loads(artifact.read_text())
                expect(page.get_by_role('link', name='Подготовить или открыть план обучения', exact=True)).to_be_visible()
                page.get_by_role('link', name='Подготовить или открыть план обучения', exact=True).click()
                expect(page.get_by_text('Для запроса: Проверка интерфейса, без роботов', exact=True)).to_be_visible()
                expect(page.get_by_role('button', name='Запустить подготовленное обучение', exact=True)).to_be_disabled()
                page.request.post(base + '/fixture/awaiting_physical_trial')
                page.goto(base + '/swarm')
                expect(page.get_by_text('ждёт разрешения испытания', exact=True)).to_be_visible()
                page.request.post(base + '/fixture/accepted')
                expect(page.get_by_text('новый навык принят', exact=True)).to_be_visible()
                page.reload()
                expect(page.get_by_text('новый навык принят', exact=True)).to_be_visible()
                page.get_by_role('button', name='Продолжить сохранённую цель', exact=True).click()
                expect(page.get_by_text('Нужно измерить место на полу, подъезд и позу выкладки.', exact=True)).to_be_visible()
                page.goto(base + '/today')
                expect(page.get_by_text('новый навык принят.', exact=False)).to_be_visible()
                assert not errors
                import sqlite3
                database = tmp_path / 'catalog.sqlite3'
                with sqlite3.connect(database) as db:
                    assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0
                browser.close()
            assert 'AssertionError' not in (tmp_path / 'server.log').read_text()
            assert 'Traceback' not in (tmp_path / 'server.log').read_text()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
