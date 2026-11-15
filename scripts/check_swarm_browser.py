"""The swarm screen in a real browser: simulation on, «принеси печенье», agents/plan/memory visible,
back to the real robots. Screenshots and page errors go to DATA/validation/swarm-browser/.

  .venv-sim/bin/python scripts/check_swarm_browser.py [http://127.0.0.1:8030]
"""
import asyncio
import json
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

OUT = Path.home() / '.local/share/explorer3/validation/swarm-browser'


async def main(base):
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'started': time.time(), 'steps': [], 'page_errors': []}
    async with async_playwright() as p:
        # the Chromium already installed on Homebrain (Playwright's own build may be a newer revision)
        shells = sorted((Path.home() / '.cache/ms-playwright').glob('chromium_headless_shell-*/chrome-linux/headless_shell'))
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox'],
                                          executable_path=str(shells[-1]) if shells else None)
        page = await browser.new_page(viewport={'width': 1500, 'height': 1300})
        page.on('pageerror', lambda e: report['page_errors'].append(str(e)))
        try:
            await page.goto(base + '/swarm')
            await page.get_by_text('Агенты', exact=True).wait_for()
            report['steps'].append('страница открыта')
            await page.get_by_role('button', name='Симуляция без роботов').click()
            await page.get_by_text('СИМУЛЯЦИЯ:', exact=False).wait_for(timeout=30000)
            report['steps'].append('симуляция включена')
            await page.get_by_role('button', name='принеси печенье').click()
            await page.get_by_text('Координатор задачи', exact=True).wait_for()
            await page.wait_for_timeout(15000)
            await page.screenshot(path=str(OUT / 'during.png'), full_page=True)
            report['steps'].append('команда отдана, план и агенты на экране')
            end = time.time() + 420
            final = None
            while time.time() < end:
                text = await page.locator('body').inner_text()
                for marker in ('доставлено', 'не найден', 'не взял', 'не доехал', 'нужен человек'):
                    if marker in text:
                        final = marker
                if final:
                    break
                await page.wait_for_timeout(3000)
            await page.wait_for_timeout(2000)
            await page.screenshot(path=str(OUT / 'after.png'), full_page=True)
            report['outcome_marker'] = final
            report['steps'].append(f'итог на экране: {final}')
        finally:
            try:
                await page.get_by_role('button', name='К настоящим роботам').click(timeout=5000)
                await page.wait_for_timeout(3000)
                report['steps'].append('возврат к настоящим роботам')
            except Exception as exc:
                report['steps'].append(f'возврат не нажат: {exc}')
            await browser.close()
    report['finished'] = time.time()
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8030'))
