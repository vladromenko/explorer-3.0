"""Isolated UI: no hardware controllers, job worker or scheduled execution."""
from nicegui import ui
from explorer3 import catalog, training_pages
from explorer3.paths import prepare
from explorer3.jobs import Jobs

prepare()
catalog.initialize()
training_pages.install('', Jobs(None))
from explorer3 import recording_pages, fleet_config
from explorer3.recording_readiness import evaluate,item
from explorer3.paths import DATA,ROOT
fleet_config.atomic_write(fleet_config.path(DATA), __import__('json').loads((ROOT/'configs/fleet.json').read_text()))
class FixtureReadiness:
    at=0
    def snapshot(self):
        robots={r:evaluate(r,{},*fleet_config.robot_allowed(r,DATA)) for r in ('tank','rosmaster')}
        robots['so101']={'checks':[item('owner','Владелец SO-101',False,'legacy_8020','Откройте SO-101 и нажмите «Взять SO-101 в Explorer»')]}
        return {'general':[item('disk','Свободное место',False,7,'Перенесите записи на внешний диск')],'robots':robots}
recording_pages.install(FixtureReadiness())
recording_pages.install_collection()
ui.run(host='127.0.0.1', port=8037, reload=False, show=False, tailwind=False,
       language='ru', title='Explorer software checks')
