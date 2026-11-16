"""Isolated browser fixture: production pages, no devices, no job worker."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

if not os.environ.get('EXPLORER3_DATA'):
    raise RuntimeError('An isolated EXPLORER3_DATA directory is mandatory')
from nicegui import app, ui
from explorer3 import catalog, knowledge, swarm, swarm_pages, today_pages, training_pages
from explorer3.paths import DATA, prepare
from explorer3.jobs import Jobs
from explorer3.cognitive import broker, store

prepare()
catalog.initialize()
store.initialize()

class NoDevice:
    def start(self, *a, **kw):
        raise AssertionError('Browser fixture must never start a robot')
    def cancel(self, *a):
        pass
    def release(self):
        pass

swarm.SWARM = swarm.Swarm(tank=NoDevice(), ros=NoDevice())
control = SimpleNamespace(robot='tank', owner=None, stop=lambda *a: None)
swarm_pages.readiness = lambda control: [('Изолированная проверка', True, 'устройства отключены')]
swarm_pages.camera_jpeg = lambda robot: None
swarm_pages.render_map = lambda: None
swarm_pages.install(app, ui, control, lambda exc: ui.notify(str(exc), type='negative'))
today_pages.install(app, ui)
training_pages.install('', Jobs(None))

packet_path = os.environ.get('EXPLORER3_TEST_PACKET')
packet = json.loads(Path(packet_path).read_text()) if packet_path else None
if packet:
    sha = store.digest(packet)
    directory = DATA / 'world/skill_packets'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (sha + '.json')).write_text(store.encoded(packet))
    knowledge.STORE.parent.mkdir(parents=True, exist_ok=True)
    entry = knowledge.add('cube', 'browser_fixture', {}, domain='sim')
    entry['skill_packet_sha256'] = sha
    knowledge._save([entry])

@app.post('/fixture/{scenario}')
def fixture(scenario: str):
    # Only this isolated server contains these deterministic UI fixtures.
    with catalog.connect() as db:
        db.execute("DELETE FROM cognitive_requests WHERE decision_id='acquisition'")
    context = {'source_domain': 'sim'}
    if scenario == 'incompatible':
        context['size_m'] = [.1] * 3
    request = broker.prepare_acquisition('fixture', 'Проверка интерфейса, без роботов', 'rosmaster', 'cube', {}, domain='sim', context=context)
    if scenario in ('data', 'awaiting_physical_trial', 'accepted'):
        request = broker.select_acquisition_packet(request['id'], sha)
        if scenario != 'data':
            request.update(state=scenario, operator_next='Изолированный пример состояния; физических испытаний не было.')
            with catalog.connect() as db:
                broker._save_acquisition(db, request, 'browser_fixture')
    return request

ui.run(host='127.0.0.1', port=int(os.environ['EXPLORER3_TEST_PORT']), reload=False, show=False, tailwind=False,
       language='ru', title='Explorer isolated acquisition checks')
