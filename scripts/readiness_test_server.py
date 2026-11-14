"""Isolated UI: no hardware controllers, job worker or scheduled execution."""
from nicegui import ui
from explorer3 import catalog, training_pages
from explorer3.paths import prepare
from explorer3.jobs import Jobs

prepare()
catalog.initialize()
training_pages.install('', Jobs(None))
ui.run(host='127.0.0.1', port=8037, reload=False, show=False, tailwind=False,
       language='ru', title='Explorer software checks')
