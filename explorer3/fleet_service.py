"""Managed shared-world subprocess; same app, no access to robot agents."""
import os
from .sim_service import SimulatorProcess

service=SimulatorProcess('explorer3.sim.fleet_worker',
    int(os.environ.get('EXPLORER3_FLEET_API_PORT','8038')),
    int(os.environ.get('EXPLORER3_FLEET_VIEWER_PORT','8039')),'fleet-simulator')
