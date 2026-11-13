import asyncio
from types import SimpleNamespace
import pytest
from fastapi import HTTPException
from explorer3 import rosmaster_control


def test_local_camera_refuses_before_reading_config_or_opening_network(monkeypatch):
    monkeypatch.setattr('explorer3.local_work.hardware_allowed',lambda *_:False)
    monkeypatch.setattr(rosmaster_control.httpx,'AsyncClient',lambda **_:pytest.fail('Network forbidden'))
    with pytest.raises(HTTPException) as error:asyncio.run(rosmaster_control.camera(SimpleNamespace()))
    assert error.value.status_code==503
