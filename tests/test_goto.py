"""«Подъехать к …»: prompt translation and the controller, with a fake detector (no camera, no robot)."""
import time

import numpy as np
import pytest

pytest.importorskip('cv2')
from explorer3 import goto


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes
    def detect(self, rgb, prompt, conf=.35):
        time.sleep(.01)
        return self.boxes, 10.0


def make(monkeypatch, boxes):
    monkeypatch.setattr(goto.Detector, 'shared', classmethod(lambda cls: FakeDetector(boxes)))
    return goto.GoTo('tank', 'к носку')


def drive(g, frames=8):
    rgb = np.zeros((400, 640, 3), np.uint8)
    out = None
    for _ in range(frames):
        out = g.command(rgb)
        time.sleep(.03)
    return out


def test_prompt_translation_any_case():
    assert goto.to_prompt('подъедь к носку') == 'sock'
    assert goto.to_prompt('к красной коробке') == 'red box'
    assert goto.to_prompt('go to the red ball') == 'red ball'


def test_turns_toward_object_on_the_right(monkeypatch):
    g = make(monkeypatch, [[520, 250, 600, 300, .9]])           # right of centre, far
    v, w = drive(g)
    assert w < 0 and v == 0                                       # rotate first (clockwise)
    g.cancel()


def test_drives_forward_when_centred_and_stops_close(monkeypatch):
    g = make(monkeypatch, [[300, 230, 340, 260, .9]])           # centred, bottom edge -> ~1 m
    v, w = drive(g)
    assert v > 0 and abs(w) < .2 and g.state == 'approach'
    assert 0.5 < g.distance < 1.6
    g.detector.boxes = [[290, 330, 350, 398, .9]]               # bottom of frame -> arrived
    time.sleep(.05)
    assert drive(g) == [0., 0.] and g.state == 'arrived' and not g.active


def test_searches_then_gives_up(monkeypatch):
    g = make(monkeypatch, [])
    g.searched = 2 * np.pi + 1
    assert drive(g, 2) == [0., 0.] and g.state == 'not_found'
