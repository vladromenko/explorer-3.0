"""Compare actual rendered projection with measured ROS calibration, no training."""
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from explorer3.paths import ROOT,DATA
from explorer3.sim.tank_model import camera_attributes,installed_geometry_matches


def test_rendered_points_project_to_measured_pixels():
    info=json.loads((ROOT/'configs/tank-hardware-evidence.json').read_text())['installed_camera']['intrinsics']
    k=np.array(info['k']).reshape(3,3)
    for x,y in [(0.,0.),(.12,.08),(-.15,-.1)]:
        root=ET.Element('mujoco')
        visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='640',offheight='400')
        world=ET.SubElement(root,'worldbody')
        ET.SubElement(world,'camera',name='calibrated',**camera_attributes())
        ET.SubElement(world,'geom',type='sphere',size='.006',pos=f'{x} {-y} -1',rgba='1 0 0 1')
        model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        with mujoco.Renderer(model,height=400,width=640) as renderer:
            renderer.update_scene(data,camera='calibrated');image=renderer.render().astype(int)
        rows,cols=np.where(image[:,:,0]>image[:,:,1]+20)
        assert len(rows)>5
        expected=k@np.array([x,y,1.])
        # Image array indices address pixel centers at index + 0.5.
        np.testing.assert_allclose([cols.mean()+.5,rows.mean()+.5],expected[:2],atol=.8)


def test_geometry_evidence_is_bound_to_exact_assets():
    entries=json.loads((DATA/'cache/source/mentorpi-t1-cad/manifest.json').read_text())['files']
    assert installed_geometry_matches(entries)
    changed=[dict(e) for e in entries];changed[0]['sha256']='0'*64
    assert not installed_geometry_matches(changed)


def test_intrinsic_randomization_and_restore():
    from explorer3.sim.mobile import MobileDock
    env=MobileDock('tank')
    try:
        base=env.model.cam_intrinsic.copy()
        env.reset(seed=1,options={'variation':'varied'})
        varied=env.model.cam_intrinsic.copy()
        assert not np.array_equal(base[:,:2],varied[:,:2])
        np.testing.assert_array_equal(base[:,2:],varied[:,2:])
        snapshot=env.snapshot()
        env.reset(seed=2,options={'variation':'nominal'})
        np.testing.assert_array_equal(base,env.model.cam_intrinsic)
        env.restore(snapshot)
        np.testing.assert_array_equal(varied,env.model.cam_intrinsic)
    finally:env.close()
