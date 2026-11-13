"""Pinned Menagerie model, with an explicit pick/place task extension."""
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from explorer3.paths import DATA
from .scene import laboratory

MENAGERIE_REVISION = '370c43965323c0944b433b202aac5058b5f907a9'


def so101_model():
    source = Path(os.environ.get('EXPLORER3_SIM_ASSETS', str(DATA / 'cache/source/mujoco_menagerie/robotstudio_so101')))
    if not (source / 'so101.xml').is_file():
        raise FileNotFoundError('Install the pinned Menagerie assets with scripts/setup-sim.sh')
    tree = ET.parse(source / 'so101.xml')
    root = tree.getroot()
    root.set('model', 'Explorer SO101 pick-place (uncalibrated)')
    root.find('compiler').set('meshdir', str(source / 'assets'))
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.set('timestep', '0.0005')
    option.set('iterations', '100')
    ET.SubElement(root.find('default'),'geom',solref='.001 1',solimp='.95 .99 .001 .5 2')
    # Upstream's soft pad defaults allow several mm of numerical intrusion under
    # the capped servo force. Retain all original collision shapes and friction.
    for name in ('collision', 'collision_gripper', 'collision_gripper_mesh'):
        geom=root.find(f".//default[@class='{name}']/geom")
        geom.set('solref','.001 1')
        geom.set('solimp','.95 .99 .001 .5 2')
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='640', offheight='480')
    world = root.find('worldbody')
    ET.SubElement(world, 'light', pos='0 0 2', dir='0 0 -1')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='2 2 0.05', rgba='.25 .3 .35 1')
    ET.SubElement(world, 'camera', name='overview', pos='.7 -.7 .65', xyaxes='.707 .707 0 -.4 .4 .82')
    cube = ET.SubElement(world, 'body', name='task_cube', pos='.24 -.06 .017')
    ET.SubElement(cube, 'freejoint', name='cube_free')
    ET.SubElement(cube, 'geom', name='cube', type='box', size='.016 .016 .016', mass='.035',
                  friction='1 .03 .003', solref='.001 1', solimp='.95 .99 .001 .5 2', rgba='1 .65 .05 1')
    box = ET.SubElement(world, 'body', name='target_box', pos='.24 .10 0')
    ET.SubElement(box, 'geom', name='bin_bottom', type='box', pos='0 0 .004', size='.055 .045 .004', rgba='.15 .6 .2 1')
    for name, pos, size in [('left','0 -.045 .02','.06 .005 .02'),
                            ('right','0 .045 .02','.06 .005 .02'),
                            ('front','.055 0 .02','.005 .04 .02'),
                            ('back','-.055 0 .02','.005 .04 .02')]:
        ET.SubElement(box, 'geom', name='bin_'+name, type='box', pos=pos, size=size, rgba='.15 .6 .2 1')
    presentation=laboratory(root)
    xml = ET.tostring(root, encoding='unicode')
    manifest = {'upstream':'google-deepmind/mujoco_menagerie/robotstudio_so101',
                'revision':MENAGERIE_REVISION, 'license':'Apache-2.0 (upstream model LICENSE)','presentation':presentation,
                'calibration':'uncalibrated_to_local_robot', 'extension':'free cube + open bin + dwell success',
                'xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),
                'numerical_contact':{'timestep_s':.0005,'solref':[.001,1],'solimp':[.95,.99,.001,.5,2],
                    'iterations':100,'purpose':'bound numerical interpenetration, not fitted material compliance'}}
    # Include mesh hashes, not only the XML filename references.
    manifest['assets_sha256'] = hashlib.sha256(b''.join(
        p.name.encode()+hashlib.sha256(p.read_bytes()).digest() for p in sorted((source/'assets').glob('*')) if p.is_file()
    )).hexdigest()
    return xml, manifest


def rosmaster_arm_model():
    """Manufacturer model; task/controller additions have separate provenance."""
    from scipy.spatial.transform import Rotation
    import numpy as np
    from .manufacturer import native_tree, ARM_JOINTS
    from .vendor_rosmaster.arm_model import DCW2_ON_ARM4, REP103_OPTICAL
    def numbers(values):return ' '.join(str(float(x)) for x in values)
    root,manifest=native_tree()
    root.set('model','Explorer ROSMASTER · original Yahboom M3Pro')
    ET.SubElement(root,'option',timestep='.0005',integrator='implicitfast',iterations='100')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='640',offheight='480')
    ET.SubElement(root,'statistic',center='.12 0 .2',extent='.6')
    defaults=ET.SubElement(root,'default')
    ET.SubElement(defaults,'joint',damping='.2',armature='.003')
    ET.SubElement(defaults,'geom',friction='1 .03 .003',condim='3',solref='.001 1',solimp='.95 .99 .001 .5 2')
    for constraint in root.findall('equality/joint'):
        constraint.set('solref','.001 1');constraint.set('solimp','.99 .999 .001 .5 2')
    world=root.find('worldbody')
    ET.SubElement(world,'light',pos='0 0 2',dir='0 0 -1')
    ET.SubElement(world,'geom',name='floor',type='plane',size='2 2 .05',rgba='.25 .3 .35 1')
    wrist=root.find(".//body[@name='arm4']")
    rotation=Rotation.from_euler('xyz',DCW2_ON_ARM4[1]).as_matrix() @ REP103_OPTICAL @ np.diag([1,-1,-1])
    ET.SubElement(wrist,'camera',name='front_cam',pos=numbers(DCW2_ON_ARM4[0]),
        quat=numbers(Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]),fovy='60')
    ET.SubElement(root.find(".//body[@name='arm5']"),'site',name='gripperframe',pos=root.find(".//body[@name='Gripping']").get('pos'),size='.003',rgba='1 0 0 1')
    actuators=ET.SubElement(root,'actuator')
    for i,joint in enumerate(ARM_JOINTS):
        limits=root.find(f".//joint[@name='{joint}']").get('range')
        ET.SubElement(actuators,'position',name=f'motor{i+1}',joint=joint,
            ctrlrange=limits,kp='60',kv='3',forcerange='-3 3')
    cube=ET.SubElement(world,'body',name='task_cube',pos='.22 -.04 .018');ET.SubElement(cube,'freejoint',name='cube_free')
    ET.SubElement(cube,'geom',name='cube',type='box',size='.016 .016 .016',mass='.035',rgba='1 .65 .05 1')
    box=ET.SubElement(world,'body',name='target_box',pos='.22 .08 0')
    ET.SubElement(box,'geom',name='bin_bottom',type='box',pos='0 0 .004',size='.055 .045 .004',rgba='.15 .6 .2 1')
    for name,pos,size in [('left','0 -.045 .02','.06 .005 .02'),('right','0 .045 .02','.06 .005 .02'),
                           ('front','.055 0 .02','.005 .04 .02'),('back','-.055 0 .02','.005 .04 .02')]:
        ET.SubElement(box,'geom',name='bin_'+name,type='box',pos=pos,size=size,rgba='.15 .6 .2 1')
    presentation=laboratory(root)
    xml=ET.tostring(root,encoding='unicode')
    manifest.update(xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),presentation=presentation,
        numerical_contact={'timestep_s':.0005,'solref':[.001,1],'solimp':[.95,.99,.001,.5,2],'iterations':100,'purpose':'bound numerical interpenetration, not fitted material compliance'},
        controller_assumptions=['position servo gains/torque caps','lag','friction','servo-to-gripper alignment pending physical check','camera FOV pending intrinsics'],
        task_geometry='simulated 32mm cube and bin; not a measurement of user objects')
    return xml,manifest
