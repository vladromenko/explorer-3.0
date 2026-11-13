"""Pinned Menagerie model, with an explicit pick/place task extension."""
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from explorer3.paths import DATA

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
    option.set('timestep', '0.002')
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
                  friction='1 .03 .003', rgba='1 .65 .05 1')
    box = ET.SubElement(world, 'body', name='target_box', pos='.24 .10 0')
    ET.SubElement(box, 'geom', name='bin_bottom', type='box', pos='0 0 .004', size='.055 .045 .004', rgba='.15 .6 .2 1')
    for name, pos, size in [('left','0 -.045 .02','.06 .005 .02'),
                            ('right','0 .045 .02','.06 .005 .02'),
                            ('front','.055 0 .02','.005 .04 .02'),
                            ('back','-.055 0 .02','.005 .04 .02')]:
        ET.SubElement(box, 'geom', name='bin_'+name, type='box', pos=pos, size=size, rgba='.15 .6 .2 1')
    xml = ET.tostring(root, encoding='unicode')
    manifest = {'upstream':'google-deepmind/mujoco_menagerie/robotstudio_so101',
                'revision':MENAGERIE_REVISION, 'license':'Apache-2.0 (upstream model LICENSE)',
                'calibration':'uncalibrated_to_local_robot', 'extension':'free cube + open bin + dwell success',
                'xml_sha256':hashlib.sha256(xml.encode()).hexdigest()}
    # Include mesh hashes, not only the XML filename references.
    manifest['assets_sha256'] = hashlib.sha256(b''.join(
        p.name.encode()+hashlib.sha256(p.read_bytes()).digest() for p in sorted((source/'assets').glob('*')) if p.is_file()
    )).hexdigest()
    return xml, manifest


def rosmaster_arm_model():
    """Use the saved URDF transforms; collision shapes/masses are explicit surrogates."""
    from scipy.spatial.transform import Rotation
    import numpy as np
    from .vendor_rosmaster.arm_model import CHAIN,DCW2_ON_ARM4,REP103_OPTICAL
    def numbers(values):return ' '.join(str(float(x)) for x in values)
    def quat(rotation):return numbers(rotation.as_quat()[[3,0,1,2]])
    root=ET.Element('mujoco',model='Explorer ROSMASTER provisional arm')
    ET.SubElement(root,'compiler',angle='radian',autolimits='true')
    ET.SubElement(root,'option',timestep='.002',integrator='implicitfast')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='640',offheight='480')
    ET.SubElement(root,'statistic',center='.15 0 .2',extent='.55')
    defaults=ET.SubElement(root,'default')
    ET.SubElement(defaults,'joint',damping='.2',armature='.003')
    ET.SubElement(defaults,'geom',friction='1 .03 .003',rgba='.25 .45 .65 1',condim='3')
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',pos='0 0 2',dir='0 0 -1')
    ET.SubElement(world,'geom',name='floor',type='plane',size='2 2 .05',rgba='.25 .3 .35 1')
    parent=ET.SubElement(world,'body',name='chassis')
    ET.SubElement(parent,'geom',type='box',pos='-.0125 0 .07',size='.1425 .12 .07',mass='2',rgba='.12 .16 .22 1')
    limits=[(-np.pi/2,np.pi/2)]*4+[(-np.pi/2,np.pi)]
    for i,(xyz,rpy,axis) in enumerate(CHAIN):
        parent=ET.SubElement(parent,'body',name=f'arm{i+1}',pos=numbers(xyz),quat=quat(Rotation.from_euler('xyz',rpy)))
        ET.SubElement(parent,'joint',name=f'joint{i+1}',type='hinge',axis=numbers(axis),range=numbers(limits[i]))
        end=CHAIN[i+1][0] if i<4 else (0,0,.035)
        ET.SubElement(parent,'geom',name=f'link{i+1}',type='capsule',fromto='0 0 0 '+numbers(end),size='.012',mass='.1')
        if i==3:
            camera_rot=Rotation.from_euler('xyz',DCW2_ON_ARM4[1]).as_matrix() @ REP103_OPTICAL @ np.diag([1,-1,-1])
            ET.SubElement(parent,'camera',name='front_cam',pos=numbers(DCW2_ON_ARM4[0]),quat=quat(Rotation.from_matrix(camera_rot)),fovy='60')
    ET.SubElement(parent,'site',name='gripperframe',pos='0 0 .085',size='.003',rgba='1 0 0 1')
    for name,sign in [('jaw_right',1),('jaw_left',-1)]:
        jaw=ET.SubElement(parent,'body',name=name,pos=f'{sign*.006} 0 .04')
        ET.SubElement(jaw,'joint',name=name,type='slide',axis=f'{sign} 0 0',range='0 .055',damping='1')
        ET.SubElement(jaw,'geom',name=name+'_pad',type='box',pos='0 0 .025',size='.004 .013 .025',mass='.025',rgba='.8 .8 .8 1')
    equality=ET.SubElement(root,'equality');ET.SubElement(equality,'joint',joint1='jaw_left',joint2='jaw_right',polycoef='0 1 0 0 0')
    actuators=ET.SubElement(root,'actuator')
    for i,limit in enumerate(limits):
        ET.SubElement(actuators,'position',name=f'motor{i+1}',joint=f'joint{i+1}',ctrlrange=numbers(limit),kp='60',kv='3',forcerange='-3 3')
    ET.SubElement(actuators,'position',name='gripper',joint='jaw_right',ctrlrange='-.024 .055',kp='500',kv='10',forcerange='-12 12')
    cube=ET.SubElement(world,'body',name='task_cube',pos='.22 -.04 .018');ET.SubElement(cube,'freejoint',name='cube_free')
    ET.SubElement(cube,'geom',name='cube',type='box',size='.016 .016 .016',mass='.035',rgba='1 .65 .05 1')
    box=ET.SubElement(world,'body',name='target_box',pos='.22 .08 0')
    ET.SubElement(box,'geom',name='bin_bottom',type='box',pos='0 0 .004',size='.055 .045 .004',rgba='.15 .6 .2 1')
    for name,pos,size in [('left','0 -.045 .02','.06 .005 .02'),('right','0 .045 .02','.06 .005 .02'),
                           ('front','.055 0 .02','.005 .04 .02'),('back','-.055 0 .02','.005 .04 .02')]:
        ET.SubElement(box,'geom',name='bin_'+name,type='box',pos=pos,size=size,rgba='.15 .6 .2 1')
    xml=ET.tostring(root,encoding='unicode')
    return xml,{'source':'user console 8090 cubebot.arm_model CHAIN and camera transforms',
        'xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),'calibration':'provisional_geometry_uncalibrated_dynamics',
        'assumed':['capsule collisions','link masses 0.1kg','motor gains/force caps','parallel-jaw surrogate','friction'],
        'provenance':json.loads((Path(__file__).parent/'vendor_rosmaster/provenance.json').read_text())}
