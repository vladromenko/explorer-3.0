"""Original Yahboom M3Pro URDF/meshes via MuJoCo's upstream URDF importer.

No hand-drawn replacement links. CAD inertias are retained, but are not
measurements of the user's complete robot; controller dynamics remain provisional.
"""
import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from explorer3.paths import DATA

ARCHIVE_SHA256='7f9083d301fef50633445272505e9acb561a4b6abe362dd946d7deaaf4caff0f'
ARM_JOINTS=['arm1_Joint','arm2_Joint','arm3_Joint','arm4_Joiint','arm5_Joint','rlink1_Joint']


def source_directory():
    return Path(os.environ.get('EXPLORER3_M3PRO_ASSETS',str(DATA/'cache/source/yahboom-downloads/m3pro-orin/m3pro_ws/src/M3Pro')))


def native_tree(*, lock_arm=False):
    source=source_directory();urdf=source/'urdf/M3Pro.urdf'
    if not urdf.is_file():raise FileNotFoundError('Нужна исходная модель Yahboom M3Pro; см. docs/SIM_MODELS.md')
    tree=ET.parse(urdf).getroot()
    # Upstream contains an empty visual on an inertial-only link. The native
    # importer requires geometry; removing that empty tag removes no geometry.
    for link in tree.findall('link'):
        for tag in ('visual','collision'):
            for element in link.findall(tag):
                if element.find('geometry') is None:link.remove(element)
    assets={}
    for mesh in tree.findall('.//mesh'):
        name=Path(mesh.get('filename')).name;path=source/'meshes'/name
        mesh.set('filename',str(path));assets[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    settings=ET.SubElement(tree,'mujoco')
    ET.SubElement(settings,'compiler',discardvisual='false',fusestatic='false',balanceinertia='true')
    for joint in tree.findall('joint'):
        if lock_arm and joint.get('name') in ARM_JOINTS[:5]:
            index=ARM_JOINTS.index(joint.get('name'))
            origin=joint.find('origin');axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
            angle=np.deg2rad([0,40,-90,-65,0][index])
            rotation=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy'),sep=' '))*Rotation.from_rotvec(axis/np.linalg.norm(axis)*angle)
            origin.set('rpy',' '.join(str(x) for x in rotation.as_euler('xyz')))
        if lock_arm or joint.get('name') in ('Camera_Joint','lwheel1_Joint','lwheel2_Joint','rwheel1_Joint','rwheel2_Joint'):
            joint.set('type','fixed')
            mimic=joint.find('mimic')
            if mimic is not None:joint.remove(mimic)
    model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'))
    with tempfile.TemporaryDirectory(prefix='explorer-m3pro-') as directory:
        destination=Path(directory)/'converted.xml';mujoco.mj_saveLastXML(str(destination),model)
        result=ET.parse(destination).getroot()
    srdf=source.parents[2]/'yahboomcar_ws/src/M3Pro_config/config/M3Pro.srdf'
    semantic=ET.parse(srdf).getroot()
    contact=ET.SubElement(result,'contact')
    names={b.get('name') for b in result.findall('.//body')}
    for pair in semantic.findall('disable_collisions'):
        if pair.get('link1') in names and pair.get('link2') in names:
            ET.SubElement(contact,'exclude',body1=pair.get('link1'),body2=pair.get('link2'))
    # Keep source visual meshes; mark the duplicate collision meshes invisible.
    # Collision exclusions are the manufacturer's SRDF list, including linked
    # gripper parts and the pan bearing inside the base mesh.
    for geom in result.findall('.//geom'):
        if geom.get('contype')!='0':geom.set('group','3');geom.set('rgba','0 0 0 0')
    manifest={'source':'Yahboom official ROSMASTER M3 PRO code archive, orin/M3Pro_ws-0.0.1.zip',
        'source_url':'https://drive.google.com/file/d/1WxZbB2ulfWldWYRKiT5zheso-8EtqfdV/view',
        'archive_sha256':ARCHIVE_SHA256,'srdf_sha256':hashlib.sha256(srdf.read_bytes()).hexdigest(),
        'gripper_closed_rad':float(semantic.find("group_state[@name='close']/joint[@name='rlink1_Joint']").get('value')),'urdf_sha256':hashlib.sha256(urdf.read_bytes()).hexdigest(),
        'meshes_sha256':assets,'geometry':'original URDF link transforms, visual/collision STL, gripper mimic constraints',
        'inertias':'original manufacturer CAD values, not identified whole-robot mass',
        'calibration':'manufacturer_geometry_unverified_local_dynamics',
        'license':'package.xml says Apache 2.0 but retains TurtleBot metadata; assets cached locally, no redistribution',
        'adaptations':['empty visual tags removed','camera pan and wheels fixed for arm task' if not lock_arm else 'fixed resting joint pose for docking visual model',
                        'MuJoCo native URDF import; balanceinertia enabled; manufacturer SRDF collision exclusions','duplicate collision geometry invisible, still active']}
    return result,manifest
