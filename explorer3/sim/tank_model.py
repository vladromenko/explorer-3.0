"""Pinned MentorPi T1 CAD mirror, imported through MuJoCo's URDF loader.

Geometry is original package data, not a hand-drawn Tank. The mirror is not an
attested manufacturer release; compare installed files at C. CAD masses do not
represent a complete assembled robot with the user's Pi and battery.
"""
import hashlib,json,os,tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from explorer3.paths import DATA


def compatible_mesh(path):
    # MuJoCo's binary STL decoder caps a mesh at 200k faces. Convert the
    # original triangles losslessly to OBJ, without decimation or new geometry.
    content=path.read_bytes()
    count=int.from_bytes(content[80:84],'little')
    if count<=200000:return path
    if len(content)!=84+50*count:raise ValueError('Invalid binary STL')
    checksum=hashlib.sha256(content).hexdigest()
    output=path.with_name(path.stem+'-'+checksum[:12]+'.obj')
    if not output.exists():
        triangles=np.frombuffer(content,offset=84,count=count,dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')]))['vertices']
        vertices,indices=np.unique(triangles.reshape(-1,3),axis=0,return_inverse=True)
        temporary=output.with_suffix('.tmp')
        with temporary.open('w') as stream:
            stream.write('# Exact original STL triangles; source sha256 '+checksum+'\n')
            np.savetxt(stream,vertices,fmt='v %.9g %.9g %.9g')
            np.savetxt(stream,indices.reshape(-1,3)+1,fmt='f %d %d %d')
        temporary.replace(output)
    return output


def native_tree():
    source=Path(os.environ.get('EXPLORER3_TANK_ASSETS',str(DATA/'cache/source/mentorpi-t1-cad')))
    manifest_path=source/'manifest.json'
    if not manifest_path.exists():raise FileNotFoundError('Нужны исходные meshes T1: scripts/cache_tank_model.py')
    evidence=json.loads(manifest_path.read_text())
    for entry in evidence['files']:
        path=source/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Tank source checksum changed: '+entry['path'])
    root=ET.parse(source/'urdf/tank.xacro').getroot()
    for item in ET.parse(source/'urdf/depth_cam.xacro').getroot():
        if item.tag in {'link','joint'}:root.append(item)
    for item in list(root):
        if item.tag.startswith('{'):root.remove(item)  # unused M_PI property, no generated geometry
    for mesh in root.findall('.//mesh'):
        path=source/mesh.get('filename').removeprefix('package://mentorpi_description/')
        mesh.set('filename',str(compatible_mesh(path)))
    settings=ET.SubElement(root,'mujoco');ET.SubElement(settings,'compiler',discardvisual='false',fusestatic='false',balanceinertia='true')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    with tempfile.TemporaryDirectory(prefix='explorer-t1-') as directory:
        destination=Path(directory)/'converted.xml';mujoco.mj_saveLastXML(str(destination),model)
        result=ET.parse(destination).getroot()
    for geom in result.findall('.//geom'):
        if geom.get('contype')!='0':geom.set('group','3');geom.set('rgba','0 0 0 0')
    evidence.update(source='MentorPi T1 package CAD from pinned public mirror',
        geometry='original STL chassis/tracks, board, depth camera and lidar; original fixed URDF transforms',
        original_cad_mass_kg=sum(float(x.get('mass')) for x in result.findall('.//inertial')),
        calibration='geometry_unverified_against_installed_robot; dynamics_uncalibrated',
        camera='URDF depth_cam mount; optical center and FOV still require calibration',
        adaptations=['native MuJoCo URDF import','one rigid planar chassis; tracks represented by chassis CAD mesh',
                     'published complete-kit mass used for surrogate dynamics, CAD partial inertias not a whole-robot measurement'])
    return result,evidence
