"""Bounded shared-room variants with licensed object meshes and real contacts.

Furniture dimensions and prop masses are declared simulation assumptions, not
measurements of the operator's room. No renderer-only obstacle masquerades as a
physical object. Visual meshes and collision geometry have separate purposes.
"""
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ASSETS=Path(__file__).with_name('vendor_props')
PRESETS={
    'workshop':{'label':'Мастерская','wall':[.66,.69,.70,1],'floor':[.35,.38,.39,1],
        'wood':[.45,.30,.17,1],'light':1.,'props':{'bottle':[1.65,.50,.845],'can':[1.9,.55,.825],'lemon':[1.75,.78,.805]}},
    'storage':{'label':'Зона хранения','wall':[.53,.61,.65,1],'floor':[.27,.30,.33,1],
        'wood':[.36,.39,.42,1],'light':.85,'props':{'bottle':[1.9,.70,.845],'can':[1.62,.77,.825],'lemon':[1.77,.47,.805]}},
    'inspection':{'label':'Сборочный участок','wall':[.77,.75,.69,1],'floor':[.44,.43,.40,1],
        'wood':[.54,.45,.31,1],'light':1.1,'props':{'bottle':[1.76,.78,.845],'can':[1.63,.48,.825],'lemon':[1.95,.60,.805]}}}
MASSES={'bottle':.08,'can':.15,'lemon':.07}


def vec(value):return ' '.join(str(x) for x in value)


def geom(parent,name,kind='box',**values):
    args={'name':name,'type':kind,'contype':'3','conaffinity':'3','friction':'.65 .005 .0001',
          'solref':'.003 1','solimp':'.95 .99 .001 .5 2',**values}
    return ET.SubElement(parent,'geom',**{k:vec(v) if isinstance(v,(list,tuple)) else str(v) for k,v in args.items()})


def prop_asset(root,name,position):
    source=ET.parse(ASSETS/'objects'/f'{name}.xml').getroot();asset=root.find('asset')
    prefix='prop_'+name+'_'
    for element in source.find('asset'):
        element=copy.deepcopy(element)
        if 'name' in element.attrib:element.set('name',prefix+element.get('name'))
        if element.get('texture'):element.set('texture',prefix+element.get('texture'))
        if element.get('file'):element.set('file',str((ASSETS/'objects'/element.get('file')).resolve()))
        # A bottle surface is not a second skybox; keep the original mesh and
        # use a restrained tinted material for opaque renderers.
        if element.tag=='texture' and element.get('type')=='skybox':element.set('type','2d')
        if element.tag=='material':
            element.set('reflectance','.05');element.set('specular','.25');element.set('shininess','.3')
            element.set('texrepeat','1 1');element.set('texuniform','false')
        asset.append(element)
    original=source.find(".//body[@name='object']").find('geom')
    body=ET.SubElement(root.find('worldbody'),'body',name='prop_'+name,pos=vec(position))
    ET.SubElement(body,'freejoint',name='prop_'+name+'_free')
    mesh=prefix+original.get('mesh');material=prefix+original.get('material')
    geom(body,'prop_'+name+'_visual','mesh',mesh=mesh,material=material,contype=0,conaffinity=0,mass=0,group=2)
    # Separate collision geometry, with explicit total mass and metre units.
    # Closed mesh shapes use a single convex hull; no hollow vessel claims.
    geom(body,'prop_'+name+'_collision','mesh',mesh=mesh,mass=MASSES[name],rgba=[0,0,0,0],
         group=3,condim=6,friction=[.65,.005,.001])


def furnish(root,preset='workshop',*,layout=None):
    if preset not in PRESETS:raise ValueError('Unknown shared-room preset')
    lock=json.loads((ASSETS/'manifest.json').read_text())
    for name,record in lock['files'].items():
        if hashlib.sha256((ASSETS/name).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Scene asset hash changed: '+name)
    cfg=PRESETS[preset];world=root.find('worldbody');asset=root.find('asset')
    for name,color in [('room_wall',cfg['wall']),('room_wood',cfg['wood']),('room_metal',[.16,.19,.21,1]),
                       ('room_floor',cfg['floor']),('room_trim',[.32,.37,.40,1])]:
        ET.SubElement(asset,'material',name=name,rgba=vec(color),specular='.15',shininess='.15')
    floor=world.find("geom[@name='floor']");floor.set('material','room_floor');floor.attrib.pop('rgba',None)
    floor.set('friction','.65 .005 .0001');floor.set('contype','3');floor.set('conaffinity','3')
    # Open front wall for the inspector camera; camera rays still see side/back
    # walls and every collision structure present in the shared physical world.
    for name,pos,size in [('back',[2.8,0,1.3],[.05,2.3,1.3]),('left',[.3,2.3,1.3],[2.5,.05,1.3])]:
        geom(world,'room_'+name,pos=pos,size=size,material='room_wall')
    for y in [-2.245,2.245]:geom(world,'skirting_'+str(y),pos=[.3,y,.06],size=[2.5,.015,.06],material='room_trim')
    geom(world,'back_skirting',pos=[2.745,0,.06],size=[.015,2.25,.06],material='room_trim')
    from .layout import placement
    support=copy.deepcopy(layout) if layout is not None else placement()
    height=support['effective_so101_height_m']
    table=support['so101_support'];cx,cy=table['nominal_table_center_xy_m'];hx,hy=table['nominal_table_half_size_xy_m']
    so_table=ET.SubElement(world,'body',name='so101_support_table',pos=vec([cx,cy,0.]))
    geom(so_table,'so101_table_top',pos=[0,0,height-.025],size=[hx,hy,.025],material='room_wood')
    for x in [-hx+.06,hx-.06]:
        for y in [-hy+.06,hy-.06]:
            geom(so_table,f'so101_table_leg_{x}_{y}',pos=[x,y,(height-.05)/2],
                 size=[.025,.025,(height-.05)/2],material='room_metal')
    # Calibration mats follow the actual support class, not a shared floor assumption.
    for name,y in [('so101',-1.1),('rosmaster',0.)]:
        geom(world,name+'_calibration_mat',pos=[.12,y,(height if name=='so101' else 0.)+.00015],size=[.45,.43,.0001],
             material='explorer_floor',contype=0,conaffinity=0)
    bench=ET.SubElement(world,'body',name='workbench',pos='1.8 .6 0')
    geom(bench,'workbench_top',pos=[0,0,.72],size=[.45,.35,.025],material='room_wood')
    for x in [-.38,.38]:
        for y in [-.28,.28]:geom(bench,f'bench_leg_{x}_{y}',pos=[x,y,.35],size=[.025,.025,.35],material='room_metal')
    geom(bench,'bench_lower_shelf',pos=[0,0,.18],size=[.41,.30,.015],material='room_metal')
    rack=ET.SubElement(world,'body',name='storage_rack',pos='2.45 -1.25 0')
    for y in [-.5,.5]:
        for x in [-.16,.16]:geom(rack,f'rack_post_{x}_{y}',pos=[x,y,.65],size=[.018,.018,.65],material='room_metal')
    for z in [.10,.55,1.0]:geom(rack,f'rack_shelf_{z}',pos=[0,0,z],size=[.20,.53,.012],material='room_trim')
    # Open plastic tote, assembled from five collision walls; no solid hull
    # spanning the opening, so later placement tests can use its real interior.
    tote=ET.SubElement(world,'body',name='storage_tote',pos='1.65 1.7 .02')
    for name,pos,size in [('bottom',[0,0,.012],[.23,.17,.012]),('front',[0,-.16,.14],[.23,.01,.14]),
                          ('back',[0,.16,.14],[.23,.01,.14]),('left',[-.22,0,.14],[.01,.16,.14]),('right',[.22,0,.14],[.01,.16,.14])]:
        geom(tote,'tote_'+name,pos=pos,size=size,rgba=[.12,.29,.34,1])
    for name,pos in cfg['props'].items():prop_asset(root,name,pos)
    # A loose object on the shared floor demonstrates genuine free-body physics
    # near, but outside, the nominal recipient path and camera-to-target ray.
    loose=ET.SubElement(world,'body',name='shared_ball',pos='.55 1.75 .12')
    ET.SubElement(loose,'freejoint',name='shared_ball_free')
    geom(loose,'shared_ball','sphere',size=[.045],mass=.06,rgba=[.16,.33,.65,1],condim=6)
    for light in world.findall('light'):
        if light.get('diffuse'):light.set('diffuse',vec(np.fromstring(light.get('diffuse'),sep=' ')*cfg['light']))
    lock=json.loads((ASSETS/'manifest.json').read_text())
    return {'id':preset,'label':cfg['label'],'physical_placement':support,'source_assets':lock,'design':cfg,'prop_mass_kg':MASSES,
        'friction':[.65,.005,.0001],'prop_friction':[.65,.005,.001],'units':'metres, kilograms, seconds',
        'physical_scope':'shared gravity, contact and free-body dynamics; closed props use convex collision hulls',
        'uncalibrated':['room dimensions','prop mass and friction','surface appearance'],
        'layout_sha256':hashlib.sha256(json.dumps({'room':cfg,'placement':support},sort_keys=True).encode()).hexdigest()}


def dock_frame(child):
    target=child.find(".//body[@name='target']")
    if target is None:return
    # Existing orange target is a physical printed panel in a freestanding frame.
    for y in [-.14,.14]:
        geom(target,'frame_post_'+str(y),'capsule',fromto=[.02,y,-.18,.02,y,.22],size=[.008],rgba=[.23,.25,.27,1])
        geom(target,'frame_foot_'+str(y),pos=[.02,y,-.185],size=[.10,.025,.015],rgba=[.23,.25,.27,1])
    geom(target,'frame_top','capsule',fromto=[.02,-.14,.215,.02,.14,.215],size=[.008],rgba=[.23,.25,.27,1])
