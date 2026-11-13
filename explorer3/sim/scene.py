"""Native MuJoCo presentation, adapted from Menagerie's Apache-2.0 scene.xml.

Materials and lights only: collision geometry and mechanics are unchanged.
"""
import xml.etree.ElementTree as ET


def laboratory(root, *, mobile=False):
    asset=root.find('asset')
    if asset is None:asset=ET.SubElement(root,'asset')
    ET.SubElement(asset,'texture',name='explorer_sky',type='skybox',builtin='gradient',
        rgb1='.28 .34 .42',rgb2='.07 .10 .15',width='512',height='3072')
    ET.SubElement(asset,'texture',name='explorer_grid',type='2d',builtin='checker',mark='edge',
        rgb1='.18 .22 .26',rgb2='.25 .29 .33',markrgb='.40 .45 .50',width='512',height='512')
    # A checker texture contains 2 cells across: 5 repeats/m = 10 cm cells.
    ET.SubElement(asset,'material',name='explorer_floor',texture='explorer_grid',texuniform='true',
        texrepeat='5 5',reflectance='0',specular='.1',shininess='.2',rgba='1 1 1 1')
    ET.SubElement(asset,'material',name='explorer_plain',rgba='.30 .35 .40 1',
        reflectance='.02',specular='.1',shininess='.2')
    visual=root.find('visual')
    if visual is None:visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'headlight',ambient='.2 .2 .2',diffuse='.35 .35 .35',specular='.05 .05 .05')
    ET.SubElement(visual,'quality',shadowsize='2048',offsamples='4')
    ET.SubElement(visual,'rgba',haze='.16 .21 .28 1')
    ET.SubElement(visual,'map',znear='.005',zfar='20',fogstart='3',fogend='8')
    world=root.find('worldbody')
    for light in world.findall('light'):world.remove(light)
    ET.SubElement(world,'light',name='studio_key',pos='.6 -.5 2',dir='-.2 .15 -1',
        type='directional',diffuse='.55 .54 .52',specular='.15 .15 .15',castshadow='true')
    ET.SubElement(world,'light',name='studio_fill',pos='-.5 .8 1.5',dir='.2 -.2 -1',
        type='directional',diffuse='.18 .22 .28',specular='.05 .05 .05',castshadow='false')
    floor=world.find("geom[@name='floor']")
    floor.set('material','explorer_floor');floor.set('rgba','1 1 1 1');floor.set('size','0 0 .05')
    # Semantic object colours, with restrained highlights instead of a flat preview.
    for name,colour in [('cube','.95 .49 .11 1'),('bin_bottom','.12 .48 .40 1')]:
        geom=world.find(f".//geom[@name='{name}']")
        if geom is not None:geom.set('rgba',colour)
    for geom in world.findall('.//geom'):
        if geom.get('name','').startswith('bin_'):geom.set('rgba','.12 .48 .40 1')
    from .variation import RECIPE
    import json
    custom=root.find('custom')
    if custom is None:custom=ET.SubElement(root,'custom')
    ET.SubElement(custom,'text',name='explorer_variation_recipe',data=json.dumps(RECIPE,sort_keys=True))
    return {'renderer':'native MuJoCo/Studio','style':'laboratory-v1','checker_cell_m':.1,
            'reference':'Menagerie robotstudio_so101/scene.xml (Apache-2.0)',
            'policy_camera':'same physical scene, no object coordinates or debug overlays'}
