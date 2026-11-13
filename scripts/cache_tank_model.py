"""Cache a pinned public mirror of MentorPi T1 CAD; no redistribution in Git.

The manufacturer's open T1 branch omits these meshes. This mirror's chassis
transform/mesh path matches the official T1 mapping lesson; installed-robot
identity and the incomplete license declaration remain explicitly unverified.
"""
import hashlib,json
from pathlib import Path
import requests
from explorer3.paths import DATA
REPO='gongyizhi0425/Shipcleaning-Robot'
REV='e63726dcd51620fd1778672a1296c6ca3d7c29af'
PREFIX='robot_code/src/simulations/mentorpi_description/'
FILES=['package.xml','urdf/tank.xacro','urdf/depth_cam.xacro','urdf/inertial_matrix.xacro',
       'meshes/tank/base_chaais.STL','meshes/tank/body_board.STL','meshes/tank/cam_Link.STL',
       'meshes/tank/cam_link_asc.STL','meshes/tank/depth_cam.STL','meshes/tank/lidar_Link.STL']

def cache():
    root=DATA/'cache/source/mentorpi-t1-cad';root.mkdir(parents=True,exist_ok=True);records=[]
    for name in FILES:
        url=f'https://raw.githubusercontent.com/{REPO}/{REV}/{PREFIX}{name}'
        response=requests.get(url,timeout=45);response.raise_for_status();content=response.content;lfs=None
        if content.startswith(b'version https://git-lfs.github.com/spec/v1'):
            fields=dict(line.split(' ',1) for line in content.decode().strip().splitlines());lfs=fields['oid'].removeprefix('sha256:')
            url=f'https://media.githubusercontent.com/media/{REPO}/{REV}/{PREFIX}{name}'
            response=requests.get(url,timeout=90);response.raise_for_status();content=response.content
            if len(content)!=int(fields['size']) or hashlib.sha256(content).hexdigest()!=lfs:raise ValueError('LFS checksum mismatch')
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        records.append({'path':name,'url':url,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),'lfs_sha256':lfs})
        print(name,len(content),flush=True)
    manifest={'source_repository':REPO,'revision':REV,'files':records,
        'official_crosscheck':'https://wiki.hiwonder.com/projects/MentorPi-T1/en/latest/docs/6.mapping_courses.html#brief-analysis-of-robot-s-main-body-model',
        'provenance':'public third-party mirror of manufacturer package, not an official Hiwonder release',
        'license':'package.xml says TODO; assets cached locally, not redistributed',
        'installed_robot_match':'unverified until stage C read-only copy'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));return root
if __name__=='__main__':cache()
