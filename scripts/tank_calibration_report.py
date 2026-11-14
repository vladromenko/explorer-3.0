#!/usr/bin/env python3
"""Read installed calibration hashes/values; never invoke calibration or motion."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
import hashlib,json
from pathlib import Path
paths=[
'/home/ubuntu/ros2_ws/src/calibration/config/imu_calib.yaml',
'/home/ubuntu/ros2_ws/install/calibration/share/calibration/config/imu_calib.yaml',
'/home/ubuntu/ros2_ws/src/driver/controller/config/calibrate_params.yaml',
'/home/ubuntu/ros2_ws/install/controller/share/controller/config/calibrate_params.yaml',
'/home/ubuntu/explorer3/native/config/calibrate_params.yaml']
files=[]
for name in paths:
 p=Path(name)
 files.append({'path':name,'exists':p.is_file(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
               'yaml':p.read_text() if p.is_file() else None})
source=Path('/home/ubuntu/ros2_ws/src/driver/controller/controller/odom_publisher_node.py')
lines=source.read_text().splitlines() if source.exists() else []
print(json.dumps({'files':files,'odometry_source':str(source),'odometry_sha256':hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None,
 'odometry_evidence':[line for line in lines if any(word in line for word in ('odom_raw','linear_correction_factor_tank','linear_x * dt','angular_z * dt','encoder'))],
 'read_only':True,'calibration_started':False}))
'''


def capture():
    command=shlex.join(['docker','exec','-u','ubuntu','MentorPi','python3','-c',REMOTE])
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','tank',command],capture_output=True,text=True,timeout=20,check=True)
    return json.loads(result.stdout)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=capture();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'read_only':True,'files':[{k:f[k] for k in ('path','exists','sha256')} for f in report['files']],'output':str(args.output)},indent=2))
