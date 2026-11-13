"""Offline upstream YOLOE / SAM3 image analysis. Never produces motor commands."""
import argparse
import hashlib
import json
import os
import time
from importlib.metadata import version as package_version
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .paths import DATA

SAM_REV='3c879f39826c281e95690f02c7821c4de09afae7'


def analyze(image,output,method,prompt,device='cpu'):
    image=Path(image).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    rgb=Image.open(image).convert('RGB');width,height=rgb.size
    if width*height>4096*4096:raise ValueError('Image too large')
    if not 1<=len(prompt)<=200:raise ValueError('Нужно короткое название объекта')
    import torch
    torch.set_num_threads(2)
    started=time.monotonic();root=DATA/'cache/vision';masks=[];boxes=[];scores=[]
    from .vision_assets import verify
    assets=verify(root,method)
    if method=='yoloe':
        # The upstream text encoder resolves its local file from cwd. A worker
        # process owns this directory; never change the UI process cwd.
        if not (root/'mobileclip2_b.ts').is_file():raise ValueError('Text encoder not cached')
        os.chdir(root);os.environ['YOLO_AUTOINSTALL']='false'
        settings=root/'settings';settings.mkdir(exist_ok=True);os.environ['YOLO_CONFIG_DIR']=str(settings)
        from ultralytics import YOLOE,settings as preferences
        preferences.update({'sync':False})
        checkpoint=root/'yoloe-26s-seg.pt'
        if not checkpoint.exists():raise ValueError('YOLOE weights not cached')
        model=YOLOE(str(checkpoint));model.set_classes([prompt])
        loaded=time.monotonic()
        result=model.predict(rgb,device=device,imgsz=640,conf=.25,retina_masks=True,verbose=False)[0]
        boxes=result.boxes.xyxy.cpu().numpy();scores=result.boxes.conf.cpu().numpy()
        if result.masks is not None:
            masks=[np.asarray(Image.fromarray((m.cpu().numpy()*255).astype(np.uint8)).resize((width,height),Image.Resampling.NEAREST))>127 for m in result.masks.data]
        version={'package':'ultralytics','version':package_version('ultralytics'),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    elif method=='sam3':
        from transformers import Sam3Model,Sam3Processor
        checkpoint=root/'sam3'
        if not (checkpoint/'model.safetensors').exists():raise ValueError('SAM3 weights not cached')
        dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32
        model=Sam3Model.from_pretrained(str(checkpoint),local_files_only=True,dtype=dtype).to(device).eval()
        processor=Sam3Processor.from_pretrained(str(checkpoint),local_files_only=True)
        loaded=time.monotonic()
        inputs=processor(images=rgb,text=prompt,return_tensors='pt').to(device)
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
            result=model(**inputs)
        result=processor.post_process_instance_segmentation(result,threshold=.5,mask_threshold=.5,target_sizes=inputs['original_sizes'].tolist())[0]
        boxes=result['boxes'].float().cpu().numpy();scores=result['scores'].float().cpu().numpy()
        masks=[m.cpu().numpy().astype(bool) for m in result['masks']]
        version={'package':'transformers','version':package_version('transformers'),'repository':'facebook/sam3','revision':SAM_REV}
    else:raise ValueError('Unknown vision model')
    if device.startswith('cuda'):torch.cuda.synchronize()
    finished=time.monotonic()
    boxes=np.asarray(boxes,dtype=float).reshape(-1,4);scores=np.asarray(scores,dtype=float)
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():raise ValueError('Non-finite perception result')
    overlay=np.array(rgb).copy();colour=np.array([45,212,191])
    for mask in masks:overlay[mask]=(.6*overlay[mask]+.4*colour).astype(np.uint8)
    rendered=Image.fromarray(overlay);draw=ImageDraw.Draw(rendered)
    for box,score in zip(boxes,scores):
        draw.rectangle(box.tolist(),outline=(45,212,191),width=2);draw.text((float(box[0]),max(0.,float(box[1])-14)),f'{prompt}: {score:.2f}',fill=(255,255,255))
    rendered.save(output/'overlay.png')
    np.savez_compressed(output/'masks.npz',masks=np.asarray(masks,dtype=bool).reshape(-1,height,width))
    report={'method':method,'model':version,'input_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
        'assets':assets,'prompt':prompt,'device':device,'image_size':[width,height],'boxes_xyxy':boxes.tolist(),'scores':scores.tolist(),
        'mask_pixels':[int(m.sum()) for m in masks],'load_seconds':loaded-started,'inference_seconds':finished-loaded,
        'training_started':False,'motor_commands':False,'ground_truth':False,'skill_verified':False,
        'limitation':'Model proposals require review; absent detections are retained, not task failures',
        'overlay':str(output/'overlay.png'),'masks':str(output/'masks.npz')}
    (output/'result.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--method',choices=['yoloe','sam3'],required=True);p.add_argument('--prompt',required=True);p.add_argument('--device',default='cpu')
    args=p.parse_args();result=analyze(**vars(args));print(json.dumps({k:v for k,v in result.items() if k not in ['model','boxes_xyxy']}))
