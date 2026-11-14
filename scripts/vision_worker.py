"""Persistent open-vocabulary detector (SAM 3 by default, YOLOE optional) for live "go to <object>" control.

Protocol (one JSON object per line): stdin {"prompt": "sock", "jpeg_b64": "..."}
stdout {"boxes": [[x1,y1,x2,y2,score], ...], "ms": 87.0} or {"error": "..."}.
Runs in its own process because the YOLOE text encoder resolves files from the cwd.
Never produces motor commands.
"""
import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

DATA = Path(os.environ.get('EXPLORER3_DATA', str(Path.home() / '.local/share/explorer3')))


def load_sam3(root):
    import torch
    from transformers import Sam3Model, Sam3Processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model = Sam3Model.from_pretrained(str(root / 'sam3'), local_files_only=True, dtype=dtype).to(device).eval()
    processor = Sam3Processor.from_pretrained(str(root / 'sam3'), local_files_only=True)

    def detect(frame_bgr, prompt, conf):
        from PIL import Image
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(device)
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            out = model(**inputs)
        r = processor.post_process_instance_segmentation(out, threshold=max(conf, .3), mask_threshold=.5,
                                                         target_sizes=inputs['original_sizes'].tolist())[0]
        return [[float(x) for x in b] + [float(sc)] for b, sc in zip(r['boxes'], r['scores'])]
    return detect, device


def load_yoloe(root):
    os.chdir(root)
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    (root / 'settings').mkdir(exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(root / 'settings')
    from ultralytics import YOLOE, settings
    settings.update({'sync': False})
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOE(str(root / 'yoloe-26s-seg.pt'))
    state = {'prompt': None}

    def detect(frame_bgr, prompt, conf):
        if prompt != state['prompt']:
            model.set_classes([prompt])
            state['prompt'] = prompt
        r = model.predict(frame_bgr, device=device, imgsz=640, conf=conf, verbose=False)[0]
        return [b + [sc] for b, sc in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist())]
    return detect, device


def main():
    method = sys.argv[1] if len(sys.argv) > 1 else 'sam3'
    root = DATA / 'cache/vision'
    detect, device = (load_sam3 if method == 'sam3' else load_yoloe)(root)
    print(json.dumps({'ready': True, 'device': device, 'method': method}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            prompt = request['prompt'].strip()[:80]
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(request['jpeg_b64']), np.uint8), cv2.IMREAD_COLOR)
            started = time.monotonic()
            boxes = detect(frame, prompt, float(request.get('conf', .3)))
            print(json.dumps({'boxes': boxes, 'ms': round((time.monotonic() - started) * 1000, 1)}), flush=True)
        except Exception as exc:  # report and keep serving
            print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}), flush=True)


if __name__ == '__main__':
    main()
