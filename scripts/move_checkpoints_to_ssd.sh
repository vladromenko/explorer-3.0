#!/bin/bash
# Move Explorer checkpoints to the external SSD and bind-mount them back (paths stay the same).
# Run only when no training is running. Idempotent.
set -e
D=/home/jetsonnx/.local/share/explorer3
SSD=/mnt/explorer-data/explorer3/checkpoints
mountpoint -q /mnt/explorer-data || { echo "SSD not mounted"; exit 1; }
findmnt "$D/checkpoints" >/dev/null && { echo "already on SSD"; exit 0; }
mkdir -p "$SSD"
rsync -a "$D/checkpoints/" "$SSD/"
diff -rq "$D/checkpoints" "$SSD"
grep -q "$SSD $D/checkpoints" /etc/fstab || echo "$SSD $D/checkpoints none bind,nofail,x-systemd.requires-mounts-for=/mnt/explorer-data 0 0" | sudo tee -a /etc/fstab >/dev/null
mv "$D/checkpoints" "$D/checkpoints.nvme-old"; mkdir "$D/checkpoints"
sudo systemctl daemon-reload; sudo mount "$D/checkpoints"
diff -rq "$D/checkpoints.nvme-old" "$D/checkpoints" && rm -rf "$D/checkpoints.nvme-old"
echo "checkpoints on SSD"
