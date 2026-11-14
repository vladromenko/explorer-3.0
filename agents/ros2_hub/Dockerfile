FROM ros:jazzy-ros-base@sha256:00b8d47a6e6384a2834e3d01fa87ae767876543baa574f2d64a75c04ea33c698
RUN apt-get update && apt-get install -y --no-install-recommends python3-httpx python3-opencv libnss-mdns && rm -rf /var/lib/apt/lists/*
# Resolve the existing .local robot names through the host Avahi socket.
# No Avahi daemon or ROS discovery is exposed on the LAN by this container.
RUN sed -i 's/^hosts:.*/hosts: files mdns4_minimal [NOTFOUND=return] dns/' /etc/nsswitch.conf
ENV PYTHONPATH=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp/explorer3 ROS_LOG_DIR=/tmp/roslog
ENV ROS_DOMAIN_ID=73 ROS_LOCALHOST_ONLY=1 EXPLORER3_DATA=/data
WORKDIR /workspace
CMD ["python3", "-m", "explorer3.ros2_hub"]
