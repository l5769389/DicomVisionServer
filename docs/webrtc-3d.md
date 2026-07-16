# Experimental 3D WebRTC transport

The `feature/webrtc-3d` branch keeps Socket.IO for viewer operations, progress,
metadata, and errors. Only rendered 3D RGB frames move to a WebRTC video track.
WebP remains the default and automatic fallback while the two paths are measured.

For local/LAN testing, host ICE candidates are normally sufficient. Public cloud
deployments should configure STUN and TURN using a JSON array:

```bash
export DICOMVISION_WEBRTC_ICE_SERVERS='[
  {"urls":"stun:stun.example.com:3478"},
  {
    "urls":["turn:turn.example.com:3478?transport=udp","turns:turn.example.com:5349?transport=tcp"],
    "username":"dicomvision",
    "credential":"replace-me"
  }
]'
```

Run the same benchmark for both transports before changing the default:

```bash
uv run python scripts/benchmark_3d_transport.py \
  --server http://127.0.0.1:8100 \
  --folder /path/to/dicom \
  --transport webp

uv run python scripts/benchmark_3d_transport.py \
  --server http://127.0.0.1:8100 \
  --folder /path/to/dicom \
  --transport webrtc
```

Do not remove WebP until WebRTC has been verified on LAN, public cloud with TURN,
mobile Safari, Chromium, and the packaged Electron clients.
