# Experimental 3D WebRTC transport

The `feature/webrtc-3d` branch keeps Socket.IO for viewer operations, progress,
metadata, errors, and settled lossless stills. Interactive 3D RGB previews move
to a WebRTC video track.
The transport is selected once when the server starts; it is not a viewer UI
preference and cannot be hot-switched.

Copy `.env.example` to `.env` and select one transport:

```bash
DICOMVISION_3D_TRANSPORT=webrtc
DICOMVISION_WEBRTC_VIDEO_CODEC=vp8
DICOMVISION_WEBRTC_VIDEO_BITRATE_BPS=4000000
DICOMVISION_WEBRTC_VIDEO_FPS=30
DICOMVISION_WEBRTC_INITIAL_BURST_FRAMES=2
DICOMVISION_3D_FINAL_WEBP_METHOD=auto
```

`webp` uses the stable Socket.IO image path. In `webrtc` mode, continuous 3D
rotate, pan, zoom, and other preview frames bypass WebP encoding and use the
latest rendered video frame only. When an operation settles, the server sends a
lossless WebP final still over Socket.IO and the client layers it above the video.
This preserves low interaction latency without leaving the viewport on a
rate-controlled VP8 frame. WebRTC negotiation failure still falls back to WebP
so an unsupported browser does not show a blank viewport.

Final renders use an adaptive settle delay. The server only waits long enough to
complete a 35 ms pan/zoom or 50 ms rotate preview-to-final spacing, with a small
minimum delay for cancellation when another gesture starts. This avoids the old
fixed 60/100 ms wait when the last preview is already visible.

With `DICOMVISION_3D_FINAL_WEBP_METHOD=auto`, the first real final frame is sampled
at up to 256 px and lossless WebP methods 0, 1, and 2 are compared once. The
process caches the fastest method, preferring a smaller payload when timings are
within 10%. Set the value to an integer from 0 through 6 to bypass calibration.

The first rendered image is repeated briefly to initialize the decoder. Later
renders emit one latest-state frame only, reducing post-interaction playout lag.
The default VP8 bitrate is raised from aiortc's 500 kbps to 4 Mbps because medical
volume rendering contains substantially more fine texture than a webcam stream.
The default encoder ceiling is 30 fps, matching the measured VTK cadence so the
rate controller spends more of that bitrate on each rendered frame. RTP timestamps
still follow actual render arrival time rather than pretending frames are evenly
spaced.

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

The WebRTC report separates interactive `render_to_frame_ms` from the settled
`final_lossless_webp` latency and payload size.

Restart the server after changing any transport or codec value. Do not remove
WebP until WebRTC has been verified on LAN, public cloud with TURN, mobile Safari,
Chromium, and the packaged Electron clients.
