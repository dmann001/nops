# 🚀 Quick Setup Guide - NOPS


## ⚙️ Complete Setup Instructions

### Start the Server

```bash
# Install dependencies (first time only)
pip install -r requirements.txt

# Start the server
python server.py
```

**You should see:**
```
HTTP server: http://0.0.0.0:5000/imu
UDP listener: udp://0.0.0.0:65000
UDP listener thread started
```

**Note your IP address** (e.g., `10.75.173.146`)

--

## 📊 What You'll See

### In Browser (index.html):

**Top Section - Browser Data:**
- Accelerometer, Gyroscope, Magnetometer (from browser)
- Statistics (packets sent from browser)

**Bottom Section - iPhone Data:** ⭐
- Accelerometer, Gyroscope, Magnetometer (from iPhone)
- Server Statistics (total packets from all sources)
- Connection Status (✓ Connected / Waiting...)

### In Server Logs:

```
Received packet #10 [UDP] - Accel: (0.12, -0.45, 9.81), ...
Received packet #20 [HTTP] - Accel: (0.15, -0.42, 9.80), ...
```

---



---



