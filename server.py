"""
NOPS - Building Real-time Adaptive Motion Positioning System
===============================================================

Flask-based server for real-time IMU data collection and dead-reckoning navigation.

Endpoints:
    GET  /                  - Main monitoring dashboard
    GET  /trail_tracker     - Trail visualization interface
    POST /imu               - Receive IMU sensor data (HTTP)
    GET  /trail_data        - Get current trail and position
    POST /reset_trail       - Reset dead-reckoning state
    GET  /stats             - Server statistics
    GET  /health            - Health check
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import logging
from datetime import datetime
import os
import socket
import threading
from threading import Lock
from collections import deque
import time

# Import advanced IMU dead-reckoning module
from imu_dead_reckoning import AdvancedTrailTracker

# Configure dual logging system: file (UTF-8) + console (ASCII-safe)
import sys

# File handler: UTF-8 encoding for comprehensive logging including special characters
file_handler = logging.FileHandler('bramps_server.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Console handler: ASCII-safe output for terminal compatibility
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Initialize logger with both handlers
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS for cross-origin requests

# Global statistics tracking for server metrics
stats = {
    'total_packets': 0,           # Total packets received (UDP + HTTP)
    'last_packet_time': None,     # Timestamp of most recent packet
    'first_packet_time': None,    # Timestamp of first packet (for rate calculation)
    'packets_per_second': 0,      # Moving average packet rate
    'last_data': None,            # Most recent sensor data
    'udp_packets': 0,             # UDP packet counter
    'http_packets': 0,            # HTTP POST packet counter
    'last_udp_time': None,        # Last UDP packet timestamp
    'last_http_time': None        # Last HTTP POST timestamp
}

# Thread-safe circular buffer for recent sensor data (debugging/analysis)
recent_data = []
recent_data_lock = threading.Lock()

# Initialize dead-reckoning trail tracker with configurable sample rate
# Optimal range: 50-100 Hz for balance between accuracy and computational cost
expected_hz = int(os.environ.get('EXPECTED_HZ', 100))
trail_tracker = AdvancedTrailTracker(expected_hz=expected_hz)

def log_imu_data(packet_num, source, sensor_data, accel_mag, gyro_mag, mag_mag):
    """
    Log formatted IMU sensor data with dual-output support.
    
    Args:
        packet_num: Sequential packet number
        source: Data source ('UDP' or 'HTTP')
        sensor_data: Dict containing accelerometer, gyroscope, magnetometer readings
        accel_mag: Acceleration magnitude (m/s²)
        gyro_mag: Angular velocity magnitude (rad/s)
        mag_mag: Magnetic field magnitude (μT)
    """
    # Log to file with emojis (UTF-8 encoding)
    file_handler.stream.write(f"📱 IMU Packet #{packet_num} [{source}]\n")
    file_handler.stream.write(f"   📊 Accelerometer: X={sensor_data['accel_x']:+7.3f}, Y={sensor_data['accel_y']:+7.3f}, Z={sensor_data['accel_z']:+7.3f} (|a|={accel_mag:.3f} m/s²)\n")
    file_handler.stream.write(f"   🔄 Gyroscope:     X={sensor_data['gyro_x']:+7.3f}, Y={sensor_data['gyro_y']:+7.3f}, Z={sensor_data['gyro_z']:+7.3f} (|ω|={gyro_mag:.3f} rad/s)\n")
    file_handler.stream.write(f"   🧲 Magnetometer:  X={sensor_data['mag_x']:+7.3f}, Y={sensor_data['mag_y']:+7.3f}, Z={sensor_data['mag_z']:+7.3f} (|B|={mag_mag:.3f} μT)\n")
    if 'latency_ms' in sensor_data:
        file_handler.stream.write(f"   ⏱️  Latency: {sensor_data['latency_ms']} ms\n")
    file_handler.stream.write(f"   📈 Rate: {stats['packets_per_second']:.1f} packets/sec | Total: {stats['total_packets']} packets\n")
    file_handler.stream.write("-" * 80 + "\n")
    file_handler.stream.flush()
    
    # Log to console without emojis (ASCII-safe)
    logger.info(f"IMU Packet #{packet_num} [{source}]")
    logger.info(f"   Accelerometer: X={sensor_data['accel_x']:+7.3f}, Y={sensor_data['accel_y']:+7.3f}, Z={sensor_data['accel_z']:+7.3f} (|a|={accel_mag:.3f} m/s²)")
    logger.info(f"   Gyroscope:     X={sensor_data['gyro_x']:+7.3f}, Y={sensor_data['gyro_y']:+7.3f}, Z={sensor_data['gyro_z']:+7.3f} (|w|={gyro_mag:.3f} rad/s)")
    logger.info(f"   Magnetometer:  X={sensor_data['mag_x']:+7.3f}, Y={sensor_data['mag_y']:+7.3f}, Z={sensor_data['mag_z']:+7.3f} (|B|={mag_mag:.3f} uT)")
    if 'latency_ms' in sensor_data:
        logger.info(f"   Latency: {sensor_data['latency_ms']} ms")
    logger.info(f"   Rate: {stats['packets_per_second']:.1f} packets/sec | Total: {stats['total_packets']} packets")
    logger.info("-" * 60)

def process_sensor_data(data, source='HTTP'):
    """
    Process incoming IMU sensor data and update tracking state.
    
    Handles flexible field name mapping for compatibility with various mobile apps.
    Updates dead-reckoning position, server statistics, and data buffers.
    
    Args:
        data: Dictionary containing raw sensor readings from mobile device
        source: Data source identifier ('HTTP' or 'UDP')
        
    Returns:
        Processed sensor data dictionary or None on error
    """
    try:
        # Log raw data for debugging (first packet only, or if all zeros)
        if stats['total_packets'] == 0 or (stats['total_packets'] % 50 == 0):
            logger.info(f"Raw data received [{source}]: {json.dumps(data, indent=2)[:500]}")
        
        # Flexible field name mapping - handle different naming conventions
        # Try multiple possible field names for each sensor
        def get_value(data, *possible_keys):
            """Try multiple possible key names, return first found or 0.0"""
            for key in possible_keys:
                if key in data and data[key] is not None:
                    value = data[key]
                    # Convert to float if it's a number
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return 0.0
            return 0.0
        
        # Extract sensor data with flexible field name matching
        sensor_data = {
            'accel_x': get_value(data, 'accel_x', 'accelX', 'accelerationX', 'acceleration_x', 'ax', 'a_x'),
            'accel_y': get_value(data, 'accel_y', 'accelY', 'accelerationY', 'acceleration_y', 'ay', 'a_y'),
            'accel_z': get_value(data, 'accel_z', 'accelZ', 'accelerationZ', 'acceleration_z', 'az', 'a_z'),
            'gyro_x': get_value(data, 'gyro_x', 'gyroX', 'rotationRateX', 'rotation_rate_x', 'gx', 'g_x', 'omega_x'),
            'gyro_y': get_value(data, 'gyro_y', 'gyroY', 'rotationRateY', 'rotation_rate_y', 'gy', 'g_y', 'omega_y'),
            'gyro_z': get_value(data, 'gyro_z', 'gyroZ', 'rotationRateZ', 'rotation_rate_z', 'gz', 'g_z', 'omega_z'),
            'mag_x': get_value(data, 'mag_x', 'magX', 'magneticFieldX', 'magnetic_field_x', 'mx', 'm_x', 'magnetometerX'),
            'mag_y': get_value(data, 'mag_y', 'magY', 'magneticFieldY', 'magnetic_field_y', 'my', 'm_y', 'magnetometerY'),
            'mag_z': get_value(data, 'mag_z', 'magZ', 'magneticFieldZ', 'magnetic_field_z', 'mz', 'm_z', 'magnetometerZ'),
            'timestamp': data.get('timestamp', data.get('time', int(datetime.now().timestamp() * 1000))),
            'server_received_at': datetime.now().isoformat(),
            'source': source  # Track data source
        }
        
        # Log if we got zeros (might indicate format mismatch)
        if (sensor_data['accel_x'] == 0.0 and sensor_data['accel_y'] == 0.0 and sensor_data['accel_z'] == 0.0 and
            sensor_data['gyro_x'] == 0.0 and sensor_data['gyro_y'] == 0.0 and sensor_data['gyro_z'] == 0.0 and
            sensor_data['mag_x'] == 0.0 and sensor_data['mag_y'] == 0.0 and sensor_data['mag_z'] == 0.0):
            logger.warning(f"All sensor values are zero! Raw data keys: {list(data.keys())}")
            logger.warning(f"Sample values: {json.dumps({k: v for k, v in list(data.items())[:10]})}")
        
        # Calculate latency (if client timestamp provided)
        if 'timestamp' in data:
            client_time = data['timestamp']
            server_time = int(datetime.now().timestamp() * 1000)
            latency = server_time - client_time
            sensor_data['latency_ms'] = latency
        
        # Update statistics
        stats['total_packets'] += 1
        current_time = datetime.now()
        
        if stats['first_packet_time'] is None:
            stats['first_packet_time'] = current_time
        
        stats['last_packet_time'] = current_time
        stats['last_data'] = sensor_data
        
        # Track source-specific stats
        if source == 'UDP':
            stats['udp_packets'] += 1
            stats['last_udp_time'] = current_time
        elif source == 'HTTP':
            stats['http_packets'] += 1
            stats['last_http_time'] = current_time
        
        # Calculate packets per second (simple moving average)
        if stats['first_packet_time']:
            elapsed = (current_time - stats['first_packet_time']).total_seconds()
            if elapsed > 0:
                stats['packets_per_second'] = stats['total_packets'] / elapsed
        
        # Convert timestamp to seconds for calculations
        timestamp_sec = sensor_data.get('timestamp') / 1000.0

        # Update the trail tracker with new data, including magnetometer for heading correction
        trail_tracker.update(
            accel_x=sensor_data.get('accel_x', 0),
            accel_y=sensor_data.get('accel_y', 0),
            accel_z=sensor_data.get('accel_z', 0),
            gyro_z=sensor_data.get('gyro_z', 0),
            timestamp=timestamp_sec,
            mag_x=sensor_data.get('mag_x'),
            mag_y=sensor_data.get('mag_y'),
            mag_z=sensor_data.get('mag_z')
        )

        # Store recent data (keep last 100 packets) - Thread safe
        with recent_data_lock:
            recent_data.append(sensor_data)
            if len(recent_data) > 100:
                recent_data.pop(0)
            current_length = len(recent_data)
        
        # Debug logging for data storage
        if stats['total_packets'] % 10 == 0:
            logger.info(f"Data stored - recent_data length: {current_length}, total_packets: {stats['total_packets']}")
        
        # Enhanced IMU logging - every 5th packet for better monitoring
        if stats['total_packets'] % 5 == 0:
            # Calculate magnitudes for better understanding
            accel_magnitude = (sensor_data['accel_x']**2 + sensor_data['accel_y']**2 + sensor_data['accel_z']**2)**0.5
            gyro_magnitude = (sensor_data['gyro_x']**2 + sensor_data['gyro_y']**2 + sensor_data['gyro_z']**2)**0.5
            mag_magnitude = (sensor_data['mag_x']**2 + sensor_data['mag_y']**2 + sensor_data['mag_z']**2)**0.5
            
            # Use safe logging function
            log_imu_data(stats['total_packets'], source, sensor_data, accel_magnitude, gyro_magnitude, mag_magnitude)
        
        return sensor_data
    except Exception as e:
        logger.error(f"Error processing sensor data: {e}", exc_info=True)
        return None

def udp_listener(udp_port=8888):
    """
    Dedicated UDP listener thread for high-frequency sensor streaming.
    
    Runs in background daemon thread to receive JSON-formatted IMU data
    from mobile applications. UDP is preferred for real-time streaming
    due to lower latency compared to HTTP.
    
    Args:
        udp_port: UDP port number (default: 8888)
    """
    try:
        # Create UDP socket
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_socket.bind(('0.0.0.0', udp_port))
        
        logger.info(f"UDP listener started on port {udp_port}")
        logger.info(f"Ready to receive UDP packets at {udp_port}")
        
        while True:
            try:
                # Receive UDP packet (max 4096 bytes)
                data, addr = udp_socket.recvfrom(4096)
                
                # Decode and parse JSON
                try:
                    data_str = data.decode('utf-8')
                    sensor_json = json.loads(data_str)
                    
                    # Log first UDP packet to see format
                    if stats['udp_packets'] == 0:
                        logger.info(f"First UDP packet received from {addr}")
                        logger.info(f"Raw JSON keys: {list(sensor_json.keys())}")
                        logger.info(f"Sample values: {json.dumps({k: v for k, v in list(sensor_json.items())[:15]}, indent=2)}")
                    
                    # Process the sensor data
                    process_sensor_data(sensor_json, source='UDP')
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from {addr}: {e}")
                    logger.warning(f"Received data (first 200 chars): {data_str[:200]}")
                except UnicodeDecodeError as e:
                    logger.warning(f"Invalid encoding from {addr}: {e}")
                except Exception as e:
                    logger.error(f"Error processing UDP packet from {addr}: {e}")
                    
            except socket.error as e:
                logger.error(f"UDP socket error: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error in UDP listener: {e}", exc_info=True)
                
    except Exception as e:
        logger.error(f"Failed to start UDP listener: {e}", exc_info=True)
    finally:
        if 'udp_socket' in locals():
            udp_socket.close()
        logger.info("UDP listener stopped")

@app.route('/')
def index():
    """Root endpoint - serve HTML page or return server status"""
    # Try to serve the iPhone monitor page first, then fall back to index.html
    try:
        return send_from_directory('.', 'iphone_monitor.html')
    except:
        try:
            return send_from_directory('.', 'index.html')
        except Exception as e:
            # If HTML file not found, return JSON status
            logger.warning(f"Could not serve HTML files: {e}")
            return jsonify({
                'status': 'running',
                'service': 'BRAMPS IMU Data Collection Server',
                'endpoints': {
                    '/': 'GET - iPhone IMU Monitor (HTML interface)',
                    '/imu': 'POST - Receive IMU sensor data',
                    '/stats': 'GET - Get server statistics',
                    '/health': 'GET - Health check',
                    '/debug': 'GET - Debug information'
                },
                'note': 'If you see this JSON, no HTML files found in server directory'
            })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })

# --- Trail Endpoints ---
@app.route('/trail_data', methods=['GET'])
def get_trail_data():
    """Returns the current dead-reckoning trail and stats."""
    return jsonify(trail_tracker.get_trail_data())

@app.route('/reset_trail', methods=['POST'])
def reset_trail():
    """Resets the dead-reckoning trail."""
    trail_tracker.reset()
    return jsonify({'status': 'ok', 'message': 'Trail reset successfully.'})

@app.route('/set_expected_hz', methods=['POST'])
def set_expected_hz():
    """Change the expected data rate (Hz) dynamically.
    
    Recommended: 50-100 Hz for best accuracy
    """
    global trail_tracker  # Declare global at the start of the function
    
    try:
        data = request.get_json()
        if not data or 'hz' not in data:
            return jsonify({'error': 'Missing "hz" parameter'}), 400
        
        new_hz = int(data['hz'])
        
        if new_hz < 10 or new_hz > 200:
            return jsonify({
                'error': f'Invalid Hz value. Must be between 10 and 200. Recommended: 50-100 Hz'
            }), 400
        
        # Create new tracker with new Hz (preserve current position if desired)
        current_pos = trail_tracker.position.copy()
        current_heading = trail_tracker.heading
        
        trail_tracker = AdvancedTrailTracker(expected_hz=new_hz)
        trail_tracker.position = current_pos
        trail_tracker.heading = current_heading
        
        logger.info(f"Expected Hz changed to: {new_hz}")
        return jsonify({
            'status': 'ok',
            'message': f'Expected data rate changed to: {new_hz} Hz',
            'expected_hz': new_hz
        })
    except ValueError:
        return jsonify({'error': 'Invalid Hz value. Must be a number.'}), 400
    except Exception as e:
        logger.error(f"Error setting expected Hz: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/get_expected_hz', methods=['GET'])
def get_expected_hz():
    """Get the current expected data rate (Hz)."""
    return jsonify({
        'expected_hz': trail_tracker.expected_hz,
        'velocity_damping': trail_tracker.velocity_damping,
        'note': 'Recommended: 50-100 Hz for best accuracy'
    })

@app.route('/diagnostics', methods=['GET'])
def diagnostics():
    """Get diagnostic information about the current state and recent sensor data."""
    state = trail_tracker.get_state()
    
    # Get recent sensor data
    recent_sensor = None
    with recent_data_lock:
        if recent_data:
            recent_sensor = recent_data[-1]
    
    return jsonify({
        'current_state': state,
        'recent_sensor_data': recent_sensor,
        'parameters': {
            'accel_move_threshold': trail_tracker.accel_move_threshold,
            'accel_deadzone': trail_tracker.accel_deadzone,
            'velocity_threshold': trail_tracker.velocity_threshold,
            'stationary_threshold': trail_tracker.stationary_threshold,
            'expected_hz': trail_tracker.expected_hz
        },
        'stats': {
            'total_packets': stats['total_packets'],
            'packets_per_second': stats['packets_per_second'],
            'last_packet_time': stats['last_packet_time'].isoformat() if stats['last_packet_time'] else None
        }
    })


@app.route('/imu', methods=['POST', 'OPTIONS'])
def receive_imu_data():
    """
    Receive IMU and magnetometer data from client
    
    Expected JSON format:
    {
        "accel_x": float,
        "accel_y": float,
        "accel_z": float,
        "gyro_x": float,
        "gyro_y": float,
        "gyro_z": float,
        "mag_x": float,
        "mag_y": float,
        "mag_z": float,
        "timestamp": int (milliseconds since epoch)
    }
    """
    if request.method == 'OPTIONS':
        # Handle preflight request
        return '', 200
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            logger.warning("Received empty request body")
            return jsonify({'error': 'No data received'}), 400
        
        # Validate required fields
        required_fields = ['timestamp']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            logger.warning(f"Missing required fields: {missing_fields}")
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'received_fields': list(data.keys())
            }), 400
        
        # Process sensor data using shared function
        sensor_data = process_sensor_data(data)
        
        if sensor_data is None:
            return jsonify({'error': 'Failed to process sensor data'}), 500
        
        # Return success response
        return jsonify({
            'status': 'success',
            'packet_number': stats['total_packets'],
            'received_at': sensor_data['server_received_at']
        }), 200
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return jsonify({'error': 'Invalid JSON format', 'details': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get server statistics"""
    current_time = datetime.now()
    
    # Check if UDP data was received recently (within last 5 seconds)
    udp_recent = False
    if stats['last_udp_time']:
        time_since_udp = (current_time - stats['last_udp_time']).total_seconds()
        udp_recent = time_since_udp < 5.0
    
    return jsonify({
        'total_packets': stats['total_packets'],
        'packets_per_second': round(stats['packets_per_second'], 2),
        'first_packet_time': stats['first_packet_time'].isoformat() if stats['first_packet_time'] else None,
        'last_packet_time': stats['last_packet_time'].isoformat() if stats['last_packet_time'] else None,
        'last_data': stats['last_data'],
        'recent_packets_count': len(recent_data),
        'udp_packets': stats['udp_packets'],
        'http_packets': stats['http_packets'],
        'last_udp_time': stats['last_udp_time'].isoformat() if stats['last_udp_time'] else None,
        'last_http_time': stats['last_http_time'].isoformat() if stats['last_http_time'] else None,
        'udp_recent': udp_recent,  # True if UDP data received in last 5 seconds
        'time_since_last_udp': (current_time - stats['last_udp_time']).total_seconds() if stats['last_udp_time'] else None
    })

@app.route('/recent', methods=['GET'])
def get_recent_data():
    """Get recent sensor data packets (last 100)"""
    limit = request.args.get('limit', 100, type=int)
    
    # Thread-safe access to recent_data
    with recent_data_lock:
        data_length = len(recent_data)
        data_to_return = recent_data[-limit:] if recent_data else []
    
    # Debug logging with more details
    logger.info(f"Recent data request - limit: {limit}, available packets: {data_length}")
    logger.info(f"Total packets received: {stats['total_packets']}, UDP packets: {stats['udp_packets']}")
    
    if data_length > 0:
        logger.info(f"Latest packet sample: {json.dumps(data_to_return[-1], indent=2)[:300]}")
    else:
        logger.warning("No recent data available despite receiving packets!")
        logger.info(f"Stats: {json.dumps(stats, default=str, indent=2)}")
    
    return jsonify({
        'count': min(data_length, limit),
        'data': data_to_return
    })

@app.route('/trail_tracker')
def trail_tracker_page():
    """Serve the trail tracker page."""
    return send_from_directory('.', 'trail_tracker.html')

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check server state"""
    return jsonify({
        'stats': stats,
        'recent_data_count': len(recent_data),
        'sample_recent_data': recent_data[-1] if recent_data else None,
        'all_recent_data': recent_data[-5:] if len(recent_data) >= 5 else recent_data
    })

if __name__ == '__main__':
    # Get ports from environment variables or use defaults
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')  # Listen on all interfaces
    udp_port = int(os.environ.get('UDP_PORT', 8888))  # UDP port for mobile apps
    
    logger.info(f"Starting BRAMPS IMU Data Collection Server")
    logger.info(f"HTTP server: http://{host}:{port}/imu")
    logger.info(f"UDP listener: udp://{host}:{udp_port}")
    
    # Start UDP listener in a separate thread
    udp_thread = threading.Thread(target=udp_listener, args=(udp_port,), daemon=True)
    udp_thread.start()
    logger.info("UDP listener thread started")
    
    # Run Flask app (this blocks)
    # Note: debug=False to avoid conflicts with UDP listener thread
    app.run(host=host, port=port, debug=False)

