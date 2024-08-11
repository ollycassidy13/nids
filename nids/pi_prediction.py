import torch
import pandas as pd
from nids.logging import log_prediction
from nids.flow import Flow
from sklearn.preprocessing import StandardScaler
from scapy.all import IP, TCP, UDP, IPv6
import pickle
import RPi.GPIO as GPIO
import time
import threading
from queue import Queue

# Load the metadata including class mapping and feature names
with open('nids/model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    LABEL_TO_TRAFFIC_TYPE = metadata['class_mapping']
    FEATURE_NAMES = metadata['feature_names']

flows = {}

def preprocess_packet(packet, scaler):
    if IP in packet:
        flow_key = (packet[IP].src, packet[IP].dst, packet[IP].sport, packet[IP].dport)
    elif IPv6 in packet:
        flow_key = (packet[IPv6].src, packet[IPv6].dst, packet[IPv6].sport, packet[IPv6].dport)
    else:
        return None, None, None, None

    if flow_key not in flows:
        flows[flow_key] = Flow(packet)
    else:
        flows[flow_key].update(packet)
    
    flow = flows[flow_key]
    features = flow.get_features()

    # Create DataFrame with consistent feature names
    df = pd.DataFrame([features])
    df = pd.get_dummies(df)
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)  # Ensure consistent feature order

    scaled_data = scaler.transform(df)
    src_ip = flow.src_ip
    dst_ip = flow.dst_ip
    return torch.tensor(scaled_data, dtype=torch.float32), features, src_ip, dst_ip

# Define the GPIO pin numbers for the LEDs
GREEN_LED_PIN = 27
RED_LED_PIN = 22
ORANGE_LED_PIN = 23

# Setup GPIO mode and pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(ORANGE_LED_PIN, GPIO.OUT)

# Create a queue for red and orange LED events
led_queue = Queue()

# Lock for mutual exclusivity
led_lock = threading.Lock()

def control_led(pin, duration):
    with led_lock:
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(pin, GPIO.LOW)

def led_worker():
    while True:
        pin, duration = led_queue.get()
        control_led(pin, duration)
        led_queue.task_done()

# Start the LED worker thread
led_thread = threading.Thread(target=led_worker, daemon=True)
led_thread.start()

def run_prediction(packet, model, scaler):
    model.eval()
    try:
        data_tensor, original_data, src_ip, dst_ip = preprocess_packet(packet, scaler)
        if data_tensor is None:
            print(f"Packet ignored: {packet.summary()}")
            return
        
        # Make prediction
        with torch.no_grad():
            output = model(data_tensor)
            _, prediction = torch.max(output, 1)
            traffic_type = LABEL_TO_TRAFFIC_TYPE.get(prediction.item(), "Unknown")
        
        # Log detailed information
        log_prediction(packet, prediction, original_data, traffic_type, src_ip, dst_ip)
        print(f'Prediction: {prediction.item()} ({traffic_type}), Source IP: {src_ip}, Destination IP: {dst_ip}')
        
        # Control LEDs based on prediction
        if traffic_type == "Unknown":
            led_queue.put((ORANGE_LED_PIN, 2))
        elif prediction.item() != 0:
            led_queue.put((RED_LED_PIN, 2))
        else:
            threading.Thread(target=control_led, args=(GREEN_LED_PIN, 2)).start()

    except Exception as e:
        print(f"Error during prediction: {e}")

    finally:
        # Ensure all LEDs are turned off after processing
        GPIO.output(GREEN_LED_PIN, GPIO.LOW)
        GPIO.output(RED_LED_PIN, GPIO.LOW)
        GPIO.output(ORANGE_LED_PIN, GPIO.LOW)

# Clean up GPIO on exit
def cleanup_gpio():
    GPIO.cleanup()
