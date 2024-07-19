from scapy.all import sniff
import logging

# Set up logging to log to both the terminal and a file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler("packet_sniffer.log"),
    logging.StreamHandler()
])

def packet_callback(packet):
    # This function will be called for each packet captured
    packet_info = f"Packet: {packet.summary()}"
    logging.info(packet_info)

def main():
    # Start sniffing packets
    print("Starting packet sniffer...")
    sniff(prn=packet_callback, store=False)

if __name__ == "__main__":
    main()
