from scapy.all import get_if_list

# List all available network interfaces
interfaces = get_if_list()
print("Available network interfaces:")
for iface in interfaces:
    print(iface)
