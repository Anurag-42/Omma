# 02 - Enabling SSH and Setting Static IP on Jetson Nano

Purpose:  
Enable remote access to the Jetson Nano over the local network using SSH, and assign a static IP address for consistent connectivity.

---

## Steps

Note: LAN looked like this: the router and jetson were connected via an Ethernet Cable and my mac connected wirelessly to the LAN

### 1. Enable SSH on Jetson Nano

- Connect a monitor, keyboard, and mouse directly to the Jetson Nano.
- Open a terminal on the Jetson Nano and run:

sudo systemctl enable ssh   # Ensures SSH starts on boot

sudo systemctl start ssh    # Starts SSH service immediately

sudo systemctl status ssh   # Verify that SSH is active (look for "active (running)")

### 2. Identify Network Details

On your computer (which was mac for me), run:
ipconfig

### 3. Set Static IP on Jetson Nano (Ethernet Connection)

On Jetson's terminal, run: nmcli connection show

Identify the active Ethernet connection name (commonly "Wired connection 1").

Assign a static IP:

sudo nmcli connection modify "Wired connection 1" ipv4.method manual ipv4.addresses 192.168.88.150/24 ipv4.gateway 192.168.1.1 ipv4.dns "8.8.8.8 8.8.4.4"

sudo nmcli connection up "Wired connection 1"

### 4. SSH Jetson from Mac

ssh omma@192.168.88.150
pw: jamesomma

Find your Mac’s IP address to identify the LAN subnet. Example: My Mac had the IP 192.168.88.1, so I chose 192.168.88.150 for the Jetson’s static IP.

