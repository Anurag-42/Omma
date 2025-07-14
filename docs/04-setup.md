# Offline Installation of libcamera on Raspberry Pi (Bookworm)

Raspberry Pi running Raspberry Pi OS Bookworm with no internet access. Docker on Mac is used to simulate and fetch packages. Camera used: OwlSight - 64MP OV64A40 Autofocus. The goal is to update/install libcamera and all dependencies offline.

Initial issues on Pi:

sudo apt update
sudo apt install libcamera-apps libcamera0 libcamera-dev -y

This resulted in broken dependencies (libcamera-ipa, libcamera0), showing held packages.

To resolve this, use Docker on Mac to simulate Bookworm OS, download all required .deb packages (including dependencies), then SCP them to the Pi and install offline.

Docker setup on Mac (open Docker app first):

docker run -it --rm debian:bookworm bash

Inside container:

apt update
apt install -y wget curl gnupg nano

curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key \
  | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.com/debian bookworm main" >> /etc/apt/sources.list

apt update
apt download libcamera-apps libcamera0 libcamera-ipa

mkdir /debs && cd /debs
apt-get install --download-only --reinstall --print-uris libcamera-apps \
  | grep .deb | cut -d"'" -f2 | xargs wget


Bundle packages for transfer:

mkdir libcamera_offline
mv *.deb libcamera_offline/
tar -czvf libcamera_offline.tar.gz libcamera_offline


On Mac terminal, check container ID:

docker ps
docker cp <container_id>:/libcamera_offline.tar.gz .

Transfer to Raspberry Pi (replace with actual IP; ensure LAN connection):

scp libcamera_offline.tar.gz omma@192.168.88.151:/home/omma/

On Raspberry Pi:

cd ~
tar -xzvf libcamera_offline.tar.gz
cd libcamera_offline

sudo apt purge libcamera-apps libcamera0 libcamera-ipa -y
sudo apt autoremove -y

sudo dpkg -i *.deb
sudo apt --fix-broken install


If errors like libcamera-ipa depends on libpisp1 or similar show up, go back to Docker and download those additional .deb files using the same method.

