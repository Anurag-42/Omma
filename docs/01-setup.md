#01 -> Flashing the Jetson Nano SD Card

##
Purpose:
To flash the JetPack OS image onto the SD Card for the Jetson Nano.

## Steps
- Download the JetPack image from NVIDIA.
- Identify the SD Card device manually using diskutil list (mac/Linux). So, at first before inserting the SD card reader, see all the devices and then once again to see which one has been added. If prompted with eject, ignore or initialize, just ignore.
- Unmount the disk associated with SD Card reader so that noone is using it while u flash it by doing: diskutil unmountDisk /dev/disk5 (/dev/disk5 is what I identified as SD Card reader)
- Flash using the command: sudo dd if=jetson-image.img of=/dev/diskX bs=1m status=progress
- Then Eject the SD Card safely


## References:
- [NVIDIA Jetson Nano Getting Started Guide](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
