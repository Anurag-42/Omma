Camera Setup and Configuration (OV64A40 on Raspberry Pi)
========================================================

1. Physical Connection
----------------------
- Directly connect the camera’s ribbon cable to the MIPI camera port on the Raspberry Pi (do NOT use the Display port).
- Orientation: The blue face of the ribbon cable should face away from the board on both the Raspberry Pi and the camera module sides.

2. Configuration File Changes
-----------------------------
a. Open the configuration file:
   sudo nano /boot/firmware/config.txt

b. Add the following lines (under the [all] section or at the end of the file):
   dtoverlay=ov64a40,link-frequency=360000000
   camera_auto_detect=0

c. Save and exit (Ctrl+X, then Y, then Enter).

d. Reboot the Raspberry Pi to apply changes:
   sudo reboot

Reason: Editing the config file and rebooting is essential to ensure your Raspberry Pi loads the correct driver and settings for the OV64A40 camera. This allows libcamera-hello and other tools to detect and use the camera reliably.

3. Verifying Camera Detection
-----------------------------
- Check that the camera is detected by running:
   libcamera-hello

  If the camera is set up correctly, this command will launch a preview window or output information about the connected camera.
