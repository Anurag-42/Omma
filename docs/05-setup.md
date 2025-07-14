
Capturing Images on Raspberry Pi and Transferring to Mac
------------------------------------------------------------

Step 1: Create Bash Script to Capture Images

Create a script named `capture_images.sh` inside the `images` folder:

#!/bin/bash

for i in $(seq 1 100)
do
  libcamera-jpeg -o image${i}.jpg
  echo "Captured image${i}.jpg"
  sleep 20
done

Step 2: Run the Script

Make it executable and run:

chmod +x capture_images.sh
./capture_images.sh

This will capture 100 images using `libcamera-jpeg` every 20 seconds and save them as
`image1.jpg`, `image2.jpg`, ..., `image100.jpg` in the current folder.

Step 3: Transfer Images to Mac

From your Mac terminal, run:

scp -r omma@192.168.88.151:/home/omma/images ~/Downloads/

 This will copy all images from your Pi to your Mac's Downloads folder.



