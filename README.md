# Omma Project Documentation

This repo gives a comprehesive overview of how the bug detection hardware setup and algorithmic design was done for omma

## Documentation

All Setup and process documentation is in "docs" folder:

1. [01- Flashing RPI SD Card] (docs/01-setup.md)
2. [02 - Enabling SSH in RPI and Static IP setup) (docs/02-setup.md)
3. [03 - Camera Setup and Detection] (docs/03-setup.md)
4. [04 - libcamera dependencies installation via docker + scp] (docs/04-setup.md)
5. [05- image collection and sending it to my laptop] (docs/05-setup.md
6. customized_rcnn.ipynb -> our trained model
7. Deployment.py -> code to deploy the model in farms (assuming same setup that we had in the Omma office; box with camera + headless or headed RPI monitor
8. .gitattributes -> the pointer to the model weights for a file called best.pth. To access this, the deployer must clone this repo with git lfs installed
---
