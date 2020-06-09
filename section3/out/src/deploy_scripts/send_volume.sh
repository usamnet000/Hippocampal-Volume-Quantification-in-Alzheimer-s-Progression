#!/bin/bash

# This script sends a study to the Orthanc server

# In your test data directory you will find three different studies - you may change the dir here
# to try all three out

# apt-get install dcmtk
# deploy_scripts/send_volume.sh
# python3 inference_dcm.py '../TestVolumes'
storescu 127.0.0.1 4242 -v -aec HIPPOAI +r +sd /data/TestVolumes/Study1
