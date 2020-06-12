#!/bin/bash

# This script sends a study to the Orthanc server

# In your test data directory you will find three different studies - you may change the dir here
# to try all three out
#cp -r /data/TestVolumes /home/workspace/
# chmod -R 777 ./
# apt-get install dcmtk
# deploy_scripts/send_volume.sh
# python3 inference_dcm.py '../TestVolumes'
storescu 127.0.0.1 4242 -v -aec HIPPOAI +r +sd /home/workspace/out/report.dcm #/data/TestVolumes/Study1
