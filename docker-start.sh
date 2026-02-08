#!/bin/bash

sudo docker build -t my-whisper-app .
sudo docker run -dit -p 7860:7860 --name whisper-container -v ./outputs/:/Whisper-WebUI/outputs -v ./models/:/Whisper-WebUI/models my-whisper-app
