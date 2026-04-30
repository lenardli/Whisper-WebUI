#!/bin/bash

sudo docker build -t whisper-giga-app .
sudo docker run -dit -p 7860:7860 --name whisper-container -v ./outputs/:/Whisper-WebUI/outputs -v ./models/:/Whisper-WebUI/models -v ./:/Whisper-WebUI/ whisper-giga-app
