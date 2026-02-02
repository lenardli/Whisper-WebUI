#!/bin/bash

sudo docker build -t my-whisper-app .
sudo docker run -d -p 7860:7860 --name whisper-container my-whisper-app
