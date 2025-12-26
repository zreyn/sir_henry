#!/bin/bash

sudo docker compose down voice-agent
sudo docker compose build voice-agent
sudo docker compose up -d voice-agent