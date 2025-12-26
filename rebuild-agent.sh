#!/bin/bash

sudo docker compose --env-file .env.user down voice-agent
sudo docker compose --env-file .env.user build voice-agent
sudo docker compose --env-file .env.user up -d voice-agent