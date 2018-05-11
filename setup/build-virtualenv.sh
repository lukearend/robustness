#!/bin/sh

virtualenv -p python3 ~/envs/robust

source ~/envs/robust/bin/activate

pip install -r requirements.txt
