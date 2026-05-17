#!/bin/bash

env="source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate"
name=$(hostname)

if ! grep -q "$env" ~/.bashrc; then
    echo $env | sudo tee -a ~/.bashrc > /dev/null
fi

source ~/.bashrc

if [ ! -f influxdata-archive_compat.key ]; then
    # Download and install the InfluxData GPG key (overwrite if exists)
    curl -fsSL https://repos.influxdata.com/influxdata-archive.key -o influxdata-archive_compat.key
    cat influxdata-archive_compat.key | gpg --dearmor --yes | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
    echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list

    sudo apt-get update
    sudo apt-get install influxdb2 influxdb2-cli -y
    sudo systemctl start influxdb

    influx setup \
      --username $name \
      --org "Stanford" \
      --bucket "Asst4" \
      --force
fi
