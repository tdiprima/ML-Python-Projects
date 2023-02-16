#!/bin/bash

nohup jupyter notebook &

# https://stackoverflow.com/questions/17385794/how-to-get-the-process-id-to-kill-a-nohup-process
echo $!
