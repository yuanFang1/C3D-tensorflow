#!/bin/bash

# ./convert_video_to_images.sh ../videoFile/ucf-11 5
for folder in $1/*
do
    for subfolder in $folder/*
    do
        for file in "$subfolder"/*.avi
        do
            if [[ ! -d "${file[@]%.avi}" ]]; then
                mkdir -p "${file[@]%.avi}"
            fi
            ffmpeg -i "$file" -vf fps=$2 "${file[@]%.avi}"/%05d.jpg
        done
    done
done