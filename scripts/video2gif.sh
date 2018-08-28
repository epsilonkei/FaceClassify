#!/usr/bin/env bash
# $1 is video file name
# $2 is output gif file name

ffmpeg -ss 2.6 -t 1.3 -i $1 fps=15,scale=320:-1:flags=lanczos,palettegen palette.png
ffmpeg -ss 2.6 -t 1.3 -i $1 -i palette.png -filter_complex "fps=15,scale=400:-1:flags=lanczos[x];[x][1:v]paletteuse" $2
rm -f palette.png
