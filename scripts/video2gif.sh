#!/usr/bin/env bash
# $1 is video file name
# $2 is output gif file name
# Note: this scripts only create gif file use data from 0[s] in 2.45sec duration

ffmpeg -ss 0 -t 2.45 -i $1 -vf fps=15,scale=320:-1:flags=lanczos,palettegen palette.png
ffmpeg -ss 0 -t 2.45 -i $1 -i palette.png -filter_complex "fps=15,scale=400:-1:flags=lanczos[x];[x][1:v]paletteuse" $2
rm -f palette.png
