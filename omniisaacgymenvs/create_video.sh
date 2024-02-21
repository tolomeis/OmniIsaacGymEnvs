#!/bin/bash

# Set variables
FPS=25              # Adjust with your actual video FPS
EPOCH_INTERVAL=50   # Gap between labeled videos (epochs)

INPUT_DIR="videos"       # Folder containing input videos
OUTPUT_DIR="$INPUT_DIR/$1"     # Folder for processed videos
FINAL_VIDEO="$2"    # Name of the final concatenated video

STARTING_L=53
TARGET_L=5
# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Counter for processed videos
processed_count=0

# Loop through videos in the directory (excluding subdirectories)
for file in "$INPUT_DIR"/*.mp4; do

  # Calculate approximate epoch based on position
  epoch=$(( processed_count * EPOCH_INTERVAL ))

  output_video="$OUTPUT_DIR/output_epoch_${epoch}.mp4"

  # Speed up, label, and output video
  ffmpeg -i "$file" -vf "setpts=$TARGET_L/$STARTING_L*PTS,drawtext=fontfile=/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf:fontsize=24:fontcolor=white:x=(w-text_w)/2:y=20:text='Epoch $epoch'" -c:v libx264 -c:a copy "$output_video"
  # Add video to list for concatenation
  echo "file 'output_epoch_${epoch}.mp4'" >> "$OUTPUT_DIR/filelist.txt"

  # Increment counter
  processed_count=$(( processed_count + 1 ))
done

# Concatenate all videos
ffmpeg -f concat -safe 0 -protocol_whitelist file -i "$OUTPUT_DIR/filelist.txt" -c copy "$OUTPUT_DIR/$FINAL_VIDEO"

echo "All videos processed and concatenated into '$OUTPUT_DIR/$FINAL_VIDEO'"

# Cleanup: remove temporary file (optional)
# rm "$OUTPUT_DIR/filelist.txt"
