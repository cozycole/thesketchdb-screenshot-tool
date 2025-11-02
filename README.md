# theSketchDb Screenshot Tool

## Necessary dependencies

```bash
pip install insightface torchvision hdbscan
```

## yt-dlp

```bash
yt-dlp --download-sections "*4:12-8:04" \ 
    -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \ 
    --extractor-args "youtube:player-client=default,-tv_simply" https://www.youtube.com/watch?v=0h-j4qFR3wg
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" --extractor-args "youtube:player-client=default,-tv_simply" https://www.youtube.com/watch?v=FPhj6C0VzyU
```

## Get video dimensions

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 input.mp4
```
