# theSketchDb Screenshot Tool

1) Iterate through frames, crop faces and store them in temp dir.
2) Group the images into person clusters.
3) For each cluster choose top 3 images across the timeframe the person is in the sketch.

## Compiling dlib w cuda support

Downloaded cuda toolkit v13 following: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation

git clone https://github.com/davisking/dlib.git

```bash
cd dlib
python setup.py install --set USE_AVX_INSTRUCTIONS=ON --set DLIB_USE_CUDA=ON --set CMAKE_CUDA_ARCHITECTURES=86
```

specifying the CUDA arch was important since it installs for several different ones, some of which aren't supported by v13 and the compilation fails
NVIDIA GeForce RTX 3090 Ti is compatible with 86 as (as seen here: https://developer.nvidia.com/cuda-gpus)
Now 
```python
import dlib
print(dlib.DLIB_USE_CUDA) # True
```

You also need to install  libcublas11 for some reason, it errors otherwise but there isn't any docs on it.


```bash
sudo apt install libcublas11
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
