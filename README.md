# RA-opencv

Augmented Reality with OpenCV using Python.
 
## Usage

* Place the image of the surface to be tracked inside the `references` folder;
* Replace `TRAIN_IMAGE_RELATIVE_PATH` with the relative path of the train image you want to track;
* Replace `OBJ_MODEL_RELATIVE_PATH` with the relative path of the model you want to render. To change the size of the rendered model change the scale parameter `MODEL_OBJ_SCALE` to a suitable number. This might require some trial and error.

## Running 

Open a terminal session inside the project folder and run:

```sh
# Activate virtualenv
virtualenv python_ra
source python_ra/bin/activate
# to deactivate: run `deactivate`
```
Then:

```sh
# Install from the requirements
pip install -r requirements.txt
python3 src/ra.py
```

### Command line arguments

* `--rectangle`, `-r`: Draws the projection of the reference surface on the video frame as a blue rectangle;
* `--matches`, `-ma`: Draws matches between reference surface and video frame.

```sh
python3 src/ra.py -r -ma
```

## Troubleshooting

**If you get the message**:

```
Unable to capture video
```

printed to your terminal, the most likely cause is that your OpenCV installation has been compiled without FFMPEG support. Pre-built OpenCV packages such as the ones downloaded via pip are not compiled with FFMPEG support, which means that you have to build it manually.


