## Face in vivo detection

### Env

1. python: 3.6.0
2. tensorflow: 1.11.0
3. keras: 2.2.4
4. cv2(OpenCV): 3.4.1

### File Info

1. `haarcascade_frontalface_default.xml`: `Lib` that included in `OpenCV`. We use it to extract the face in the picture.
2. `data_translate.py`: extract the pictures from the video and normalize them.
3. `data_supply.py`: read the images and combine five continuous of them into one 3-dimension pictures.
4. `model.py`: define the network structure.
5. `main.py`: entrace.


You can find our specific process in the `LOG.md`.