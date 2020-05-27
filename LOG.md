## Experiment Process

### First Try

1. For each video, we drew 5 or more frames at the same interval. 
2. We combine the same person's continuous five (grayscale) pictures into one picture with five channels.
3. So the size of `Training Dataset` or `Test Dataset` is 20(10:pos, 10:neg).

Test Accuracy: 50%

### Second Try

1. Reduce the interval so that we can get more picture from one video.
2. Save the 128 X 128 instead of 64 X 64(Enlarget the picture).
3. Adjust the network stucture.

Test Accuracy: 52.14%