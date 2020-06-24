## Experiment Process

### First Try

1. For each video, we drew 5 or more frames at the same interval. 
2. We combine the same person's continuous five (grayscale) pictures into one picture with five channels.
3. So the size of `Training Dataset` or `Test Dataset` is 20(10:pos, 10:neg).

Test Accuracy: 50%

### Second Try

1. Reduce the interval so that we can get more picture from one video.
2. Save the 128 X 128 instead of 64 X 64(Enlarget the picture).
3. Adjust the network stucture correspondingly.

Test Accuracy: 52.14%


### Third Try(Detail)

1. Normalization(52.14%)
2. Six Frames(52.19%)
3. Adjust the DNN(52.19%)
4. Seven Frames(52.25%)
5. More Dataset(~54%)

Finally, we found that it's the output vector's form which impedes the training process.
We change the (single output & binary_crossentropy) to (double outputs & categorical_crossentropy).

The latest repository stores the best model we found.
   
