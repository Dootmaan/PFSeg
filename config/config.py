# sr scale 2x

# original experiment
# input_img_size=(64,96,96)
# crop_size=(32,64,64) 


# patch-free experiment (DSRL)
input_img_size=(64,96,96)
crop_size=input_img_size

# patch-free experiment (ResUnet). The model input is LABEL_SR, so it's actually doing (64,96,96) patch on (128,192,192) image
# input_img_size=(64,96,96)
# crop_size=(32,48,48)  # crop image will not be used in this experiment, instead the (64,96,96) label_sr gt will be used.

# crop_size=(24,32,32)
# crop_size=(48,64,64)
# crop_size=(16,24,24)
# crop_size=(8,16,16)