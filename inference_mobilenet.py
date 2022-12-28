#%%
from PIL import Image
import numpy as np
import onnxruntime
import torch
import cv2

def preprocess_image(image_path, height, width, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.LANCZOS)
    image_data = np.asarray(image).astype(np.float32)
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

#%%

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_sample(session, image_file, categories):
    output = session.run([], {'input':preprocess_image(image_file, 224, 224)})[0]
    output = output.flatten()
    output = softmax(output) # this is optional
    top5_catid = np.argsort(-output)[:5]
    for catid in top5_catid:
        print(categories[catid], output[catid])
    # write the result to a file
    with open("result.txt", "w") as f:
        for catid in top5_catid:
            f.write(categories[catid] + " " + str(output[catid]) + " \r")



#%%
# create main function
if __name__ == "__main__":
    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    
    # Create Inference Session
    session = onnxruntime.InferenceSession("mobilenet_v2_float.onnx")

    # get image from camera
    cap = cv2.VideoCapture(0)
    cap.set(3,640) # set Width
    cap.set(4,480) # set Height

    # capture image from camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, -1) # Flip camera vertically
    cv2.imwrite('capture.jpg', frame)
    cap.release()
    cv2.destroyAllWindows()

    run_sample(session, 'capture.jpg', categories)

# %%
