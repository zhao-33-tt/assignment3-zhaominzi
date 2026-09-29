from jetson.inference import detectNet
from jetson.utils import videoSource, videoOutput
#import jetson.inference
#import jetson.utils

net = detectNet("ssd-mobilenet-v2",threshold=0.5)
camera=videoSource("/home/nvidia/jetson-inference/data/video/test4.mp4")
display=videoOutput("/home/nvidia/jetson-inference/data/video/output_test4.mp4")

frame_count = 0
while display.IsStreaming():
    img = camera.Capture()
    if img is None:
        continue
    frame_count +=1
    detections=net.Detect(img)
    for detection in detections:
        print(f"---Frame{frame_count}---")
        print(detection)
    display.Render(img)
    display.SetStatus("Object Dtection| Network {:.0f} FPS".format(net.GetNetworkFPS()))
