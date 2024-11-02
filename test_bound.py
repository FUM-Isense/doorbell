import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from DoorbellTask import DoorbellTask
from iSenseTasksUtils.SpeechEngine import TextToSpeechPlayer
import pyrealsense2 as rs


doorbell = DoorbellTask(True)
target_image = cv2.imread("johnson.PNG")
doorbell.Set_Target_Image(target_image)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline_profile = pipeline.start(config)

device = pipeline_profile.get_device()
color_sensor = device.query_sensors()[1]
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.exposure, 125)

detection = 7
frame_count = 0

while True:
    frame_count += 1
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    rgb_frame = np.asanyarray(color_frame.get_data())
    
    # print(filename)
    # image_path = os.path.join(folder_path, filename)
    # image_path = f'zoom_test.png'
    # image = cv2.imread(image_path)

    try:
        if frame_count % detection == 0:
            rgb_frame,found,target_data = doorbell.Find_target_doorbell(rgb_frame)
            print(target_data)
    except:
        print("Error")
    
    # if found:
    #     totall_found += 1
    #     found_data.append(target_data)

    #     if len(found_data) >= 3:
    #         avg_x = round(sum(data[0] for data in found_data) / len(found_data))
    #         avg_y = round(sum(data[1] for data in found_data) / len(found_data))

    #         column = "right" if avg_x == 1 else "left"
    #         row = avg_y

    #         avg_message = f"Found on Column {column} Row {row}"
    #         print(avg_message)
    #         speech.say(avg_message)

    #         found_data = []
            
    cv2.imshow("Test", rgb_frame)
    cv2.waitKey(0)
    
cap.release()
speech.stop()
print(f"Found: {totall_found}")