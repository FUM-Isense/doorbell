import argparse
from collections import Counter
import random
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
from DoorbellTask import DoorbellTask
from TextToSpeechPlayer import TextToSpeechPlayer
from pynput import keyboard

capture_step = 0
capture_flag_name = False
capture_flag_doorbell = False

def on_press(key):
    global capture_step
    try:
        if key.char == 'b':  # Detect when 'b' is pressed
            capture_step += 1
    except AttributeError:
        pass
    
    
def main(show_cv2,vertical):
    global capture_step, doorbell, speech, pipeline, capture_flag_doorbell, capture_flag_name
    
    proccess_frame = 5
    counter = 0
    vote_count = 1
    found_data = [] 
        
    while True:
        try:      
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                rgb_frame = np.asanyarray(color_frame.get_data())
                
                if vertical:
                    rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                if capture_step == 1 and not capture_flag_name:                          
                    counter += 1
                    if counter % proccess_frame == 0:
                        found_name = doorbell.Set_Target_Image(rgb_frame)
                        if found_name != "":
                            found_data.append(found_name)
                            print(f"Target : {found_name}")
                            
                        if len(found_data) > 1:
                            counter = Counter(found_data)
                            most_common_name, count = counter.most_common(1)[0]    

                            doorbell.target_name = most_common_name
                            speech.say(f"Target found {doorbell.target_name}")
                            
                            capture_flag_name = True
                            proccess_frame = 3
                            counter = 0
                            vote_count = 3
                            found_data = []   
                            counter = 0
                            
                    if show_cv2:
                        cv2.imshow("Target Img", rgb_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):  # فشردن کلید 'q' برای خروج
                            break

                if capture_step == 2 and not capture_flag_doorbell:                          
                    counter += 1
                    
                    if counter % proccess_frame == 0:
                        res_img,found,target_data = doorbell.Find_target_doorbell(rgb_frame)
                    
                        if found:
                            found_data.append(target_data)

                            if len(found_data) >= vote_count:
                                avg_x = round(sum(data[0] for data in found_data) / len(found_data))
                                avg_y = round(sum(data[1] for data in found_data) / len(found_data))

                                column = "right" if avg_x == 1 else "left"
                                row = avg_y

                                avg_message = f"Found on Column {column} Row {row}"
                                print(avg_message)
                                speech.say(avg_message)
                                capture_flag_doorbell = True
                    else:
                        res_img = rgb_frame.copy()
                        
                    if show_cv2:
                        cv2.imshow("Test", res_img)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):  # فشردن کلید 'q' برای خروج
                            break
        except Exception as ex:
            print("error")
            
        finally:
            speech.stop()
            cv2.destroyAllWindows()
        
if __name__ == "__main__":
    
    print("Doorbell Start")
    parser = argparse.ArgumentParser(description="Process optional arguments.")

    parser.add_argument('-v', type=bool, default=True, help='First optional argument')
    parser.add_argument('-cv2', type=bool, default=False, help='Second optional argument')

    args = parser.parse_args()

    vertical = args.v
    show_cv2 = args.cv2
    
    doorbell = DoorbellTask(True)
    # doorbell.Set_Target_Image(target_image)
    # rnd_name = random.choice(doorbell.names)
    # doorbell.target_name = rnd_nameqq


    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device("234322308671")
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline_profile = pipeline.start(config)

    device = pipeline_profile.get_device()
    color_sensor = device.query_sensors()[1]
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    color_sensor.set_option(rs.option.exposure, 125)

    counter = 0
    proccess_frame = 5

    speech = TextToSpeechPlayer()
    speech.say("Start")


    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    

    print("Doorbell Running...")
    main(show_cv2,vertical)
