import numpy as np
from ultralytics import YOLOv10
import cv2
from boxmot import BYTETracker
from utils import create_video_writer

# Initialize the tracker
tracker = BYTETracker(per_class=True)

# Initialize YOLO model
model = YOLOv10('weights/yolov10s.pt')

# Open the input video
INPUT_PATH = 'assets/cars.mp4'

# Define the output video path
OUTPUT_PATH = f'output/BYTETracker_{INPUT_PATH.split(".")[0].split("/")[1]}.mp4'

LIMITS = [173,205,540,205]

class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
               6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
               11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
               16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
               22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
               27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
               32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
               36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 
               41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 
               48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 
               54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
               60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 
               66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 
               72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 
               78: 'hair drier', 79: 'toothbrush'}

# Open the video
cap = cv2.VideoCapture(INPUT_PATH)

# Check if the video was opened
if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

# Create the video writer
out = create_video_writer(cap, OUTPUT_PATH)

car_count = []

# Loop through the video frames
while cap.isOpened():
    ret, im = cap.read()
    if not ret:
        break
    
    try:
        # Run the YOLO model on the frame
        results = model(im)

        dets = []

        # Draw LIMITS line
        cv2.line(im, (LIMITS[0], LIMITS[1]), (LIMITS[2], LIMITS[3]), (0, 0, 255), 2)

        # Iterate over each bbox detected
        for result in results:
            for detection in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = detection
                
                dets.append([x1, y1, x2, y2, conf, int(cls)])

        # Convert to ndarray    
        dets = np.array(dets)

        # Update tracker with detections
        tracks = tracker.update(dets, im)

        # Iterate over each bbox to get xyxys, ids...
        for track in tracks:
            # Convert all xyxys to int 
            xyxys = [int(xyxy) for xyxy in track[:4]]
            ids = track[4]
            conf = track[5]
            cls = track[6]
            
            x1, y1, x2, y2 = xyxys[:4]
            
            # Get width and height
            w, h = x2 - x1, y2 - y1

            # Get center X and center Y of the bbox
            cx, cy = x1 + w // 2, y1 + h // 2

            # Check if detected bbox class name is a vehicle
            if class_names[int(cls)] in [class_names[2], class_names[5], class_names[6], class_names[7]]:
                # Draw a circle in the center of the bbox
                cv2.circle(im, (cx, cy), 3, (255, 255, 255), cv2.FILLED)

                # Check if the circle has crossed the limits
                if LIMITS[0] < cx < LIMITS[2] and LIMITS[1] - 10 < cy < LIMITS[1] + 10:
                    # Check if current id was already counted
                    if car_count.count(int(ids)) == 0:
                        # Count the vehicles by appending the id to the car count list
                        car_count.append(int(ids))

                        # Change the limits line color to green a vehicle is counted
                        cv2.line(im, (LIMITS[0], LIMITS[1]), (LIMITS[2], LIMITS[3]), (0, 255, 0), 2)
        
        # Put vehicles count text and counter
        cv2.putText(im, f"Vehicles count: {str(len(car_count))}", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Plot the results from the tracker
        im = tracker.plot_results(im, show_trajectories=False)

        # Display the frame
        cv2.imshow("Car counter", im)
        
        # Write the frame to the output video
        out.write(im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {e}")
        break


cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Tracking video saved to {OUTPUT_PATH}")
