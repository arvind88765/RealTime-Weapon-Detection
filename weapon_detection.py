import cv2
import numpy as np
import os
import threading

# Load YOLO model with weights and config file paths
net = cv2.dnn.readNet("D:\\wp_detect\\yolov3_training_2000.weights", "D:\\wp_detect\\yolov3_testing.cfg")
classes = ["Arm"]  # Ensure this matches the class name in your YOLO model

# Get output layer names for YOLO
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def display_menu():
    """Display menu for user selection"""
    print("\n" + "="*40)
    print("Welcome to the Detection System")
    print("="*40)
    print("1. Image Detection")
    print("2. Video Detection")
    print("3. Real-time (Webcam)")
    print("="*40)
    choice = input("Please choose an option (1/2/3): ").strip()
    return choice

def get_image_file():
    """Prompt user to select or provide an image file"""
    files = [f for f in os.listdir("D:\\wp_detect\\") if f.endswith(('.jpg', '.jpeg', '.png'))]
    if files:
        print("\nAvailable image files:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
    else:
        print("\nNo image files found in the directory.")
    choice = input("Enter file name from the list or provide the full path: ").strip()

    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(files):
            return os.path.join("D:\\wp_detect\\", files[index])
        else:
            print("Invalid choice.")
            return None
    elif os.path.isfile(choice):
        return choice
    elif os.path.isfile("D:\\wp_detect\\" + choice):
        return "D:\\wp_detect\\" + choice
    else:
        print("Invalid file path.")
        return None

def get_video_file():
    """Prompt user to select or provide a video file"""
    files = [f for f in os.listdir("D:\\wp_detect\\") if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if files:
        print("\nAvailable video files:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
    else:
        print("\nNo video files found in the directory.")
    choice = input("Enter file name from the list or provide the full path: ").strip()

    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(files):
            return os.path.join("D:\\wp_detect\\", files[index])
        else:
            print("Invalid choice.")
            return None
    elif os.path.isfile(choice):
        return choice
    elif os.path.isfile("D:\\wp_detect\\" + choice):
        return "D:\\wp_detect\\" + choice
    else:
        print("Invalid file path.")
        return None

def process_image(image_path):
    """Process and detect arms in an image"""
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to read the image.")
        return
    
    height, width, _ = img.shape

    # Prepare image for YOLO processing
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    class_ids = []
    confidences = []
    boxes = []
    detected = False  # Flag to check if arms are detected

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Minimum confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected = True

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if detected:
        print("Arm detected in the image")
    else:
        print("No arms detected in the image")

    # Draw bounding boxes on the image
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    # Show the image with detection
    cv2.imshow("Image Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_source):
    """Process and detect arms in a video frame by frame"""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to read a frame from the video.")
            break

        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layer_names)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indexes) > 0:
            print("Arm detected in frame")

        # Draw bounding boxes on the frame
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

        cv2.imshow("Video Detection", img)
        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

def process_frame_for_detection(frame):
    """This function will handle YOLO detection on the frame."""
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Minimum confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw detection boxes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    return frame

def process_real_time():
    """Detect arms in real-time using a webcam feed"""
    cap = cv2.VideoCapture(1)  # Open the default webcam
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    def run_detection():
        while True:
            ret, img = cap.read()
            if not ret:
                break
            img_with_detections = process_frame_for_detection(img)
            cv2.imshow("Real-time Detection", img_with_detections)
            if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
                break

    detection_thread = threading.Thread(target=run_detection)
    detection_thread.start()

    detection_thread.join()  # Wait for the thread to finish

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function for user selection and processing"""
    choice = display_menu()
    if choice == '1':
        image_path = get_image_file()
        if image_path:
            process_image(image_path)
    elif choice == '2':
        video_source = get_video_file()
        if video_source:
            process_video(video_source)
    elif choice == '3':
        process_real_time()
    else:
        print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
