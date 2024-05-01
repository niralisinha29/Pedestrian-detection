import cv2

def load_pretrained_model():
    # Using HOG + SVM model for pedestrian detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

def detect_pedestrians(frame, model):
    """
    Detect pedestrians in an input frame.
    :param frame: Input frame from a video.
    :param model: Pretrained HOG descriptor with SVM-based people detector.
    :return: Frame with detection bounding boxes.
    """
    # Perform the detection
    rects, _ = model.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # Draw bounding boxes on the detected pedestrians
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame
