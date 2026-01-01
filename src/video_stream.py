import cv2
import threading
import time

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
             print(f"Warning: Could not open video source {src}")
             self.grabbed = False
             self.frame = None
             return

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
        # Start the thread to read frames from the video stream
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while True:
            # If the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # Otherwise, read the next frame from the stream
            (grabbed, frame) = self.stream.read()
            if not grabbed:
                self.stopped = True
                return
            
            self.grabbed = grabbed
            self.frame = frame

    def read(self):
        # Return the most recently read frame
        return self.frame

    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True
        self.thread.join()
        self.stream.release()
