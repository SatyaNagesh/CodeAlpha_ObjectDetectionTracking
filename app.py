import cv2
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Object Detection & Tracking with YOLOv8")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: '0' for webcam, or path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO model name (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--track", action="store_true",
                        help="Enable object tracking (uses BoT-SORT)")
    args = parser.parse_args()

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    if args.source == "0":
        cap = cv2.VideoCapture(0)
        source_name = "Webcam"
    else:
        cap = cv2.VideoCapture(args.source)
        source_name = args.source

    if not cap.isOpened():
        print(f"Error: Could not open video source '{args.source}'")
        return

    print(f"Running detection on {source_name}")
    print("Press 'q' to quit | 's' to save screenshot | '+'/'-' to adjust confidence")

    conf = args.conf

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        if args.track:
            results = model.track(frame, conf=conf, persist=True, verbose=False)
        else:
            results = model(frame, conf=conf, verbose=False)

        annotated_frame = results[0].plot()

        cv2.putText(annotated_frame, f"Conf: {conf:.2f} | {'Tracking ON' if args.track else 'Detection ON'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "q:quit s:screenshot +/-:confidence",
                    (10, annotated_frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Object Detection & Tracking - CodeAlpha", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit requested.")
            break
        elif key == ord('s'):
            filename = f"screenshot_{cv2.getTickCount()}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('+') or key == ord('='):
            conf = min(1.0, conf + 0.05)
            print(f"Confidence threshold: {conf:.2f}")
        elif key == ord('-') or key == ord('_'):
            conf = max(0.0, conf - 0.05)
            print(f"Confidence threshold: {conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
