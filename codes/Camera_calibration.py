import cv2
import numpy as np
import apriltag
import time
import os

# Constants for the Apriltag detection
tag_family = 'tag36h11'
detector = apriltag.Detector(apriltag.DetectorOptions(families=tag_family))

# Criteria for termination of the iterative process of refining the corner points
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def capture_images(num_images=10):
    cap = cv2.VideoCapture(0)
    captured_images = []
    savedir = "captured_images"

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    print(f"Press 's' to capture {num_images} images. Press 'q' to quit early.")
    start_time = time.time()
    frame_count = 0

    while len(captured_images) < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue

        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        else:
            fps = frame_count / elapsed_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(frame, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)),
                         (0, 255, 0), 2)
            cv2.putText(frame, str(tag.tag_id), tuple(tag.corners[0, :].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Capture', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            captured_images.append(gray)
            img_path = os.path.join(savedir, f"image_{len(captured_images)}.png")
            cv2.imwrite(img_path, gray)
            print(f"Captured image {len(captured_images)}/{num_images} and saved to {img_path}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_images


def estimate_tag_size(tag):
    # Estimate the size of the tag based on the distance between its corners
    d1 = np.linalg.norm(tag.corners[0] - tag.corners[1])
    d2 = np.linalg.norm(tag.corners[1] - tag.corners[2])
    d3 = np.linalg.norm(tag.corners[2] - tag.corners[3])
    d4 = np.linalg.norm(tag.corners[3] - tag.corners[0])
    avg_size = np.mean([d1, d2, d3, d4])
    return avg_size


def calibrate_camera(images, image_size):
    obj_points = []
    img_points = []

    for img in images:
        tags = detector.detect(img)
        if tags:
            for tag in tags:
                tag_size = estimate_tag_size(tag)
                objp = np.array([[0, 0, 0], [tag_size, 0, 0], [tag_size, tag_size, 0], [0, tag_size, 0]],
                                dtype=np.float32)
                img_points.append(tag.corners.astype(np.float32))
                obj_points.append(objp)

    if not obj_points or not img_points:
        print("No valid data for calibration.")
        return None, None, None, None

    print(f"Number of object points: {len(obj_points)}")
    print(f"Number of image points: {len(img_points)}")
    print(f"Image size: {image_size}")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    return mtx, dist, rvecs, tvecs


def main():
    num_images = int(input("Enter the number of images to capture: "))

    print("Capturing images...")
    images = capture_images(num_images)

    if not images:
        print("No images captured. Exiting.")
        return

    image_size = images[0].shape[::-1]  # (width, height)

    print("Calibrating camera...")
    mtx, dist, rvecs, tvecs = calibrate_camera(images, image_size)

    if mtx is None:
        print("Calibration failed. Exiting.")
        return

    print("Camera matrix:")
    print(mtx)
    print("Distortion coefficients:")
    print(dist)

    # Show the calibration result on one of the images
    img = images[0]
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imshow('Calibrated Image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
