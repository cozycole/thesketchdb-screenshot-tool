import numpy as np
import cv2

def resize_for_detection(frame, target_width=1280):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    resized = cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized

def remove_borders(frame, tol=15, safety_crop=2):
    """
    Removes solid-color borders (e.g. black bars) from an image.
    Tolerant of mild gradients and compression artifacts.
    """
    if frame is None or frame.size == 0:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Border color estimate = median of corners
    corners = np.concatenate([
        gray[0:5, 0:5].ravel(),
        gray[0:5, -5:].ravel(),
        gray[-5:, 0:5].ravel(),
        gray[-5:, -5:].ravel()
    ])
    border_color = np.median(corners)

    # Build mask: significant deviation from border color
    diff = np.abs(gray.astype(np.int16) - border_color)
    mask = diff > tol

    # Morphological cleanup: erode small noise at edges
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    # Find bounding box of content
    coords = cv2.findNonZero(mask)
    if coords is None:
        return frame

    x, y, w, h = cv2.boundingRect(coords)

    # Apply safety crop to remove residual thin lines
    x = max(0, x + safety_crop)
    y = max(0, y + safety_crop)
    w = max(1, w - 2 * safety_crop)
    h = max(1, h - 2 * safety_crop)

    cropped = frame[y:y+h, x:x+w]

    return cropped

# def remove_borders(frame, tol=15):
#     """
#     Removes solid-color borders (e.g. black bars) from an image.
#
#     Args:
#         frame: BGR image (np.ndarray)
#         tol: tolerance (0–255), how much variation from border color to allow
#
#     Returns:
#         cropped: cropped image with borders removed
#     """
#     if frame is None or frame.size == 0:
#         return frame
#
#     # Convert to grayscale for simplicity
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Detect border color (average of corners)
#     corners = np.concatenate([
#         gray[0:5, 0:5].ravel(),
#         gray[0:5, -5:].ravel(),
#         gray[-5:, 0:5].ravel(),
#         gray[-5:, -5:].ravel()
#     ])
#     border_color = np.median(corners)
#
#     # Build mask of pixels differing significantly from border color
#     mask = np.abs(gray.astype(np.int16) - border_color) > tol
#
#     # Find bounding box of non-border area
#     coords = cv2.findNonZero(mask.astype(np.uint8))
#     if coords is None:
#         return frame  # whole frame is one color, nothing to crop
#
#     x, y, w, h = cv2.boundingRect(coords)
#     cropped = frame[y:y+h, x:x+w]
#
#     return cropped

def crop_profile_image(frame, box):
    """
    Create a square profile image with head (including hair) in top half,
    torso filling bottom half. Always returns an in-frame crop, no padding.
    
    Args:
        frame: BGR image from cv2
        box: bounding box (x1, y1, x2, y2)
    
    Returns:
        profile_img: square BGR image
    """
    x1, y1, x2, y2 = map(int, box)
    H, W = frame.shape[:2]

    # Clamp input box within frame
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)

    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    rel = h / H
    # Interpolation factor
    t = np.clip((rel - 0.1) / (0.6 - 0.1), 0, 1)

    # Far shot (0.2 top, 1.0 bottom)
    # Close shot (0.45 top, 0.35 bottom)
    top_margin_ratio = 0.2 + (0.4 - 0.2) * t
    bottom_margin_ratio = 1.2 + (0.3 - 1.2) * t

    # Derived values
    nose_x = x1 + w / 2
    
    crop_y1 = int(y1 - h*top_margin_ratio) # top of image
    crop_y2 = int(y2 + h*bottom_margin_ratio) # bottom of image
    crop_height = crop_y2 - crop_y1
    crop_x1 = int(nose_x - crop_height / 2) # left of image
    crop_x2 = int(nose_x + crop_height / 2) # right of image
    
    # in a while loop since sometimes after a resize, it needs to be shifted
    while True:
        if (crop_x1 < 0 and crop_x2 > W) or (crop_y1 < 0 and crop_y2 > H):
            scale = min(W / (crop_x2 - crop_x1), H / (crop_y2 - crop_y1))
            crop_y1 = int(crop_y1 * scale)
            crop_y2 = int(crop_y2 * scale) # bottom of image
            crop_height = crop_y2 - crop_y1
            crop_x1 = int(nose_x - crop_height / 2) # left of image
            crop_x2 = int(nose_x + crop_height / 2) # right of image
            print("Resized crop dimensions:", crop_x1, crop_y1, crop_x2, crop_y2)
        else:
            # Adjust boundaries individually if only one side exceeds
            if crop_x1 < 0:
                print("crop_x_left < 0")
                shift = -crop_x1
                crop_x1 += shift
                crop_x2 += shift
            if crop_x2 > W:
                print("crop_x_right > W")
                shift = crop_x2 - W
                crop_x1 -= shift
                crop_x2 -= shift

            if crop_y1 < 0:
                print("crop_y_top < 0")
                shift = -crop_y1
                crop_y1 += shift
                crop_y2 += shift
            if crop_y2 > H:
                print("crop_y_bottom > H")
                shift = crop_y2 - H
                crop_y1 -= shift
                crop_y2 -= shift
            break

    # Clamp again to ensure within frame
    crop_x_left = max(0, crop_x1)
    crop_y_top = max(0, crop_y1)
    crop_x_right = min(W, crop_x2)
    crop_y_bottom = min(H, crop_y2)

    cropped = frame[crop_y_top:crop_y_bottom, crop_x_left:crop_x_right]
    return cropped

def crop_thumbnail_16x9(frame, box):
    """
    Create a 16:9 thumbnail with the person centered as much as possible
    without going outside frame boundaries. Includes head and torso.
    Never pads with black borders; slides or rescales instead.

    Args:
        frame: BGR image from cv2
        box: (x1, y1, x2, y2)

    Returns:
        thumb_img: 16:9 BGR image
    """
    x1, y1, x2, y2 = map(int, box)
    H, W = frame.shape[:2]

    # Clamp input box within frame
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)

    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    rel = h / H
    # Interpolation factor
    t = np.clip((rel - 0.1) / (0.6 - 0.1), 0, 1)
    
    # Far shot → (0.3 top, 0.7 bottom)
    # Close shot → (0.45 top, 0.35 bottom)
    top_margin_ratio = 0.2 + (0.4 - 0.2) * t
    bottom_margin_ratio = 1.2 + (0.3 - 1.2) * t
    # Derived values
    nose_x = x1 + w / 2
    
    crop_y1 = int(y1 - h*top_margin_ratio) # top of image
    crop_y2 = int(y2 + h*bottom_margin_ratio) # bottom of image
    crop_height = crop_y2 - crop_y1
    print("crop height", crop_height)
    crop_x1 = int(nose_x - crop_height * (16/9) / 2) # left of image
    crop_x2 = int(nose_x + crop_height * (16/9) / 2) # right of image
    crop_width = crop_x2 - crop_x1
    print("crop width", crop_width)
    
    print("Crop dimensions:", crop_x1, crop_y1, crop_x2, crop_y2)
    # scale = min(W / (crop_x2 - crop_x1), H / (crop_y2 - crop_y1))
    # print("SCALE: ",scale)
    # return crop_x1, crop_y1, crop_x2, crop_y2
    # Check if crop is larger than frame in both dimensions
    while True:
        if (crop_x1 < 0 and crop_x2 > W) or (crop_y1 < 0 and crop_y2 > H):
            print("Scaled!")
            scale = min(W / (crop_x2 - crop_x1), H / (crop_y2 - crop_y1))
            crop_y1 = int(crop_y1 * scale)
            crop_y2 = int(crop_y2 * scale) # bottom of image
            crop_x1 = int(crop_x1 * scale)
            crop_x2 = int(crop_x2 * scale) # right of image
            print("Resized crop dimensions:", crop_x1, crop_y1, crop_x2, crop_y2)
        else:
            # Adjust boundaries individually if only one side exceeds
            if crop_x1 < 0:
                print("crop_x_left < 0")
                shift = -crop_x1
                crop_x1 += shift
                crop_x2 += shift
            if crop_x2 > W:
                print("crop_x_right > W")
                shift = crop_x2 - W
                crop_x1 -= shift
                crop_x2 -= shift
    
            if crop_y1 < 0:
                print("crop_y_top < 0")
                shift = -crop_y1
                crop_y1 += shift
                crop_y2 += shift
            if crop_y2 > H:
                print("crop_y_bottom > H")
                shift = crop_y2 - H
                crop_y1 -= shift
                crop_y2 -= shift
            break

    # Clamp again to ensure safety
    crop_x_left = max(0, crop_x1)
    crop_y_top = max(0, crop_y1)
    crop_x_right = min(W, crop_x2)
    crop_y_bottom = min(H, crop_y2)
    # Extract region
    cropped = frame[crop_y_top:crop_y_bottom, crop_x_left:crop_x_right]

    # Ensure 16:9 output (resize only, no pad)
    crop_h, crop_w = cropped.shape[:2]
    target_ratio = 16 / 9
    current_ratio = crop_w / crop_h

    if abs(current_ratio - target_ratio) > 0.01:
        if current_ratio > target_ratio:
            # Too wide → crop sides
            new_w = int(crop_h * target_ratio)
            offset = (crop_w - new_w) // 2
            cropped = cropped[:, offset:offset + new_w]
        else:
            # Too tall → crop top/bottom
            new_h = int(crop_w / target_ratio)
            offset = (crop_h - new_h) // 2
            cropped = cropped[offset:offset + new_h, :]

    # Optional: resize to fixed output dimensions (e.g., 1280x720)
    # cropped = cv2.resize(cropped, (1280, 720), interpolation=cv2.INTER_AREA)

    return cropped


if __name__ == "__main__":
    test_img_path = "/tmp/face_cluster_p7qzhfvp/frames/frame_0680.jpg"
    img = cv2.imread(test_img_path)
    new_img = remove_borders(img)
    cv2.imwrite("test_image.jpg", new_img)
