import cv2

def resize_image_with_ratio(image, ratio):
    
    # Get the original dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the new dimensions based on the specified ratio
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize the image while maintaining aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    return resized_image

if __name__ == '__main__':
    img = cv2.imread('eky.jpg', cv2.IMREAD_UNCHANGED)
    img = resize_image_with_ratio(img, 0.4)
    cv2.imwrite('eky_resized.jpg', img)