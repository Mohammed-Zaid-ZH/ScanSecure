import cv2
from pyzbar.pyzbar import decode

# Load the QR code image
image = cv2.imread('qrcode.png')

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image. Please check the file path.")
else:
    # Convert image to grayscale for better detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Decode the QR code
    decoded_objects = decode(gray_image)

    if len(decoded_objects) == 0:
        print("No QR code detected.")
    else:
        for obj in decoded_objects:
            print("Link:", obj.data.decode('utf-8'))
