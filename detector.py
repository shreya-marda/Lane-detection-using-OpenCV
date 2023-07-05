import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("road.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]
width = image.shape[1]

#Now let us define our region of interest ROI
#our region of interest will only be the road
ROI_vertices = [
    (0,height-100),
    (width/2, height/2),
    (width,height-100)
]

#Function to mask every other thing other than our region of interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8) #number of channels for colored image is always 3 
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0,255,0), 5)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, gamma=0)
    return img

imgray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
canny_edge = cv2.Canny(imgray, 100,200)
cropped_image = region_of_interest(canny_edge, np.array([ROI_vertices], np.int32))

#Draw lines using probabilistic Hough line transform
lines = cv2.HoughLinesP(cropped_image, rho=6, theta=np.pi/180, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)

image_with_lines = draw_the_lines(image, lines)

plt.imshow(image_with_lines)
#plt.imshow(canny_edge)
#plt.imshow(cropped_image)
#plt.imshow(image)
plt.show()
