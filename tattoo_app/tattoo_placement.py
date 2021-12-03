# To use Inference Engine backend, specify location of plugins:
# source /opt/intel/computer_vision_sdk/bin/setupvars.sh
import cv2 as cv
import numpy as np
import argparse
from PIL import Image
#from imutils import face_utils, rotate_bound


#make tattoo transparent
with open('tattoo.txt') as f:
    tat_type = f.readlines()


if tat_type[0] == "bird":
    img = Image.open('bird.png')
elif tat_type[0] == "skull":
    img = Image.open('skull.png')
else:
    img = Image.open('scorpion.png')


rgba = img.convert("RGBA")
datas = rgba.getdata()
  
newData = []
for item in datas:
    if item[0] > 100 and item[1] > 100 and item[2] > 100:  
        # replacing it with a transparent value
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)
  
rgba.putdata(newData)
rgba.save("tattoo.png", "PNG")





# Adjust the given sprite to the head's width and position
# in case of the sprite not fitting the screen in the top, the sprite should be trimed
def adjust_sprite2head(sprite, back_width, back_ypos, ontop=True):
    (h_sprite, w_sprite) = (sprite.shape[0], sprite.shape[1])
    factor = .6 * back_width / w_sprite
    sprite = cv.resize(
        sprite, (0, 0), fx=factor, fy=factor
    )  # adjust to have the same width as head
    (h_sprite, w_sprite) = (sprite.shape[0], sprite.shape[1])

    y_orig = (
        back_ypos + (h_sprite/2) if ontop else back_ypos
	
    )  # adjust the position of sprite to end where the head begins
    if (
        y_orig < 0
    ):  # check if the head is not to close to the top of the image and the sprite would not fit in the screen
        sprite = sprite[abs(y_orig) : :, :, :]  # in that case, we cut the sprite
        y_orig = 0  # the sprite then begins at the top of the image

    return (sprite, y_orig)


# Applies tattoo to back
#def apply_sprite(image, path2sprite, w, x, y, angle, ontop=True):
def apply_sprite(image, path2sprite, w, x, y, ontop=True):
    sprite = cv.imread(path2sprite, -1)

    #sprite = rotate_bound(sprite, angle)
    (sprite, y_final) = adjust_sprite2head(sprite, w, y, ontop)
    
    x = x + (w/3)
    
    image = draw_sprite(image, sprite, x, y_final)

def draw_sprite(frame, sprite, x_offset, y_offset):


    (h, w) = (sprite.shape[0], sprite.shape[1])
    (imgH, imgW) = (frame.shape[0], frame.shape[1])

    if y_offset + h >= imgH:  # if sprite gets out of image in the bottom
        sprite = sprite[0 : imgH - y_offset, :, :]

    if x_offset + w >= imgW:  # if sprite gets out of image to the right
        sprite = sprite[:, 0 : imgW - x_offset, :]

    if x_offset < 0:  # if sprite gets out of image to the left
        sprite = sprite[:, abs(x_offset) : :, :]
        w = sprite.shape[1]
        x_offset = 0

    # for each RGB chanel
    for c in range(3):
        # chanel 4 is alpha: 255 is not transpartne, 0 is transparent background
        y_offset = int(y_offset)
        x_offset = int(x_offset)
        frame[y_offset : y_offset + h, x_offset : x_offset + w, c] = sprite[:, :, c] * (
            sprite[:, :, 3] / 255.0
        ) + frame[y_offset : y_offset + h, x_offset : x_offset + w, c] * (
            1.0 - sprite[:, :, 3] / 255.0
        )
    return frame






BODY_PARTS = { "Neck": 0, "RShoulder": 1, "LShoulder": 2, "RHip": 3}

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
inWidth = 250
inHeight = 250
#inScale = 0.1
#inWidth = 360
#inHeight = 360
inScale = 0.2


#net = cv.dnn.readNet('graph_opt.pb')
#protoFile = "./pose_deploy.prototxt"
protoFile = "./pose_deploy_linevec.prototxt"
weightsFile = "./pose_iter_440000.caffemodel"
#net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
net = cv.dnn.readNetFromTensorflow("./graph_opt.pb")
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 5)
(x, y, w, h) = (0, 0, 10, 10)  # whatever initial values

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break
    #frame = cv.resize(frame,(400,400))
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]



    inp = cv.dnn.blobFromImage(frame, inScale, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()

    assert(len(BODY_PARTS) <= out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]
        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > inScale else None)

    #(x, y, w, h) = (points[1][1], points[1][0], points[2]-, back.height())
	
    if points[1] and points[2] and points[3] is not None:

    	x = points[1][0]
    	y = points[1][1]
    	w = abs(points[1][0]-points[2][0])
    	h = abs(points[1][1]-points[3][1])

    	#shape = face_utils.shape_to_np(shape)

    	#incl = calculate_inclination(shape[17], shape[26]) 	


    	#apply_sprite(frame, "tattoo.png", w, x, y, incl)
    	apply_sprite(frame, "tattoo.png", w, x, y)

  
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))





    cv.imshow('OpenPose using OpenCV', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
quit()


