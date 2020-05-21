import cv2

src_img = cv2.imread('myAvatar.png')
points = []
def mouse_edit(event, x, y, flags, param):
    global points,img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
        print(len(points))


cv2.namedWindow("edit")

cv2.setMouseCallback("edit", mouse_edit)

started = True
while started:
    img = src_img.copy()
    for p in points:
        img = cv2.circle(img, (p[0], p[1]), 1, (0, 255, 0), 1)

    cv2.imshow('edit',img)
    key = cv2.waitKey(1)
    if key in [ord('q'), 202, 27]:
        started = False
        break
    if key in [ord('d')]:
        if len(points)>0:
            points = points[:len(points)-1]


print("-----------------------")
print(points)