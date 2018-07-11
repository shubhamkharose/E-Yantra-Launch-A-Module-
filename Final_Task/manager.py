import cv2, math
import numpy as np


def uniform_split(img):
    '''
    * Function Name: uniform_split(Image)
    * Input: ROI image
    * Output: Dictionary of cropped images,Dictionary (Key :grid cordinates,Value : absolute pixel cordinate)
    * Logic: Cropped image using numpy indexing.
    * Example Call: uniform_split(img)
    '''
    abs_cords = {}
    split = []
    cropped_imagess = {}
    height, width = dst.shape[:2]
    gridXdist = int(width / 9.0)
    gridYdist = int(height / 6.0)
    for yaxis in range(0, 6):
        for xaxis in range(0, 9):
            box = [xaxis * gridXdist, yaxis * gridYdist, (xaxis + 1) * gridXdist, (yaxis + 1) * gridYdist]
            crop = img[box[1] + 1:box[3] - 1, box[0] + 1:box[2] - 1]
            split.append(crop)
            cropped_imagess[(xaxis + 1, yaxis + 1)] = crop
            abs_cords[(xaxis + 1, yaxis + 1)] = (box[0] + gridXdist / 2, box[1] + gridYdist / 2)
            cv2.imshow("cropped", crop)
            cv2.waitKey(0)
    # print abs_cords
    return cropped_imagess, abs_cords


def filterLines(lines):
    '''
    * Function Name: filterLines(lines)
    * Input: Lines list from Hough lines function
    * Output: List of Filtered horizontal lines,List of filtered vertical lines
    * Logic: Filtering based on thresholding lines based on angle and and distance between lines
    * Example Call: filterLines(lines)
    '''
    distinct_lines = []
    vertical_lines = []
    horizontal_lines = []
    m_v = []
    m_h = []

    for rho, theta in lines[0]:
        if abs(math.cos(theta)) <= 0.001:
            if rho >= 0:
                horizontal_lines.append(rho)
        if abs(math.cos(theta)) >= 0.998:
            if rho >= 0:
                vertical_lines.append(rho)

    vertical_lines = sorted(vertical_lines)
    horizontal_lines = sorted(horizontal_lines)
    last_v = 0
    last_h = 0
    i = 0
    while True:
        i = 0
        while True:
            if i + 1 >= len(vertical_lines):
                try:
                    m_v.append(vertical_lines[i])
                except Exception, e:
                    pass
                    # print e
                break
            if vertical_lines[i + 1] - vertical_lines[i] <= 40:
                var = (vertical_lines[i + 1] + vertical_lines[i]) / 2
                m_v.append(var)
                i += 2
            else:
                m_v.append(vertical_lines[i])
                i += 1
        vertical_lines = m_v
        if len(m_v) == last_v:
            break
        else:
            last_v = len(m_v)
        m_v = []

    while True:
        i = 0
        while True:
            if i + 1 >= len(horizontal_lines):
                try:
                    m_h.append(horizontal_lines[i])
                except Exception, e:
                    # print e
                    pass
                break
            if horizontal_lines[i + 1] - horizontal_lines[i] <= 40:
                var = (horizontal_lines[i + 1] + horizontal_lines[i]) / 2
                m_h.append(var)
                i += 2
            else:
                m_h.append(horizontal_lines[i])
                i = i + 1
        horizontal_lines = m_h

        if len(m_h) == last_h:
            break
        else:
            last_h = len(m_h)
        m_h = []
    return m_h, m_v




# Subgriding 3 times

# h,v = subgrid(horizontals,verticals)

# h,v = subgrid(h,v)

# h,v = subgrid(h,v)


def image_to_boxes(horis, vers):
    '''
    * Function Name: image_to_boxes(horis,vers)
    * Input: Horizontal lines list,vertical lines list
    * Output: Dictionary of cropped images,Dictionary (Key :grid cordinates,Value : absolute pixel cordinate),list of rectangular coordintes of grid box
    * Logic: Cropped image using numpy indexing.
    * Example Call: image_to_boxes
    '''
    # cv2.imshow('original',dst )
    px_cords = {}  # Absolute pixel coordinates
    cropped_img = {}  # Contains cropped images and grid position
    box = {}  # Contains absolute coordinates of cropped box
    for i in range(0, len(horis)):
        for j in range(0, len(vers)):
            if (i + 1) == len(horis) or (j + 1) == len(vers):
                break
            cropped_img[(j + 1, i + 1)] = dst[horis[i]:horis[i + 1], vers[j]:vers[j + 1]]
            px_cords[(j + 1, i + 1)] = ((vers[j] + vers[j + 1]) / 2, (horis[i] + horis[i + 1]) / 2)
            box[(j + 1, i + 1)] = [(vers[j], horis[i], vers[j + 1], horis[i + 1])]
            cv2.imwrite('images\\cropped arena\\' + str(j + 1) + str(i + 1) + '.jpg', cropped_img[(j + 1, i + 1)])
            # cv2.circle(dst,px_cords[(j+1,i+1)],4,[0,0,255],-1)
            # cv2.imshow('circles',cropped_img[(j+1,i+1)])
            # cv2.waitKey(0)
    return cropped_img, px_cords, box




def isEmpty(img):
    '''
    * Function Name: isEmpty(img)
    * Input: image to check its mean hsv values
    * Output: Mean of hsv values, Mean for red color,Mean for green color,mean for blue color
    * Logic: Compute hsv mask for red blue and green color and calculate mean's.
    * Example Call: isEmpty(img)
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    HSVLOWR = np.array([157, 43, 100])
    HSVHIGHR = np.array([179, 200, 203])

    HSVLOWG = np.array([45, 77, 1])
    HSVHIGHG = np.array([108, 255, 140])

    HSVLOWB = np.array([89, 43, 127])
    HSVHIGHB = np.array([151, 255, 255])

    maskr = cv2.inRange(hsv, HSVLOWR, HSVHIGHR)
    maskg = cv2.inRange(hsv, HSVLOWG, HSVHIGHG)
    maskb = cv2.inRange(hsv, HSVLOWB, HSVHIGHB)

    # cv2.imshow('redmask',maskr)
    # cv2.imshow('greenmask',maskg)
    # cv2.imshow('bluemask',maskb)
    # cv2.waitKey(0)
    # print 'Mask mean '
    # print maskr.mean()
    # print maskg.mean()
    # print maskb.mean()
    return int(maskr.mean() + maskg.mean() + maskb.mean()), maskr.mean(), maskg.mean(), maskb.mean()


def filterDups(arr):
    '''
    * Function Name: filterDups(array)
    * Input: List of contours
    * Output: Filtered contours
    * Logic: Filtering based on area
    * Example Call: uniform_split(img)
    '''
    temp = [arr[0]]
    check = 0
    for i in range(0, len(arr)):
        index = arr[i][1]
        check = 0
        for j in range(0, len(temp)):
            if temp[j][1] == index:
                check = 1
            if not check:
                temp.append(arr[i])
    return temp


def filterContours(arr, minArea):
    '''
    * Function Name: filterContours(arr,minArea)
    * Input: list of contours,Threshold area
    * Output: List of filtered contours
    * Logic: Thresholding based on AREA
    * Example Call: filterContours([..,..,],54)
    '''
    temp = []
    for i in range(0, len(arr)):
        area = arr[i][0]
        if area >= minArea:
            temp.append(arr[i])
    return temp


def distance(pointA, pointB):
    # print "Point in distance",pointA,pointB[0]
    return math.sqrt(math.pow(pointA[0] - int(pointB[0]), 2) + math.pow(pointA[1] - int(pointB[1]), 2))


def blackBorder(img):
    '''
    * Function Name: blackBorder(img)
    * Input: image
    * Output: Image with black border
    * Logic: Corrects some errors that we dont know why they come
    * Example Call: blackBorder(img)
    '''
    rows, cols = img.shape
    for i in range(0, rows):
        img[i][0] = 0
        img[i][1] = 0
        img[i][2] = 0
        img[i][-1] = 0
        img[i][-2] = 0
        img[i][-3] = 0
    for i in range(0, cols):
        img[0][i] = 0
        img[1][i] = 0
        img[2][i] = 0
        img[-1][i] = 0
        img[-2][i] = 0
        img[-3][i] = 0
    return img


def isCircle(center, points):
    '''
    * Function Name: isCircle(center,contour points)
    * Input: center of circle,contour points
    * Output: Boolean value
    * Logic: Calculates error in calculated average radius and radius calculated from cv2.minEnclosingcircle() function
    * Example Call: filterContours([..,..,],54)
    '''
    (x, y), radius = cv2.minEnclosingCircle(points)
    centera = (int(x), int(y))
    radius = int(radius)
    # print "Enclosing circle radius",radius
    dist = 0
    av_radius = 0
    count = 0
    score = 0
    error = 0
    temp_points = []
    new_error = 0
    # print "Points in isCircle",points[0][0][0]
    area_of_enc_circle = math.pi * radius * radius
    area_of_contour = cv2.contourArea(points)
    # print 'Area of enclosing circle',area_of_enc_circle
    # print 'Area of Contour',area_of_contour
    error_in_area = (area_of_enc_circle - area_of_contour) / area_of_contour * 100.0
    # print 'Error in Area',abs(error_in_area)
    for i in range(0, len(points)):
        temp_points.append([int(points[i][0][0]), int(points[i][0][1])])

    for i in range(0, len(points)):
        dist += distance(center, points[i][0])
        count += 1
        # print list(points[i][0])
        # temp_points.remove(list(points[i][0]))

    av_radius = dist / float(count)
    # print 'Average radius of circle:',av_radius
    for i in range(0, len(temp_points)):
        error += math.pow(av_radius - distance(center, temp_points[i]), 2)
    for i in range(0, len(temp_points)):
        new_error += math.pow(av_radius - distance(centera, temp_points[i]), 2)
    av_error = error / float(len(points))
    av_error_on_acenter = new_error / float(len(points))
    print 'Average Error based on ACtual circle', av_error_on_acenter
    print "Average Error in circle", av_error
    percent_error = (av_radius - radius) / float(radius) * 100.0
    print "Percentage error in Radius: ", abs(percent_error)
    if av_error <= 0.9:
        if abs(percent_error) < 12:
            return True
        else:
            return False
    else:
        return False


def getPropertiesM(img):
    '''
    * Function Name: getPropertiesM(img)
    * Input: Image
    * Output: [color, shape, area, perimeter]
    * Logic: Masks image then calculates shape using approxPolyDp() function
    * Example Call: getPropertiesM(img)
    '''
    areas = []
    shape = 0
    approx_cnt = {}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tm, cr, cg, cb = isEmpty(img)
    color = [[cr, 'red'], [cg, 'green'], [cb, 'blue']]
    color = sorted(color, reverse=True)
    HSVLOW = np.array([0, 63, 3])
    HSVHIGH = np.array([104, 255, 255])
    color_v = color[0][1]
    print "Color", color_v
    if color_v == 'red':
        HSVLOW = np.array([0, 87, 132])
        HSVHIGH = np.array([179, 182, 255])
    elif color_v == 'green':
        HSVLOW = np.array([78, 41, 13])
        HSVHIGH = np.array([112, 255, 160])
    elif color_v == 'blue':
        HSVLOW = np.array([23, 64, 82])
        HSVHIGH = np.array([168, 255, 253])

    # apply the range on a mask
    mask = cv2.inRange(hsv, HSVLOW, HSVHIGH)
    # corners =

    # cv2.waitKey(0)
    # res = cv2.bitwise_and(hsv,hsv, mask =mask)
    # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(mask, 158, 378)
    # cv2.imshow('Edges',edges)
    # cv2.waitKey(0)
    # edges = blackBorder(edges)
    kernel = np.ones((2 * 1 + 1, 2 * 1 + 1), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    erode = cv2.erode(dilation, kernel, iterations=1)
    cv2.imshow('mask', erode)
    cv2.imshow('original', img)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in range(0, len(contours)):
        epsilon = 0.01 * 12 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        area = cv2.contourArea(approx)
        areas.append((area, i))
        approx_cnt[i] = approx
    print "Areas List", areas
    areas = sorted(areas, reverse=True)
    areas = filterDups(areas)
    areas = filterContours(areas, 20)
    M = cv2.moments(contours[areas[0][1]])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    area = areas[0][0]
    corners = approx_cnt[areas[0][1]]
    perimeter = cv2.arcLength(approx_cnt[areas[0][1]], True)
    cv2.drawContours(img, [corners], 0, 255, 2)
    # cv2.imshow('img',img)

    if len(corners) == 3:
        shape = 'triangle'

    if len(corners) >= 4:
        if isCircle((cx, cy), contours[areas[0][1]]):
            shape = 'circle'
        else:
            shape = 'square'

    print([color[0][1], shape, area, perimeter])
    cv2.waitKey(0)
    return [color[0][1], shape, area, perimeter]



def error(a, b):
    '''
    * Function Name: error(a,b)
    * Input: interger values
    * Output: Percentage error
    * Example Call: error(5,6)
    '''
    return abs(100.0 * (b - a) / (a + 0.01))


def Preprocessor(img):
    '''
    * Function Name: Preprocessor(img)
    * Input: Image
    * Output: m, obj, obstacles, new_match
    * Logic: Calculates roi then provides matches using features extracted from above function.
    * Example Call: Preprocessor(img)
    '''
    global dst
    botposition = [5,3]
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dst = cv2.bilateralFilter(gray, 2 * 11 + 1, 75, 75)

    gblur = cv2.GaussianBlur(dst, (2 * 5 + 1, 2 * 5 + 1), 0)

    th3 = cv2.adaptiveThreshold(gblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 2 * 9 + 1, 2)

    edges = cv2.Canny(th3, 400, 450)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_val = 0
    max_index = 0
    s_max = 0
    s_index = 0
    areas = []
    empty_locations = [(5,3)]

    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        areas.append((area, i))
        if area >= max_val:
            s_max = max_val
            s_index = max_index
            max_val = area
            max_index = i

    areas = sorted(areas, reverse=True)

    areas = [areas[x] for x in range(0, 20)]
    # print areas
    for i in range(0, len(areas)):
        error = 100.0 * (max_val - areas[i][0]) / max_val
        if 3 <= error <= 10:
            max_index = areas[i][1]
            break

    # print max_val,max_index,s_max,s_index
    epsilon = 0.001 * 60 * cv2.arcLength(contours[max_index], True)
    approx = cv2.approxPolyDP(contours[max_index], epsilon, True)

    ps1 = np.float32([approx[0][0], approx[3][0], approx[1][0], approx[2][0]])
    # print 'approx', approx
    rows, cols, ch = img.shape
    # pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts1 = np.float32([approx[1][0], approx[0][0], approx[2][0], approx[3][0]])
    pts2 = np.float32([[0, 0], [615, 0], [0, 421], [615, 421]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (615, 421))
    cv2.imwrite('images\\results\\ROI.png', dst)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    r, c, chans = dst.shape

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)
    kernel = np.ones((2, 2), np.uint8)
    erode = cv2.erode(threshold_img, kernel, iterations=1)
    can = cv2.Canny(erode, 100, 200)
    lines = cv2.HoughLines(can, 1, np.pi / 180, 120)
    LOI = []

    mh, mv = filterLines(lines)
    horizontals = [int(x) for x in mh]
    verticals = [int(x) for x in mv]

    # ---------------------- Minor Corrections -----------------------#
    horizontals[0] = 1
    horizontals[-1] = r - 1
    verticals[0] = 1
    verticals[-1] = c - 1

    if len(horizontals) != 7 or len(verticals) != 10:
        print "Unable to detect flex lines..."
        print 'Spliting uniformly'
        gridx = max(r, c) / 9
        gridy = min(r, c) / 6
        horizontals = [i * gridy for i in range(0, 7)]
        verticals = [i * gridx for i in range(0, 10)]

    cropped, pxc, box = image_to_boxes(horizontals, verticals)

    occupied_locations = []
    for location in cropped:
        if location == (5,3):
            continue
        # print "Location Checking",location
        mean, r, g, b = isEmpty(cropped[location])
        if mean <= 2.5:
            empty_locations.append(location)
        else:
            occupied_locations.append(location)

    for rho in horizontals:
        theta = 1.57
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for rho in verticals:
        theta = 0.00
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)

    obj_properties = {}
    for location in occupied_locations:
        print 'Location Processing', location
        obj_properties[location] = getPropertiesM(cropped[location])
        # print obj_properties[location]

    matches = {}
    weights = [4, 4, 5, 5, 7, 8, 1]

    cv2.imwrite('images\\Results\\grid.jpg', dst)
    # cv2.imshow('ne',dst)
    # cv2.waitKey(0)
    objects = {}
    obstacles = []
    markers = {}
    m = markers.keys()
    obj = objects.keys()

    for location in obj_properties:
        if obj_properties[location][2] > 1200:
            obstacles.append(location)
        else:
            if location[0] == 1:
                markers[location] = obj_properties[location]
            else:
                objects[location] = obj_properties[location]

    print "Markers"
    print markers
    print "Objects"
    print objects
    print "Obstacles"
    print obstacles
    for location in occupied_locations:
        matches[location] = []
        total_error = 0.0
        min_error = 9999999999
        match_location = (0, 0)
        for new_location in occupied_locations:
            total_error = 0
            if new_location != location:
                if obj_properties[location][0] == obj_properties[new_location][0]:
                    if obj_properties[location][1] == obj_properties[new_location][1]:
                        area_error = (obj_properties[location][2] - obj_properties[new_location][2]) / min(
                            obj_properties[location][2], obj_properties[new_location][2]) * 100.0
                        if abs(area_error) <= 20:
                            matches[location].append(new_location)
                        else:
                            obstacles.append(location)
                    else:
                        obstacles.append(location)
                else:
                    obstacles.append(location)
    new_match = {}
    for location in matches:
        if len(matches[location]) != 0:
            new_match[location] = matches[location]


    return m, obj, obstacles, new_match



'''
while True:

    cap = cv2.VideoCapture(1)
    _,frame = cap.read()
    Preprocessor(frame)

    '''''
