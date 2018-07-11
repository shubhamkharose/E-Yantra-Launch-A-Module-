import cv2
import math
import numpy as np
import serial
from matplotlib import pyplot as plt
import heapq
import time
import manager

'''
* Team Id : 1224
* Author List : Shubham Kharose,Hrushikesh Budhale,Nilesh Sutar,Prateek Pawar
* Filename: main.py
* Theme: Launch a Module
* Functions: getContours(),canny(),getBorder(),getBotPosition(),sendcommand(),creategraph(),modgraph(),addgraph(),picktoobjsrun(),addjustAngle(),shortpath(),furistic(),movebot(),distances(),rotateangle(),Preprocessor()
* Global Variables:cap,serial,flag,pts1
'''


#4ft -121.92 cm
#6ft -182.88 cm

cap=cv2.VideoCapture(0)
ser=serial.Serial('COM12',9600)

flag=0
pts1=[]
scale = 0

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

def pxTomm(img,mm):
    global scale
    r,c,ch = img.shape
    scale = r/1219.2

def getContours(img):
    '''
    * Function Name: getContours
    * Input: image
    * Output: contours and approximate contours
    * Example Call: getContours(image)
    '''
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    approx = []
    if not len(contours):
        return [],[]
    epsilon = 0.04*cv2.arcLength(contours[0],True)
    approx = cv2.approxPolyDP(contours[0],epsilon,True)
    return contours,approx


def canny(img):
    '''
    * Function Name: canny
    * Input: image
    * Output: Edges image
    * Example Call: canny(image)
    '''
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,38,157,apertureSize = 3)
    cv2.imshow('canny',edges)
    cv2.waitKey(0)
    return edges


def getBorder(img):
    '''
    * Function Name: getBorder
    * Input: image
    * Output: Inner border coordinates
    * Example Call: getBorder(img)
    '''
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
    empty_locations = []

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
    p1 = list(approx[0][0])
    p2 = list(approx[1][0])
    p3 = list(approx[2][0])
    p4 = list(approx[3][0])
    #print "Approx",approx
    pp = [p1,p2,p3,p4]
    nn = []
    for i in range(0,len(pp)):
        nn.append([sum(pp[i]),i])
    #print "sorted nn",sorted(nn)[0]
    nn = sorted(nn)
    pts1 = np.float32([pp[nn[0][1]], pp[nn[2][1]], pp[nn[1][1]], pp[nn[3][1]]])
    return pts1

def actualBotPosition(posBot,center):
    new_bot_pos = []
    flex_line_slope = 0
    botHeight = 120
    camHeight = 2270.0
    hori_distance = distances(posBot[0],posBot[1],center[0],center[1])
    error = botHeight*scale*hori_distance/(camHeight*scale)
    if posBot[0] - center[0]:
        flex_line_slope = (posBot[1] - center[1])/(posBot[0] - center[0])
    else:
        flex_line_slope = (posBot[1] - center[1])*99999999
    theta_flex = math.atan(flex_line_slope)

    new_bot_pos[0] = posBot[0] - error*math.cos(theta_flex)
    new_bot_pos[1] = posBot[1] - error*math.sin(theta_flex)

    return new_bot_pos

def getBotPosition(rv):
    '''
    * Function Name: getBotPosition
    * Input: integer variable
    * Output: Cordinates of various markers on bot
    * Example Call: getBotPosition(1)
    '''
    i=0
    print("In get Bot Position")
    while(i<2):
        ret,img = cap.read()
        i+=1
        time.sleep(0.25)

    global flag
    global pts1
    if(flag==0):
        #cv2.imshow("Imagess",img)
        pts1=getBorder(img)
        i=0
        while(i<2):
            ret,img = cap.read()
            i+=1
            time.sleep(0.25)
        
    flag=1
    rows, cols, ch = img.shape
    pts2 = np.float32([[0, 0], [615, 0], [0, 421], [615, 421]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (615, 421))


    
    hh1,ww1,c=img.shape
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hh,ww,cc=img.shape
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    HSVLOW=np.array([0,0,0])
    HSVHIGH=np.array([179,255,91])
    mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)

    kernel = np.ones((2 * 1 + 1, 2 * 1 + 1), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(dilation, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(erosion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    temparea = 0
    max = 0
    areas = []
    for index in range(0,len(contours)):
        epsilon = 0.01 * 5 * cv2.arcLength(contours[index], True)
        approx = cv2.approxPolyDP(contours[index], epsilon, True)
        area = cv2.contourArea(approx)
        areas.append((area,index))

    areas = sorted(areas,reverse=True)
   #print 'Areas', areas
    contours_of_interest= []
    for l in areas:
        if l[0] > 500 or 100<= l[0] <= 200:
            contours_of_interest.append(l)

    M1 = cv2.moments(contours[areas[0][1]])
    cx1 = int(M1['m10']/M1['m00'])
    cy1 = int(M1['m01']/M1['m00'])

    M2 = cv2.moments(contours[areas[1][1]])
    cx2 = int(M2['m10']/M2['m00'])
    cy2 = int(M2['m01']/M2['m00'])

    M3 = cv2.moments(contours[areas[2][1]])
    cx3 = int(M3['m10']/M3['m00'])
    cy3 = int(M3['m01']/M3['m00'])

    cv2.circle(img,(cx1,cy1),2,[0,0,255],2)
    cv2.circle(img,(cx2,cy2),2,[0,0,255],2)
    cv2.circle(img,(cx3,cy3),2,[0,0,255],2)
    x = (cx2 + cx3)/2
    y = (cy2 + cy3)/2
    cv2.circle(img,(x,y),2,[0,0,255],2)
    cv2.imshow('dhakd',img)
    print("enter")
    cv2.waitKey(0)
    if(rv==1):
        return ((x,y),(cx1,cy1),hh,ww,img)
    else:
        hh,ww,c=img.shape
        xindex=x*6/hh+1
        yindex=y*9/ww+1
        return ((xindex,yindex),hh,ww,img)


def sendCommand(a,no):
    '''
    * Function Name: sendCommand
    * Input: (type of command,data for command)
    * Output: None
    * Example Call: sendCommand(1,0)
    '''
    print('In command Send',a,no)
    s=(int) (a*1000+no)
    command = str(s)
    stringn = command.rstrip()
    command = command.rstrip()
    command = command.split(" ")

    command = [int(x) for x in command]
    for data in command:
        ser.write(stringn)
    ser.read(5)

    ser.flushInput()
    ser.flushOutput()


def creategraph(occupied_grids):
    '''
    * Function Name: creategraph
    * Input: Matrix containing all x-y co-ordinates of object cells
    * Output: Dictionary containing adjacent list representation of all cells in grid(graph)
    * Logic: Checking in four directions whether adjacent cell is occupied or not
    * Example Call: creategraph(img,Matrix)
    '''
    adjacent={}
    i=1
    while(i<=9):
        j=1
        while(j<=6):
                adjacent[(i,j)]=[]

                #Checking Adjacent cell in Left Direction
                if((i-1)>0 and(i-1,j) not in occupied_grids):
                        adjacent[(i,j)].append((i-1,j))
                #Checking Adjacent cell in Right Direction
                if((i+1)<=9 and(i+1,j) not in occupied_grids):
                        adjacent[(i,j)].append((i+1,j))
                #Checking Adjacent cell in Upward Direction
                if((j-1)>0 and(i,j-1) not in occupied_grids):
                        adjacent[(i,j)].append((i,j-1))
                #Checking Adjacent cell in Downward Direction
                if((j+1)<=6 and(i,j+1) not in occupied_grids):
                        adjacent[(i,j)].append((i,j+1))
                        
                j+=1
        i+=1
    '''for i in adjacent:
            print(i, adjacent[i])'''
    return adjacent

def modgraph(adjacent,i):
    '''
    * Function Name: modgraph
    * Input: Graph,grid coordinate
    * Output: None
    * Example Call: modgraph(adjacent,(1,2))
    '''
    #As we have picked object i and j modify graph edges by including i and j's grids
    a=i[0]
    b=i[1]
    if(a-1>=1):
        adjacent[(a-1,b)].append(i)
    if(b-1>=1):
        adjacent[(a,b-1)].append(i)
    if(a+1<=9):
        adjacent[(a+1,b)].append(i)
    if(b+1<=6):
        adjacent[(a,b+1)].append(i)

def addgraph(adjacent,i):
    '''
    * Function Name: addgraph
    * Input: Graph,grid coordinate
    * Output: None
    * Example Call: addgraph(adjacent,(1,2))
    '''
    #As we have picked object i and j modify graph edges by including i and j's grids
    a=i[0]
    b=i[1]
    if(a-1>=1):
        adjacent[(a-1,b)].remove(i)
    if(b-1>=1):
        adjacent[(a,b-1)].remove(i)
    if(a+1<=9):
        adjacent[(a+1,b)].remove(i)
    if(b+1<=6):
        adjacent[(a,b+1)].remove(i)

def pick2objsrun(adjacent,cur,object_grids,marker_grids,matches):
    '''
    * Function Name: pick2objsrun
    * Input: graph,current position,object position list,marker position list,matches object dictionary
    * Output: Next position of bot
    * Example Call: pick2objsrun(graph,(1,2),[],[],{})
    '''
    print('In pick and run')
    a=0
    b=0
    p=0
    q=0
    r=0
    s=0
    pmin=qmin=rmin=smin=0
    min1=1111111111
    sum1=0
    for i in object_grids:
        for j in object_grids:
            if(i!=j and i not in matches[j]):
                sum1=0
                p=(shortpath(adjacent,cur,i))
                modgraph(adjacent,i)
                q=(shortpath(adjacent,i,j))
                modgraph(adjacent,j)
                mj=mi=0
                for k in marker_grids:
                    if(k in matches[j]):
                        mj=k
                        break
                for k in marker_grids:
                    if(k in matches[i]):
                        mi=k
                        break
                r=(shortpath(adjacent,j,mj))
                s=(shortpath(adjacent,mj,mi))
                if(len(p)==0 or len(q)==0 or len(r)==0 or len(s)==0):
                    continue
                sum1=len(p)+len(q)+len(r)+len(s)

                if(sum1<min1):
                    min1=sum1
                    a=i
                    b=j
                    ma=mi
                    mb=mj
                    pmin=p
                    qmin=q
                    rmin=r
                    smin=s
                addgraph(adjacent,i)
                addgraph(adjacent,j)
    print pmin
    print qmin
    print rmin
    print smin

    print("MOVING BOT")
    movebot(cur,pmin)
    adjustAngle(a)
    sendCommand(1,000)    #Pick

    movebot(pmin[len(pmin)-1],qmin)
    adjustAngle(b)
    sendCommand(1,000)    #Pick

    movebot(qmin[len(qmin)-1],rmin)
    adjustAngle(mb)
    sendCommand(2,000)    #Place

    movebot(rmin[len(rmin)-1],smin)
    adjustAngle(ma)
    sendCommand(2,000)    #Place
    
    modgraph(adjacent,a)
    modgraph(adjacent,b)
    object_grids.remove(a)
    object_grids.remove(b)
    marker_grids.remove(ma)
    marker_grids.remove(mb)

    
    for k in object_grids:
        f1=1
        if(a!=k and a in matches[k]):
            f1=0
            for ll in marker_grids:
                if(ll in matches[k]):
                    f1=1
                    break
        if(f1==0):
            object_grids.remove(k)
    
    for k in object_grids:
        f1=1
        if(b!=k and b in matches[k]):
            f1=0
            for ll in marker_grids:
                if(ll in matches[k]):
                    f1=1
                    break
        if(f1==0):
            object_grids.remove(k)
    
    #occupied_grids.remove(i)
    #occupied_grids.remove(j)
    return smin[len(smin)-1]

def adjustAngle(i):
    '''
    * Function Name: adjustAngle
    * Input: bot coordinates
    * Output: None
    * Example Call: adjustAngle((1,2))
    '''
    print('IN ADJUST ANGLE')
    while(1):
        botb,botf,hh,ww,img=getBotPosition(1) #botb and botf in form of pixels
        j=[]
        if(i==0):
            return

        j.append((i[0]-0.5)*(ww/9))
        j.append((i[1]-0.5)*(hh/6))
        ang=rotateangle(botb,botf,j)
        if(ang-5<=5 and ang+5>=-5):
            break
        if(ang>0):
            sendCommand(4,ang-5)
        else:
            ang=-1*ang
            sendCommand(5,ang+5)


def shortpath(adjacent,src,desti):
    '''
    * Function Name: shortpath(adjacent,a,b)
    * Input: Dictionary adjacent containining ajacent list like representation,source and destination
    * Output: List containing destination,shortest path and length to destination
    * Logic: A-Star algorithm by implementing Priority queue and heap
    * Example Call: shortpath(adjacent,a,b)
    '''
    print('In ShortPath')
    #outlist:List containing shortest path
    outlist=[]
    frontier=PriorityQueue()
    #put source in priority queue
    frontier.put(src,0)
    came_from ={}
    cost_so_far ={}
    came_from[src] = None
    cost_so_far[src] = 0
    while not frontier.empty():
       #if priority queue is not empty,get element from priority queue as current
       current=frontier.get()
       #if current is adjacent to desination
       if (desti==0 or current in adjacent[desti]):
           break
    
       for next in adjacent[current]:
           #for all neighbouring cells of current setting new_cost
           new_cost = cost_so_far[current] + 1
           if next not in came_from:
               cost_so_far[next] = new_cost
               #set priority by new_cost and heuristic value
               priority = new_cost+heuristic(next,desti)
               frontier.put(next, priority)
               came_from[next] = current

    #print(current,desti)
    if(desti==0 or current not in adjacent[desti]):
        #if path is not present
        return []
    else:
        lst = []
        while(1):
            lst.append(current)
            if(current==src):
                break
            current = came_from[current]
        lst.reverse()
        #print(lst)
        #print(src,desti,outlist)
        return lst

def heuristic(cur,desti):
   # Manhattan distance on a square grid
   return abs(cur[0]-desti[0])+abs(cur[1]-desti[1])



def movebot(cur,pathlist):
    '''
    * Function Name: movebot
    * Input: current path,Path list
    * Output: None
    * Example Call: movebot((1,2),[])
    '''
    print('MoveBot',cur,pathlist)
    flg=0
    if(pathlist==0):
        return
    for i in pathlist:
        if(i[0]!=1):
            while(1):
                    botb,botf,hh,ww,img=getBotPosition(1) #botb and botf in form of pixels
                    j=[]
                    print(i)
                    if(i==0):
                        return
                    
                    j.append((int)((i[0]-0.5)*(ww/9)))
                    j.append((int)((i[1]-0.5)*(hh/6)))
                    cv2.circle(img,(j[0],j[1]),2,[0,255,0],2)
                    
                    cv2.circle(img,(botb[0],botb[1]),2,[0,255,0],2)
                    
                    cv2.circle(img,(botf[0],botf[1]),2,[0,255,0],2)
                    cv2.imshow('DESTINATION',img)
                    dx=abs(j[0]-botb[0])
                    dy=abs(j[1]-botb[1])
                    if(dx>dy):
                        maxd=dx
                    else:
                        maxd=dy
                        
                    dist = maxd*1828.8/ww
                    
                    if(dist<=50):
                        break
                    
                    ang=rotateangle(botb,botf,j)
                    if(ang-5<=10 and ang+5>=-10):
                        break
                    if(ang>0):
                        sendCommand(4,ang-5)
                    else:
                        ang=-1*ang
                        sendCommand(5,ang+5)
                    

            while(1):
                botb,botf,hh,ww,img=getBotPosition(1) #botb and botf in form of pixels
                j=[]
                j.append((int)((i[0]-0.5)*(ww/9)))
                j.append((int)((i[1]-0.5)*(hh/6)))
                dx=abs(j[0]-botb[0])
                dy=abs(j[1]-botb[1])
                cv2.circle(img,(j[0],j[1]),2,[0,255,0],2)

                cv2.circle(img,(botb[0],botb[1]),2,[0,255,0],2)

                cv2.circle(img,(botf[0],botf[1]),2,[0,255,0],2)
                cv2.imshow('DESTINATION',img)
                if(dx>dy):
                    maxd=dx
                else:
                    maxd=dy
                hh,ww,cc=img.shape
                dist=maxd*1828.8/ww
                if(i[0]==1 or i[0]==9 or i[1]==1 or i[1]==6):
                    dist-=50
                if(dist<=50):
                    break

                sendCommand(3,dist)




def distances(x1,y1,x2,y2):
    '''
    * Function Name: distances()
    * Input: Coordinates of points
    * Output: distance between points
    * Example Call: distances(1,2,3,4)
    '''
    return math.sqrt(pow((x2-x1),2)+pow((y2-y1),2))
    
def rotateangle(botb,botf,nextp):
    '''
    * Function Name: rotateangle()
    * Input: marker positions,nextp = Next coordinates on path
    * Output: Angle
    * Example Call: rotateangle((1,2),(2,2),(3,2))
    '''
    cx1=botb[0]
    cy1=botb[1]
    cx2=botf[0]
    cy2=botf[1]
    cx3=nextp[0]
    cy3=nextp[1]
    print('Rotate Angle')
   
    if ((cx2 - cx1) == 0):
        slope1 = 9999999
        if(cy2>cy1):
            slope1=-1*slope1
            
    else:
        slope1 = math.fabs((cy2 - cy1)/float(cx2 - cx1))

    theta1 = math.atan(slope1)*180/math.pi

    if(cx2>cx1 and cy2>cy1):
        theta1=-1*theta1

    if(cx2<cx1 and cy2>cy1):
        theta1=-1*(180-theta1)

    if(cx2<cx1 and cy2<cy1):
        theta1=-1*(180+theta1)

    print (theta1)



    if (cx3 - cx1) == 0:
        slope2 = 9999999
        if(cy3>cy1):
            slope2=-1*slope2
            
    else:
        slope2 = math.fabs((cy3 - cy1)/float(cx3 - cx1))
        
    theta2 = math.atan(slope2)*180/math.pi

    if(cx3>cx1 and cy3>cy1):
        theta2=-1*theta2

    if(cx3<cx1 and cy3>cy1):
        theta2=-1*(180-theta2)

    if(cx3<cx1 and cy3<cy1):
        theta2=-1*(180+theta2)

    print (theta2)

        
    fangel=theta2-theta1

    if(((fangel>=-10)and(fangel<=10))or((abs(fangel-180)>=-10)and(abs(fangel-180)<=10))or((abs(fangel+180)>=-10)and(abs(fangel+180)<=10))):
        sdd1 = distances(cx2,cy2,cx1,cy1)
        sdd2 = distances(cx2,cy2,cx3,cy3)
        sdd3 = distances(cx1,cy1,cx3,cy3)
        sdd = sdd1+sdd2 - sdd3
        print sdd
        if(sdd<=5 and sdd>=-5):
            fangel=0
        else:
            fangel=-180
        
    if(fangel<-180):
        fangel=360+fangel
    elif(fangel>180):
        fangel=-1*(360-fangel)
    print(fangel)
    '''
        Negative Angle Right direction
        Positive Angle Left direction
    '''
    return fangel

###########################################################################

def Preprocessor(img):
    '''
    * Function Name: Preprocessor
    * Input: Image
    * Output: marker position list,object position marker,obstacle position list,match dictionary
    * Logic: Features detection using color masking and Matching of markers and objects by comparing various features extracted
    * Example Call: rotateangle((1,2),(2,2),(3,2))
    '''
    #marker_grids,object_grids,obstacles,matches = manager.Preprocessor(img)
    marker_grids=[(1,1)
        ,(1,2),(1,3),(1,4),(1,5),(1,6)]
    object_grids=[(2,6),(3,3),(4,4),(4,5),(8,1),(9,6),(4,1)]
    obstacles=[(3,2),(5,4),(9,5),(6,5),(6,6)]
    matches={}
    matches[(2,6)]=[(1,6)]
    matches[(3,3)]=[(1,1),(1,4),(8,1)]
    matches[(4,4)]=[(1,5)]
    matches[(4,5)]=[(4,1),(1,2)]
    matches[(8,1)]=[(1,1),(1,4),(3,3)]
    matches[(9,6)]=[(1,3)]
    matches[(4,1)]=[(4,5),(1,2)]
    matches[(1,5)]=[(4,4)]
    matches[(1,1)]=[(3,3),(1,4),(8,1)]
    matches[(1,4)]=[(1,1),(8,1),(3,3)]
    matches[(1,2)]=[(4,5),(4,1)]
    matches[(1,3)]=[(9,6)]
    matches[(1,6)]=[(2,6)]
    #return marker_grids,obstacles,matches,object_grids
    return marker_grids,obstacles,matches,object_grids


i = 0
while(i<2):
        ret,img = cap.read()
        i+=1
        time.sleep(0.5)
_,frame = cap.read()

marker_grids,obstacles,matches,object_grids=Preprocessor(frame)
adjacent=creategraph(object_grids+marker_grids+obstacles)

'''for j in adjacent:
    print(j,adjacent[j])'''
while(len(object_grids)>0):
    cur,hh,ww,img=getBotPosition(0)
    print(cur)
    if(len(object_grids)>1):
        cur=pick2objsrun(adjacent,cur,object_grids,marker_grids,matches)
cap.release()
cv2.destroyAllWindows()














