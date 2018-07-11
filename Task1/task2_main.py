# -*- coding: utf-8 -*-
'''
**************************************************************************
*                  IMAGE PROCESSING (e-Yantra 2016)
*                  ================================
*  This software is intended to teach image processing concepts
*  
*  Author: e-Yantra Project, Department of Computer Science
*  and Engineering, Indian Institute of Technology Bombay.
*  
*  Software released under Creative Commons CC BY-NC-SA
*
*  For legal information refer to:
*        http://creativecommons.org/licenses/by-nc-sa/4.0/legalcode 
*     
*
*  This software is made available on an “AS IS WHERE IS BASIS”. 
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*  
*  e-Yantra - An MHRD project under National Mission on Education using 
*  ICT(NMEICT)
*
* ---------------------------------------------------
*  Theme: Launch a Module
*  Filename: task2_main.py
*  Version: 1.0.0  
*  Date: November 28, 2016
*  How to run this file: python task2_main.py
*  Author: e-Yantra Project, Department of Computer Science and Engineering, Indian Institute of Technology Bombay.
* ---------------------------------------------------

* ====================== GENERAL Instruction =======================
* 1. Check for "DO NOT EDIT" tags - make sure you do not change function name of main().
* 2. Return should be a list named occupied_grids and a dictionary named planned_path.
* 3. Do not keep uncessary print statement, imshow() functions in final submission that you submit
* 4. Do not change the file name
* 5. Your Program will be tested through code test suite designed and graded based on number of test cases passed 
**************************************************************************
'''

import cv2
import numpy as np

# ******* WRITE YOUR FUNCTION, VARIABLES etc HERE



'''
* Team Id : 1224
* Author List : Shubham Kharose,Hrushikesh Budhale,Nilesh Sutar,Prateek Pawar
* Filename: task2_main.py
* Theme: Launch a Module
* Functions: detect(),obstagrid(),occgrid(),creategraph(),update(),planpath(),shortpath(),heuristic(),main()
* Global Variables:none
'''


'''Using A-star Algorithm for finding Shortest Path'''


import heapq

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def detect(c):
    '''
    * Function Name: detect
    * Input: contour
    * Output: shape of contour
    * Example Call: detect(contours[0])
    '''
    #shape: shape of contour
    #peri: perimeter of contour
    #area: area of contour
    
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        shape = "4-sided"
    else:
        shape = "Circle"
    return shape


def obstagrid(img):
    '''
    * Function Name: obstagrid
    * Input: image
    * Output: List containing x-y co-ordinates of obstacles in grid
    * Logic: By detecting obstacles by BLACK colour
    * Example Call: obstagrid(image_name)
    '''
    #hh: height of image
    #ww: width of image
    #obstacle_grid: List containing x-y co-ordinates of obstacles in grid
    
    obstacle_grids = []
    hh,ww,cc=img.shape
    j=1
    while(j<20):
        i=1
        while(i<20):
            color=img[i*hh/20,j*ww/20]
            #detecting obstacles by BLACK colour
            if(color[0]<=5 and color[1]<=5 and color[2]<=5):
                    obstacle_grids.append(((j+1)/2,(i+1)/2))   
            i+=2
        j+=2
    return obstacle_grids

        
def occgrid(img,Matrix):
    '''
    * Function Name: occgrid
    * Input: Image and Matrix containing all x-y co-ordinates,shape,area,perimeter of objects
    * Output: List containing x-y co-ordinates of all occupied cells in grid 
    * Logic: By checking for obstacles and objects seperately
    * Example Call: occgrid(img,Matrix)
    '''
    #occupied_grids: List containing x-y co-ordinates of all occupied cells in grid
    #Matrix: Dictionary containing position,shape,colour,perimeter,area of objects
    
    occupied_grids=obstagrid(img)
    for i in Matrix:
        occupied_grids.append(i)
    #sorting the list
    occupied_grids.sort()
    return occupied_grids
    

def creategraph(img,Matrix):
    '''
    * Function Name: creategraph
    * Input: Image and Matrix containing all x-y co-ordinates of object cells
    * Output: Dictionary containing adjacent list representation of all cells in grid(graph)
    * Logic: Checking in four directions whether adjacent cell is occupied or not
    * Example Call: creategraph(img,Matrix)
    '''
    #adjacent: Dictionary containing adjacent list representation of all cells in grid(graph)
    #occupied_grids: List containing x-y co-ordinates of all occupied cells in grid
    #Matrix: Dictionary containing position,shape,colour,perimeter,area of objects
   
    adjacent={}
    occupied_grids = occgrid(img,Matrix)
    i=1
    while(i<=10):
        j=1
        while(j<=10):
                adjacent[(i,j)]=[]

                #Checking Adjacent cell in Left Direction
                if((i-1)>0 and(i-1,j) not in occupied_grids):
                        adjacent[(i,j)].append((i-1,j))
                #Checking Adjacent cell in Right Direction
                if((i+1)<=10 and(i+1,j) not in occupied_grids):
                        adjacent[(i,j)].append((i+1,j))
                #Checking Adjacent cell in Upward Direction
                if((j-1)>0 and(i,j-1) not in occupied_grids):
                        adjacent[(i,j)].append((i,j-1))
                #Checking Adjacent cell in Downward Direction
                if((j+1)<=10 and(i,j+1) not in occupied_grids):
                        adjacent[(i,j)].append((i,j+1))
                        
                j+=1
        i+=1
    '''for i in adjacent:
            print(i, adjacent[i])'''
    return adjacent


        
def update(img):
    '''
    * Function Name: update
    * Input: Image
    * Output: Matrix(Dictionary) containing position,shape,colour,perimeter,area of objects
    * Logic: First Masking is done for finding objects colourwise, After thresholding shape,area and perimeter of contour is found out
    * Example Call: update(image_name)
    '''
    #levels: list containing range of BGR values for different colours
    #hh: height of image
    #ww: width of image
    #Matrix: Dictionary containing position,shape,colour,perimeter,area of objects
   
    levels= [np.array([110, 50, 50], dtype=np.uint8),np.array([130,255,255], dtype=np.uint8),np.array([50, 100, 100], dtype=np.uint8),np.array([70, 255, 255], dtype=np.uint8),np.array([0,50,50], dtype=np.uint8),np.array([10,255,255], dtype=np.uint8),np.array([30,0,0], dtype=np.uint8),np.array([40,255,255], dtype=np.uint8)]
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hh,ww,cc=img.shape
    Matrix= {}
    
    i=0
    while(i<6):
            #Masking image for particular colour
            mask  = cv2.inRange(hsv,levels[i],levels[i+1])
            res   = cv2.bitwise_and(img, img, mask= mask)

            #Applying Filter
            blur = cv2.medianBlur(res,5)

            #Converting image from BGR to GRAY
            gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
            #Thresholding the image
            ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
            #Finding Contours
            contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img,contours,-1,(0,255,0),3)
            l=len(contours)
            #print(l)
            for j in range (0,l,1):
                    M = cv2.moments(contours[j])
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    lst=[]
                    #Finding position of object
                    xindex=cx*10/hh+1
                    yindex=cy*10/ww+1
                    #print(xindex,yindex)
                    #Finding colour of object
                    if(i==0):
                            lst.append("blue")
                    elif(i==2):
                            lst.append("green")
                    else:
                            lst.append("red")
                    #Finding shape of object
                    lst.append(detect(contours[j]))
                    #Finding area of object
                    lst.append(int(cv2.contourArea(contours[j])))
                    #Finding perimeter of object
                    lst.append(int(cv2.arcLength(contours[j],True)))
                    Matrix[(xindex,yindex)]=lst
            i+=2

    return Matrix



def planpath(img,Matrix):
    '''
    * Function Name: planpath
    * Input: Image and Matrix containing all x-y co-ordinates of object cells
    * Output: Dictionary containing lists of destination,shortest path and shortest path length for objects and source as key
    * Logic: Finding matching objects positions in grid then finding shortest path by a-star algorithm
    * Example Call: planpath(img,Matrix)
    '''
    #adjacent: Dictionary containing adjacent list representation of all cells in grid(graph)
    #planned_path: Dictionary containing lists of destination,shortest path and shortest path length for objects and source as key
    
    adjacent=creategraph(img,Matrix)
    planned_path={}
    x=[None]*3
    y=[None]*3   
    for i in Matrix:
        flg=0
        for j in Matrix:
            #Finding for matching objects
            if(i is not j and Matrix[i][0]==Matrix[j][0] and Matrix[i][1]==Matrix[j][1] and abs(Matrix[i][2]-Matrix[j][2])<=5.0*Matrix[i][2]/100.0 and abs(Matrix[i][3]-Matrix[j][3])<=10.0*Matrix[i][3]/100.0):
                #call to shortpath() for finding shortest path between source and destination
                x=shortpath(adjacent,i,j)
                if(flg and x[2]<y[2]):
                        y=x
                if(flg==0):
                        y=x
                        flg=1
        if(flg==0):
            #if matching object is not present
            planned_path[i]=["NO MATCH", [], 0]
        else:
            planned_path[i]=y
            
    '''for i in planned_path:
        print(i,'--->',planned_path[i])'''
    return planned_path

def heuristic(cur,desti):
   # Manhattan distance on a square grid
   return abs(cur[0]-desti[0])+abs(cur[1]-desti[1])



def shortpath(adjacent,src,desti):
    '''
    * Function Name: shortpath(adjacent,a,b)
    * Input: Dictionary adjacent containining ajacent list like representation,source and destination
    * Output: List containing destination,shortest path and length to destination
    * Logic: A-Star algorithm by implementing Priority queue and heap
    * Example Call: shortpath(adjacent,a,b)
    '''
    #outlist:List containing destination,shortest path and length to destination
    
    outlist=[]
    outlist.append(desti)
    frontier=PriorityQueue()
    #put source in priority queue
    frontier.put(src, 0)
    came_from ={}
    cost_so_far ={}
    came_from[src] = None
    cost_so_far[src] = 0
    while not frontier.empty():
       #if priority queue is not empty,get element from priority queue as current
       current=frontier.get()
       #print(current)
       #if current is adjacent to desination
       if current in adjacent[desti]:
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

    if(current not in adjacent[desti]):
        #if path is not present
        return ["NO PATH", [], 0]
    else:
        lst = []
        while (current!=src):
            lst.append(current)
            current = came_from[current]
        lst.reverse()
        outlist.append(lst)
        outlist.append(len(lst)+1)
        #print(src,desti,outlist)
        return outlist

        
        

    
def main(image_filename):
	'''
This function is the main program which takes image of test_images as argument. 
Team is expected to insert their part of code as required to solve the given 
task (function calls etc).

***DO NOT EDIT THE FUNCTION NAME. Leave it as main****
Function name: main()

******DO NOT EDIT name of these argument*******
Input argument: image_filename

Return:
1 - List of tuples which is the coordinates for occupied grid. See Task2_Description for detail. 
2 - Dictionary with information of path. See Task2_Description for detail.
	'''

	occupied_grids = []		# List to store coordinates of occupied grid -- DO NOT CHANGE VARIABLE NAME
	planned_path = {}		# Dictionary to store information regarding path planning  	-- DO NOT CHANGE VARIABLE NAME

	##### WRITE YOUR CODE HERE - STARTS

	img = cv2.imread(image_filename)
	Matrix=update(img)
	occupied_grids =occgrid(img,Matrix)
	planned_path=planpath(img,Matrix)
	
		
	print(occupied_grids)
	print("\n\n\n\n")
	cv2.imshow("image_filename - press Esc to close",cv2.imread(image_filename))
	for i in planned_path:
			print(i,planned_path[i])
	#print(planned_path)
			
		

	# #### NO EDIT AFTER THIS

# DO NOT EDIT
# return Expected output, which is a list of tuples. See Task1_Description for detail.
	return occupied_grids, planned_path



'''
Below part of program will run when ever this file (task1_main.py) is run directly from terminal/Idle prompt.

'''
if __name__ == '__main__':

    # change filename to check for other images
    image_filename = "test_images/test_image1.jpg"

    main(image_filename)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
