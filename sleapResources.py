import matplotlib.pyplot as plt
import math
import h5py
import numpy as np
import cv2
import glob
import random
#This loads Daphnia nodes position data and return list with data and proper indexing of nodes for reference in angles function
def file(filename):
    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T#looking at dataset tracks
        node_names = [n.decode() for n in f["node_names"][:]]

    print("===filename===")
    print(filename)
    print()

    print("===HDF5 datasets===")
    print(dset_names)
    print()

    print("===locations data shape===")
    print(locations.shape)
    print()

    print("===nodes===")
    
    L1 = None
    L2 = None
    L3 = None
    L4 = None
    R1 = None
    R2 = None
    R3 = None
    R4 = None
    
    for i, name in enumerate(node_names):
        print(f"{i}: {name}")
        if name == "L1": 
            L1 = i
 
        elif name =="L2": 
            L2 = i

        elif name == "L3": 
            L3 = i

        elif name == "L4": 
            L4 = i

        elif name == "R1": 
            R1 = i

        elif name =="R2":
            R2 = i

        elif name == "R3": 
            R3 = i

        elif name == "R4": 
            R4 = i 
    print([L1,L2,L3,L4,R1,R2,R3,R4])
    
    return locations,[L1,L2,L3,L4,R1,R2,R3,R4]
#Using node position from Daphnia (SLEAP) return first and second segment angle and position data
def angles(data,node_n,reference_segment):  
    #create a copy of original node data
    daphnia_data_copy = data.copy()
    
    #using CHECK_DATA FUNCTION to assign nodes x,y position to proper variables
    x1_left,y1_left,x2_left,y2_left,x3_left,y3_left,x4_left,y4_left,x1_right,y1_right,x2_right,y2_right,x3_right,y3_right,x4_right,y4_right = check_data(data,node_n,daphnia_data_copy)
    
    #using find _angles FUNCTION to use position data to find angle data for segment 1 L/R
    Angles_left_first_segment,Angles_right_first_segment =  find_angles(x1_left,y1_left,x1_right,y1_right,x2_left,y2_left,x2_right,y2_right)
    
    #FIX_MISSING_ANGLES indicate where an first segment angle is a dummy value in graphs happens when n2 value does not truly exist but n1 should always exist (user should always make sure these dummy values are real in video)
    fix_missing_angles(Angles_left_first_segment)
    fix_missing_angles(Angles_right_first_segment)
    #call SECOND SEGMENT function for second segment angle using [n2,n3] and [n2,n4]
    Angles_left_second_segment_3,Angles_right_second_segment_3 =  find_angles(x2_left,y2_left,x2_right,y2_right,x3_left,y3_left,x3_right,y3_right)

    Angles_left_second_segment_4,Angles_right_second_segment_4 =  find_angles(x2_left,y2_left,x2_right,y2_right,x4_left,y4_left,x4_right,y4_right)
    
    #call AVERAGE_SEGMENT to find average second segment 
    average_left_second_segment = average_segment(Angles_left_second_segment_3,Angles_left_second_segment_4)
    average_right_second_segment = average_segment(Angles_right_second_segment_3, Angles_right_second_segment_4)

    return [Angles_left_first_segment,Angles_right_first_segment],[average_left_second_segment,average_right_second_segment],[ [ [x1_left,y1_left], [x2_left,y2_left], [x3_left,y3_left], [x4_left,y4_left] ], [ [x1_right,y1_right] , [x2_right,y2_right] , [x3_right,y3_right], [x4_right,y4_right] ] ]
#return x and y posiiton of nodes 
def check_data(data,node_n,daphnia_data_copy):
    
    frames_not_passed = 0
    frames_passed = 0
    
    #Left node: L1 (closest to body),L2 (tip of first segment), L3/L4 (tip of second segments)
    x1_left = []
    y1_left = []
    
    x2_left = []
    y2_left = []
    
    x3_left = []
    y3_left = []
    
    x4_left = []
    y4_left = []
    
    
    #Right node: R1 (closest to body),R2 (tip of first segment), R3/R4 (tip of second segments)
    x1_right = []
    y1_right = []
    
    x2_right = []
    y2_right = []
    
    x3_right = []
    y3_right = []
    
    x4_right = []
    y4_right = []
    
    for i in range(0,len(data)):
        #looking at every frame in Daphnia Data, caling CHECKING function 
        node_lst = checking(data,i,node_n,daphnia_data_copy)
        #if one or more node does not have position data 
        if len(node_lst) >= 1:
            print("Data Does not exist at frame: " + str(i + 1) + " ", end = " ")
            for j in node_lst: 
                print(j, end = " ")
            print()
            frames_not_passed += 1
        else:
            frames_passed += 1
        #don't really have to add on iteration could just add whole list at end of loop
        x1_left.append(daphnia_data_copy[i][node_n[0]][0][0]), y1_left.append(daphnia_data_copy[i][node_n[0]][1][0])
        x2_left.append(daphnia_data_copy[i][node_n[1]][0][0]), y2_left.append(daphnia_data_copy[i][node_n[1]][1][0])
        x3_left.append(daphnia_data_copy[i][node_n[2]][0][0]), y3_left.append(daphnia_data_copy[i][node_n[2]][1][0])                                                                                                    
        x4_left.append(daphnia_data_copy[i][node_n[3]][0][0]), y4_left.append(daphnia_data_copy[i][node_n[3]][1][0])

        x1_right.append(daphnia_data_copy[i][node_n[4]][0][0]), y1_right.append(daphnia_data_copy[i][node_n[4]][1][0])
        x2_right.append(daphnia_data_copy[i][node_n[5]][0][0]), y2_right.append(daphnia_data_copy[i][node_n[5]][1][0])
        x3_right.append(daphnia_data_copy[i][node_n[6]][0][0]), y3_right.append(daphnia_data_copy[i][node_n[6]][1][0])
        x4_right.append(daphnia_data_copy[i][node_n[7]][0][0]), y4_right.append(daphnia_data_copy[i][node_n[7]][1][0])
        
    print("Total Frames need to be checked " + str(frames_not_passed))  
    print("Total Frames passed " + str(frames_passed))
    #percent of frames that did not pass or have missing values compared to whole video only looking at nodes 
    #L1,L2,L3,L4,R1,R2,R3,R4
    print(str(round((frames_not_passed/(frames_passed + frames_not_passed))*100))+"% Error Inluding all segments" )
    
    return x1_left,y1_left,x2_left,y2_left,x3_left,y3_left,x4_left,y4_left,x1_right,y1_right,x2_right,y2_right,x3_right,y3_right,x4_right,y4_right            
        
#lets user know which frames don't have node values returned in List holding strings of nodes with NAn values in a frame, utilizing copy daphnia data because of changes made to position data in function if used original and changed were made and function was called again it will thing replaced position data if data generated from the file which is false and produce less frames that don't have labeled nodes
def checking(angle_data,iteration,node_n,new_angle_data):
    #MATH used to check if node at a frame has a value 
    import math
    
    #message_to_return = "Data Does not exist at frame: " + str(iteration + 1) + " "
    
    nodes_frames_does_not_exist = []
    #L1 should always have a values 
    #if L1 node does not have a value assign x and y position to zero 
    if(math.isnan(angle_data[iteration][node_n[0]][0][0])):
        nodes_frames_does_not_exist.append("L1")
        #new_angle_data[iteration][node_n[0]][0][0] = 0
        #new_angle_data[iteration][node_n[0]][1][0] = 0
    #if L2 node does not have a value leave x and y Nan values; will be dealed with LATER
    if(math.isnan(angle_data[iteration][node_n[1]][0][0])):
        nodes_frames_does_not_exist.append("L2")
    #if L3 node does not have a value, set it to L4 x and y positions data  
    # if L3 node has no value and L4 has no value then they both will have no values; will be dealed with LATER
    #if set the same their second segment average angle will be one value ex: (3+3)/2 = 3
    if(math.isnan(angle_data[iteration][node_n[2]][0][0])):
        nodes_frames_does_not_exist.append("L3")
        new_angle_data[iteration][node_n[2]][0][0] = new_angle_data[iteration][node_n[3]][0][0] 
        new_angle_data[iteration][node_n[2]][1][0] = new_angle_data[iteration][node_n[3]][1][0] 
    #if L4 node does not have a value, set it to L3 x and y positions data  
    if(math.isnan(angle_data[iteration][node_n[3]][0][0])):
        nodes_frames_does_not_exist.append("L4")
        new_angle_data[iteration][node_n[3]][0][0] = new_angle_data[iteration][node_n[2]][0][0]
        new_angle_data[iteration][node_n[3]][1][0] = new_angle_data[iteration][node_n[2]][1][0]   
    #if R1 node does not have a value assign x and y position to zero 
    if(math.isnan(angle_data[iteration][node_n[4]][0][0])):
        nodes_frames_does_not_exist.append("R1")
        new_angle_data[iteration][node_n[4]][0][0] = 0 #2
        new_angle_data[iteration][node_n[4]][1][0] = 0
    #if R2 node does not have a value leave x and y Nan values; will be dealed with LATER
    if(math.isnan(angle_data[iteration][node_n[5]][0][0])):
        nodes_frames_does_not_exist.append("R2")
    #if R3 node does not have a value, set it to R4 x and y positions data  
    if(math.isnan(angle_data[iteration][node_n[6]][0][0])):
        nodes_frames_does_not_exist.append("R3")
        new_angle_data[iteration][node_n[6]][0][0] = new_angle_data[iteration][node_n[7]][0][0] 
        new_angle_data[iteration][node_n[6]][1][0] = new_angle_data[iteration][node_n[7]][1][0]  
    #if R4 node does not have a value, set it to R3 x and y positions data  
    if(math.isnan(angle_data[iteration][node_n[7]][0][0])):
        nodes_frames_does_not_exist.append("R4")
        new_angle_data[iteration][node_n[7]][0][0] = new_angle_data[iteration][node_n[6]][0][0] 
        new_angle_data[iteration][node_n[7]][1][0] = new_angle_data[iteration][node_n[6]][1][0]
        
    return nodes_frames_does_not_exist

#algorithm used to find angle data for first and second segment 
def find_angles(x_left,y_left,x_right,y_right,x2_left,y2_left,x2_right,y2_right):
    #print("FINDING NEW ANGLE")
    import numpy as np
    
    #print(x_left[0])
    #print()
    
    Angles_left_segment = [] 
    Angles_right_segment = [] 
    
    #print(len(x_left))
    #print(len(y_left))
    #print(len(x_right))
    #print(len(y_right))
    #print(len(x2_left))
    #print(len(y2_left))
    #print(len(x2_right))
    #print(len(y2_right))
    
    for i in range(len(x2_left)):
     #for first segment, if n1 missing but n2 is not then the final angle (using FIX_MISSING_ANGLES) will set angle to 180
     #for second segment, if n2 missing but n3/n3 is not then throught average_segment(still none) and Hinge Data/value function will set hinge angle to zero to stand out in visual
        if(math.isnan(x_left[i]) and math.isnan(x2_left[i])==False):
            Angles_left_segment.append(None)
     #similar to above      
        if(math.isnan(x_right[i]) and math.isnan(x2_right[i])==False):
            Angles_right_segment.append(None)   
     #for first segment, if n2 is missing but n1 is not then final angle (using FIX_MISSING_ANGLES) will set angle to 180
     # for second segment, if n3/n4 is missing then throught average_segment(still none)and Hinge Data/value function will set hinge angle to zero to stand out in visual
        if(math.isnan(x2_left[i])):
            Angles_left_segment.append(None)
            
        if(math.isnan(x2_right[i])):
            Angles_right_segment.append(None)
            
############################# same criteria for difference y positions #################################################
        #left segment - Y2 AND X2 shorter than X1 and Y1 
        if (x2_left[i] < x_left[i]) and (y2_left[i] < y_left[i]):
            Angles_left_segment.append(round(math.degrees(np.arctan((y2_left[i] - y_left[i])/(x2_left[i] - x_left[i])))))
        #Right segment - X2 greater than X1 and Y2 smaller than Y1     
        if (x2_right[i] > x_right[i]) and (y2_right[i] < y_right[i]):
            Angles_right_segment.append(round(math.degrees(np.arctan((-1*(y2_right[i] - y_right[i]))/(x2_right[i] - x_right[i] )))))
            
        #left segment - Y2 AND X2 shorter than X1 and Y1 
        if (x2_left[i] < x_left[i]) and (y2_left[i] > y_left[i]):
            Angles_left_segment.append(round(math.degrees(np.arctan((y2_left[i] - y_left[i])/(x2_left[i] - x_left[i] )))))
        #Right segment - X2 greater than X1 and Y2 smaller than Y1   
        if (x2_right[i] > x_right[i]) and (y2_right[i] > y_right[i]):
            Angles_right_segment.append(-1*round(math.degrees(np.arctan(((y2_right[i] - y_right[i]))/(x2_right[i] - x_right[i] )))))
#########################################################################################################################            
        #Left segment - X2 and y2 is larger than X1 and y1 left segment need to be checked   
    
        if (x2_left[i] > x_left[i]) and (y2_left[i] > y_left[i]):
            Angles_left_segment.append(-1*180+round(math.degrees(np.arctan((y2_left[i] - y_left[i])/(x2_left[i] - x_left[i] )))))
        #Right segment - X2 is smaller than x1 and y2 is greater than y1
        if (x2_right[i] < x_right[i]) and (y2_right[i] > y_right[i]):
            Angles_right_segment.append(-180+-1*round(math.degrees(np.arctan(((y2_right[i] - y_right[i]))/(x2_right[i] - x_right[i] )))))
           
        #Left segment - X2 is greater than X1 and Y2 is smaller than Y1

        if (x2_left[i] > x_left[i]) and (y2_left[i] < y_left[i]):
            Angles_left_segment.append(180+round(math.degrees(np.arctan((y2_left[i] - y_left[i])/(x2_left[i] - x_left[i] )))))
        #right segment - X2 is is smaller then X1 and Y2 is smaller than Y1 #Works!!!
        if (x2_right[i] < x_right[i]) and (y2_right[i] < y_right[i]):
            Angles_right_segment.append(180+-1*round(math.degrees(np.arctan(((y2_right[i] - y_right[i]))/(x2_right[i] - x_right[i] )))))
        
       
    #Angle_left_second_segment = []
    #Angle_right_second_segment = []
    
    #for i in range(len(Angles_left_second_segment_1)):
        #Angle_left_second_segment.append(np.average([Angles_left_second_segment_1[i],Angles_left_second_segment_2[i]]))
        #Angle_right_second_segment.append(np.average([Angles_right_second_segment_1[i],Angles_right_second_segment_2[i]]))
    #print("Left: " + str(len(Angles_left_segment)))
    #print("Right: " + str(len(Angles_right_segment)))
    #print()
    #print()
    return Angles_left_segment,Angles_right_segment    
#for first segment when a node 2 does not truly exist but node 1 does set dummy angle to 180(choosen as it will be noticable in graph data)
def fix_missing_angles(Angle):#used for first segment
    for i in range(len(Angle)):
        if Angle[i] == None:
            Angle[i] = 180
#averaging between n3 and n4 second segments 
def average_segment(n3_angles,n4_angles):
    import numpy as np

    average_segment = []

    for i in range(len(n3_angles)):
        if(n3_angles[i] == None or n4_angles == None):#only possible if both are missing values 
            average_segment.append(None)
        else:
            average_segment.append(np.average([n3_angles[i],n4_angles[i]]))

    return average_segment 
#find hinge data and show real time plots
def angle_data(Angles,limit):
    
    fig,(ax0,ax1) = plt.subplots(2,1,figsize=(30,15))
    ax0.scatter(np.arange(limit[0], limit[1]), Angles[0][limit[0]:limit[1]], color="blue")#left first segment angle
    ax1.scatter(np.arange(limit[0], limit[1]), Angles[1][limit[0]:limit[1]], color="green")#right first segment angle
    
    ax0.set_title('First Segment Left',fontsize = 20)
    ax1.set_title('First Segment Right',fontsize = 20)
    ax = [ax0,ax1]

    for axs in ax: 
        plt.sca(axs)
        plt.ylabel('Angle (degree)',fontsize = 20)
        plt.xlabel('Time (milliseconds)',fontsize = 20)
        plt.xticks(np.arange(limit[0],limit[1]+5,5),np.arange(limit[0],limit[1]+5,5)*5,fontsize=20)
        plt.yticks(np.arange(-60,90,step = 20),fontsize=20)
        #plt.xticks(np.arange(limit[0],limit[1]+2,2),np.arange(limit[0],limit[1]+2,2)*10, fontsize=20)
        #plt.yticks(np.arange(-60,90,step = 20),fontsize=20)
    plt.show()
def overlapped(Angles,limit,plotToGraph,line):   
    plt.figure(figsize=(30,10))
    if (plotToGraph[0] == True):
        if (line == True):
            plt.plot(np.arange(limit[0], limit[1]), Angles[0][limit[0]:limit[1]],label = "left", color="blue")
            
        else:
            plt.scatter(np.arange(limit[0], limit[1]), Angles[0][limit[0]:limit[1]],label = "left", color="blue")
    if (plotToGraph[1] == True):
        if (line == True):
            plt.plot(np.arange(limit[0], limit[1]), Angles[1][limit[0]:limit[1]],label = "right",color="green")
        else: 
            plt.scatter(np.arange(limit[0], limit[1]), Angles[1][limit[0]:limit[1]],label = "right",color="green")
    plt.title("First Segment",fontsize = 24)
    plt.ylabel(''r'$\theta_{1}$ (degree)',fontsize = 20)
    plt.xlabel('Time (milliseconds)',fontsize = 20)
    plt.xticks(np.arange(limit[0],limit[1]+5,5),np.arange(limit[0],limit[1]+5,5)*5,fontsize=20)
    #plt.xticks(np.arange(0,limit[1]+2,2),np.arange(0,limit[1]+2,2)*10, fontsize=20)
    plt.yticks(np.arange(-60,90,step = 20),fontsize=20)
    plt.legend(fontsize="20")
    #plt.show()
#[ [ [x1_left,y1_left], [x2_left,y2_left] ], [ [x1_right,y1_right] , [x2_right,y2_right] ] ]

#  x1_left - [0][0][0]
# y1_left - [0][0][1]
# x2_left - [0][1][0]
# y2_left - [0][1][1]
# x1_right - [1][0][0]
# y1_right - [1][0][1]
# x2_right - [1][1][0]
# y2_right - [1][1][1] 

def Hinge_data(Angles_Data,limit,points):
    #using HINGE_VALUES FUnction to find hinge angles
    angle_hinge_left = []
    for i in range(len(Angles_Data[0][0])):
        angle_hinge_left.append(Hinge_values(Angles_Data[0][0],Angles_Data[1][0],i))

    angle_hinge_right = []
    for i in range(len(Angles_Data[0][1])):
        angle_hinge_right.append(Hinge_values(Angles_Data[0][1],Angles_Data[1][1],i))    
    
    fig,(ax0,ax1) = plt.subplots(2,1,figsize=(30,15))
    ax0.scatter(np.arange(limit[0], limit[1]), angle_hinge_left[limit[0]:limit[1]], color="blue")#left first segment angle

    ax1.scatter(np.arange(limit[0], limit[1]), angle_hinge_right[limit[0]:limit[1]], color="green")#right first segment angle
 
    
    ax0.set_title('Hinge Left',fontsize = 20)
    ax1.set_title('Hinge Right',fontsize = 20)
    ax = [ax0,ax1]

    for axs in ax: 
        plt.sca(axs)
        plt.ylabel('Angle (degree)',fontsize = 20)
        plt.xlabel('Time (milliseconds)',fontsize = 20)
        plt.xticks(np.arange(limit[0],limit[1]+5,5),np.arange(limit[0],limit[1]+5,5)*5,fontsize=20)
        plt.yticks(np.arange(0,200,step = 20),fontsize=20)
        plt.axvline(x = points[0], color = 'b')
        plt.axvline(x = points[1], color = 'r')
        #plt.xticks(np.arange(limit[0],limit[1]+2,2),np.arange(limit[0],limit[1]+2,2)*10, fontsize=20)
        #plt.yticks(np.arange(-60,90,step = 20),fontsize=20)
    #plt.show()  
    return angle_hinge_left,angle_hinge_right 
#criteria to find hinge angle, if using dummy first segment angle should be filtered when finalizing PS
def Hinge_values(First, Second, i) :
    if(Second[i] == None):
        Hinge_Angle = 0
    else:
        if (First[i] and Second[i]) >= 0 :
            New_angle = 180 - (First[i] + 90)
            Hinge_Angle = New_angle + 90 + Second[i]
        elif (First[i] > 0 ) and (Second[i] < 0) :
            New_angle = 180 - (First[i] + 90)
            Second_new_angle = 90 + Second [i]
            Hinge_Angle = New_angle + Second_new_angle
        elif (Second[i] and First[i]) < 0 :
            New_angle = 180 - (180 - (abs(First[i]) + 90))
            Second_new_angle = 90 + Second[i]
            Hinge_Angle = New_angle + Second_new_angle
        else:
            print("Error")
    return Hinge_Angle
#Method 2 to find power strokes in data     
def find_power_strokes(Angles,Hinge_Angle,Angle_Threshold):
    final_position_of_strokes = []
    for i in range(0,len(Angles)):#look at both left and right angles of first segment
        position_of_power_strokes = []
        for j in range(1,len(Angles[i])-1):#looking at all first angle angle frames starting at second frame
            #prepeak = Angles[i][j-1] - Angles[i][j]
            #postpeak = Angles[i][j] - Angles[i][j+1]
            #catch frame/position of start/end of PS when next frame first segment angle starts descending
            #The hinge angle of frame is greater than 160
            #The frame angle is higher/lower than given Angle_Threshold [30,50] not fixed
            if (Angles[i][j+1] < Angles[i][j]) and Hinge_Angle[i][j] > 160 and Angles[i][j]>Angle_Threshold[0] and Angles[i][j] < Angle_Threshold[1]: # limit = 18 need to change limits cannot be 100 percent,Angles[i][j]>28 
                #make sure there is a difference of 7 frames between start/end frames of PS, helps avoid capturing PS frames in a PS, as long as there is a at least one frame to refer too 
                if(len(position_of_power_strokes) >= 1):
                    if(j - position_of_power_strokes[-1] > 7):
                        position_of_power_strokes.append(j)
                    else:
                        continue
                else:
                    position_of_power_strokes.append(j)
        #possible implement assymetric criteria here for one list instead of two?             
        side_data = [] #store [start,end frame of PS]
                
        for k in range(0,len(position_of_power_strokes)-1):   
            side_data.append([position_of_power_strokes[k],position_of_power_strokes[k+1]])
            
        final_position_of_strokes.append(side_data)
    return final_position_of_strokes
def checkTracking(Angles,powerStrokeFrames,limit,plotToGraph,line): #specify left or right 
    overlapped(Angles,limit,plotToGraph,line)
    for i in powerStrokeFrames:
        if i[0] > limit[1]:
            break;
        elif i[0] < limit[0] or i[1] > limit [1]:
            continue;
        plt.axvline(x = i[0], color = 'black', label = 'axvline - full height',alpha=0.5)
        plt.axvline(x = i[1], color = 'black', label = 'axvline - full height',alpha=0.5)
#delete powerstroke longer than a limit
def filterData(powerStrokeFrames,limit):
    Todelete = []
    for i in powerStrokeFrames:
        if(i[1] - i[0] > limit):
            print(i)
            Todelete.append(i)
    for j in Todelete:
        powerStrokeFrames.remove(j)
#criteria to delete assymetrical data in powerstrokes found
def removeAssymetrical(powerStrokeFrames,Angle):
    todelete = []
    num = 0 
    for i in range(len(powerStrokeFrames)):  
        #min angle difference between left and right first segment cannot be greater than 15
        if(abs(displayMinAngle(powerStrokeFrames[i],Angle[0]) - displayMinAngle(powerStrokeFrames[i],Angle[1]))>15):
            print("Min comparison Change")
            print(powerStrokeFrames[i])
            num += 1
            todelete.append(powerStrokeFrames[i])
        #first angle difference between left and right first segment cannot be greater than 25
        elif(abs(displayFirstAngle(powerStrokeFrames[i],Angle[0])-displayFirstAngle(powerStrokeFrames[i],Angle[1])) > 25):
            print("First L/R angle far from limits")
            print(powerStrokeFrames[i])
            num += 1
            todelete.append(powerStrokeFrames[i])
        #last angle difference between left and right first segment cannot be greater than 25
        elif(abs(displayLastAngle(powerStrokeFrames[i],Angle[0])-displayLastAngle(powerStrokeFrames[i],Angle[1])) > 25 ):
            print("Last L/R angle far from limits")
            print(powerStrokeFrames[i])
            num += 1
            todelete.append(powerStrokeFrames[i])
        #min angle for left or right segment must be greater than -20    
        elif(displayMinAngle(powerStrokeFrames[i],Angle[0]) > -20 or displayMinAngle(powerStrokeFrames[i],Angle[1])>-20):
            print("Min point")
            print(powerStrokeFrames[i])
            num += 1
            todelete.append(powerStrokeFrames[i])
        #max angle difference between left and right first segment cannot be greater than 15    
        elif(abs(displayMaxAngle(powerStrokeFrames[i],Angle[0]) - displayMaxAngle(powerStrokeFrames[i],Angle[1]))>15):
            print("Max Change")
            print(powerStrokeFrames[i])
            num += 1
            todelete.append(powerStrokeFrames[i])
            
    for j in todelete:
        powerStrokeFrames.remove(j)
    print(num)
#find min angle in power stroke data
def displayMinAngle(powerStrokeData,Angle_data):
    return min(Angle_data[powerStrokeData[0]:powerStrokeData[1]])
#find max angle in power stroke data
def displayMaxAngle(powerStrokeData,Angle_data):
    return max(Angle_data[powerStrokeData[0]:powerStrokeData[1]])
#find first angle in power stroke data
def displayFirstAngle(powerStrokeData,Angle_data):
    return Angle_data[powerStrokeData[0]]
#find last angle in power stroke data
def displayLastAngle(powerStrokeData,Angle_data):
    return Angle_data[powerStrokeData[1]]

def graph(data,Angles,title,limits):
    data_return = []
    print("This is where the power strokes where found\n")
    print(data)
    print("\nThere is " + str(len(data)) + " PowerStrokes")
    starting_frame = int(input("What power stroke to start with?")) #skipping first item in list so if 1 will look at index1
    end_frame = int(input("What power stroke to end with with? "))
    print("\nLooking at power strokes data ")
    print(data[starting_frame:end_frame+1])
    for j in range(starting_frame,end_frame+1):
        plt.plot(Angles[data[j][0]:(data[j][1]+1)], color="blue",marker = 'o')
        data_return.append(Angles[data[j][0]:(data[j][1]+1)])
        
    max_xlimit,max_frame = max_number_power_strokes(Angles,data,[starting_frame,end_frame+1])
    
    plt.title("First " + title + " Segment Angles",fontsize = 24)
    plt.ylabel('Angle (degree)',fontsize = 20)
    plt.xlabel('Time (milliseconds)',fontsize = 20)
    plt.xticks(np.arange(0,max_xlimit,2),np.arange(0,max_xlimit,2)*5, fontsize=20)
    print(max_xlimit)
    print(max_frame)
    plt.yticks(np.arange(limits[0],limits[1],step = 20),fontsize=20)
    #plt.legend(fontsize="20")
    plt.show()
    
    return data_return
def eliminatePS_first_segment(data,Angles):
    PS_position = []
    Index_In_data = []
    for j in range(len(data)):
        current_PS = Angles[data[j][0]:(data[j][1]+1)]
        for i in current_PS:
            if(i == 180):
                PS_position.append([data[j][0],data[j][1]])
                Index_In_data.append(j)
                break
    print(PS_position)
    #return Index_In_data
    return PS_position    

def plot_averages(Daph_Data,num,data_name,color):
    
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    
    Average_plot = []
    Standard_Error = []
    
    max_powerstrokes,max_sheet,max_sheet_min_index = max_number_power_strokes_hinge(Daph_Data,num)
   # print(max_sheet)
    #print(max_sheet_min_index)
    #print("Max power Strokes: " + str(max_powerstrokes))
    for j in range(max_powerstrokes):#for each data point in longest data set
        STD = []
        Average = [] 
        #print("Data point : " + str(j))
        for i in range(num):#amount of different data looking at
                if (len(Daph_Data[i]) < j+1):
                    continue
                Average.append(Daph_Data[i][j])#appending sheet and corresponding data points
                
               # print("Sheet " + str(i) + " Data " + str(Daph_Data[i][j]))
       # print("Average: ", end = " ")
        #print(len(Average))
        #print()
        #print("Mean: " + str(np.mean(Average)))
        #print(Average)
        #print()
        Average_plot.append(np.mean(Average)) #once done iterating through all datasets find average
        Standard_Error.append(np.std(Average)/np.sqrt(np.size(Average)))#find standard error from timesheet
        #Standard_Error.append(np.std(Average)**2)#find standard error from timesheet

    #for average_time in Daph_Data:
        #total_time = []
        #print(len(average_time))
        #total_time.append(len(average_time))
        #print(np.mean(total_time))
        
    #Standard_Error_time.append(np.std(len(Average)/np.sqrt(len(Average))))
        #print(Average_plot)
        
        #print()
    #print(Standard_Error)
    N = np.array(Average_plot)
    ranges = np.arange(0,max_powerstrokes)
    plt.fill_between(ranges,N-Standard_Error,N+Standard_Error,color = color,alpha=0.2)    
    plt.plot(Average_plot,label = data_name,linestyle='none', marker = '.',c = color)
    #plt.scatter(ranges,Average_plot,label = data_name, marker = '.',c = color)
    plt.errorbar(ranges,Average_plot,yerr = Standard_Error,fmt='none',color = color)
    plt.ylabel(''r'$\theta_{1}$ (degree)',fontsize = 20)
    plt.xlabel('Time (milliseconds)',fontsize = 20)
    #side = input("Left or right?")
    #plt.title(data_name + " First Segment " + side)
    plt.tick_params(bottom = False)
    plt.yticks(np.arange(-90,100,step = 20),fontsize=15)
    plt.xticks(np.arange(0,max_powerstrokes+2,2),np.arange(0,max_powerstrokes+2,2)*5, fontsize=15)
    plt.legend(frameon = False,fontsize=12,loc = 4)
    #plt.show()
    
    return Average_plot,Standard_Error    

def Final_average(Daph_Data,num,data_name,color,error,limit):
    
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    
    Average_plot = []
    Standard_Error = []
    
    for i in range(len(error[0])):
        Standard_Error.append(np.average([error[0][i],error[1][i]]))
        
        #Standard_Error.append(np.sqrt(((error[0][i]**2) + (error[1][i]**2))/len(error)))

    
    max_powerstrokes,max_sheet,max_sheet_min_index = max_number_power_strokes_hinge(Daph_Data,num)
   # print(max_sheet)
    #print(max_sheet_min_index)
    #print("Max power Strokes: " + str(max_powerstrokes))
    for j in range(max_powerstrokes):#for each data point in longest data set
       
        Average = [] 
        #print("Data point : " + str(j))
        for i in range(num):#amount of different data looking at
                if (len(Daph_Data[i]) < j+1):
                    continue
                Average.append(Daph_Data[i][j])#appending sheet and corresponding data points
                
               # print("Sheet " + str(i) + " Data " + str(Daph_Data[i][j]))
       # print("Average: ", end = " ")
        #print(len(Average))
        #print()
        #print("Mean: " + str(np.mean(Average)))
        #print(Average)
        #print()
        Average_plot.append(np.mean(Average)) #once done iterating through all datasets find average
        #Standard_Error.append(np.std(Average)/np.sqrt(np.size(Average)))#find standard error from timesheet
        #Standard_Error.append(np.std(Average)**2)#find standard error from timesheet

    #for average_time in Daph_Data:
        #total_time = []
        #print(len(average_time))
        #total_time.append(len(average_time))
        #print(np.mean(total_time))
        
    #Standard_Error_time.append(np.std(len(Average)/np.sqrt(len(Average))))
        #print(Average_plot)
        
        #print()
    #print(Standard_Error)
    N = np.array(Average_plot)
    ranges = np.arange(0,max_powerstrokes)
    plt.fill_between(ranges,N-Standard_Error,N+Standard_Error,color = color,alpha=0.2)    
    plt.plot(Average_plot,label = data_name,linestyle='none', marker = '.',c = color)
    #plt.scatter(ranges,Average_plot,label = data_name, marker = '.',c = color)
    plt.errorbar(ranges,Average_plot,yerr = Standard_Error,fmt='none',color = color)
    plt.ylabel(''r'$\theta_{1}$ (degree)',fontsize = 20)
    plt.xlabel('Time (milliseconds)',fontsize = 20)
    #side = input("Left or right?")
    #plt.title(data_name + " First Segment " + side)
    plt.tick_params(bottom = False)
    plt.yticks(np.arange(limit[0],limit[1],step = 20),fontsize=15)
    plt.xticks(np.arange(0,max_powerstrokes+2,5),np.arange(0,max_powerstrokes+2,5)*5, fontsize=15)
    plt.legend(frameon = False,fontsize=12,loc = 4)
    #plt.show()
    
    return Average_plot,Standard_Error    

def eliminatePS_Hinge(data,Angles):
    PS_position = []
    Index_In_data = []
    for j in range(len(data)):
        current_PS = Angles[data[j][0]:(data[j][1]+1)]
        for i in current_PS:
            if(i == 0):
                PS_position.append([data[j][0],data[j][1]])
                Index_In_data.append(j)
                break
    print(PS_position)
    #return Index_In_data
    return PS_position    
def plot_average_hinge(Daph_Data,num,data_name,color):
    import numpy as np
    import math
    
    Average_plot = []
    Standard_Error = []
    max_powerstrokes,max_sheet,max_sheet_min_index = max_number_power_strokes_hinge(Daph_Data,num)
    L_minimum_index = minimum(Daph_Data)
    ran_List_L = min_in_sheet(L_minimum_index[max_sheet_min_index], L_minimum_index,Daph_Data)
    
    #print("Max power Strokes: " + str(max_powerstrokes))
    
    track_index_for_average = np.zeros(num)
    
    print("Sheets " + str(len(Daph_Data)))
    first_frame = int(input("First frame: "))
    last_frame = int(input("Last frame: "))
    for i in range(first_frame,last_frame+1):
        #print(ran_List_L[i])
        #print()
        plt.plot(ran_List_L[i],Daph_Data[i],label = "Position in databse " + str(i),marker = 'o')
        #plt.set(xlabel=None)
        #plt.xticks(np.arange(5,max_powerstrokes+10,5),np.arange(5,max_powerstrokes+10,5)-5)
        #plt.legend()
    plt.show()


    
    for j in range(max_powerstrokes):#making sure it goes up to 35data points for longest point
        STD = []
        Average = [] 
        #print("Data point : " + str(j))
        #print(num)
        for i in range(num):#look at all data sheets
                #print(Daph_Data[i][j])
                #print(ran_List_L[i])
                if(ran_List_L[i][0] > j or (ran_List_L[i][-1] <j) ):
                    continue
                #print("point" + str(j))
                #print("sheet" + str(i))
                #print(Daph_Data[i][int(track_index_for_average[i])])
                #print(track_index_for_average[i])
                #print()
                Average.append(Daph_Data[i][int(track_index_for_average[i])])
                track_index_for_average[i] += 1
                #if (len(Daph_Data[i]) < j+1):#aligns first point with of all sheets starting with zero unless all sheets are zero
                    #continue
               # print("Sheet " + str(i) + " Data " + str(Daph_Data[i][j]))
       # print("Average: ", end = " ")
       # print(Average)
        #print("Mean: " + str(np.mean(Average)))
        #print("Point " + str(j) + " has " + str(len(Average)) + " lists averaging on")
        #print()
        #print(Average)
        Average_plot.append(np.mean(Average))
        #Standard_Error.append(np.std(Average)/np.sqrt(np.size(Average)))
        Standard_Error.append(np.std(Average))
        #print(Average_plot)
        
        #print()
    N = np.array(Average_plot)
    ranges = np.arange(0,max_powerstrokes)
        
    #plt.figure(figsize=(8,4))
    plt.fill_between(ranges,N-Standard_Error,N+Standard_Error,color = color,alpha=0.2)    
    plt.plot(Average_plot,label = data_name, marker = 'o',linestyle = 'none', color = color)
    plt.errorbar(ranges,Average_plot,yerr = Standard_Error,fmt='none',color = color)
    #plt.ylabel(''r'$\theta_{2}$ (degree)',fontsize = 20)
    #plt.xlabel('Time (milliseconds)',fontsize = 15)
    #side = input("Left or right?")
    #plt.title(data_name + " Hinge " + side)
    plt.tick_params(bottom = False)
    plt.yticks(np.arange(60,220,step = 20),fontsize=15)
    plt.xticks(np.arange(0,max_powerstrokes+2,2),[], fontsize=15)
    plt.legend(frameon = False,fontsize=12,loc = 4)
    #np.arange(0,max_powerstrokes+2,2)*10
    #x_data  = np.arange(0,max_powerstrokes+2,2)*10
    #plt.show()
    
    return Average_plot,Standard_Error  #######################LOOOK AT THIS      

 
def SafeDataAnalysis(data,Angle,Angle_2,angle_hinge_left,angle_hinge_right,Position_Data,Average_Plot_L,Average_Plot_L_Error,Average_Plot_R,Average_Plot_R_Error,Average_Hinge_L,Average_Hinge_L_Error,Average_Hinge_R,Average_Hinge_R_Error,Average_Plot_First,Average_Plot_First_Error,Average_Hinge,Average_Hinge_Error,name,angle_min_threshold,angle_max_threshold,cutoff):
    from xlwt import Workbook
    import xlwt
    #Saving all the data into an excel sheet
    #This is pretty self expenatory
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    
    

    sheet1.write(0,0, 'X1_Left')
    sheet1.write(0,1, 'Y1_Left')
    
    sheet1.write(0,2, 'X2_Left')
    sheet1.write(0,3, 'Y2_Left')
    
    sheet1.write(0,4, 'X3_Left')
    sheet1.write(0,5, 'Y3_Left')
    
    sheet1.write(0,6, 'X4_Left')
    sheet1.write(0,7, 'Y4_Left')
    
    
    
    sheet1.write(0,8, 'X1_Right')
    sheet1.write(0,9, 'Y1_Right')
    
    sheet1.write(0,10, 'X2_Right')
    sheet1.write(0,11, 'Y2_Right')
    
    sheet1.write(0,12, 'X3_Right')
    sheet1.write(0,13, 'Y3_Right')
    
    sheet1.write(0,14, 'X4_Right')
    sheet1.write(0,15, 'Y4_Right')
    
    sheet1.write(0,16, 'Segment 1 angle Left')
    sheet1.write(0,17, 'Segment 1 angle Right')
    
    sheet1.write(0,18, 'Segment 2 angle Left')
    sheet1.write(0,19, 'Segment 2 angle Right')
    
    sheet1.write(0,20, 'Hinge Theta angle Left')
    sheet1.write(0,21, 'Hinge Theta angle Right')
    
    sheet1.write(0,22,' ')
    
    sheet1.write(0,23,'Power Strokes I')
    sheet1.write(0,24,'Power Strokes F')
    
    sheet1.write(0,25,' ')
    
    sheet1.write(0,26,'Theta Average_L')
    
    sheet1.write(0,27,'Theta Average_L Error')
    
    sheet1.write(0,28,'Theta Average_R')
    
    sheet1.write(0,29,'Theta Average_R Error')
    
    sheet1.write(0,30,'Theta Hinge Average_L')
    
    sheet1.write(0,31,'Theta Hinge Average_L Error')

    sheet1.write(0,32,'Theta Hinge Average_R')
    
    sheet1.write(0,33,'Theta Hinge Average_R Error')

    sheet1.write(0,34,'Average Theta Segment')
    
    sheet1.write(0,35,'Average Theta Segment Error')

    sheet1.write(0,36,'Average Theta Hinge')
    
    sheet1.write(0,37,'Average Theta Hinge Error')
    
    sheet1.write(0,38,' ')
    
    sheet1.write(0,39,'angle_min_threshold')
    
    sheet1.write(0,40,'angle_max_threshold')
    
    sheet1.write(0,41,'Frame Cutoff')
    

    #Now writting the data
    length = len(data[0][0][0])
    for i in range(1, length+1):
        sheet1.write(i, 0, data[0][0][0][i-1])
        sheet1.write(i, 1, data[0][0][1][i-1])
        
        sheet1.write(i, 2, data[0][1][0][i-1])
        sheet1.write(i, 3, data[0][1][1][i-1])
        
        sheet1.write(i, 4, data[0][2][0][i-1])
        sheet1.write(i, 5, data[0][2][1][i-1])
        
        sheet1.write(i, 6, data[0][3][0][i-1])
        sheet1.write(i, 7, data[0][3][1][i-1])
        
        
        sheet1.write(i, 8, data[1][0][0][i-1])
        sheet1.write(i, 9, data[1][0][1][i-1])
        
        sheet1.write(i, 10, data[1][1][0][i-1])
        sheet1.write(i, 11, data[1][1][1][i-1])
        
        sheet1.write(i, 12, data[1][2][0][i-1])
        sheet1.write(i, 13, data[1][2][1][i-1])
        
        sheet1.write(i, 14, data[1][3][0][i-1])
        sheet1.write(i, 15, data[1][3][1][i-1])
        
        sheet1.write(i, 16, Angle[0][i-1])
        sheet1.write(i, 17, Angle[1][i-1])
        
        sheet1.write(i, 18, Angle_2[0][i-1])#Second segment angle
        sheet1.write(i, 19, Angle_2[1][i-1])
        
        sheet1.write(i, 20, angle_hinge_left[i-1])#Hinge Angle
        sheet1.write(i, 21, angle_hinge_right[i-1])
        
    for j in range(1,len(Position_Data)+1):
        sheet1.write(j, 23, Position_Data[j-1][0])
        sheet1.write(j, 24, Position_Data[j-1][1])
        
    for k in range(1,len(Average_Plot_L)+1):
        
        sheet1.write(k,26,Average_Plot_L[k-1])
        sheet1.write(k,27,Average_Plot_L_Error[k-1])
        sheet1.write(k,28,Average_Plot_R[k-1])
        sheet1.write(k,29,Average_Plot_R_Error[k-1])
        
        sheet1.write(k,30,Average_Plot_First[k-1])
        sheet1.write(k,31,Average_Plot_First_Error[k-1])
        
    for m in range(1,len(Average_Hinge_L)+1):
        sheet1.write(m,32,Average_Hinge_L[m-1])
        sheet1.write(m,33,Average_Hinge_L_Error[m-1])
        sheet1.write(m,34,Average_Hinge_R[m-1])
        sheet1.write(m,35,Average_Hinge_R_Error[m-1])
        
        sheet1.write(m,36,Average_Hinge[m-1])
        sheet1.write(m,37,Average_Hinge_Error[m-1])

    sheet1.write(1,39,angle_min_threshold)
    sheet1.write(1,40,angle_max_threshold)
    sheet1.write(1,41,cutoff)
    
    wb.save(name)
    
    
def max_number_power_strokes(Angle,Data,num):
    max_powerstrokes = 0
    for j in range(num[0],num[1]):
        #print(len((Daphnia_drugged_1hr_100Micro_t1[i][0][0])))
        if max_powerstrokes < len(Angle[Data[j][0]:(Data[j][1]+1)]):
            max_powerstrokes = len(Angle[Data[j][0]:(Data[j][1]+1)])
            max_frames = Data[j]
    return max_powerstrokes,max_frames

def max_number_power_strokes_hinge(Daph_Data,num):#find longest data in database and saves length, data and index using first value as reference
    max_powerstrokes = len(Daph_Data[0])
    max_sheet = Daph_Data[0]
    max_sheet_index = 0
    for i in range(1,num):
        #print(len((Daphnia_drugged_1hr_100Micro_t1[i][0][0])))
        if max_powerstrokes < len((Daph_Data[i])):
            max_powerstrokes = len((Daph_Data[i]))
            max_sheet = Daph_Data[i]
            max_sheet_index = i
    return max_powerstrokes,max_sheet,max_sheet_index
def min_in_sheet(target,min_index_sheets,data):
    range_List  = []
    indexx = 0 
    for i in min_index_sheets:#looking at all minimum index for all sheet even max
        #print(target)
        #print(i)
        #print(abs(target-i))
        #print(indexx)
        #print()
        if(i<target):#if other sheet minimum value index is lower than index of longest sheet
            change = target - i #changing x value
            range_List.append(range(change,len(data[indexx])+change))#allows build from minimum points onwards
        else:
            change = i - target
            range_List.append(range(0-change,len(data[indexx])-change))
            
        indexx +=1#change to next sheet 
    
    return range_List

def minimum(Data):#look at sheet of data with hinge angle in a power strokes and return minimum index of each sheet with lowest angle
    index_min = []
    #Average_min = []
    for minn in Data:#for each power stroke observe hinge angle values
        A = min(minn)#find min of hinge angle data
        minn = np.array(minn)
        #print(minn)
        #print(min(minn), end = ' ')
        #Average_min.append(A)#add to list 
        index_min.append(np.where(minn==A)[0][0])#find apporpirate lowest index in for loop of lowest value
    #print()
    #print(Average_min)
    #print("This is the minimum for " + name +": "+  str(np.mean(Average_min)))
    #print(index_min)
    
    return(index_min)
############################################ For Overall Average Notebook ###############################################


def FormatData(name):
    import pandas as pd
    
    #code to extract the data from the excel sheet
    
    data = pd.read_excel(name)
    
    Theta_first_Average_L = data['Theta Average_L'].dropna().tolist()
    Theta_first_Average_L_error = data['Theta Average_L Error'].dropna().tolist()
    Theta_first_Average_R = data['Theta Average_R'].dropna().tolist()
    Theta_first_Average_R_error = data['Theta Average_R Error'].dropna().tolist()
    Theta_first_Average = data['Average Theta Segment'].dropna().tolist()
    Theta_first_Average_error = data['Average Theta Segment Error'].dropna().tolist()
    Hinge_Average_L =data['Theta Hinge Average_L'].dropna().tolist()
    Hinge_Average_L_error = data['Theta Hinge Average_L Error'].dropna().tolist()
    Hinge_Average_R = data['Theta Hinge Average_R'].dropna().tolist()
    Hinge_Average_R_error = data['Theta Hinge Average_R Error'].dropna().tolist()
    Hinge_Average = data['Average Theta Hinge'].dropna().tolist()
    Hinge_Average_error = data['Average Theta Hinge Error'].dropna().tolist()

    
    return [Theta_first_Average_L, Theta_first_Average_L_error, Theta_first_Average_R, Theta_first_Average_R_error, Theta_first_Average, Theta_first_Average_error, Hinge_Average_L, Hinge_Average_L_error, Hinge_Average_R, Hinge_Average_R_error, Hinge_Average, Hinge_Average_error]



def Format_Lst(lst):
    Data = []
    for i in lst:
        Daphnia_data = FormatData(i)
        Data.append(Daphnia_data)
    return Data


def plot_segment_lst(data,limits,names):
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ([ax0, ax1],[ax2, ax3]) = plt.subplots(2, 2, figsize=(20, 14))
    
    max_xlimit = 0
    
    left_first = []
    left_error = []
    
    right_first = []
    right_error = []
    
    for i in data:
        left_first.append(i[0])
        left_error.append(i[1])
        
        if len(i[0]) > max_xlimit:
            max_xlimit = len(i[0])
            
        ax0.plot(i[0],marker = 'o')
        N = np.array(i[0])
        ranges = np.arange(0,len(i[0]))
        ax0.fill_between(ranges,N-i[1],N+i[1],alpha=0.8)  
        ax0.errorbar(ranges,i[0],yerr = i[1],fmt='none')

        
    for i in data:
        right_first.append(i[2])
        right_error.append(i[3])
        
        ax1.plot(i[2],marker = 'o')
        N = np.array(i[2])
        ranges = np.arange(0,len(i[2]))
        ax1.fill_between(ranges,N-i[3],N+i[3],alpha=0.8)  
        ax1.errorbar(ranges,i[2],yerr = i[3],fmt='none')
        
    Final_error_l = []
    Final_error_r = []
    
    for j in range(0,max_xlimit):
        Average_error_l = []
        Average_error_r = []
        for k in left_error:
            if(len(k)-1 < j):
                continue
            else:
                Average_error_l.append(k[j])
        Final_error_l.append(np.average(Average_error_l))
        for m in right_error:
            if(len(m)-1 < j):
                continue
            else:
                Average_error_r.append(m[j])
        Final_error_r.append(np.average(Average_error_r))
        
    
    left_average,left_average_left_error = plot_averages(left_first,len(left_first),"Left Average","Gray")
    
    right_average,average_right_error = plot_averages(right_first,len(right_first),"Left Average","Gray")
        
    ax2.plot(left_average,marker='o')
    N = np.array(left_average)
    ranges = np.arange(0,len(left_average))
    ax2.fill_between(ranges,N-Final_error_l,N+Final_error_l,alpha=0.8)  
    ax2.errorbar(ranges,left_average,yerr = Final_error_l,fmt='none')

    ax3.plot(right_average,marker='o')
    N = np.array(right_average)
    ranges = np.arange(0,len(right_average))
    ax3.fill_between(ranges,N-Final_error_r,N+Final_error_r,alpha=0.8)  
    ax3.errorbar(ranges,right_average,yerr = Final_error_r,fmt='none')
        
    ax = [ax0,ax1,ax2,ax3]
    
    ax0.set_title("First Left Segment Angles",fontsize = 20)
    ax1.set_title("First Right Segment Angles",fontsize = 20)
    ax2.set_title("Average Left Segment",fontsize = 20)
    ax3.set_title("Average Right Segment",fontsize = 20)
    
    for axs in ax:
        plt.sca(axs)
        plt.ylabel('Angle (degree)',fontsize = 20)
        plt.xlabel('Time (milliseconds)',fontsize = 20)
        plt.xticks(np.arange(0,max_xlimit,2),np.arange(0,max_xlimit,2)*5, fontsize=20)
        plt.yticks(np.arange(limits[0],limits[1],step = 20),fontsize=20)
        plt.legend(names)
    plt.show()
    
    print()
    return left_average,right_average,Final_error_l,Final_error_r

def information(data):
    mindata = round(min(data),2)
    maxdata = round(max(data),2)
    print(f'The Min for this plot is {mindata} and the max for this plot is {maxdata}, that max the absolute {abs(mindata) + maxdata}' )
    

############################################ Functions not used ##########################################################



def Average_graph(data,Angles,title):
    print("This is where the power strokes where found\n")
    print(data)
    print("\nThere is " + str(len(data)) + " PowerStrokes")
    starting_powerstroke = int(input("What power stroke to start with?")) #skipping first item in list so if 1 will look at index 1
    end_powerstroke = int(input("What power stroke to end with with? "))#power strokes to stop at in list
    print("\nLooking at power strokes data ")
    print(data[starting_powerstroke:end_powerstroke+1])
    
    Average_plot = []
    
    max_xlimit = max_number_power_strokes(Angles,data,[starting_powerstroke,end_powerstroke+1])
    print(max_xlimit)
    
    list_of_angles = []
    for j in range(starting_powerstroke,end_powerstroke+1):
        list_of_angles.append(Angles[data[j][0]:(data[j][1]+1)])
        #plt.plot(Angles[data[j][0]:(data[j][1]+1)], color="blue",marker = 'o')
    #print(list_of_angles)

    for j in range(max_xlimit):
        #print(j)
        Average = [] 
        #count = 0
        for i in range(0,len(list_of_angles)):
            if (len(list_of_angles[i]) < j+1):
                continue
            #count +=1 
            #print(list_of_angles[i])
            Average.append(list_of_angles[i][j])
        #print("This is the count: " + str(count))
        Average_plot.append(np.mean(Average))
    print("This is the average list")
        
    print(Average_plot)
        
    plt.plot(Average_plot, color="blue",marker = 'o')
    plt.title("First " + title + " Segment Angles",fontsize = 24)
    plt.ylabel('Angle (degree)',fontsize = 20)
    plt.xlabel('Time (milliseconds)',fontsize = 20)
    plt.xticks(np.arange(0,max_xlimit,2),np.arange(0,max_xlimit,2)*10, fontsize=20)
    plt.yticks(np.arange(-60,100,step = 20),fontsize=20)
    #plt.legend(fontsize="20")
    #plt.show()

    return Average_plot


def plot_images(r, array,pic):#plot randome images shown below ; r as random images to choose and array where images will be choosen
    plt.imshow(pic[r])
    plt.title(r)
    plt.scatter(array[0][1][0][r], array[0][1][1][r], color="blue", label="Left Hinge Angle")
    plt.scatter(array[0][0][0][r], array[0][0][1][r], color="blue", marker="x")
    plt.scatter(array[1][1][0][r], array[1][1][1][r], color="green", label="Right Hinge Angle")
    plt.scatter(array[1][0][0][r], array[1][0][1][r], color="green", marker="x")
                                                          
    plt.show()
