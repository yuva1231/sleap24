import matplotlib.pyplot as plt
import numpy as np
#import functions 
import pandas as pd


    
def file(filename):
    import h5py
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
    for i, name in enumerate(node_names):
        print(f"{i}: {name}")
    print()
    
    return locations.transpose()

def checking(middle_data,iteration_frames):
    import math
    
    node_missing = []
    for iteration_daphnia in range(0,len(middle_data)):
        if(math.isnan(middle_data[iteration_daphnia][0][0][iteration_frames])):
            node_missing.append(iteration_daphnia)
    return node_missing        
    
    
def daphnia_data(data):
    import math
    #left node L1 (closest to body),L2 (tip of first segment)
    x_middle = []
    y_middle = []
    
    for i in range(0,len(data[0][0][0])):
        instance_lst = []
        instance_lst = checking(data,i)
        if len(instance_lst) >= 1:
            print("Data does not exits at frame: " + str(i+1) + " ", end = " ")
            for j in instance_lst: 
                print(j, end = " ")
            print()
            
    for n in range(0,len(data)):
        x_middle.append(data[n][0][0])
        y_middle.append(data[n][1][0])
       
    return [x_middle,y_middle]
    #return "Error check data"

def filters_sleap(data,cutoff):
    X = data[0]#x data 
    Y = data[1]#y data
    Y = [(cutoff - i) for i in Y]
    return [X],[Y]


def lengths(x):
    if isinstance(x,list):
        yield len(x)
        for y in x:
            yield from lengths(y)



#Finds the mean square displacement
def msd(x, y):
    import math
    import numpy as np
    
    #calculates the msd
    u = 0
    allmsd = []
    while u < len(x):
        r = []
        for i in range(0, len(x[u])):
            if math.isnan(x[u][i]) == False and math.isnan(y[u][i]) == False:
                r.append(np.sqrt(x[u][i]**2 + y[u][i]**2))
        MSD = []
        for t in range(0, len(r)):
            diff = []
            i = 0
            while i < (len(r)-t):
                diff.append(r[t + i] - r[i])
                i = i + 1
            diff_sq = [a**2 for a in diff]
            MSD.append(np.mean(diff_sq))
        allmsd.append(MSD)
        u = u + 1
        
    average = np.zeros(max(lengths(allmsd)))
    count = np.zeros(max(lengths(allmsd)))
    std = np.zeros((max(lengths(allmsd)), len(allmsd)))
    j = 0
    for msd in allmsd:
        for i in range(0, len(msd)):
            average[i] = average[i] + msd[i]
            std[i][j] = msd[i]
            count[i] = count[i] + 1;
        j = j + 1
    
    for i in range(0, len(count)):
        average[i] = average[i]/count[i];

    st = []
    for s in std:
        st.append(np.var(s))
    #, average, st
    return allmsd, average, st

    
#Find the speed, change in speed, direction and change in direction    
def velfinder(x, y):
    import numpy as np
    
    speed = []
    direc = []
    for i in range(0, len(x)):
        velo = []
        dire = []
        for j in range(0, len(x[i])-1):
            xdir = x[i][j+1] - x[i][j]
            ydir = y[i][j+1] - y[i][j]
            if(xdir != 0):
                dire.append(np.arctan(ydir/xdir))
            else:
                dire.append(np.arctan(ydir/0.001))
            velo.append(np.sqrt(xdir**2 + ydir**2))

        speed.append(velo)
        direc.append(dire)
        
    speedacc = []
    diffdirec = []
    for i in speed:
        speedacc.append(np.diff(i))
        
    for i in direc:
        diffdirec.append(np.diff(i))
    return speed, speedacc, direc, diffdirec

def PowerStrokes(speed, acc, threshold, peak_difference, peak_together):
    import numpy as np
    
    #This code below finds all the peaks and then puts them into bouts
    positions_of_peaks = []
    peak_displacement = []
    
    for spe in speed:
        last_peak = 0;
        j = 0;
        peaks = []
        pd = []
        for i in range(1, len(spe)-1):
            
            #Filters out double peaks
            if spe[i+1] < spe[i] and spe[i-1] < spe[i]:
                
                #Only if it is a peak
                pre_peak_displacement = spe[i+1] - spe[i]#post peak displacement
                post_peak_displacement = spe[i] - spe[i-1] #pre peak displacement
                max_peak_displacement = max(abs(pre_peak_displacement), abs(post_peak_displacement))
                
                pd.append([i, max_peak_displacement])
                
                
                #(i-last_peak > 2) is a 2 frame buffer in between videos
                # has to be above th eminimum peak value 
                if spe[i] > threshold and  max_peak_displacement > peak_difference and (i-last_peak > 1):     
                    peaks.append(i)   
                    last_peak = i
        
        #How high the peaks are
        peak_displacement.append(pd)
        
        positions_of_peaks.append(peaks)
        j += 1;
        
        
        
    #Putting the peaks into BOUTS based on the 'peak_together value'
    ps = positions_of_peaks 
    multiple_power_strokes = []
    
    for j in range(0, len(ps)):
        count = 1;
        last_i = positions_of_peaks[0][0]-2;
        mps = []

        for i in range(0, len(ps[j])-1):
            
            if (ps[j][i+1] - ps[j][i]) > peak_together:
                mps.append([last_i, ps[j][i], count])
                last_i = ps[j][i+1]
                count = 1
            else:
                count += 1
                
                
        multiple_power_strokes.append(mps)
        
    return positions_of_peaks, multiple_power_strokes, peak_displacement

#-----------------------------------------------------------------------------------------------------------------------------------
#Looking at Power Stroke Bouts and POST Power Stroke Bouts

def PowerStrokeBout(power_strokes, x, y, speed):
    power_stroke_bout = []
    
    #looping through all the power strokes 
    for i in range(0, len(power_strokes)):
        #print(i)
        psc = []
        for j in range(0, len(power_strokes[i])):
            #print(j)
            
            #this is always true except for the first Power Stroke Bout
            if(power_strokes[i][j][0]-1 > 0):
                
                #total displacement
                t_displacement = np.sum(speed[i][power_strokes[i][j][0]-1:power_strokes[i][j][1]+1])
                
                #total horizontal displacement
                xdisp = x[i][power_strokes[i][j][0]-1:power_strokes[i][j][1]+1]
                xdispdiff = np.diff(xdisp)
                xdispdiffabs = [abs(i) for i in xdispdiff]
                xdispdiffabsum = np.sum(xdispdiffabs)
                #total vertical displacement
                
                ydisp = y[i][power_strokes[i][j][0]-1:power_strokes[i][j][1]+1]
                ydispdiff = np.diff(ydisp)
                ydispdiffabs = [abs(i) for i in ydispdiff]
                ydispdiffabsum = np.sum(ydispdiffabs)
                
                #net vertical displacment
                vertical_displacement_r = y[i][power_strokes[i][j][1]+1] - y[i][power_strokes[i][j][0]-1]
                
                #net horizontal displacment
                horizontal_displacement_r = x[i][power_strokes[i][j][1]+1] - x[i][power_strokes[i][j][0]-1]
                
                #making sure that we dont divide by 0
                if(t_displacement == 0):
                    t_displacement = 0.001;

                #NET displacement
                total_displ = np.sqrt(vertical_displacement_r**2 + horizontal_displacement_r**2)

                #ration between net displacement and total displcament (NDGR. ask moumita!)
                NDGR = total_displ/t_displacement

                #ratio between net vertical and net horizontal displacement
                displacement_ratio = vertical_displacement_r/horizontal_displacement_r
                
                            
                #This is the format in which we return the power stroke bouts


                #[start frame, 
                #end frame, 
                #number of power strokes in that time
                psc.append([power_strokes[i][j][0]-1, #start frame
                            power_strokes[i][j][1]+1, #end frame
                            power_strokes[i][j][2], #number of power strokes during this interval
                            displacement_ratio, #ratio between net vertical and net horizontal displacement
                            vertical_displacement_r, #vertical displacement
                            horizontal_displacement_r, #horizontal displacement
                            NDGR, #ration between net displacement and total displcament (NDGR. ask moumita!)
                            xdispdiffabsum,#total horizontal displacement
                            ydispdiffabsum,#total vertical displacement
                            t_displacement#total displacement
                           ])
            
                
                
            else:#making sure that the first Power Stroke Bout doesnt start at -1
                
                
                #total displacement
                t_displacement = np.sum(speed[i][power_strokes[i][j][0]:power_strokes[i][j][1]+1])
                
                #total horizontal displacement
                xdisp = x[i][power_strokes[i][j][0]:power_strokes[i][j][1]+1]
                xdispdiff = np.diff(xdisp)
                xdispdiffabs = [abs(i) for i in xdispdiff]
                xdispdiffabsum = np.sum(xdispdiffabs)
                
                #total vertical displacement
                ydisp = y[i][power_strokes[i][j][0]:power_strokes[i][j][1]+1]
                ydispdiff = np.diff(ydisp)
                ydispdiffabs = [abs(i) for i in ydispdiff]
                ydispdiffabsum = np.sum(ydispdiffabs)
                
                #net vertical displacment
                vertical_displacement_r = y[i][power_strokes[i][j][1]+1] - y[i][power_strokes[i][j][0]]
                
                #net horizontal displacment
                horizontal_displacement_r = x[i][power_strokes[i][j][1]+1] - x[i][power_strokes[i][j][0]]
            
            
                #making sure that we dont divide by 0
                if(t_displacement == 0):
                    t_displacement = 0.001;

                #NET displacement
                total_displ = np.sqrt(vertical_displacement_r**2 + horizontal_displacement_r**2)

                #ration between net displacement and total displcament (NDGR. ask moumita!)
                NDGR = total_displ/t_displacement

                #ratio between net vertical and net horizontal displacement
                displacement_ratio = vertical_displacement_r/horizontal_displacement_r
                
                 #This is the format in which we return the power stroke bouts
            
            
                #[start frame, 
                #end frame, 
                #number of power strokes in that time
                psc.append([0, #start frame
                            power_strokes[i][j][1]+1, #end frame
                            power_strokes[i][j][2], #number of power strokes during this interval
                            displacement_ratio, #ratio between net vertical and net horizontal displacement
                            vertical_displacement_r, #net vertical displacement
                            horizontal_displacement_r, #net horizontal displacement
                            NDGR, #ration between net displacement and total displcament (NDGR. ask moumita!)
                            xdispdiffabsum,#total horizontal displacement
                            ydispdiffabsum,#total vertical displacement
                            t_displacement#total displacement
                           ])
            
            
            
        power_stroke_bout.append(psc)
        
       
        
    return power_stroke_bout
    
def PostPowerStrokeBout(power_strokes, x, y, speed):
    post_power_stroke_bout = []
    
    first_event = True

    for i in range(0, len(power_strokes)):
        psc = []
        for j in range(0, len(power_strokes[i])-1):
            #total displacement
            t_displacement = np.sum(speed[i][power_strokes[i][j][1]+1:power_strokes[i][j+1][0]-1])
            
            #net vertical displacement
            vertical_displacement_r = y[i][power_strokes[i][j+1][0]-1] - y[i][power_strokes[i][j][1]+1]
            
            #net horizontal displacement
            horizontal_displacement_r = x[i][power_strokes[i][j+1][0]-1] - x[i][power_strokes[i][j][1]+1]
            
            #total horizontal displacement
            xdisp = x[i][power_strokes[i][j][1]+1:power_strokes[i][j+1][0]-1]
            xdispdiff = np.diff(xdisp)
            xdispdiffabs = [abs(i) for i in xdispdiff]
            xdispdiffabsum = np.sum(xdispdiffabs)

            #total vertical displacement
            ydisp = y[i][power_strokes[i][j][1]+1:power_strokes[i][j+1][0]-1]
            ydispdiff = np.diff(ydisp)
            ydispdiffabs = [abs(i) for i in ydispdiff]
            ydispdiffabsum = np.sum(ydispdiffabs)
           
            
            #making sure we dont divide by 0
            if(horizontal_displacement_r != 0):
                
                #ratio between net vertical and net horizontal displacement
                displacement_ratio = vertical_displacement_r/horizontal_displacement_r
            else:
                #just making it bigger than the threshold (which is the slop that the ration needs to be above)
                #no need to make it bigegr than that
                displacement_ratio = 10 
              
            
            #making sure we dont divide by 0
            if(t_displacement == 0):
                t_displacement = 0.001;
           
        
            #NET displacement
            total_displ = np.sqrt(vertical_displacement_r**2 + horizontal_displacement_r**2)
            
            #ration between net displacement and total displcament (NDGR. ask moumita!)
            NDGR = total_displ/t_displacement
            
            
            
            #If the first power stroke is not right in the beginning of the video the code below sets a POST power stroke 
            #as the first event. This is mainly necessary when the daphnia is very inactive. Otherwise it doesnt 
            #matter very much
            if(first_event == True):
                
                t_displacement = np.sum(speed[i][0:power_strokes[i][j+1][0]-1])

                vertical_displacement_r = y[i][power_strokes[i][j+1][0]-1] - y[i][0]
                horizontal_displacement_r = x[i][power_strokes[i][j+1][0]-1] - x[i][0]
                total_displ = np.sqrt(vertical_displacement_r**2 + horizontal_displacement_r**2)
                
                #total horizontal displacement
                xdisp = x[i][0:power_strokes[i][j+1][0]-1]
                xdispdiff = np.diff(xdisp)
                xdispdiffabs = [abs(i) for i in xdispdiff]
                xdispdiffabsum = np.sum(xdispdiffabs)

                #total vertical displacement
                ydisp = y[i][0:power_strokes[i][j+1][0]-1]
                ydispdiff = np.diff(ydisp)
                ydispdiffabs = [abs(i) for i in ydispdiff]
                ydispdiffabsum = np.sum(ydispdiffabs)
                
                if(horizontal_displacement_r != 0):
                    displacement_ratio = vertical_displacement_r/horizontal_displacement_r
                else:
                    displacement_ratio = 10
                if(t_displacement == 0):
                    t_displacement = 0.001; 
                    
                psc.append([0, #first frame
                            power_strokes[i][j][0], #last frame
                            power_strokes[i][j][0], #length of the interval (DIFFERENT from Power Strokes BOUT value in this place)
                            displacement_ratio, #ratio between net vertical and net horizontal displacement
                            vertical_displacement_r, #net vertical Displacement
                            horizontal_displacement_r, #net horizontal Displacement
                            NDGR, #ration between net displacement and total displcament (NDGR. ask moumita!)
                            xdispdiffabsum,#total horizontal displacement
                            ydispdiffabsum,#total vertical displacement
                            t_displacement#total displacement
                           ])
                
                #once the first event happens we can set this value to false so we never enter the loop again
                first_event = False
                post_power_stroke_bout.append(psc)
                
                
            psc.append([power_strokes[i][j][1]+1, #first frame
                        power_strokes[i][j+1][0]-1, #last frame
                        power_strokes[i][j+1][0] - 1 - power_strokes[i][j][1] + 1 -2, #length of the interval 
                        displacement_ratio,  #ratio between net vertical and net horizontal displacement
                        vertical_displacement_r, #net vertical Displacement
                        horizontal_displacement_r, #net horizontal Displacement
                        NDGR, #ration between net displacement and total displcament (NDGR. ask moumita!)
                        xdispdiffabsum,
                        ydispdiffabsum,
                        t_displacement
                       ])
                 
        
        
        
        

        t_displacement = np.sum(speed[i][power_strokes[i][j+1][1]:len(x[i])])
                                
        vertical_displacement_r = y[i][len(y[i])-1] - y[i][power_strokes[i][j+1][1]]
        horizontal_displacement_r = x[i][len(x[i])-1] - x[i][power_strokes[i][j+1][1]]
        total_displ = np.sqrt(vertical_displacement_r**2 + horizontal_displacement_r**2)

        #total horizontal displacement
        xdisp = x[i][power_strokes[i][j+1][1]:len(x[i])]
        xdispdiff = np.diff(xdisp)
        xdispdiffabs = [abs(i) for i in xdispdiff]
        xdispdiffabsum = np.sum(xdispdiffabs)

        #total vertical displacement
        ydisp = y[i][power_strokes[i][j+1][1]:len(y[i])]
        ydispdiff = np.diff(ydisp)
        ydispdiffabs = [abs(i) for i in ydispdiff]
        ydispdiffabsum = np.sum(ydispdiffabs)

        if(horizontal_displacement_r != 0):
            displacement_ratio = vertical_displacement_r/horizontal_displacement_r
        else:
            displacement_ratio = 10
        if(t_displacement == 0):
            t_displacement = 0.001; 

        psc.append([power_strokes[i][j+1][1], #first frame
                    len(x[i])-1, #last frame
                    len(x[i])-1 - power_strokes[i][j+1][1], #length of the interval (DIFFERENT from Power Strokes BOUT value in this 
                    displacement_ratio, #ratio between net vertical and net horizontal displacement
                    vertical_displacement_r, #net vertical Displacement
                    horizontal_displacement_r, #net horizontal Displacement
                    NDGR, #ration between net displacement and total displcament (NDGR. ask moumita!)
                    xdispdiffabsum,
                    ydispdiffabsum,
                    t_displacement
                   ])

        #once the first event happens we can set this value to false so we never enter the loop again
        first_event = False


        post_power_stroke_bout.append(psc)

        
    return post_power_stroke_bout
    
    

def PlotData(index, windowsize, x, y, speed, acc, direction, diffdirection, scale, VideoDimension):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    acc[0] = [i**2 for i in acc[0]]
    
    #Plotting some data to see the magnite of speed and displacements
    #The x and y values are not adjusted with the scale

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1)
    x5 = np.linspace(windowsize[0], windowsize[1], windowsize[1]-windowsize[0])
     
    y = [i*scale for i in y[index][windowsize[0]:windowsize[1]]]
    ax0.plot(x5, y)
    ax0.scatter(x5, y, s=10, color="red")
    ax0.set_ylabel("Vertical Displacement [mm]", fontsize = 20)
    #ax0.set_ylim(0, VideoDimension[1]*scale)
                 
                 
    x = [i*scale for i in x[index][windowsize[0]:windowsize[1]]]
    ax1.plot(x5, x)
    ax1.scatter(x5, x, s=10, color="red")
    ax1.set_ylabel("Horizontal Displacement [mm]", fontsize = 20)
    #ax1.set_ylim(0, VideoDimension[0]*scale)
    
    speed = [i*1 for i in speed[index][windowsize[0]:windowsize[1]]]#does not changethe scale to mm like x and y 
    ax2.plot(x5, speed)
    ax2.scatter(x5, speed, s=10, color="red")
    ax2.set_ylabel("Speed pixels/s", fontsize = 20)

    ax3.plot(x5, acc[index][windowsize[0]:windowsize[1]])
    ax3.scatter(x5, acc[index][windowsize[0]:windowsize[1]], s=10, color="red")
    ax3.set_ylabel("Acceleration^2", fontsize = 20)

    ax4.plot(x5, direction[index][windowsize[0]:windowsize[1]])
    ax4.scatter(x5, direction[index][windowsize[0]:windowsize[1]], s=10, color="red")
    ax4.set_ylabel("Direction", fontsize = 20)
    ax4.set_xlabel("Time [frames]", fontsize = 20)

    fig.set_size_inches(18, 18, forward=True)
    plt.show()

    
    return
    
    
def PlotTrajectory(x, y, trajectoryDistance, VideoDimension, frameRate, scale):
    
    #This plots the trajectory. 
    
    if(VideoDimension[1] > VideoDimension[0]):
        dimension = [5, round(max(VideoDimension)/min(VideoDimension),1)*5]
        plt.figure(figsize=(dimension[0], dimension[1]))
    else:
        dimension = [round(max(VideoDimension)/min(VideoDimension),1)*5, 5]
        plt.figure(figsize=(dimension[0], dimension[1]))
    
    xValues = [i*scale for i in x[trajectoryDistance[0]:trajectoryDistance[1]]]
    yValues = [i*scale for i in y[trajectoryDistance[0]:trajectoryDistance[1]]]
    
    plt.scatter(xValues, yValues,s=10);
    #plt.plot(xValues, yValues);
    
    print("Trajectory of the Daphnia [", (trajectoryDistance[0]/frameRate)*1000, "," , (trajectoryDistance[1]/frameRate )*1000 , "] ms or [ ", (trajectoryDistance[0]/frameRate), "," , (trajectoryDistance[1]/frameRate ) ,"] s")
    
    plt.xlabel("X [mm]", fontsize=20);
    plt.ylabel("Y [mm]", fontsize=20);
    plt.xlim(0, VideoDimension[0]*scale)
    plt.ylim(0, VideoDimension[1]*scale)
    
    
    plt.show()
    
    return


def FormatData(name):
    import pandas as pd
    
    #code to extract the data from the excel sheet
    
    data = pd.read_excel(name)
    x = [data['X'].dropna().tolist()]
    y = [data['Y'].dropna().tolist()]
    L = [data['L'].dropna().tolist()]
    W = [data['W'].dropna().tolist()]
    speed = [data['SPEED'].dropna().tolist()]
    acceleration = [data['ACCELERATION'].dropna().tolist()]
    direction = [data['DIRECTION'].dropna().tolist()]
    msd = [data['MSD'].dropna().tolist()]
    area = [data['AREA'].dropna().tolist()]

    scale = data['SCALE'].dropna().tolist()
    framerate = data['FRAMERATE'].dropna().tolist()
    power_strokes = []
    power_stroke_I = [data['POWER_STROKE_I'].dropna().tolist()]
    power_stroke_F = [data['POWER_STROKE_F'].dropna().tolist()]
    power_stroke_N = [data['POWER_STROKE_N'].dropna().tolist()]
    power_stroke_R = [data['POWER_STROKE_R'].dropna().tolist()]
    power_stroke_V = [data['POWER_STROKE_V'].dropna().tolist()]
    power_stroke_H = [data['POWER_STROKE_H'].dropna().tolist()]
    power_stroke_T = [data['POWER_STROKE_T'].dropna().tolist()]
    power_stroke_H_NET = [data['POWER_STROKE_H_NET'].dropna().tolist()]
    power_stroke_V_NET = [data['POWER_STROKE_V_NET'].dropna().tolist()]
    power_stroke_D_NET = [data['POWER_STROKE_D_NET'].dropna().tolist()]
    
    for i in range(0, len(power_stroke_H[0])):
        power_strokes.append([power_stroke_I[0][i], 
                              power_stroke_F[0][i], 
                              power_stroke_N[0][i], 
                              power_stroke_R[0][i], 
                              power_stroke_V[0][i], 
                              power_stroke_H[0][i], 
                              power_stroke_T[0][i],
                              power_stroke_H_NET[0][i], 
                              power_stroke_V_NET[0][i], 
                              power_stroke_D_NET[0][i]])
            
        
        
    post_power_strokes = []
    post_power_stroke_I = [data['POST_POWER_STROKE_I'].dropna().tolist()]
    post_power_stroke_F = [data['POST_POWER_STROKE_F'].dropna().tolist()]
    post_power_stroke_N = [data['POST_POWER_STROKE_N'].dropna().tolist()]
    post_power_stroke_R = [data['POST_POWER_STROKE_R'].dropna().tolist()]
    post_power_stroke_V = [data['POST_POWER_STROKE_V'].dropna().tolist()]
    post_power_stroke_H = [data['POST_POWER_STROKE_H'].dropna().tolist()]
    post_power_stroke_T = [data['POST_POWER_STROKE_T'].dropna().tolist()]
    post_power_stroke_H_NET = [data['POST_POWER_STROKE_H_NET'].dropna().tolist()]
    post_power_stroke_V_NET = [data['POST_POWER_STROKE_V_NET'].dropna().tolist()]
    post_power_stroke_D_NET = [data['POST_POWER_STROKE_D_NET'].dropna().tolist()]
    
    for i in range(0, len(post_power_stroke_H[0])):
        post_power_strokes.append([post_power_stroke_I[0][i], 
                                   post_power_stroke_F[0][i], 
                                   post_power_stroke_N[0][i], 
                                   post_power_stroke_R[0][i], 
                                   post_power_stroke_V[0][i], 
                                   post_power_stroke_H[0][i], 
                                   post_power_stroke_T[0][i],
                                   post_power_stroke_H_NET[0][i], 
                                   post_power_stroke_V_NET[0][i], 
                                   post_power_stroke_D_NET[0][i]
                                  ])

    peak_displacement_I = [data['PEAK_DISPLACEMENT_I'].dropna().tolist()]
    peak_displacement_V = [data['PEAK_DISPLACEMENT_V'].dropna().tolist()]
    
    peak_displacement = []
    for i in range(0, len(post_power_stroke_H[0])):
        peak_displacement.append([peak_displacement_I[0][i], peak_displacement_V[0][i]*scale[0]])

    
    return x, y, L, W, speed, acceleration, direction, msd, area, scale, power_strokes, post_power_strokes, peak_displacement, framerate




def SafeDataAnalysis(x, y, msd, direction,speed, acc, power_stroke, post_power_stroke, peak_displacement, minPrePostPeakSlope, minThreshold, accThreshold, framerate, videoDimension, enclosureDimension, scale, name,Bout_size):
    from xlwt import Workbook
    import xlwt
    #Saving all the data into an excel sheet
    #This is pretty self expenatory
    wb = Workbook()
    
    for daphnia in range(0,len(x)):
        sheet = wb.add_sheet('Sheet ' + str(daphnia+1))

        sheet.write(0, 0, 'X')
        sheet.write(0, 1, 'Y')
        #sheet.write(0, 2, 'L')
        #sheet.write(0, 3, 'W')
        #sheet.write(0, 4, 'O')

        sheet.write(0, 2, '')

        sheet.write(0, 3, 'MSD')
        sheet.write(0, 4, 'SPEED')
        sheet.write(0, 5, 'ACCELERATION')
        sheet.write(0, 6, 'DIRECTION')
        #sheet.write(0, 7, 'AREA')

        sheet.write(0, 7, '')

        sheet.write(0, 8, 'POWER_STROKE_I')
        sheet.write(0, 9, 'POWER_STROKE_F')
        sheet.write(0, 10, 'POWER_STROKE_N')
        sheet.write(0, 11, 'POWER_STROKE_R')
        sheet.write(0, 12, 'POWER_STROKE_V')
        sheet.write(0, 13, 'POWER_STROKE_H')
        sheet.write(0, 14, 'POWER_STROKE_T')
        sheet.write(0, 15, 'POWER_STROKE_H_NET')
        sheet.write(0, 16, 'POWER_STROKE_V_NET')
        sheet.write(0, 17, 'POWER_STROKE_D_NET')

        sheet.write(0, 18, '')

        sheet.write(0, 19, 'POST_POWER_STROKE_I')
        sheet.write(0, 20, 'POST_POWER_STROKE_F')
        sheet.write(0, 21, 'POST_POWER_STROKE_N')
        sheet.write(0, 22, 'POST_POWER_STROKE_R')
        sheet.write(0, 23, 'POST_POWER_STROKE_V')
        sheet.write(0, 24, 'POST_POWER_STROKE_H')
        sheet.write(0, 25, 'POST_POWER_STROKE_T')
        sheet.write(0, 26, 'POST_POWER_STROKE_H_NET')
        sheet.write(0, 27, 'POST_POWER_STROKE_V_NET')
        sheet.write(0, 28, 'POST_POWER_STROKE_D_NET')

        sheet.write(0, 29, '')

        sheet.write(0, 30, 'PEAK_DISPLACEMENT_I')
        sheet.write(0, 31, 'PEAK_DISPLACEMENT_V')

        sheet.write(0, 32, '')

        sheet.write(0, 33, 'MIN_PRE_POST_PEAK_SLOPE')
        sheet.write(0, 34, 'SPEED_THRESHOLD')
        sheet.write(0, 35, 'ACCELERATION_THRESHOLD')
        sheet.write(0, 36, 'VIDEO_DIMENSION_PIXELS')
        sheet.write(0, 37, 'ENCLOSURE_DIMENSION_MM')
        #sheet.write(0, 42, 'DAPHNIA_LENGTH')
        #sheet.write(0, 43, 'DAPHNIA_WIDTH')
        sheet.write(0, 38, 'SCALE')
        sheet.write(0, 39, 'FRAMERATE')
        sheet.write(0,40,  'Bout size')


        #Now writting the data
        minValue = min(len(x[daphnia]), len(y[daphnia]))
        for i in range(1, minValue):
            sheet.write(i, 0, x[daphnia][i])
            sheet.write(i, 1, y[daphnia][i])
            #sheet.write(i, 2, L[daphnia][i])
            #sheet.write(i, 3, W[daphnia][i])
            #sheet.write(i, 4, O[daphnia][i])
        #-------------------------------------
        for i in range(1, len(msd[daphnia])):
            sheet.write(i, 3, msd[daphnia][i])   
        for i in range(1, len(speed[daphnia])):
            sheet.write(i, 4, speed[daphnia][i])     
        for i in range(1, len(acc[daphnia])):
            sheet.write(i, 5, acc[daphnia][i])    
        for i in range(1, len(direction[daphnia])):
            sheet.write(i, 6, direction[daphnia][i])  
        #for i in range(1, len(area[daphnia])):
            #sheet.write(i, 10, area[daphnia][i])
        #-------------------------------------
        i = 1
        for ps in power_stroke[daphnia]:
            sheet.write(i, 8, ps[0])
            i += 1

        i = 1
        for ps in power_stroke[daphnia]:
            sheet.write(i, 9, ps[1])
            i += 1

        i = 1
        for ps in power_stroke[daphnia]:
            sheet.write(i, 10, ps[2])
            i += 1

        i = 1
        for ps in power_stroke[daphnia]:
            sheet.write(i, 11, ps[3])
            i += 1

        i = 1
        for ps in power_stroke[daphnia]:
            sheet.write(i, 12, ps[4])
            i += 1

        i = 1
        for ps in power_stroke[daphnia]:
            sheet.write(i, 13, ps[5])
            i += 1

        i = 1
        for ps in power_stroke[daphnia]:
            sheet.write(i, 14, ps[6])
            i += 1

        i = 1
        for ps in power_stroke[daphnia]:
            sheet.write(i, 15, ps[7])
            i += 1

        i = 1
        for ps in power_stroke[daphnia]:
            sheet.write(i, 16, ps[8])
            i += 1

        i = 1
        for ps in power_stroke[daphnia]:
            sheet.write(i, 17, ps[9])
            i += 1

        #-------------------------------------
        i = 1
        for ps in post_power_stroke[daphnia]:
            sheet.write(i, 19, ps[0])
            i += 1

        i = 1
        for ps in post_power_stroke[daphnia]:
            sheet.write(i, 20, ps[1])
            i += 1

        i = 1
        for ps in post_power_stroke[daphnia]:
            sheet.write(i, 21, ps[2])
            i += 1

        i = 1
        for ps in post_power_stroke[daphnia]:
            sheet.write(i, 22, ps[3])
            i += 1

        i = 1
        for ps in post_power_stroke[daphnia]:
            sheet.write(i, 23, ps[4])
            i += 1

        i = 1
        for ps in post_power_stroke[daphnia]:
            sheet.write(i, 24, ps[5])
            i += 1

        i = 1
        for ps in post_power_stroke[daphnia]:
            sheet.write(i, 25, ps[6])
            i += 1

        i = 1
        for ps in post_power_stroke[daphnia]:
            sheet.write(i, 26, ps[7])
            i += 1

        i = 1
        for ps in post_power_stroke[daphnia]:
            sheet.write(i, 27, ps[8])
            i += 1

        i = 1
        for ps in post_power_stroke[daphnia]:
            sheet.write(i, 28, ps[9])
            i += 1   


        #-------------------------------------
        i = 1
        for ps in peak_displacement[daphnia]:
            sheet.write(i, 30, ps[0])
            i += 1

        i = 1
        for ps in peak_displacement[daphnia]:
            sheet.write(i, 31, ps[1])
            i += 1

        #-------------------------------------   
        sheet.write(1, 33, minPrePostPeakSlope)
        sheet.write(1, 34, minThreshold)
        sheet.write(1, 35, accThreshold)
        sheet.write(1, 36, videoDimension)
        sheet.write(1, 37, enclosureDimension)
        #sheet.write(1, 42, daphniaLength)
        #sheet.write(1, 43, daphniaWidth)
        sheet.write(1, 38, scale)
        sheet.write(1, 39, framerate)
        sheet.write(1, 40, Bout_size)

    
    wb.save(name)
