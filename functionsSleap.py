import matplotlib.pyplot as plt
import numpy as np
import functions 
import pandas as pd


#Filters the data provided from ctrax. getting ride of ghost daphnia. Short paths. etc
#This is not NEEDED if only one daphnia is succesfully tracked
def filters(address, cutoff):
    import pandas as pd
    import numpy as np
    
    x = []
    y = []
    O = []
    W = []
    L = []
    nd = [] #first turn numpy array into a list
    
    
    data = pd.read_csv(address)
    columnames = []
    for i in range(0, int(len(data.columns))):
            columnames.append('X'+str(i))
            columnames.append('Y'+str(i))
            columnames.append('L'+str(i))
            columnames.append('W'+str(i))
            columnames.append('O'+str(i))
     
    
    data = data.dropna(axis = 1, how ='all') #getting ride of all columns that dont have numbers
    
    data.columns = columnames[:len(data.columns)] #rename the column heads
    rows = data.shape[0]
    data = np.array(data)
    data = data.transpose()

    for i in range(0, len(data)):
        col = data[i]
        col = col[col != -1]
        if(len(col) > cutoff):
            nd.append(col)
        
    #for i in range(0, len(data)):
       # col = data[i]
       # col = col[col != 0]
       # if(len(col) > 300):
          #  nd.append(col)
   
            
    data = np.array(nd, dtype=object) #turn lsit back into a numpy array

    i = 1
    while i < len(nd)-1:
        x.append(data[i])
        y.append(data[i+1])
        L.append(data[i+2])
        W.append(data[i+3])
        O.append(data[i+4])
        i = i + 6
    
    return x, y, L, W, O




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


#Finds the Mean Square Change in Direction
def msdDirection(x):
    import math
    import numpy as np
    
    #calculates the msd
    allmsd = []
    r = []
    for i in range(0, len(x)):
            r.append(np.sqrt(x[i]**2))
    MSD = []
    for t in range(0, len(r)):
        diff = []
        i = 0
        while i < (len(r)-t):
            diff.append(r[t + i] - r[i])
            i = i + 1
        diff_sq = [a**2 for a in diff]
        MSD.append(np.mean(diff_sq))
        
    return MSD

#Finds the direction from the provided orientation from CTrax
def dirfinder(O):
    import numpy as np
    D = []
    for dum1 in O:
        n = int(len(dum1)-50)
        #getting rid of single value peaks
        for i in range(0, n-1):
            if(np.sign(dum1[i]) != np.sign(dum1[i+1]) and np.sign(dum1[i]) == np.sign(dum1[i+2])):
                dum1 = np.delete(dum1, i+1)

        for i in range(0, n-1):
            g = 0
            if(abs(dum1[i+1] - dum1[i]) > 2  and dum1[i+1] > dum1[i]):
                g = abs(dum1[i+1]) + abs(dum1[i])

                if(np.sign(dum1[i]) == np.sign(dum1[i+1]) and np.sign(dum1[i]) == -1):
                    g = abs(dum1[i]) - abs(dum1[i+1])
                elif(np.sign(dum1[i]) == np.sign(dum1[i+1]) and np.sign(dum1[i]) == 1):
                    g = abs(dum1[i+1]) - abs(dum1[i])


                for j in range(i+1, n):
                    dum1[j] = dum1[j] - g


            elif(abs(dum1[i+1] - dum1[i])>2 and dum1[i+1] < dum1[i]):
                g = abs(dum1[i+1]) + abs(dum1[i])

                if(np.sign(dum1[i]) == np.sign(dum1[i+1]) and np.sign(dum1[i]) == -1):
                    g = abs(dum1[i+1]) - abs(dum1[i])
                elif(np.sign(dum1[i]) == np.sign(dum1[i+1]) and np.sign(dum1[i]) == 1):
                    g = abs(dum1[i]) - abs(dum1[i+1])


                for j in range(i+1, n):
                    dum1[j] = dum1[j] + g

            if(g > 10):
                print("Error: Difference is wrong")
                
        D.append(dum1[:n])
    return D

#Finds how the area of the fitted elipse (from Ctrax) changes every frame
def areafinder(L, W):
    
    area = []
    for i in range(0, len(L)):
        a = []
        for j in range(0, len(L[i])):
            a.append(3.141*(L[i][j]/2)*(W[i][j]/2))

        area.append(a)
    return area
    
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
    
    
    
def safedata(x, y, L, W, O, name):
    from xlwt import Workbook
    import xlwt
    
    for i in range(0, len(x)):
        if(len(x[i]) == len(y[i]) == len(O[i]) == len(W[i]) == len(L[i])):
            print(i, "All the same length.")
        else:
            print(i, "Length x, y, L, W, O:", len(x[i]), len(y[i]), len(L[i]), len(W[i]), len(O[i]))


    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')

    i = 0
    k = 0
    while i < 5*len(x):
        sheet1.write(0,   i, 'X'+str(k))
        sheet1.write(0, i+1, 'Y'+str(k))
        sheet1.write(0, i+2, 'L'+str(k))
        sheet1.write(0, i+3, 'W'+str(k))
        sheet1.write(0, i+4, 'O'+str(k))
        print(k+1, len(O[k]))
        for j in range(1, len(O[k])-1):
            sheet1.write(j,   i, x[k][j])
            sheet1.write(j, i+1, y[k][j])
            sheet1.write(j, i+2, L[k][j])
            sheet1.write(j, i+3, W[k][j])
            sheet1.write(j, i+4, O[k][j])
        i = i + 6
        k = k + 1

    wb.save(name)
    
    
    
    
#---------------------------------------------------------------------------------------------------------------------------   
#---------------------------------------------------------------------------------------------------------------------------      
#---------------------------------------------------------------------------------------------------------------------------      
#The code below is used for the movement analysis of a single daphnia
    
    
    
    
    
    
    
    
def SafeDataAnalysis(x, y, L, W, O, msd, speed, acc, direction, area, power_stroke, post_power_stroke, peak_displacement, minPrePostPeakSlope, minThreshold, accThreshold, framerate, daphniaLength, daphniaWidth, videoDimension, enclosureDimension, scale, name):
    from xlwt import Workbook
    import xlwt
    #Saving all the data into an excel sheet
    #This is pretty self expenatory
    

    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')

    sheet1.write(0,   0, 'X')
    sheet1.write(0, 1, 'Y')
    sheet1.write(0, 2, 'L')
    sheet1.write(0, 3, 'W')
    sheet1.write(0, 4, 'O')
    
    sheet1.write(0, 5, '')
    
    sheet1.write(0, 6, 'MSD')
    sheet1.write(0, 7, 'SPEED')
    sheet1.write(0, 8, 'ACCELERATION')
    sheet1.write(0, 9, 'DIRECTION')
    sheet1.write(0, 10, 'AREA')
    
    sheet1.write(0, 11, '')
    
    sheet1.write(0, 12, 'POWER_STROKE_I')
    sheet1.write(0, 13, 'POWER_STROKE_F')
    sheet1.write(0, 14, 'POWER_STROKE_N')
    sheet1.write(0, 15, 'POWER_STROKE_R')
    sheet1.write(0, 16, 'POWER_STROKE_V')
    sheet1.write(0, 17, 'POWER_STROKE_H')
    sheet1.write(0, 18, 'POWER_STROKE_T')
    sheet1.write(0, 19, 'POWER_STROKE_H_NET')
    sheet1.write(0, 20, 'POWER_STROKE_V_NET')
    sheet1.write(0, 21, 'POWER_STROKE_D_NET')
    
    sheet1.write(0, 22, '')
    
    sheet1.write(0, 23, 'POST_POWER_STROKE_I')
    sheet1.write(0, 24, 'POST_POWER_STROKE_F')
    sheet1.write(0, 25, 'POST_POWER_STROKE_N')
    sheet1.write(0, 26, 'POST_POWER_STROKE_R')
    sheet1.write(0, 27, 'POST_POWER_STROKE_V')
    sheet1.write(0, 28, 'POST_POWER_STROKE_H')
    sheet1.write(0, 29, 'POST_POWER_STROKE_T')
    sheet1.write(0, 30, 'POST_POWER_STROKE_H_NET')
    sheet1.write(0, 31, 'POST_POWER_STROKE_V_NET')
    sheet1.write(0, 32, 'POST_POWER_STROKE_D_NET')
   
    sheet1.write(0, 33, '')
    
    sheet1.write(0, 34, 'PEAK_DISPLACEMENT_I')
    sheet1.write(0, 35, 'PEAK_DISPLACEMENT_V')
    
    sheet1.write(0, 36, '')
    
    sheet1.write(0, 37, 'MIN_PRE_POST_PEAK_SLOPE')
    sheet1.write(0, 38, 'SPEED_THRESHOLD')
    sheet1.write(0, 39, 'ACCELERATION_THRESHOLD')
    sheet1.write(0, 40, 'VIDEO_DIMENSION_PIXELS')
    sheet1.write(0, 41, 'ENCLOSURE_DIMENSION_MM')
    sheet1.write(0, 42, 'DAPHNIA_LENGTH')
    sheet1.write(0, 43, 'DAPHNIA_WIDTH')
    sheet1.write(0, 44, 'SCALE')
    sheet1.write(0, 45, 'FRAMERATE')

    
    #Now writting the data
    minValue = min(len(x), len(y), len(L), len(W), len(O))
    for i in range(1, minValue):
        sheet1.write(i, 0, x[i])
        sheet1.write(i, 1, y[i])
        sheet1.write(i, 2, L[i])
        sheet1.write(i, 3, W[i])
        sheet1.write(i, 4, O[i])
    #-------------------------------------
    for i in range(1, len(msd)):
        sheet1.write(i, 6, msd[i])   
    for i in range(1, len(speed)):
        sheet1.write(i, 7, speed[i])     
    for i in range(1, len(acc)):
        sheet1.write(i, 8, acc[i])    
    for i in range(1, len(direction)):
        sheet1.write(i, 9, direction[i])  
    for i in range(1, len(area)):
        sheet1.write(i, 10, area[i])
    #-------------------------------------
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 12, ps[0])
        i += 1
    
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 13, ps[1])
        i += 1
        
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 14, ps[2])
        i += 1
    
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 15, ps[3])
        i += 1
    
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 16, ps[4])
        i += 1
        
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 17, ps[5])
        i += 1
    
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 18, ps[6])
        i += 1
        
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 19, ps[7])
        i += 1
        
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 20, ps[8])
        i += 1
    
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 21, ps[9])
        i += 1
        
    #-------------------------------------
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 23, ps[0])
        i += 1
    
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 24, ps[1])
        i += 1
        
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 25, ps[2])
        i += 1
        
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 26, ps[3])
        i += 1
    
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 27, ps[4])
        i += 1
    
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 28, ps[5])
        i += 1
        
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 29, ps[6])
        i += 1
        
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 30, ps[7])
        i += 1
    
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 31, ps[8])
        i += 1
        
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 32, ps[9])
        i += 1   
        
        
    #-------------------------------------
    i = 1
    for ps in peak_displacement:
        sheet1.write(i, 34, ps[0])
        i += 1
    
    i = 1
    for ps in peak_displacement:
        sheet1.write(i, 35, ps[1])
        i += 1
        
    #-------------------------------------   
    sheet1.write(1, 37, minPrePostPeakSlope)
    sheet1.write(1, 38, minThreshold)
    sheet1.write(1, 39, accThreshold)
    sheet1.write(1, 40, videoDimension)
    sheet1.write(1, 41, enclosureDimension)
    sheet1.write(1, 42, daphniaLength)
    sheet1.write(1, 43, daphniaWidth)
    sheet1.write(1, 44, scale)
    sheet1.write(1, 45, framerate)
    
    
    wb.save(name)
    
    
    
def PowerStrokes(speed, acc, threshold, peak_difference, peak_together):
    import numpy as np
    
    #This code below finds all the peaks and then puts them into bouts
    positions_of_peaks = []
    peak_displacement = []
    
    last_peak = 0;
    j = 0;
    for spe in speed:
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
        psc = []
        for j in range(0, len(power_strokes[i])):
            
            
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
                            xdispdiffabsum,
                            ydispdiffabsum,
                            t_displacement
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
                            vertical_displacement_r, #vertical displacement
                            horizontal_displacement_r, #horizontal displacement
                            NDGR, #ration between net displacement and total displcament (NDGR. ask moumita!)
                            xdispdiffabsum,
                            ydispdiffabsum,
                            t_displacement
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
                            xdispdiffabsum,
                            ydispdiffabsum,
                            t_displacement
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
    
    

    
def PlotBehavior(power_strokes, min_power_strokes_ratio, min_displacement, post_power_strokes, min_post_power_strokes_ratio, min_vertical_displacement, x_restrictions, x, y, speed, scale, Radius, what_to_plot):
   
    import matplotlib.pyplot as plt
    import numpy as np
    
    hopping = [];zooming = [];sinking = [];drifting = [];hovering = [];hovering2 = []
    
    
    plt.figure(figsize=(20, 6))
    
    
   
    #In the Code below we categrozie the power stroke bouts into hopping, zooming and hovering based on 
    #min_power_strokes_ratio, totalDisplacement, min_displacement
    for ps in power_strokes:      
        #The displacement from the beginning of the bout to the end
        totalDisplacement = np.sqrt(ps[4]**2 + ps[5]**2)
        
        xcoord = np.linspace(int(ps[0]), int(ps[1]), int(ps[1])-int(ps[0]))
        ycoord = np.ones(len(xcoord))
        xcoord = [i/30 for i in xcoord]
        
        if(abs(ps[3]) > min_power_strokes_ratio and totalDisplacement > Radius*(1/scale)  and ps[4] > min_displacement*(1/scale)):

            plt.plot(xcoord, ycoord, color="red", marker='o') #Hopping
            if (ps[1] < 1200 and ps[0] < 1200):
                hopping.append(ps) 

        elif(totalDisplacement > Radius*(1/scale)):
            plt.plot(xcoord, ycoord, color="green", marker='o') #Zooming
            if (ps[1] < 1200 and ps[0] < 1200):
                zooming.append(ps)

        elif(totalDisplacement < Radius*(1/scale)):
            plt.plot(xcoord, ycoord, color="blue", marker='o') #Hovering
            if (ps[1] < 1200 and ps[0] < 1200):
                hovering.append(ps)

  
    #In the Code below we categrozie the POST power stroke bouts into sinking, drifting and hovering based on 
    #min_post_power_strokes_ratio, min_vertical_displacement, totalDisplacement           
    for ps in post_power_strokes:
        
        #print(ps[0], ps[1], int(ps[1])-int(ps[0]))
        if(int(ps[1])-int(ps[0]) < 0):
            xcoord = np.linspace(0, 1, 1)
            ycoord = np.zeros(len(xcoord))
            xcoord = [i/30 for i in xcoord]
        else:
            xcoord = np.linspace(int(ps[0]), int(ps[1]), int(ps[1])-int(ps[0]))
            ycoord = np.zeros(len(xcoord))
            xcoord = [i/30 for i in xcoord]
            
        #The displacement from the beginning of the bout to the end
        totalDisplacement = np.sqrt(ps[4]**2 + ps[5]**2)

        #Implementing the values that determien the categories
        if(abs(ps[3]) > min_post_power_strokes_ratio and ps[4] < min_vertical_displacement*(1/scale) and totalDisplacement > Radius*(1/scale)):
            plt.plot(xcoord, ycoord, color="red", marker='x') #Sinking
            if (ps[1] < 1200 and ps[0] < 1200):
                sinking.append(ps)

        elif(totalDisplacement > Radius*(1/scale)):
            plt.plot(xcoord, ycoord, color="green", marker='x') #Drifting
            if (ps[1] < 1200 and ps[0] < 1200):
                drifting.append(ps)

        elif(totalDisplacement < Radius*(1/scale)):
            plt.plot(xcoord, ycoord, color="blue", marker='x') #Hovering
            if (ps[1] < 1200 and ps[0] < 1200):
                hovering2.append(ps)

                    
                    
                    
                    
    #-------------------------------------------------------
    x5 = np.linspace(x_restrictions[0], x_restrictions[1], x_restrictions[1]-x_restrictions[0],dtype=int)
    x5 = [i/30 for i in x5]
    
    
    #-------------------------------------------------------
    #Plot the Speed 
    minimumSpeedIntervalValue = min(speed[0][x_restrictions[0]:x_restrictions[1]])
    speedData = [i - minimumSpeedIntervalValue for i in speed[0][x_restrictions[0]:x_restrictions[1]]]
    maximumSpeedIntervalValue = max(speedData)
    speedData = [i/maximumSpeedIntervalValue for i in speedData]
    
    if(what_to_plot[0] == True):
    
        for event in hopping:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], alpha=0.5, color="red")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], speedData[int(event[0]):int(event[1])+1], alpha=0.5, color="red")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
                
        for event in zooming:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], alpha=0.5, color="green")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], speedData[int(event[0]):int(event[1])+1], alpha=0.5, color="green")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
        for event in hovering:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], alpha=0.5, color="blue")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], speedData[int(event[0]):int(event[1])+1], alpha=0.5, color="blue")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")

        for event in sinking:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], alpha=0.5, color="red")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], speedData[int(event[0]):int(event[1])+1], alpha=0.5, color="red")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
        for event in drifting:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], alpha=0.5, color="green")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], speedData[int(event[0]):int(event[1])+1], alpha=0.5, color="green")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
        for event in hovering2:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], alpha=0.5, color="blue")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], speedData[int(event[0]):int(event[1])+1], alpha=0.5, color="blue")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
           
        
        
    #-------------------------------------------------------
    #Plot the Vertical Displacement 

    minimumYIntervalValue = min(y[0][x_restrictions[0]:x_restrictions[1]])
    ydata = [i - minimumYIntervalValue for i in y[0][x_restrictions[0]:x_restrictions[1]]]
    maximumYIntervalValue = max(ydata)
    ydata = [i/maximumYIntervalValue for i in ydata]

    if(what_to_plot[1] == True):
        for event in hopping:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], ydata[int(event[0])-1:int(event[1])+1], alpha=0.5, color="red")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], ydata[int(event[0]):int(event[1])+1], alpha=0.5, color="red")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
        for event in zooming:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], ydata[int(event[0])-1:int(event[1])+1], alpha=0.5, color="green")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], ydata[int(event[0]):int(event[1])+1], alpha=0.5, color="green")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
        for event in hovering:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], ydata[int(event[0])-1:int(event[1])+1], alpha=0.5, color="blue")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], ydata[int(event[0]):int(event[1])+1], alpha=0.5, color="blue")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")

        for event in sinking:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], ydata[int(event[0])-1:int(event[1])+1], alpha=0.5, color="red")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], ydata[int(event[0]):int(event[1])+1], alpha=0.5, color="red")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
        for event in drifting:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], ydata[int(event[0])-1:int(event[1])+1], alpha=0.5, color="green")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], ydata[int(event[0]):int(event[1])+1], alpha=0.5, color="green")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
        for event in hovering2:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], ydata[int(event[0])-1:int(event[1])+1], alpha=0.5, color="blue")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], ydata[int(event[0]):int(event[1])+1], alpha=0.5, color="blue")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
    
        
    #-------------------------------------------------------
    #Plot the Horizontal Displacement 
        
    minimumXIntervalValue = min(x[0][x_restrictions[0]:x_restrictions[1]])
    xdata = [i - minimumXIntervalValue for i in x[0][x_restrictions[0]:x_restrictions[1]]]
    maximumXIntervalValue = max(xdata)
    xdata = [i/maximumXIntervalValue for i in xdata]
    
    if(what_to_plot[2] == True):
        for event in hopping:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], xdata[int(event[0])-1:int(event[1])+1], alpha=0.5, color="red")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], xdata[int(event[0]):int(event[1])+1], alpha=0.5, color="red")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
        for event in zooming:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], xdata[int(event[0])-1:int(event[1])+1], alpha=0.5, color="green")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], xdata[int(event[0]):int(event[1])+1], alpha=0.5, color="green")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
        for event in hovering:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], xdata[int(event[0])-1:int(event[1])+1], alpha=0.5, color="blue")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], xdata[int(event[0]):int(event[1])+1], alpha=0.5, color="blue")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")

        for event in sinking:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], xdata[int(event[0])-1:int(event[1])+1], alpha=0.5, color="red")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], xdata[int(event[0]):int(event[1])+1], alpha=0.5, color="red")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
        for event in drifting:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], xdata[int(event[0])-1:int(event[1])+1], alpha=0.5, color="green")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], xdata[int(event[0]):int(event[1])+1], alpha=0.5, color="green")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
        for event in hovering2:
            if(event[0]!=0):
                plt.plot(x5[int(event[0])-1:int(event[1])+1], xdata[int(event[0])-1:int(event[1])+1], alpha=0.5, color="blue")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
            else:
                plt.plot(x5[int(event[0]):int(event[1])+1], xdata[int(event[0]):int(event[1])+1], alpha=0.5, color="blue")
                #plt.scatter(x5[int(event[0])-1:int(event[1])+1], speedData[int(event[0])-1:int(event[1])+1], s=10, color="red")
    
    #-----------------------------
    #Final Comands
    plt.xlabel("Time [sec]", fontsize=20)
    plt.xlim(x_restrictions[0]/30, x_restrictions[1]/30)
    plt.tick_params(axis='x',top = True,direction="in", which='both', labelsize=14, width=2, length=4)
    plt.yticks([]) #So that there are no y-ticks
    #plt.legend()
    plt.show()
    #plt.close()
    
    return hopping, zooming, sinking, drifting, hovering, hovering2

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
def PlotData2(index, windowsize, x, y, speed, acc, direction, diffdirection, scale, VideoDimension):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    acc[0] = [i**2 for i in acc[0]]
    
    #Plotting some data to see the magnite of speed and displacements
    #The x and y values are not adjusted with the scale
    
    speed = [(i-min(speed[index][windowsize[0]:windowsize[1]]))/max(speed[index][windowsize[0]:windowsize[1]]) for i in speed[index][windowsize[0]:windowsize[1]]]
        
    y = [(i-min(y[index][windowsize[0]:windowsize[1]]))/max(y[index][windowsize[0]:windowsize[1]]) for i in y[index][windowsize[0]:windowsize[1]]]
    print(x[index][windowsize[0]:windowsize[1]])
    x = [(i-min(x[index][windowsize[0]:windowsize[1]]))/max(x[index][windowsize[0]:windowsize[1]]) for i in x[index][windowsize[0]:windowsize[1]]]
    
    print(x)
    
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(10, 30))
    x5 = np.linspace(windowsize[0], windowsize[1], windowsize[1]-windowsize[0])
    x5ticks = [round(i/30, 3) for i in x5]
    
    ax0.plot(x5, y)
    ax0.scatter(x5, y, s=10, color="red")
    ax0.set_ylabel("Vertical Displacement [mm]", fontsize = 20)
    #ax0.set_ylim(0, VideoDimension[1]*scale)
                 
    ax1.plot(x5, x)
    ax1.scatter(x5, x, s=10, color="red")
    ax1.set_ylabel("Horizontal Displacement [mm]", fontsize = 20)
    #ax1.set_ylim(0, VideoDimension[0]*scale)
    
    ax2.plot(x5, speed)
    ax2.scatter(x5, speed, s=10, color="red")
    ax2.set_ylabel("Speed [mm/s]", fontsize = 20)

    ax3.plot(x5, acc[index][windowsize[0]:windowsize[1]])
    ax3.scatter(x5, acc[index][windowsize[0]:windowsize[1]], s=10, color="red")
    ax3.set_ylabel("Acceleration^2", fontsize = 20)
    
    fig.set_size_inches(25, 25, forward=True)
    plt.show()
    
    
    
    
    fig, (ax4) = plt.subplots(1, 1)
    
    vertical_horizontal_scale = max(y)/max(x)
    
    x = [(i-min(x))/max(x) for i in x]
    y = [(i-min(y))/max(y) for i in y]
    #speed = [(i-min(speed))/max(speed) for i in speed]
    x = [i/vertical_horizontal_scale for i in x]
    #y = [i/vertical_horizontal_scale for i in x]
    ax4.plot(x5, speed, label = 'Speed [mm/s]')
    ax4.scatter(x5, speed, s=10)
    
    ax4.plot(x5, x, label = 'Horizontal Displacement [mm]',color = "darkorange")
    ax4.scatter(x5, x, s=10,color = "darkorange")
    
    ax4.plot(x5, y, label = 'Vertical Displacement [mm]', color = "green")
    ax4.scatter(x5, y, s=10,color = "green")
    
    x5new = []
    for h in range(0, len(x5)):
        if(h % 4 == 0):
            x5new.append(x5[h])
            
            
    ax4.set_xticks(x5new)
    x5ticksnew = [round(i/30, 2) for i in x5new]
    ax4.set_xticklabels(x5ticksnew, rotation=80, fontsize=16)
    
    ax4.set_yticklabels([])
    ax4.set_xlabel("Time [seconds]", fontsize = 20)
    ax4.tick_params(left = False)


    fig.set_size_inches(20, 6, forward=True)
    plt.legend(loc = "upper left")
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
    plt.plot(xValues, yValues);
    
    print("Trajectory of the Daphnia [", (trajectoryDistance[0]/frameRate)*1000, "," , (trajectoryDistance[1]/frameRate )*1000 , "] ms or [ ", (trajectoryDistance[0]/frameRate), "," , (trajectoryDistance[1]/frameRate ) ,"] s")
    
    plt.xlabel("X [mm]", fontsize=20);
    plt.ylabel("Y [mm]", fontsize=20);
    plt.xlim(0, VideoDimension[0]*scale)
    plt.ylim(0, VideoDimension[1]*scale)
    
    
    plt.show()
    
    return
    
def PlotHist(x, y, peak_displacement, speed, acc):
    
    for index in range(0, len(x)):
        
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        histData = [i[1] for i in peak_displacement[index]]


        #This is a Histogram of the Average change in speed. This lets us filter out the most common peaks (really small ones)
        # minPrePostPeakSlope (Change this value in the previous code accordingly)
        ax0.hist(histData, bins=30)
        ax0.set_xlabel("Distance Between Peaks", fontsize=20)
        ax0.set_ylabel("Frequency", fontsize=20)
        m = int(max(histData))
        a = np.linspace(0, m, 2*m)
        a = [round(i, 1) for i in a]
        ax0.set_xticks(a)
        ax0.set_xticklabels(a, rotation=90)


        #This is a Histogram of the Average speed. This lets us filter out the most common speeds
        # minThreshold (Change this value in the previous code accordingly)
        ax1.hist(speed[index], bins=30)
        ax1.set_xlabel("Speed", fontsize=20)
        ax1.set_ylabel("Frequency", fontsize=20)
        m = int(max(speed[index]))
        a = np.linspace(0, m, 2*m)
        a = [round(i, 1) for i in a]
        ax1.set_xticks(a)
        ax1.set_xticklabels(a, rotation=90)



        #This is a Histogram of the Average acceleration. This lets us filter out the most common accelerations
        # accThreshold (Change this value in the previous code accordingly)
        ax2.hist(acc[index]**2, bins=30)
        ax2.set_xlabel("Acceleration Acerage^2 (Every 6 values average)", fontsize=20)
        ax2.set_ylabel("Frequency", fontsize=20)
        m = int(max(acc[index]**2))
        a = np.linspace(0, m, int(0.5*m))
        a = [round(i, 1) for i in a]
        ax2.set_xticks(a)
        ax2.set_xticklabels(a, rotation=90)


        fig.set_size_inches(18, 18, forward=True)
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





def PlotHZSD(all_hopping, all_zooming, all_sinking, all_drifting, all_hovering, all_hovering2, scale, minHopSlope, minSinkSlope, minSinkDist, minHopDist, indices, Radius, Lim):
    fig, (ax0, ax1) = plt.subplots(1, 2)
    #print(all_hopping)

    #checks a boolean value, so that we only print out the label for the first graph and no others
    LabelPrint = True
    
    
    #making different signs for the markers
    #markers = ["o", "x", "+", "v", ">", "<", "+", "s", "G", "F", "B", "L", "M"]
    markers = ["o", "x", "+", "v", ">", "<", "P", "s", "^","h","H","D","d","$f$",
               "*", "$a$", "$b$", "$c$", "$d$", "$e$", "$g$", "$h$", "$i$","$j$","$k$","$l$","$m$","$n$"]
    c = 0
    
    
    
    
    
    #Plotting all of the data as dots and triangles
    #----------------------------------------------------------------------------------------------------------------------
    
    
    for hopping in all_hopping[indices[0]:indices[1]] :
        for ps1 in hopping:

            if(LabelPrint):
                ax0.scatter(ps1[5]*scale[c], ps1[4]*scale[c], s=10*(ps1[1]-ps1[0]), color="red", marker=markers[c],  alpha=0.6, label= "Hopping")
                LabelPrint = False
            else:
                ax0.scatter(ps1[5]*scale[c], ps1[4]*scale[c], s=10*(ps1[1]-ps1[0]), color="red", marker=markers[c], alpha=0.6)
        c += 1
        
    LabelPrint = True 
    c = 0
    for zooming in all_zooming[indices[0]:indices[1]]:
        for ps in zooming:
            if(LabelPrint):
                ax0.scatter(ps[5]*scale[c], ps[4]*scale[c], s=10*(ps[1]-ps[0]), color="green", marker=markers[c],alpha=0.6, label= "Zooming")
                LabelPrint = False
            else:
                ax0.scatter(ps[5]*scale[c], ps[4]*scale[c], s=10*(ps[1]-ps[0]), color="green", marker=markers[c],alpha=0.6)
        c += 1
    
    LabelPrint = True 
    c = 0
    for hovering in all_hovering[indices[0]:indices[1]]:
        for ps in hovering:
            if(LabelPrint):
                ax0.scatter(ps[5]*scale[c], ps[4]*scale[c], s=10*(ps[1]-ps[0]), color="blue", marker=markers[c],alpha=0.6, label= "Hovering")
                LabelPrint = False
            else:
                ax0.scatter(ps[5]*scale[c], ps[4]*scale[c], s=10*(ps[1]-ps[0]), color="blue", marker=markers[c],alpha=0.6)

        c += 1

    xStraightLine = np.linspace(-100, 100, 200)
    yhorizontal = np.zeros(200)

    print("Size of Object plotted reflect the number of power strokes in a row (larger = more)")
    ax0.plot(xStraightLine, minHopSlope*xStraightLine, '--', color="black", label="Slope = " + str(minHopSlope))
    ax0.plot(xStraightLine, -minHopSlope*xStraightLine, '--', color="black")
    ax0.plot(xStraightLine, yhorizontal+minHopDist, '--', color="black")
    xCircle = np.linspace(0, 2*3.141592653, 100)
    ax0.plot(Radius*np.cos(xCircle), Radius*np.sin(xCircle),'--', color="black")
    ax0.set_ylabel("Vertical Displacement [mm]", fontsize=20)
    ax0.set_xlabel("Horizontal Displacement [mm]", fontsize=20)
    ax0.set_title("During Power Stroke Bout", fontsize=20)
    ax0.legend(loc="upper right")
    ax0.tick_params(axis='x', labelsize=18)
    ax0.tick_params(axis='y', labelsize=18)
    ax0.set_xlim(-8, 8)
    ax0.set_ylim(-8, 8)



#-------------------------------------------------------------------------------------------------------------------------------------
    LabelPrint = True
    c = 0
    for sinking in all_sinking[indices[0]:indices[1]]:
        for ps in sinking:
            if(LabelPrint):
                ax1.scatter(ps[5]*scale[c], ps[4]*scale[c], s=10*(ps[1]-ps[0]), color="red", marker=markers[c],alpha=0.6, label= "Sinking")
                LabelPrint = False
            else:
                ax1.scatter(ps[5]*scale[c], ps[4]*scale[c], s=10*(ps[1]-ps[0]), color="red", marker=markers[c],alpha=0.6)


        c += 1
    LabelPrint = True 
    
    c = 0
    for drifting in all_drifting[indices[0]:indices[1]]:
        for ps in drifting:
            if(LabelPrint):
                ax1.scatter(ps[5]*scale[c], ps[4]*scale[c], s=10*(ps[1]-ps[0]), color="green", marker=markers[c],alpha=0.6, label= "Drifting")
                LabelPrint = False
            else:
                ax1.scatter(ps[5]*scale[c], ps[4]*scale[c], s=10*(ps[1]-ps[0]), color="green", marker=markers[c],alpha=0.6)


        c += 1
        
        
        
    LabelPrint = True 
    
    c = 0
    for hovering2 in all_hovering2[indices[0]:indices[1]]:
        for ps in hovering2:
            if(LabelPrint):
                ax1.scatter(ps[5]*scale[c], ps[4]*scale[c], s=10*(ps[1]-ps[0]), color="blue", marker=markers[c],alpha=0.6, label= "Hovering")
                LabelPrint = False
            else:
                ax1.scatter(ps[5]*scale[c], ps[4]*scale[c], s=10*(ps[1]-ps[0]), color="blue", marker=markers[c],alpha=0.6)

        c += 1
        
        
    xStraightLine = np.linspace(-100, 100, 200)
    yhorizontal = np.zeros(200)

    ax1.plot(xStraightLine,  minSinkSlope*xStraightLine, '--', color="black", label="Slope = " + str(minSinkSlope))
    ax1.plot(xStraightLine, -minSinkSlope*xStraightLine, '--', color="black")
    ax1.plot(xStraightLine, yhorizontal+minSinkDist, '--', color="black")
    #ax0.set_tick_params(axis='both',top = True, right= True,direction="in", which='both', labelsize=14, width=2, length=4)
    xCircle = np.linspace(0,6.268, 100)
    ax1.plot(Radius*np.cos(xCircle), Radius*np.sin(xCircle), '--', color="black")
    ax1.set_ylabel("Vertical Displacement [mm]", fontsize=20)
    ax1.set_xlabel("Horizontal Displacement [mm]", fontsize=20)
    ax1.set_title("After Power Stroke Bout", fontsize=20)
    ax1.legend(loc="upper right")
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-8, 8)

    fig.set_size_inches((20, 10), forward=True)
    plt.show()
    
    return


def safeHZSDdata(all_hopping, all_zooming, all_sinking, all_drifting, name):
    from xlwt import Workbook
    import xlwt
    
    
    #saves the clasification data into an separate excel sheet (INCOMPLETE)
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True)
    sheet1.write(0, 0, 'Hopping_I')
    sheet1.write(0, 1, 'Hopping_F')
    sheet1.write(0, 2, 'Hopping_N')
    sheet1.write(0, 3, 'Hopping_R')
    sheet1.write(0, 4, 'Hopping_V')
    sheet1.write(0, 5, 'Hopping_H')
    
    prev = 0;
    for hopping in all_hopping:
        i = 0
        for j in range(prev, prev + len(hopping)):
            sheet1.write(j+1, 0, hopping[i][0])
            sheet1.write(j+1, 1, hopping[i][1])
            sheet1.write(j+1, 2, hopping[i][2])
            sheet1.write(j+1, 3, hopping[i][3])
            sheet1.write(j+1, 4, hopping[i][4])
            sheet1.write(j+1, 5, hopping[i][5])
            i += 1
        prev = prev + i
        
    sheet1.write(0, 7, 'Zooming_I')
    sheet1.write(0, 8, 'Zooming_F')
    sheet1.write(0, 9, 'Zooming_N')
    sheet1.write(0, 10, 'Zooming_R')
    sheet1.write(0, 11, 'Zooming_V')
    sheet1.write(0, 12, 'Zooming_H')

    
    prev = 0;
    for zooming in all_zooming:
        i = 0
        for j in range(prev, prev + len(zooming)):
            sheet1.write(j+1, 7, zooming[i][0])
            sheet1.write(j+1, 8, zooming[i][1])
            sheet1.write(j+1, 9, zooming[i][2])
            sheet1.write(j+1, 10, zooming[i][3])
            sheet1.write(j+1, 11, zooming[i][4])
            sheet1.write(j+1, 12, zooming[i][5])
            i += 1
        prev = prev + i



    sheet2 = wb.add_sheet('Sheet 2', cell_overwrite_ok=True)
    sheet2.write(0, 0, 'Sinking_I')
    sheet2.write(0, 1, 'Sinking_F')
    sheet2.write(0, 2, 'Sinking_R')
    sheet2.write(0, 3, 'Sinking_V')
    sheet2.write(0, 4, 'Sinking_H')
    
    prev = 0
    for sinking in all_sinking:
        i = 0
        for j in range(prev, prev + len(sinking)):
            sheet2.write(j+1, 0, sinking[i][0])
            sheet2.write(j+1, 1, sinking[i][1])
            sheet2.write(j+1, 2, sinking[i][2])
            sheet2.write(j+1, 3, sinking[i][3])
            sheet2.write(j+1, 4, sinking[i][4])
            i += 1
        prev = prev + i
        
    sheet2.write(0, 6, 'Drifting_I')
    sheet2.write(0, 7, 'Drifting_F')
    sheet2.write(0, 8, 'Drifting_R')
    sheet2.write(0, 9, 'Drifting_V')
    sheet2.write(0, 10, 'Drifting_H')
    
    prev = 0;
    for drifting in all_drifting:
        i = 0
        for j in range(prev, prev + len(drifting)):
            sheet2.write(j+1, 6, drifting[i][0])
            sheet2.write(j+1, 7, drifting[i][1])
            sheet2.write(j+1, 8, drifting[i][2])
            sheet2.write(j+1, 9, drifting[i][3])
            sheet2.write(j+1, 10, drifting[i][4])
            i += 1
        prev = prev + i
    
    wb.save(name);
    return





def Results(names, minHopSlope, minHopDist, minSinkSlope, minSinkDist, Radius, Lim, Size,show_all_graphs):
    
    
    print("red   = Hopping and Sinking")    
    print("green = Zoomin and Drifting")
    print("blue  = Hovering")
    
    #Initializing all the lists
    all_hopping = [];all_zooming = [];all_sinking = [];all_drifting = [];all_scale = [];all_hopfrac = [];
    all_sinkfrac = []; all_pslength = []; all_ppslength = []; all_number_of_ps = []; all_driftfrac = [];all_zoomfrac = [];
    all_pslengthframes = []; all_hovering = []; all_hovering2 = [];all_hoverfrac = [];all_hoverfrac2 = [];all_hopping_NGDR = [];all_zooming_NGDR = [];all_hovering_NGDR = [];all_sinking_NGDR = [];all_drifting_NGDR = [];all_hovering2_NGDR = [];eccentricity = [];

    
    #going through all the videos
    for name in names:
        
        #getting all the data from the excel files
        x, y, L, W, speed, acceleration, direction, msd, area, scale, power_strokes, post_power_strokes, peak_displacement, framerate = functions.FormatData(name)
                
        #calculating the eccentricity
        #eccen = []
        for i in range(0, len(L[0])):
            eccentricity.append(np.sqrt(1-W[0][i]**2/L[0][i]**2))
        #eccentricity.append(eccen)
        
        
        
        
        #appending the scale to our scale array
        all_scale.append(scale[0])
        
        #Plotting the gehavior and extraing the times in hopps sinks etc.... (same format as power strokes)
        hopping, zooming, sinking, drifting, hovering, hovering2 = functions.PlotBehavior(power_strokes, minHopSlope, minHopDist, post_power_strokes, minSinkSlope, minSinkDist, [Size[0], Size[1]], x, y, speed, scale[0], Radius, show_all_graphs)
        
        #calculating the fraction of times it hopps, zooms, drift,sinks,......
        hopfrac = len(hopping)/len(power_strokes)
        zoomfrac = len(zooming)/len(power_strokes)
        hoverfrac = len(hovering)/len(power_strokes)
        driftfrac = len(drifting)/len(post_power_strokes)
        sinkfrac = len(sinking)/len(post_power_strokes)
        hoverfrac2 = len(hovering2)/len(post_power_strokes)
        
        
        number_of_ps = 0
        
        #How long each power stroke bout is in ms
        for ps in power_strokes:
            all_pslength.append(ps[2])
            number_of_ps += ps[2]
            all_pslengthframes.append(((ps[1]-ps[0]+1)/framerate[0])*1000)
        
        #how long each post power stroke bout is in ms
        for pps in post_power_strokes:
            all_ppslength.append(((pps[1] - pps[0])/framerate[0])*1000)
        
               
        #I am subtracting 3 values because somehow the last 2 values of the acceleration are string values instead of float
        #I think there is some issue with the acceleration
        absacc = [abs(i) for i in acceleration[0][:len(acceleration[0])-3]] #absolute value of the acceleration

        
        #calculating the mean acceleration during the bouts (NOT NGDR)
        for ps in hopping:
            if(ps[0] == -1):
                ps[0] = 0
            dummy = np.mean(absacc[int(ps[0]):int(ps[1])+1])
            all_hopping_NGDR.append(dummy*1000*scale[0]/framerate[0]**2)
        for ps in zooming:
            if(ps[0] == -1):
                ps[0] = 0
            dummy = np.mean(absacc[int(ps[0]):int(ps[1])+1])
            all_zooming_NGDR.append(dummy*1000*scale[0]/framerate[0]**2)
        for ps in hovering:
            if(ps[0] == -1):
                ps[0] = 0
            dummy = np.mean(absacc[int(ps[0]):int(ps[1])+1])    
            all_hovering_NGDR.append(dummy*1000*scale[0]/framerate[0]**2)
            
            
        for ps in sinking:
            if(ps[0] == -1):
                ps[0] = 0
            dummy = np.mean(absacc[int(ps[0]):int(ps[1])+1])
            all_sinking_NGDR.append(dummy*1000*scale[0]/framerate[0]**2)
        for ps in drifting:
            if(ps[0] == -1):
                ps[0] = 0
            dummy = np.mean(absacc[int(ps[0]):int(ps[1])+1])
            all_drifting_NGDR.append(dummy*1000*scale[0]/framerate[0]**2)  
        for ps in hovering2:
            if(ps[0] == -1):
                ps[0] = 0
            dummy = np.mean(absacc[int(ps[0]):int(ps[1])+1])
            all_hovering2_NGDR.append(dummy*1000*scale[0]/framerate[0]**2)   
            
        
        all_number_of_ps.append(number_of_ps)
        
        all_hopfrac.append(hopfrac);
        all_sinkfrac.append(sinkfrac);
        all_zoomfrac.append(zoomfrac);
        all_driftfrac.append(driftfrac);
        all_hoverfrac.append(hoverfrac);
        all_hoverfrac2.append(hoverfrac2);
        
        all_hopping.append(hopping);
        all_sinking.append(sinking);
        all_drifting.append(drifting);
        all_zooming.append(zooming)
        all_hovering.append(hovering)
        all_hovering2.append(hovering2)

        print("There are a total of ", len(power_strokes), " Power Strokes. Out of these there are ", len(hopping), " Hops, and ", len(zooming), " Zooms and ", len(hovering), " Hoverings.")
        print("There are a total of ", len(post_power_strokes), " Post Power Strokes. Out of these there are ", len(sinking), " Sinks, and ", len(drifting), " Drifts and", len(hovering2), " Hovering.")
        #print("The two following numbers should be 0. If not the above calculations dont work out, ", len(power_strokes)-len(hopping)-len(zooming)-len(hovering) , " and ", len(post_power_strokes)-len(sinking)-len(drifting)-len(hovering2), " .")
        print(len(power_strokes)-len(hopping)-len(zooming)-len(hovering) + len(post_power_strokes)-len(sinking)-len(drifting)-len(hovering2) == 0)
        
        #Plotting the resulting graph of the hopping, drifting, zooming, etc..... (with the lines and hovering circle)
    functions.PlotHZSD(all_hopping, all_zooming, all_sinking, all_drifting, all_hovering, all_hovering2,all_scale, minHopSlope, minSinkSlope, minSinkDist, minHopDist, [0,len(all_hopping)], Radius, Lim)
        
    return all_hopfrac, all_sinkfrac, all_hoverfrac, all_zoomfrac, all_driftfrac, all_hoverfrac2, all_hopping, all_sinking, all_drifting, all_zooming, all_hovering, all_hovering2, all_number_of_ps, all_ppslength, all_pslengthframes, all_pslength, all_hopping_NGDR, all_zooming_NGDR, all_hovering_NGDR, all_sinking_NGDR, all_drifting_NGDR, all_hovering2_NGDR,eccentricity




def DirectionalDiffusion(all_hopping, all_zooming, all_hovering, all_sinking, all_drifting, all_hovering2, names):
    i = 0
    direction_diffusion_hopping = [];direction_diffusion_zooming = [];direction_diffusion_hovering = [];direction_diffusion_sinking = [];direction_diffusion_drifting = [];direction_diffusion_hovering2 = []
    
    len_of_data = 0 #how long does the bout have to be for us to do calculations with it
    
    for name in names:
        x, y,L, W, speed, acceleration, direction, msd, area, scale, power_strokes, post_power_strokes, peak_displacement, framerate = functions.FormatData(name)

        derivative_direction = np.diff(direction[0]) #taking the derivative of the direction

        for event in all_hopping[i]:
            if(event[1] - event[0] > len_of_data):
                
                #dummy = functions.msdDirection(derivative_direction[int(event[0]):int(event[1])])
                dummy = np.var(derivative_direction[int(event[0]):int(event[1])])
                #dummy = functions.msd(x[int(event[0]):int(event[1])], y[int(event[0]):int(event[1])])
                direction_diffusion_hopping.append(dummy)
        
        
        for event in all_zooming[i]:
            if(event[1] - event[0] > len_of_data):
                #dummy = functions.msdDirection(derivative_direction[int(event[0]):int(event[1])])
                dummy = np.var(derivative_direction[int(event[0]):int(event[1])])
                #dummy = functions.msd(x[int(event[0]):int(event[1])], y[int(event[0]):int(event[1])])
                direction_diffusion_zooming.append(dummy)
            
       
        for event in all_hovering[i]:
            if(event[1] - event[0] > len_of_data):
                #dummy = functions.msdDirection(derivative_direction[int(event[0]):int(event[1])])
                dummy = np.var(derivative_direction[int(event[0]):int(event[1])])
                #dummy = functions.msd(x[int(event[0]):int(event[1])], y[int(event[0]):int(event[1])])
                direction_diffusion_hovering.append(dummy)

        
        
        
        
        
        for event in all_sinking[i]:
            if(event[1] - event[0] > len_of_data):
                #dummy = functions.msdDirection(derivative_direction[int(event[0]):int(event[1])])
                dummy = np.var(derivative_direction[int(event[0]):int(event[1])])
                #dummy = functions.msd(x[int(event[0]):int(event[1])], y[int(event[0]):int(event[1])])
                direction_diffusion_sinking.append(dummy)
            
        
        for event in all_drifting[i]:
            if(event[1] - event[0] > len_of_data):
                #dummy = functions.msdDirection(derivative_direction[int(event[0]):int(event[1])])
                dummy = np.var(derivative_direction[int(event[0]):int(event[1])])
                #dummy = functions.msd(x[int(event[0]):int(event[1])], y[int(event[0]):int(event[1])])
                direction_diffusion_drifting.append(dummy)

        
        for event in all_hovering2[i]:
            if(event[1] - event[0] > len_of_data):
                #dummy = functions.msdDirection(derivative_direction[int(event[0]):int(event[1])])
                dummy = np.var(derivative_direction[int(event[0]):int(event[1])])
                #dummy = functions.msd(x[int(event[0]):int(event[1])], y[int(event[0]):int(event[1])])
                direction_diffusion_hovering2.append(dummy)
        
        
        #print(len(direction_diffusion_hopping), len(direction_diffusion_zooming), len(direction_diffusion_hovering), len(direction_diffusion_sinking), len(direction_diffusion_drifting), len(direction_diffusion_hovering2))

        i += 1
        
    return direction_diffusion_hopping, direction_diffusion_zooming, direction_diffusion_hovering, direction_diffusion_sinking, direction_diffusion_drifting, direction_diffusion_hovering2
    
####################################################Sleap########################################################################
def SafeDataAnalysis_sleap(x, y,msd, speed, acc,power_stroke, post_power_stroke, peak_displacement, minPrePostPeakSlope, minThreshold, accThreshold, framerate, daphniaLength, daphniaWidth, videoDimension, enclosureDimension, scale, name,bout_size):
    from xlwt import Workbook
    import xlwt
    #Saving all the data into an excel sheet
    #This is pretty self expenatory
    

    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')

    sheet1.write(0,   0, 'X')
    sheet1.write(0, 1, 'Y')
    sheet1.write(0, 2, 'L')
    sheet1.write(0, 3, 'W')
    sheet1.write(0, 4, 'O')
    
    sheet1.write(0, 5, '')
    
    sheet1.write(0, 6, 'MSD')
    sheet1.write(0, 7, 'SPEED')
    sheet1.write(0, 8, 'ACCELERATION')
    sheet1.write(0, 9, 'DIRECTION')
    sheet1.write(0, 10, 'AREA')
    
    sheet1.write(0, 11, '')
    
    sheet1.write(0, 12, 'POWER_STROKE_I')
    sheet1.write(0, 13, 'POWER_STROKE_F')
    sheet1.write(0, 14, 'POWER_STROKE_N')
    sheet1.write(0, 15, 'POWER_STROKE_R')
    sheet1.write(0, 16, 'POWER_STROKE_V')
    sheet1.write(0, 17, 'POWER_STROKE_H')
    sheet1.write(0, 18, 'POWER_STROKE_T')
    sheet1.write(0, 19, 'POWER_STROKE_H_NET')
    sheet1.write(0, 20, 'POWER_STROKE_V_NET')
    sheet1.write(0, 21, 'POWER_STROKE_D_NET')
    
    sheet1.write(0, 22, '')
    
    sheet1.write(0, 23, 'POST_POWER_STROKE_I')
    sheet1.write(0, 24, 'POST_POWER_STROKE_F')
    sheet1.write(0, 25, 'POST_POWER_STROKE_N')
    sheet1.write(0, 26, 'POST_POWER_STROKE_R')
    sheet1.write(0, 27, 'POST_POWER_STROKE_V')
    sheet1.write(0, 28, 'POST_POWER_STROKE_H')
    sheet1.write(0, 29, 'POST_POWER_STROKE_T')
    sheet1.write(0, 30, 'POST_POWER_STROKE_H_NET')
    sheet1.write(0, 31, 'POST_POWER_STROKE_V_NET')
    sheet1.write(0, 32, 'POST_POWER_STROKE_D_NET')
   
    sheet1.write(0, 33, '')
    
    sheet1.write(0, 34, 'PEAK_DISPLACEMENT_I')
    sheet1.write(0, 35, 'PEAK_DISPLACEMENT_V')
    
    sheet1.write(0, 36, '')
    
    sheet1.write(0, 37, 'MIN_PRE_POST_PEAK_SLOPE')
    sheet1.write(0, 38, 'SPEED_THRESHOLD')
    sheet1.write(0, 39, 'ACCELERATION_THRESHOLD')
    sheet1.write(0, 40, 'VIDEO_DIMENSION_PIXELS')
    sheet1.write(0, 41, 'ENCLOSURE_DIMENSION_MM')
    sheet1.write(0, 42, 'DAPHNIA_LENGTH')
    sheet1.write(0, 43, 'DAPHNIA_WIDTH')
    sheet1.write(0, 44, 'SCALE')
    sheet1.write(0, 45, 'FRAMERATE')
    sheet1.write(0, 46, 'Peak_together')

    
    #Now writting the data
    minValue = min(len(x), len(y))
    for i in range(1, minValue):
        sheet1.write(i, 0, x[i])
        sheet1.write(i, 1, y[i])
        sheet1.write(i, 2, 0)
        sheet1.write(i, 3, 0)
        sheet1.write(i, 4, 0)
    #-------------------------------------
    for i in range(1, len(msd)):
        sheet1.write(i, 6, msd[i])   
    for i in range(1, len(speed)):
        sheet1.write(i, 7, speed[i])     
    for i in range(1, len(acc)):
        sheet1.write(i, 8, acc[i])    
    for i in range(1, len(acc)):#direction
        sheet1.write(i, 9, 0)  
    for i in range(1, len(acc)):#area
        sheet1.write(i, 10, 0)
    #-------------------------------------
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 12, ps[0])
        i += 1
    
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 13, ps[1])
        i += 1
        
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 14, ps[2])
        i += 1
    
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 15, ps[3])
        i += 1
    
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 16, ps[4])
        i += 1
        
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 17, ps[5])
        i += 1
    
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 18, ps[6])
        i += 1
        
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 19, ps[7])
        i += 1
        
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 20, ps[8])
        i += 1
    
    i = 1
    for ps in power_stroke:
        sheet1.write(i, 21, ps[9])
        i += 1
        
    #-------------------------------------
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 23, ps[0])
        i += 1
    
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 24, ps[1])
        i += 1
        
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 25, ps[2])
        i += 1
        
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 26, ps[3])
        i += 1
    
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 27, ps[4])
        i += 1
    
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 28, ps[5])
        i += 1
        
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 29, ps[6])
        i += 1
        
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 30, ps[7])
        i += 1
    
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 31, ps[8])
        i += 1
        
    i = 1
    for ps in post_power_stroke:
        sheet1.write(i, 32, ps[9])
        i += 1   
        
        
    #-------------------------------------
    i = 1
    for ps in peak_displacement:
        sheet1.write(i, 34, ps[0])
        i += 1
    
    i = 1
    for ps in peak_displacement:
        sheet1.write(i, 35, ps[1])
        i += 1
        
    #-------------------------------------   
    sheet1.write(1, 37, minPrePostPeakSlope)
    sheet1.write(1, 38, minThreshold)
    sheet1.write(1, 39, accThreshold)
    sheet1.write(1, 40, videoDimension)
    sheet1.write(1, 41, enclosureDimension)
    sheet1.write(1, 42, daphniaLength)
    sheet1.write(1, 43, daphniaWidth)
    sheet1.write(1, 44, scale)
    sheet1.write(1, 45, framerate)
    sheet1.write(1, 46, bout_size)
    
    
    wb.save(name)
    
    
    
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
    
    return locations

def checking(angle_data,iteration):
    import math
    if(math.isnan(angle_data[iteration][0][0][0])):
            print("Data does not exits at frame: " + str(iteration+1) + " for middle")
            
def angles(data):
    import math
    #left node L1 (closest to body),L2 (tip of first segment)
    x_middle = []
    y_middle = []
    
    for i in range(0,len(data)):
        x_middle.append(data[i][0][0][0])
        y_middle.append(data[i][0][1][0])
        
        checking(data,i)
        
    check_in = input("Continue? ")    
    if (check_in == "yes" or "y"):
        return [x_middle,y_middle]
    return "Error check data"

def filters_sleap(data,cutoff):
    X = data[0]#x data 
    Y = data[1]#y data
    Y = [(cutoff - i) for i in Y]
    return [X],[Y]
    

def Results_sleap(names, minHopSlope, minHopDist, minSinkSlope, minSinkDist, Radius, Lim, Size,show_all_graphs):
    
    
    print("red   = Hopping and Sinking")    
    print("green = Zoomin and Drifting")
    print("blue  = Hovering")
    
    #Initializing all the lists
    all_hopping = [];all_zooming = [];all_sinking = [];all_drifting = [];all_scale = [];all_hopfrac = [];
    all_sinkfrac = []; all_pslength = []; all_ppslength = []; all_number_of_ps = []; all_driftfrac = [];all_zoomfrac = [];
    all_pslengthframes = []; all_hovering = []; all_hovering2 = [];all_hoverfrac = [];all_hoverfrac2 = [];all_hopping_NGDR = [];all_zooming_NGDR = [];all_hovering_NGDR = [];all_sinking_NGDR = [];all_drifting_NGDR = [];all_hovering2_NGDR = [];eccentricity = [];

    
    #going through all the videos
    for name in names:
        
        #getting all the data from the excel files
        x, y, L, W, speed, acceleration, direction, msd, area, scale, power_strokes, post_power_strokes, peak_displacement, framerate = functions.FormatData(name)
        
        power_strokes = [i for i in power_strokes if i[1] < 1200 and i[0] < 1200]
        post_power_strokes = [i for i in post_power_strokes if i[1] < 1200 and i[0] < 1200]
        
        #calculating the eccentricity
        #eccen = []
        #for i in range(0, len(L[0])):
            #eccentricity.append(np.sqrt(1-W[0][i]**2/L[0][i]**2))
        #eccentricity.append(eccen)
        
        
        
        
        #appending the scale to our scale array
        all_scale.append(scale[0])
        
        #Plotting the gehavior and extraing the times in hopps sinks etc.... (same format as power strokes)
        hopping, zooming, sinking, drifting, hovering, hovering2 = functions.PlotBehavior(power_strokes, minHopSlope, minHopDist, post_power_strokes, minSinkSlope, minSinkDist, [Size[0], Size[1]], x, y, speed, scale[0], Radius, show_all_graphs)
        
        #calculating the fraction of times it hopps, zooms, drift,sinks,......
        #print("POWER STROKE LENGTH : " + str(len(power_strokes)))
        hopfrac = len(hopping)/len(power_strokes)
        zoomfrac = len(zooming)/len(power_strokes)
 
        hoverfrac = len(hovering)/len(power_strokes)
     
        driftfrac = len(drifting)/len(post_power_strokes)
        sinkfrac = len(sinking)/len(post_power_strokes)
        hoverfrac2 = len(hovering2)/len(post_power_strokes)
        
        
        number_of_ps = 0
        
        #How long each power stroke bout is in ms
        for ps in power_strokes:
            if ps[0]<1200 and ps[1]< 1200:
                all_pslength.append(ps[2])
                number_of_ps += ps[2]
                all_pslengthframes.append(((ps[1]-ps[0]+1)/framerate[0])*1000)
        
        #how long each post power stroke bout is in ms
        for pps in post_power_strokes:
            if pps[0]<1200 and pps[1]< 1200:
                all_ppslength.append(((pps[1] - pps[0])/framerate[0])*1000)
        
               
        #I am subtracting 3 values because somehow the last 2 values of the acceleration are string values instead of float
        #I think there is some issue with the acceleration
        absacc = [abs(i) for i in acceleration[0][:len(acceleration[0])-3]] #absolute value of the acceleration

        
        #calculating the mean acceleration during the bouts (NOT NGDR)
        for ps in hopping:
            if(ps[0] == -1):
                ps[0] = 0
            dummy = np.mean(absacc[int(ps[0]):int(ps[1])+1])
            all_hopping_NGDR.append(dummy*1000*scale[0]/framerate[0]**2)
        for ps in zooming:
            if(ps[0] == -1):
                ps[0] = 0
            dummy = np.mean(absacc[int(ps[0]):int(ps[1])+1])
            all_zooming_NGDR.append(dummy*1000*scale[0]/framerate[0]**2)
        for ps in hovering:
            if(ps[0] == -1):
                ps[0] = 0
            dummy = np.mean(absacc[int(ps[0]):int(ps[1])+1])    
            all_hovering_NGDR.append(dummy*1000*scale[0]/framerate[0]**2)
            
            
        for ps in sinking:
            if(ps[0] == -1):
                ps[0] = 0
            dummy = np.mean(absacc[int(ps[0]):int(ps[1])+1])
            all_sinking_NGDR.append(dummy*1000*scale[0]/framerate[0]**2)
        for ps in drifting:
            if(ps[0] == -1):
                ps[0] = 0
            dummy = np.mean(absacc[int(ps[0]):int(ps[1])+1])
            all_drifting_NGDR.append(dummy*1000*scale[0]/framerate[0]**2)  
        for ps in hovering2:
            if(ps[0] == -1):
                ps[0] = 0
            dummy = np.mean(absacc[int(ps[0]):int(ps[1])+1])
            all_hovering2_NGDR.append(dummy*1000*scale[0]/framerate[0]**2)   
            
        
        all_number_of_ps.append(number_of_ps)
        
        all_hopfrac.append(hopfrac);
        all_sinkfrac.append(sinkfrac);
        all_zoomfrac.append(zoomfrac);
        all_driftfrac.append(driftfrac);
        all_hoverfrac.append(hoverfrac);
        all_hoverfrac2.append(hoverfrac2);
        
        all_hopping.append(hopping);
        all_sinking.append(sinking);
        all_drifting.append(drifting);
        all_zooming.append(zooming)
        all_hovering.append(hovering)
        all_hovering2.append(hovering2)
        
 

        print("There are a total of ", len(power_strokes), " Power Strokes. Out of these there are ", len(hopping), " Hops, and ", len(zooming), " Zooms and ", len(hovering), " Hoverings.")
        print("There are a total of ", len(post_power_strokes), " Post Power Strokes. Out of these there are ", len(sinking), " Sinks, and ", len(drifting), " Drifts and", len(hovering2), " Hovering.")
        #print("The two following numbers should be 0. If not the above calculations dont work out, ", len(power_strokes)-len(hopping)-len(zooming)-len(hovering) , " and ", len(post_power_strokes)-len(sinking)-len(drifting)-len(hovering2), " .")
        print(len(power_strokes)-len(hopping)-len(zooming)-len(hovering) + len(post_power_strokes)-len(sinking)-len(drifting)-len(hovering2) == 0)
        #print(len(all_hopping))
        #print(len(all_zooming))
        #Plotting the resulting graph of the hopping, drifting, zooming, etc..... (with the lines and hovering circle)
    functions.PlotHZSD(all_hopping, all_zooming, all_sinking, all_drifting, all_hovering, all_hovering2,all_scale, minHopSlope, minSinkSlope, minSinkDist, minHopDist, [0,len(all_hopping)], Radius, Lim)
        
    return all_hopfrac, all_sinkfrac, all_hoverfrac, all_zoomfrac, all_driftfrac, all_hoverfrac2, all_hopping, all_sinking, all_drifting, all_zooming, all_hovering, all_hovering2, all_number_of_ps, all_ppslength, all_pslengthframes, all_pslength, all_hopping_NGDR, all_zooming_NGDR, all_hovering_NGDR, all_sinking_NGDR, all_drifting_NGDR, all_hovering2_NGDR,eccentricity















