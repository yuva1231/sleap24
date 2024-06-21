def MSD(data):
    import math
    import numpy as np
    u = 0
    allmsd = []
    while u < len(data):
        r = []
        for i in range(0, len(data[u])):
            if math.isnan(data[u][i]) == False:
                r.append(np.sqrt(data[u][i]**2 + data[u+1][i]**2))       
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
        u = u + 2
    
    return allmsd


def lengths(x):
    if isinstance(x,list):
        yield len(x)
        for y in x:
            yield from lengths(y)
            

def AVG(allmsd):
    import numpy as np
    average = np.zeros(max(lengths(allmsd)))
    count = np.zeros(max(lengths(allmsd)))
    for msd in allmsd:
        for i in range(0, len(msd)):
            average[i] = average[i] + msd[i]
            count[i] = count[i] + 1;
            
    for i in range(0, len(count)):
        average[i] = average[i]/count[i];
        
    return average