# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 23:13:19 2015

@author: haslam
"""
import numpy as np
count = 0
trials = 100



def seatAllocator(N):
    import random
    seatsOccupied = []
    availableSeats = range(0,N)
    #seatNum = range(0,N)
    seatsEmpty = [] #list(availableSeats)
    #print "TEST"    
    
    #while len(seatsEmpty)+len(seatsOccupied) < N:
    while len(availableSeats) != 0:
        #print "available seats: ", availableSeats 
        seat = random.choice(availableSeats)
        #print "random num: ", seat
        seatPlus = seat+1
        seatMinus = seat-1
        if seat ==0 and seat in availableSeats and seat+1 in availableSeats:
            seatsEmpty = seatsEmpty + [seatPlus]
            seatsEmpty.sort()
            availableSeats.remove(seat)
            #value = seat+1
            if seatPlus in availableSeats:
                availableSeats.remove(seatPlus)
                pass
        elif seat ==N and seat in availableSeats and seat-1 in availableSeats:
            seatsEmpty = seatsEmpty + [seatMinus]
            seatsEmpty.sort()
            availableSeats.remove(seat)
            
            if seatMinus in availableSeats:
                availableSeats.remove(seatMinus)
                #seatsEmpty = seatsEmpty + value2
                pass
        elif seat in availableSeats: #and seat+1 in seatsEmpty and seat-1 in seatsEmpty:
        
            seatsEmpty.sort()
            availableSeats.remove(seat)
            if seatPlus in availableSeats:
                availableSeats.remove(seatPlus)
                seatsEmpty = seatsEmpty + [seatPlus]
                pass
            if seatMinus in availableSeats:
                availableSeats.remove(seatMinus)
                seatsEmpty = seatsEmpty + [seatMinus]
                pass
            seatsOccupied.sort(seatsOccupied.append(seat))
    
    fraction = len(seatsOccupied)/float(N)
            
    return fraction 

values = []
while count < trials:
    values.append(seatAllocator(50000))  
    count += 1        
print "mean: ", np.mean(values)
print "std dev: ", np.std(values)    