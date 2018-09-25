import math
import random as rand
import numpy as np
import sys

'''

USAGE:
    import isovist as iso
    isovist = iso.Isovist(polygonSegments) # See **BELOW** for Data Structure of MAP
    
    isIntruderFound = isovist.IsIntruderSeen(RRTPath, UAVLocation, UAVForwardVector, UAVFieldOfVision = 45)
 

EXAMPLE:
    insovist_main.py 


**BELOW**:

    # Data Structure for MAP
    polygonSegments = []
    
    #One Polygon 
    polygonSegments.append([ 
        [ (0,0),(840,0) ], 
        [ (840,0),(840,360) ],
        [ (840,360), (0,360)],
        [ (0,360), (0,0) ]
        ])   


'''

class Isovist:

    def __init__(self, polygon_map, load=False):
        if load:
            polygon_map = self.load_polygons()
        self.polygon_map = polygon_map
        self.uniquePoints = self.GetUniquePoints()

        #BY DEFAULT field of vision is set to 40 degrees
        self.UAVFieldOfVision = 40
        self.fieldOfVision = math.radians(self.UAVFieldOfVision/2.0)
        self.full = False


        all_segs = 0
        self._segs = np.zeros((323, 2, 2))
        for polygon in self.polygon_map:
            polygon_arr = np.array(polygon)
            self._segs[all_segs:all_segs+polygon_arr.shape[0],] = polygon_arr
            all_segs += polygon_arr.shape[0]


    def load_polygons(self, fn="./paths.txt" ):
        polygonSegments = []
        for line in open( fn ):
            line = line.strip('\n')
            toList = line.split(' ')
            toList = [(float(x)/1000) for x in toList]
            
            it = iter(toList)
            toList = [toList[i:i+2] for i in range(0, len(toList), 2)]

            for pair in toList:
                pair[0] = int (pair[0] *500)
                pair[1] = int (pair[1] *500)

            temp = []
            for i in xrange(1,len(toList)):
                pair = (toList[i-1], toList[i])
                temp.append(pair)
            temp.append((toList[0],toList[-1]))
            
            polygonSegments.append(temp)

        dim = 500

        '''border'''
        polygonSegments.append([ 
            [ (-5,-5),(505,-5) ], 
            [ (505,-5),(505,505) ],
            [ (505,505), (-5,505)],
            [ (-5,505), (-5,-5) ]
            ])
        return polygonSegments

    # def IsIntruderSeen(self, RRTPath, UAVLocation, UAVForwardVector, UAVFieldOfVision = 40):
    # 	# Setting customized UAV Field of vision
    # 	self.UAVFieldOfVision =  UAVFieldOfVision

    #     self.fieldOfVision = math.radians(self.UAVFieldOfVision/2.0)
    #     self.forwardVector = UAVForwardVector

    #     intersections = self.GetIsovistIntersections(UAVLocation, UAVForwardVector)


    #     # for point in RRTPath:
    #     #     #print point
    #     #     isFound = self.FindIntruderAtPoint(point, intersections)
    #     #     if isFound:
    #     #         return True, intersections
    #     skip = 20
    #     for i in xrange(skip-1,len(RRTPath), skip):
    #         segment = (RRTPath[i-skip+1], RRTPath[i])

    #         for j in xrange(1,len(intersections)):
    #         	isovist_segment = (intersections[j-1], intersections[j])
    #         	intersect, param = self.GetIntersection(segment, isovist_segment)
    #         	if intersect != None:
    #         		return True, intersections
    #         #Check the losing of the polygon segment
            
    #         isovist_segment = (intersections[0], intersections[-1])
    #         intersect, param = self.GetIntersection(segment, isovist_segment)
    #         if intersect != None:
    #         	return True, intersections

    #     return False, intersections


    def FindIntruderAtPoint(self, pos, intersections):
        if intersections == []:
            return False
        points = intersections
        cn = 0  # the crossing number counter
        pts = points[:]
        pts.append(points[0])
        for i in range(len(pts) - 1):
            if (((pts[i][1] <= pos[1]) and (pts[i+1][1] > pos[1])) or ((pts[i][1] > pos[1]) and (pts[i+1][1] <= pos[1]))):
                    if (pos[0] < pts[i][0] + float(pos[1] - pts[i][1]) / (pts[i+1][1] - pts[i][1]) * (pts[i+1][0] - pts[i][0])):
                            cn += 1
        if bool(cn % 2)==1:
            return True
        return False

    def GetIsovistIntersections(self, agentLocation, direction, UAVFieldOfVision = 40, full_iso=False):
        self.full = full_iso
        if UAVFieldOfVision == 360:
            self.full = True

        if direction == (0,0):
            return []
        # if direction[0] <= 1:
        #     if direction[1] <= 1:
        #         direction = (direction[0]*500, direction[1]*500)
    	#Setting customized UAV Field of vision
    	self.UAVFieldOfVision =  UAVFieldOfVision
        self.fieldOfVision = math.radians(self.UAVFieldOfVision/2.0)

        self.agentLocation = agentLocation
        uniqueAngles = self.GetUniqueAngles(direction)

        intersections = []
        for angle in uniqueAngles:

            # Calculate dx & dy from angle
            dx = math.cos(angle) * 2000
            dy = math.sin(angle) * 2000

            # Ray from center of screen to mouse
            ray = [ agentLocation , (agentLocation[0]+dx, agentLocation[1]+dy) ]
            ray_arr = np.array(ray)
            # Find CLOSEST intersection
            closestIntersect = None
            closestParam = 10000000

            output = self.GetIntersection_vec(ray_arr, self._segs)

            params = output[:,2]

            param_i = np.where(params == np.nanmin(params))[0]

            if len(param_i) > 0:
                closestParam = params[param_i][0]
                closestIntersect = (output[param_i,0][0], output[param_i,1][0])
                #print "closest Intersect", closestIntersect

                if closestIntersect != None:
                    intersections.append(closestIntersect)
            

            # for polygon in self.polygon_map:
            #     if old:
            #         for segment in polygon:
            #             intersect, param = self.GetIntersection(ray, segment)
            #             print intersect, param
            #             raw_input()
                        
            #             if intersect != None:
            #                 if closestIntersect == None or param < closestParam:
            #                     closestIntersect = intersect
            #                     closestParam = param
                # if not old:
                #     print ("New method")
                    
                #     polygon_arr = np.array(polygon)
                #     _segs[all_segs:all_segs+polygon_arr.shape[0],] = polygon_arr

                #     print _segs
                #     raw_input()
                #     print "all_segs", all_segs
                #     print "poly_arr size", polygon_arr.shape
                #     all_segs += polygon_arr.shape[0]
                #     #print polygon_arr.shape
                #     output = self.GetIntersection_vec(ray_arr, polygon_arr)
                #     params = output[:,2]
                #     param_i = np.where(params == np.nanmin(params))[0]
                #     if len(param_i) > 0:
                #         closestParam = params[param_i]
                #     print closestParam
                    #print "-in:", _in.shape
                    #print _in[0:2,:,:,:]
                    # for segment in polygon:
                    #     print('seg:', segment)

            
        intersections = self.SortIntoPolygonPoints(intersections)

        if not full_iso:
            intersections.insert(0, agentLocation)

        intersections.append(intersections[0])
        return intersections

    def GetIntersection_vec(self, ray, segment):
        # RAY in parametric: Point + Direction * T1
        r_px = ray[0,0]

        r_py = ray[0,1]

        # direction
        r_dx = ray[1,0] - ray[0,0]
        r_dy = ray[1,1] - ray[0,1]

        # SEGMENT in parametric: Point + Direction*T2
        s_px = segment[:,0,0] 
        s_py = segment[:,0,1] 

        # direction
        s_dx = segment[:,1,0] - segment[:,0,0]
        s_dy = segment[:,1,1] - segment[:,0,1]

        r_mag = np.sqrt(r_dx ** 2 + r_dy ** 2) #scaler and should not be zero
        if r_mag == 0:
            output = np.zeros((segment.shape[0],2))
            output[:,] = None, None
            return output
        s_mag = np.sqrt(s_dx ** 2 + s_dy ** 2) #(27,)

        output = np.zeros((s_mag.shape[0],2))
        output[np.where(s_mag == 0.0) ,] = None, None

        # PARALLEL - no intersection
        first_par_cond = np.where((r_dx/r_mag)==(s_dx/s_mag))
        sec_par_cond = np.where((r_dy/r_mag) == (s_dy/s_mag))
        par_ind = np.intersect1d(first_par_cond, sec_par_cond)
        output[par_ind,] = None, None

        denominator = -s_dx*r_dy + r_dx*s_dy

        #print "denom:", denominator
        output[np.where(denominator==0),] = None, None
        denominator[np.where(denominator==0)] = 1

        T1 = (-r_dy * (r_px - s_px) + r_dx * ( r_py - s_py)) / denominator
        T2 = (s_dx * ( r_py - s_py) - s_dy * ( r_px - s_px)) / denominator

        #print "T1:", T1
        #print "T2:", T2

        Ti_1 = np.where(T1 >= 0)[0]
        Ti_2 = np.where(T1 <= 1)[0]
        Ti_3 = np.where(T2 >= 0)[0]
        Ti_4 = np.where(T2 <= 1)[0]
        # print "Ti_1", Ti_1
        # print "Ti_2", Ti_2
        # print "Ti_3", Ti_3
        # print "Ti_4", Ti_4

        Ti = np.intersect1d(Ti_1, Ti_2)
        Ti = np.intersect1d(Ti, Ti_3)
        Ti = np.intersect1d(Ti, Ti_4)
        #print "ti:", Ti

        x = r_px+r_dx*T2
        y = r_py+r_dy*T2
        param = T2

        final_output = np.zeros((s_mag.shape[0],3))

        final_output[:,] = None, None, None

        # print final_output[Ti,]
        # print x[Ti], y[Ti], T2[Ti]
        
        #final_output[Ti,] = x[Ti].T, y[Ti].T, T2[Ti].T

        final_output[Ti,0,] = x[Ti].T 
        final_output[Ti,1,] = y[Ti].T 
        final_output[Ti,2,] = T2[Ti].T 
        #print "final:", final_output
        
        return final_output

    # def GetIsovistIntersections_vec(self, agentLocation, direction, UAVFieldOfVision = 40, full_iso=False):
    #     self.full = full_iso
    #     if UAVFieldOfVision == 360:
    #         self.full = True
    #     if direction == (0,0):
    #         return []
    #     # if direction[0] <= 1:
    #     #     if direction[1] <= 1:
    #     #         direction = (direction[0]*500, direction[1]*500)
    #     #Setting customized UAV Field of vision
    #     self.UAVFieldOfVision =  UAVFieldOfVision
    #     self.fieldOfVision = math.radians(self.UAVFieldOfVision/2.0)

    #     self.agentLocation = agentLocation
    #     uniqueAngles = self.GetUniqueAngles(direction)

    #     intersections = []
    #     for angle in uniqueAngles:

    #         # Calculate dx & dy from angle
    #         dx = math.cos(angle) * 2000
    #         dy = math.sin(angle) * 2000

    #         # Ray from center of screen to mouse
    #         ray = [ agentLocation , (agentLocation[0]+dx, agentLocation[1]+dy) ]

    #         # Find CLOSEST intersection
    #         closestIntersect = None
    #         closestParam = 10000000

    #         for polygon in self.polygon_map:
    #             for segment in polygon:
    #                 intersect, param = self.GetIntersection(ray, segment)
                    
    #                 if intersect != None:
    #                     if closestIntersect == None or param < closestParam:
    #                         closestIntersect = intersect
    #                         closestParam = param
            
    #         if closestIntersect != None:
    #             intersections.append(closestIntersect)

    #     intersections = self.SortIntoPolygonPoints(intersections)
    #     if not full_iso:
    #         intersections.insert(0, agentLocation)
    #     return intersections

    def GetUniqueAngles(self, direction):
        alpha, beta = self.GetAgentViewRays(direction)
        alphaAngle = math.atan2(alpha[1]-self.agentLocation[1], alpha[0]-self.agentLocation[0])
        betaAngle = math.atan2(beta[1]-self.agentLocation[1], beta[0]-self.agentLocation[0])
        self.startingFieldOfVision = alpha
        uniqueAngles = [alphaAngle, betaAngle]
        for point in self.uniquePoints:
            angleBetween = self.GetRelativeAngle(point, direction)

            if not self.full:
                if math.fabs(angleBetween) <= self.fieldOfVision:
                    #find world angle
                    angle = math.atan2(point[1]-self.agentLocation[1], point[0]-self.agentLocation[0])
                    uniqueAngles.append(angle)
                    uniqueAngles.append(angle-0.01)
                    uniqueAngles.append(angle+0.01)
            elif self.full:
                angle = math.atan2(point[1]-self.agentLocation[1], point[0]-self.agentLocation[0])
                uniqueAngles.append(angle)
                uniqueAngles.append(angle-0.01)
                uniqueAngles.append(angle+0.01)

        return uniqueAngles

    def GetRelativeAngle(self, point, direction):
        #forward direction
        dx = direction[0] 
        dy = direction[1]
        dMag = math.sqrt(dx**2 + dy**2)

        #vector agentlocation to point
        uniqueVectorX = point[0] - self.agentLocation[0]
        uniqueVectorY = point[1] - self.agentLocation[1]
        uniqueMag = math.sqrt(uniqueVectorX**2 + uniqueVectorY**2)

        

        #dot product equation stuff to find angle between the two vectors
        dotProduct = uniqueVectorX * dx + uniqueVectorY * dy
        
        check = (uniqueMag * dMag)
        if check == 0:
            check = .0000001

        cosineTheta = dotProduct / check
        cosineTheta = min(1,max(cosineTheta,-1))

        angleBetween = math.acos(cosineTheta)
        return angleBetween
        
    def SortIntoPolygonPoints(self, points):
        points.sort(self.Compare)
        return points

    def Compare(self, a, b):
        if self.full:
            a_row = a[0]
            a_col = a[1]

            b_row = b[0]
            b_col = b[1]

            a_vrow = a_row - self.agentLocation[0]
            a_vcol = a_col - self.agentLocation[1]

            b_vrow = b_row - self.agentLocation[0]
            b_vcol = b_col - self.agentLocation[1]

            a_ang = math.degrees(math.atan2(a_vrow, a_vcol))
            b_ang = math.degrees(math.atan2(b_vrow, b_vcol))

            if a_ang < b_ang:
                return -1

            if a_ang > b_ang:
                return 1

            return 0 


        a_ang = self.GetRelativeAngle(a, self.startingFieldOfVision)
        b_ang = self.GetRelativeAngle(b, self.startingFieldOfVision)

        if a_ang < b_ang:
            return -1
        if a_ang > b_ang:
            return 1
        return 0 

    def GetIntersection(self, ray, segment):

        #print ray, segment
        # RAY in parametric: Point + Direction * T1
        r_px = ray[0][0]
        r_py = ray[0][1]

        # direction
        r_dx = ray[1][0] - ray[0][0]
        r_dy = ray[1][1] - ray[0][1]

        # SEGMENT in parametric: Point + Direction*T2
        s_px = segment[0][0]
        s_py = segment[0][1]

        # direction
        s_dx = segment[1][0] - segment[0][0]
        s_dy = segment[1][1] - segment[0][1]

        r_mag = math.sqrt(r_dx ** 2 + r_dy ** 2)
        s_mag = math.sqrt(s_dx ** 2 + s_dy ** 2)

        # print ("r_px", r_px)
        # print ("r_py", r_py)
        # print "-----------"
        # print ("r_dx", r_dx)
        # print ("r_dy", r_dy)
        # print "-----------"
        # print ("s_px", s_px)
        # print ("s_py", s_py)
        # print "-----------"
        # print ("s_dx", s_dx)
        # print ("s_dy", s_dy)
        # print "-----------"
        # print ("r_mag", r_mag)
        # print ("s_mag", s_mag)



        if r_mag == 0 or s_mag == 0:
        	return None, None
        # PARALLEL - no intersection
        if (r_dx/r_mag) == (s_dx/s_mag):
            if (r_dy/r_mag) == (s_dy/s_mag):
                return None, None
        
        denominator = float( -s_dx*r_dy + r_dx*s_dy )
        if denominator == 0:
            return None, None

        T1 = (-r_dy * (r_px - s_px) + r_dx * ( r_py - s_py)) / denominator
        T2 = (s_dx * ( r_py - s_py) - s_dy * ( r_px - s_px)) / denominator

        if T1 >= 0 and T1 <= 1 and T2 >= 0 and T2 <= 1:
            #Return the POINT OF INTERSECTION
            x = r_px+r_dx*T2
            y = r_py+r_dy*T2
            param = T2
            return ( x, y ), param

        return None, None


    def GetUniquePoints(self):
        points = []
        for polygon in self.polygon_map:
            for segment in polygon:
                if segment[0] not in points:
                    points.append(segment[0])
                if segment[1] not in points:
                    points.append(segment[1])
        return points


    def GetAgentViewRays(self, direction):
        dx = direction[0] * 2000
        dy = direction[1] * 2000

        alphax = dx * math.cos(self.fieldOfVision) - dy * math.sin(self.fieldOfVision)
        alphay = dy * math.cos(self.fieldOfVision) + dx * math.sin(self.fieldOfVision)

        betax = dx * math.cos(-self.fieldOfVision) - dy * math.sin(-self.fieldOfVision)
        betay = dy * math.cos(-self.fieldOfVision) + dx * math.sin(-self.fieldOfVision)
        return (alphax, alphay), (betax, betay)




def setup_plot(poly_map, locs=None, scale=1):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)

    # plot map
    x1,y1,x2,y2 = poly_map
    for i in xrange(x1.shape[0]):
        ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'grey', linewidth=1  )

    # show possible start and goal locations
    add_locations = True
    if add_locations:
        for i in xrange(len(locs)):
            ax.scatter( locs[i][0] * scale,  locs[i][1] * scale , color="Green", s = 50, marker='+', linestyle='-')
            ax.scatter( locs[i][0] * scale,  locs[i][1]  * scale, s = 75, facecolors='none', edgecolors='g')
    return fig, ax

def close_plot(fig, ax, plot_name=None):
    if plot_name is None:
        plot_name = str(int(time.time()))+".eps"
    print "plot_name:", plot_name

    ax.set_ylim(ymax = 1, ymin = 0)
    ax.set_xlim(xmax = 1, xmin = 0)

    #plt.show()
    fig.savefig(plot_name, bbox_inches='tight')
    

from my_rrt import load_polygons
from my_rrt import polygons_to_segments
from methods import *

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import time
import matplotlib

def load_polygons( fn="./paths.txt" ):
    bdata = []
    for x in open( fn ):
        tmp = np.fromstring( x, dtype=float, sep=' ' )
        tmp = np.reshape( tmp/1000, (-1,2) )
        tmp = np.vstack(( np.mean(tmp, axis=0, keepdims=True), tmp, tmp[0,:] ))
        tmp[:,1] = 1.0 - tmp[:,1]  # flip on the y axis
        bdata.append( tmp )
    return bdata

def load_isovist_map( fn="./paths.txt" ):
    
    polygonSegments = []
    for line in open( fn ):
        line = line.strip('\n')
        toList = line.split(' ')
        toList = [(float(x)/1000) for x in toList]
        
        it = iter(toList)
        toList = [toList[i:i+2] for i in range(0, len(toList), 2)]


        for pair in toList:
            pair[0] = int (pair[0] *500)
            pair[1] = int ((1-pair[1]) *500)

        temp = []
        for i in xrange(1,len(toList)):
            pair = (toList[i-1], toList[i])
            temp.append(pair)
        temp.append((toList[0],toList[-1]))
        
        polygonSegments.append(temp)

        load_segs( fn="./paths.txt" )
    dim = 500
    return polygonSegments

def do_stuff():
    locs = None
    poly_map = None
    isovist = None

    locs = [[ 0.100, 1-0.900 ],[ 0.566, 1-0.854 ],[ 0.761, 1-0.665 ],
        [ 0.523, 1-0.604 ],[ 0.241, 1-0.660 ],[ 0.425, 1-0.591 ],
        [ 0.303, 1-0.429 ],[ 0.815, 1-0.402 ],[ 0.675, 1-0.075 ],
        [ 0.432, 1-0.098 ] ]
    poly_map = polygons_to_segments( load_polygons( "./paths.txt" ) )
    isovist = Isovist( load_isovist_map() )

    #show field of view
    other = [ 0.2, 0.5]
    me = [0.6, 0.6]
    fv = direction(scale_up(other), scale_up(me))
    intersections = isovist.GetIsovistIntersections(scale_up(me), fv)
    
    fig, ax = setup_plot(poly_map, locs)

    # show last isovist
    if not intersections is None:
      intersections = np.asarray(intersections)
      intersections /= 500.0
      if not intersections.shape[0] == 0:
          patches = [ Polygon(intersections, True)]
          p = PatchCollection(patches, cmap=matplotlib.cm.Set2, alpha=0.2)
          colors = 100*np.random.rand(len(patches))
          p.set_array(np.array(colors))
          ax.add_collection(p)



    close_plot(fig, ax, plot_name=str(int(time.time()))+".eps")

# How everything was before
#     1    0.015    0.015    0.114    0.114 isovist.py:130(GetIsovistIntersections)
#     1    0.000    0.000    0.001    0.001 isovist.py:227(GetUniqueAngles)
#   732    0.003    0.000    0.003    0.000 isovist.py:251(GetRelativeAngle)
#     1    0.000    0.000    0.003    0.003 isovist.py:277(SortIntoPolygonPoints)
#   291    0.000    0.000    0.003    0.000 isovist.py:281(Compare)
# 20026    0.091    0.000    0.095    0.000 isovist.py:316(GetIntersection)
#     1    0.002    0.002    0.002    0.002 isovist.py:360(GetUniquePoints)
#     1    0.000    0.000    0.002    0.002 isovist.py:37(__init__)
#     1    0.000    0.000    0.000    0.000 isovist.py:371(GetAgentViewRays)
#     1    0.002    0.002    0.949    0.949 isovist.py:385(setup_plot)
#     1    0.000    0.000    0.380    0.380 isovist.py:403(close_plot)
#     1    0.001    0.001    0.002    0.002 isovist.py:424(load_polygons)
#     1    0.000    0.000    1.499    1.499 isovist.py:434(do_stuff)


# after

      #   1    0.001    0.001    0.023    0.023 isovist.py:139(GetIsovistIntersections)
      #  62    0.008    0.000    0.016    0.000 isovist.py:228(GetIntersection_vec)
      #   1    0.000    0.000    0.001    0.001 isovist.py:355(GetUniqueAngles)
      #   1    0.000    0.000    0.003    0.003 isovist.py:37(__init__)
      # 728    0.003    0.000    0.004    0.000 isovist.py:379(GetRelativeAngle)
      #   1    0.000    0.000    0.003    0.003 isovist.py:405(SortIntoPolygonPoints)
      # 289    0.000    0.000    0.003    0.000 isovist.py:409(Compare)
      #   1    0.002    0.002    0.002    0.002 isovist.py:507(GetUniquePoints)
      #   1    0.000    0.000    0.000    0.000 isovist.py:518(GetAgentViewRays)
      #   1    0.003    0.003    1.023    1.023 isovist.py:532(setup_plot)
      #   1    0.000    0.000    0.378    0.378 isovist.py:550(close_plot)
      #   1    0.001    0.001    0.002    0.002 isovist.py:571(load_polygons)
      #   1    0.003    0.003    0.049    0.049 isovist.py:581(load_isovist_map)
      #   1    0.000    0.000    1.481    1.481 isovist.py:609(do_stuff)
import cProfile
import re

if __name__ == '__main__':

    cProfile.run('do_stuff()')
    #do_stuff()

    #plot("test.eps")
    
    