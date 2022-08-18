import numpy as np
import math
import random
class Plane_Fitting:
    """
    fitting plane using RANSAC
    """
    def __init__(self, points, max_iteration, distance_threshold):
        # self.points = []
        # for key in points:
        #     self.points.append(points[key])
        self.points = points
        self.max_iteration = max_iteration
        self.distance_threshold = distance_threshold

    def apply(self):
        inliers_result = []
        result_plane = []
        while self.max_iteration:
            self.max_iteration -=1
            random.seed()
            inliers=[]
            while len(inliers) < 3:
                random_index = random.randint(0, len(self.points)-1)
                if random_index in inliers:
                    continue
                else :
                    inliers.append(random_index)
            print(inliers[0],inliers[1], inliers[2])
            [x1, y1, z1] = self.points[inliers[0]]
            [x2, y2, z2] = self.points[inliers[1]]
            [x3, y3, z3] = self.points[inliers[2]]
            
            a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
            b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
            c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            d = -(a * x1 + b * y1 + c * z1)
            plane_length = max(0.1, math.sqrt(a*a + b*b + c*c))

            for n, point in enumerate(self.points):
                if n in inliers:
                    continue
                else :
                    [x, y, z] = point
                    distance = math.fabs(a*x + b*y + c*z +d)
                    if distance <= self.distance_threshold:
                        inliers.append(n)
                    
            if len(inliers) > len(inliers_result):
                inliers_result.clear()
                inliers_result = inliers
                result_plane = [a, b, c, d]
            
            print(f'*****{self.max_iteration}, {len(inliers)}*****')
            print(inliers_result)
            
        if len(inliers_result) < 10 :
            return [False, 0, 0, 0]
        else :
            return result_plane

    def apply_mean(self):
        inliers_result = []
        result_plane = [0, 0, 0, 0]
        count = 0
        while self.max_iteration:
            self.max_iteration -=1
            random.seed()
            inliers=[]
            while len(inliers) < 3:
                random_index = random.randint(0, len(self.points)-1)
                if random_index in inliers:
                    continue
                else :
                    inliers.append(random_index)
            print(inliers[0],inliers[1], inliers[2])
            print(self.points)
            [x1, y1, z1] = self.points[inliers[0]]
            [x2, y2, z2] = self.points[inliers[1]]
            [x3, y3, z3] = self.points[inliers[2]]
            
            a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
            b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
            c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            d = -(a * x1 + b * y1 + c * z1)
            if d < 500:    
                norm = math.sqrt(a*a + b*b + c*c)
                result_plane[0] += (a/norm)
                result_plane[1] += (b/norm)
                result_plane[2] += (c/norm)
                result_plane[3] += (d/norm)
                print(self.max_iteration, a, b, c, d)
                count+=1

        #cv.waitKey(0)       
        result_plane[0] /= count
        result_plane[1] /= count
        result_plane[2] /= count
        result_plane[3] /= count
           
        return result_plane
    
    def apply_combination(self):
        combination = [
            [4, 1, 10],
            [4, 2, 9],
            [4, 3, 7],
            [4, 5, 10],
            [4, 6, 10],
            [1, 7, 9],
            [81, 8, 4],
            [82, 10, 7]
            ]
        result_plane = [0, 0, 0, 0]
        for com in combination:
            [x1, y1, z1] = self.points[com[0]]
            [x2, y2, z2] = self.points[com[1]]
            [x3, y3, z3] = self.points[com[2]]

            vec12 = [x2 - x1, y2 - y1, z2 - z1]
            vec13 = [x3 - x1, y3 - y1, z3 - z1]

            
            a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
            b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
            c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

            norm = math.sqrt(a*a + b*b + c*c)
            result_plane[0] += (a/norm)
            result_plane[1] += (b/norm)
            result_plane[2] += (c/norm)
        
        result_plane[0] /= len(combination) * 10
        result_plane[1] /= len(combination) * 10
        result_plane[2] /= len(combination) * 10
        result_plane[3] /= len(combination) * 10
        
        if result_plane[0] > 0:
              result_plane[0] *= -1
              result_plane[1] *= -1
              result_plane[2] *= -1

           
        return result_plane
