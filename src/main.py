from test import affine_detect
from find_obj import init_feature, filter_matches, explore_match
from multiprocessing.pool import ThreadPool
import cv2 as cv
import sys, getopt
from matplotlib import pyplot as plt
import time
import numpy as np
from matplotlib import gridspec
import time
import numpy as np
import math

class Anagrams:
    def __init__(self, x):
        self.list1=[]
        self.qtree=x
    def get_point(self,tree):
            self.list1=self.list1+tree.points2
            if 'nw' in tree.__dict__:
                self.list1=self.list1+tree.nw.points2
            if 'ne' in tree.__dict__:
                self.list1=self.list1+tree.ne.points2
            if 'se' in tree.__dict__:
                self.list1=self.list1+tree.se.points2
            if 'sw' in tree.__dict__:
                self.list1=self.list1+tree.sw.points2
            if 'nw' in tree.__dict__:
                self.get_point(tree.nw)
            if 'ne' in tree.__dict__:
                self.get_point(tree.ne)
            if 'se' in tree.__dict__:
                self.get_point(tree.se)
            if 'sw' in tree.__dict__:
                self.get_point(tree.sw)
            return self.list1

    def get_point2(self,tree):
        if tree.divided:
            self.list1=self.list1+[tree.points[-2],tree.points[-1],tree.points[0]]
            self.get_point2(tree.nw)
            self.get_point2(tree.ne)
            self.get_point2(tree.se)
            self.get_point2(tree.sw)
        return self.list1

    def get_bound(self,tree):
        if not tree.divided:
            self.list1.append(tree.boundary)
        else:
            self.list1.append(tree.nw.boundary)
            self.list1.append(tree.ne.boundary)
            self.list1.append(tree.se.boundary)
            self.list1.append(tree.sw.boundary)
            self.get_bound(tree.nw)
            self.get_bound(tree.ne)
            self.get_bound(tree.se)
            self.get_bound(tree.sw)
        return self.list1
    
    def __str__(self) -> str:
        print(self.list1)


class Point:
    """A point located at (x,y) in 2D space.

    Each Point object may be associated with a payload object.

    """

    def __init__(self, x, y, payload=None):
        self.x, self.y = x, y
        self.payload = payload

    def __repr__(self):
        return '{}: {}'.format(str((self.x, self.y)), repr(self.payload))

    def distance_to(self, other):
        try:
            other_x, other_y = other.x, other.y
        except AttributeError:
            other_x, other_y = other
        return np.hypot(self.x - other_x, self.y - other_y)

class Rect:
    """A rectangle centred at (cx, cy) with width w and height h."""

    def __init__(self, cx, cy, w, h):
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h
        self.west_edge, self.east_edge = cx - w/2, cx + w/2
        self.north_edge, self.south_edge = cy - h/2, cy + h/2

    def __repr__(self):
        return str((self.west_edge, self.east_edge, self.north_edge,
                self.south_edge))

    def contains(self, point):
        """Is point (a Point object or (x,y) tuple) inside this Rect?"""

        try:
            point_x, point_y = point.x, point.y
        except AttributeError:
            point_x, point_y = point

        return (point_x >= self.west_edge and
                point_x <  self.east_edge and
                point_y >= self.north_edge and
                point_y < self.south_edge)

    def intersects(self, other):
        """Does Rect object other interesect this Rect?"""
        return not (other.west_edge > self.east_edge or
                    other.east_edge < self.west_edge or
                    other.north_edge > self.south_edge or
                    other.south_edge < self.north_edge)

    def draw(self, ax, c='k', lw=1, **kwargs):
        x1, y1 = self.west_edge, self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        ax.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], c=c, lw=lw, **kwargs)


class QuadTree:
    """A class implementing a quadtree."""

    def __init__(self, boundary, max_points=1, depth=0):
        """Initialize this node of the quadtree.

        boundary is a Rect object defining the region from which points are
        placed into this node; max_points is the maximum number of points the
        node can hold before it must divide (branch into four more nodes);
        depth keeps track of how deep into the quadtree this node lies.

        """

        self.boundary = boundary
        self.max_points = max_points
        self.points = []
        self.depth = depth
        self.points2=[]
        self.divided = False

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = ' ' * self.depth * 2
        s = str(self.boundary) + '\n'
        s += sp + ', '.join(str(point) for point in self.points)
        if not self.divided:
            return s
        return s + '\n' + '\n'.join([
                sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
                sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])

    def divide(self):
        """Divide (branch) this node by spawning four children nodes."""
        
        cx, cy = self.boundary.cx, self.boundary.cy
        w, h = self.boundary.w / 2, self.boundary.h / 2
        self.nw = QuadTree(Rect(cx - w/2, cy - h/2, w, h),
                                        self.max_points, self.depth + 1)
        self.ne = QuadTree(Rect(cx + w/2, cy - h/2, w, h),
                                        self.max_points, self.depth + 1)
        self.se = QuadTree(Rect(cx + w/2, cy + h/2, w, h),
                                        self.max_points, self.depth + 1)
        self.sw = QuadTree(Rect(cx - w/2, cy + h/2, w, h),
                                        self.max_points, self.depth + 1)
        self.divided = True

    def insert(self, point):
        """Try to insert Point point into this QuadTree."""
        if self.depth<5:
            if not self.boundary.contains(point):
                return False
            if len(self.points) < self.max_points:
                self.points.append(point)
                return True
            if not self.divided:
                self.divide()

            return (self.ne.insert(point) or self.nw.insert(point) or self.se.insert(point) or self.sw.insert(point))
        else:
            return 1


    def __len__(self):
        """Return the number of points in the quadtree."""

        npoints = len(self.points)
        if self.divided:
            npoints += len(self.nw)+len(self.ne)+len(self.se)+len(self.sw)
        return npoints

    def draw(self, ax):
        """Draw a representation of the quadtree on Matplotlib Axes ax."""

        self.boundary.draw(ax)
        if self.divided:
            self.nw.draw(ax)
            self.ne.draw(ax)
            self.se.draw(ax)
            self.sw.draw(ax)

'''
--feature:  sift/surf/orb/akaze  
'''
#feature_name = opts.get('--feature', 'orb-flann')
#=====/*==== image 1 ===================
def find_key_des(image):
    img1 = cv.imread(image, 0)
    detector, matcher = init_feature('sift-flann')
    pool = ThreadPool(processes=cv.getNumberOfCPUs())
    kp1, desc1 = affine_detect(detector, img1, pool=pool)

    DPI = 72
    width, height = 640,480

    pts1 = cv.KeyPoint_convert(kp1)
    xs1 = np.array(pts1[:,0])
    ys1 = np.array(pts1[:,1])

    #point from image 1
    points_1 = [Point(xs1[i],ys1[i]) for i in range(len(pts1))]
    domain_1 = Rect(width/2, height/2, width, height)
    qtree_1 = QuadTree(domain_1, 3)
    for point in points_1:
        qtree_1.insert(point)
    print('Number of points in the domain =', len(qtree_1))

    x_1=Anagrams(qtree_1)
    list_1_1=x_1.get_bound(x_1.qtree)
    list_2=set(x_1.list1)
    point3=[]
    for i in list_2:
        for j in points_1:
            if i.contains(j)==True:
                point3.append(j)
                break
    temp = list(dict.fromkeys(point3))
    keypoints_1 = []
    for i in temp:
        a=str(i).split(':')[0].replace('P','')
        b=eval(a)
        keypoints_1.append(b)
    keypoints_1_temp = np.array([list(x) for x in keypoints_1], dtype='f')
    descriptor_1 = []
    for i in keypoints_1_temp:
        descriptor_1 = np.append(descriptor_1, desc1[int(np.where(pts1 == i)[0][0])])
    descriptor_1 = descriptor_1.reshape(len(keypoints_1),128)
    keypoints_1obj = cv.KeyPoint_convert(keypoints_1)
    descriptor_1 = np.array([list(x) for x in descriptor_1], dtype=np.float32)
    return keypoints_1obj,descriptor_1

keypoints_1,descriptor_1 = find_key_des('1.png')
keypoints_2,descriptor_2 = find_key_des('2.png')
img1 = cv.imread('1.png')
img2 = cv.imread('2.png')
#feature matching
bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
matches = bf.match(descriptor_1,descriptor_2)
tic = time.clock()
matches = sorted(matches, key = lambda x:x.distance)
toc = time.clock()
print(toc - tic)
print(len(matches))
img3 = cv.drawMatches(img1,keypoints_1,img2,keypoints_2,matches, None,flags=2)

plt.imshow(img3),plt.show()
detector, matcher = init_feature('sift-flann')
pool = ThreadPool(processes=cv.getNumberOfCPUs())
kp1, desc1 = affine_detect(detector, img1, pool=pool)
kp2, desc2 = affine_detect(detector, img2, pool=pool)
print(len(kp1))
print(len(kp2))
tic1 = time.clock()
matches = bf.match(desc1,desc2)
toc1 = time.clock()
print(toc1 - tic)
print(len(matches))
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv.drawMatches(img1,kp1,img2,kp2,matches, None,flags=2)

plt.imshow(img3),plt.show()
