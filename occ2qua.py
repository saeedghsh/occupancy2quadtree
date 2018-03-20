import cv2
import matplotlib.pyplot as plt
import numpy as np

import time

########## load image - extract occupancy
filename = 'test_images/test.png' # '/home/saesha/Documents/tango/E5_13/20170409130542.png'
img = np.flipud( cv2.imread( filename, cv2.IMREAD_GRAYSCALE) )

# do I need them as points?
# I mean, I could just slice the image to see if each node is empty or not.
# That's also better, if I were to extend to multi-color!
Y, X = np.where(img==0)


########################################
class Node:
    def __init__(self, idx, rmin,rmax, cmin,cmax,
                 image, depth_max=10, resolution=10):
        ''''''
        self.bounds = [rmin,rmax , cmin,cmax]
        self.idx = idx
        
        dep = len(idx) <= depth_max
        occ = image[rmin:rmax-1 , cmin:cmax-1].sum() > 0
        res = ((cmax-cmin)//2)>resolution and ((rmax-rmin)//2)>resolution

        r_half, c_half = (rmax - rmin)//2 , (cmax - cmin)//2

        if dep and occ and res:
            self.children = ( Node(idx+'0', rmin,rmin+r_half, cmin,cmin+c_half, image, depth_max, resolution),
                              Node(idx+'1', rmin,rmin+r_half, cmin+c_half,cmax, image, depth_max, resolution),
                              Node(idx+'2', rmin+r_half,rmax, cmin,cmin+c_half, image, depth_max, resolution),
                              Node(idx+'3', rmin+r_half,rmax, cmin+c_half,cmax, image, depth_max, resolution))
        else:
            self.children = ()
            # print (idx)

    # def __call__(self, image): print('call function')

    def is_point_inside(self,r,c):
        return self.idx if (rmin<r<rmax and cmin<c<cmax) else False

    def get_boundary_lines(self,):
        pass

########################################
class QuadTree:
    def __init__(self, image,
                 depth_max=10, resolution=10,
                 thr1=100,thr2=255 ):

        self.depth_max = depth_max
        self.resolution = resolution

        self.img = image
        ret, self.img_bin = cv2.threshold(img, thr1, thr2, cv2.THRESH_BINARY_INV)

        h, w = image.shape
        
        self.node = Node('0', 0,h , 0,w, self.img_bin, depth_max, resolution)

  

def extract_end_node (node):
    '''
    '''
    end_nodes = []
    
    if len(node.children) == 0:
        end_nodes.append(node)
    else:
        for n in node.children:
            end_nodes.extend( extract_end_node(n) )

    return end_nodes
    

def bounds2lines (bounds):
    '''
    takes the boundary of a rectangle in column-row coordinates
    returns line-segments (pairs of points) in x-y coordinates
    '''
    [rmin,rmax , cmin,cmax] = bounds
    return (((cmin, rmin), (cmin, rmax)),
            ((cmin, rmax), (cmax, rmax)),
            ((cmin, rmin), (cmax, rmin)),
            ((cmax, rmin), (cmax, rmax)))


tic = time.time()
# qt = QuadTree(img, depth_max=10, resolution=10)
qt = QuadTree(img, depth_max=6, resolution=1)
# qt = QuadTree(img, depth_max=9, resolution=1)
print('time to complete: {:.5f}'.format(time.time()-tic))
# lines = [bounds2lines(en.bounds) for en in extract_end_node(qt.node)]
# print('number of end nodes: {:d}'.format(len(lines)))

lines = list(set([l for en in extract_end_node(qt.node) for l in bounds2lines(en.bounds)]))
print('number of end nodes: {:d}'.format(len(lines)))

# ### for animation
# for d in range(10):
#     qt = QuadTree(img, depth_max=d, resolution=1)
#     lines = list(set([l for en in extract_end_node(qt.node)
#                       for l in bounds2lines(en.bounds)]))
#     img_cv = np.stack([img for _ in range(3)], axis=2)
#     for l in lines: cv2.line(img_cv, l[0], l[1], (0,0,255), 1)
#     cv2.imwrite('qt_depth0{:d}.png'.format(d), np.flipud(img_cv))
# # convert -delay 20 -loop 0 *.png quadtree.gif

########## plottings
if 1:
    max_size = 12
    h, w = img.shape
    H,W = [max_size, (float(w)/h)*max_size] if (h > w) else [(float(h)/w)*max_size, max_size]
       

    fig, axes = plt.subplots(1,1, figsize=(W, H))


    if 0:
        axes.imshow(img, cmap='gray', alpha=1, interpolation='nearest', origin='lower')
        # axes.imshow(img_bin, cmap='gray', alpha=1, interpolation='nearest', origin='lower')

        for l in lines:
            axes.plot( [l[0][0], l[1][0]], [l[0][1], l[1][1]], 'b-')


    else:
        # img_cv = img.copy()
        img_cv = np.stack([img for _ in range(3)], axis=2)
        for l in lines: cv2.line(img_cv, l[0], l[1], (0,0,255), 1)# 2, -1)
        axes.imshow(img_cv, alpha=1, interpolation='nearest', origin='lower')
        # cv2.imwrite(target_folder+'pts/pts_motion0{:d}.png'.format(frame_counter), np.flipud(base_img)[450:-450, 550:-550] )
        
        
    # axes.plot(X,Y, 'r,')

    axes.axis('off')
    plt.tight_layout()
    plt.show()




