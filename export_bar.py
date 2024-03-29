# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 00:35:18 2022

@author: Pablo Yubero Bernabé
https://github.com/pyubero

Export data from a bar graph figure. Execute with an input image as:
    python export_bar.py -i image_filepath

Parameters
----------
-i : string, input image filepath, needs to be compatible with OpenCV

-o : string, output csv filepath. Default="data.csv"
    
-sz: float, resize factor of the image. Specify a value different than one
     if the input image is either too small or large to display. Default=1.

-th: float, color similarity threshold. Should be a value in [0,1]. Default=0.95.

-p : int, output data precision. Number of decimals of the values exported to
    the csv file. Default=2.

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
from scipy import interpolate




def getPixel(event,x,y,flags,param):
    global color, mask
        
    if event == cv2.EVENT_LBUTTONDOWN:
        color = np.array( frame[y, x,:])
        create_mask()
        
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_RBUTTON:
        mask = update_mask(x,y)
        



def project_frame(frame, color):
    # Normalize color vector to unit length
    vec_c  = color / np.linalg.norm(color)

    # Normalize frame with unit length in color dimension
    frame_norm = np.linalg.norm(frame, axis=2)
    proj_frame = np.nan*np.zeros( frame.shape )
    for c in range(3):
        proj_frame[:,:,c] = frame[:,:,c]/frame_norm
    
    # Return the projection
    return np.matmul( proj_frame, vec_c)

def update_mask(x,y):
    global show_frame, mask
    # cv2.circle(show_frame, (x,y), 20, (0,255,0), -1)
    return cv2.circle(1*mask, (x,y), 20, (0), -1)

    
def create_mask():
    global color, show_frame, mask
    mask = project_frame(frame, color) > color_thresh
    
    # _idc = np.nonzero(mask)
    # show_frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY) 
    # show_frame = cv2.cvtColor( show_frame, cv2.COLOR_GRAY2BGR)
    # show_frame[ _idc[0], _idc[1],:]=[0,255,0]
    # show_frame = show_frame.astype('uint8')


#########################################
############ Argument parser ############
my_parser = argparse.ArgumentParser(description='Description' , fromfile_prefix_chars='@')
my_parser.add_argument('-i',                         ,type=str  ,help='Input image')
my_parser.add_argument('-o',  default = 'data.csv'   ,type=str  ,help='Output file name.')
my_parser.add_argument('-sz', default = 1            ,type=float,help='Image resize factor.')
my_parser.add_argument('-th', default = 0.95         ,type=float,help='Color similarity threshold.')
my_parser.add_argument('-p',  default = 2            ,type=int  ,help='Precision of output values.')



args = my_parser.parse_args()
filepath     = args.i
out_filename = args.o
color_thresh = args.th
precision    = args.p
resize_factor= args.sz



###################################
## LOAD IMAGE
frame     = cv2.imread(filepath)
frame  = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
height, width, _ = frame.shape

print('In the window displaying the image, click and drag to delimit the bounding box')
print('of the axis. It does not need to cover the whole figure, just some limits to calibrate')
print('the units of both axes.')

###################################
## DEFINE AXES
axes = cv2.selectROI(frame)
cv2.destroyAllWindows()

result = input('Please enter X-axis min and max as "min,max": ').split(',')
if len(result)==1:
    xmin, xmax = 0, float(result[0])
else:
    xmin = float(result[0])
    xmax = float(result[1])
print('Thank you, X-axis limits are set to (%s, %s)' % (xmin, xmax) )
print('')


result = input('Please enter Y-axis min and max as "min,max": ').split(',')
if len(result)==1:
    ymin, ymax = 0, float(result[0])
else:
    ymin = float(result[0])
    ymax = float(result[1])
print('Thank you, Y-axis limits are set to (%s, %s)' % (ymin, ymax) )


    

print('Now click on the image to select the color of the data to be extracted.')
print('Data included by your selection will be colored in intense green.')
print('Press the letter q in the keyboard when satisfied with the results.')
print('')
print('<W> For bar graphs, bars need to be disconnected from one another, if its not')
print('    the case, each color-connected components is considered a single data unit.')
print('    You can always use paint, or gimp to draw some lines separating each bar.')
print('<W> If some pixels are included there are two options:')
print('    - to specify a different color similarity threshold when running this script, using -th')
print('    - to erase manually those pixels right-clicking and dragging in the image.')
    
    
###################################
## SELECT LINES
select_new_line = True
line_id = 0

# Prepare data export file
with open(out_filename,'w+') as file:
    file.write('Identifier;x-value;y-value\n')
print('Output file created.')


# Prepare figure
plt.figure( dpi=600 )

while select_new_line:
    color = None
    mask = np.zeros( frame.shape[:2] )
    # show_frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY) 
    # show_frame = cv2.cvtColor( show_frame, cv2.COLOR_GRAY2BGR)
    
    
    while cv2.waitKey(10) != ord('q'):
        show_frame = frame.copy()
        _idc = np.nonzero(mask)
        show_frame[ _idc[0], _idc[1],:]=[0,255,0]
        show_frame = show_frame.astype('uint8')
        
        cv2.imshow('window',show_frame)
        cv2.setMouseCallback('window',getPixel) 
    cv2.destroyAllWindows()
    
    
    # Threshold image
    thresh = mask.copy() #project_frame(frame, color) > color_thresh
    
    
    # Obtain contours
    contours, _ = cv2.findContours(thresh.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    # Extract height of each contour
    data_x = []
    data_y = []    
    origin = axes[0], axes[1]+axes[3]
    
    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        bbox = cv2.boundingRect( contours_poly)
        if bbox[2]>2 and bbox[3]>2:
            data_x.append(  bbox[0]+bbox[2]/2-origin[0] )
            data_y.append(  origin[1]-bbox[1] )
    data_x = np.array( data_x )
    data_y = np.array( data_y )

    # Sort values
    I = np.argsort( data_x )
    data_x = data_x[I]
    data_y = data_y[I]
    
    # Transform to real values
    out = np.nan*np.zeros( (len(data_x), 3) )
    out[:,0] = line_id
    out[:,1] = xmin + data_x * (xmax-xmin)/axes[2]
    out[:,2] = ymin + data_y * (ymax-ymin)/axes[3]

    
    
    # Export data
    with open(out_filename,'a+') as file:
        _=[file.write( '%d;%s;%s\n' %  (o[0],o[1],o[2]) ) for o in np.round(out,precision)]; 
    print('Line %d exported!' % line_id )
    
    
    # Keep plotting
    plt.bar( out[:,1], out[:,2])
    
    
    # If you would like to export another line keep everything going
    if input('Would you like to export another curve? [y/n]: ') =='y':
        line_id+=1
        select_new_line = True
    else:
        select_new_line = False

plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.tight_layout()
plt.show()

print('Good bye!')
    
  