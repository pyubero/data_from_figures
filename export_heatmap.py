# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 00:35:18 2022

@author: Pablo
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
from scipy import interpolate
from sklearn.linear_model import LinearRegression
    



#########################################
############ Argument parser ############
my_parser = argparse.ArgumentParser(description='Description' , fromfile_prefix_chars='@')
my_parser.add_argument('-i',                         ,type=str  ,help='Input image')
my_parser.add_argument('-o',  default = 'data.csv'   ,type=str  ,help='Output file name.')
my_parser.add_argument('-sz', default = 1            ,type=float,help='Image resize factor.')
my_parser.add_argument('-th', default = 0.95          ,type=float,help='Color similarity threshold.')
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
print('Please select the plot boundaries.')
axes = cv2.selectROI(frame)
cv2.destroyAllWindows()



result = input('Please enter the number of data points in the X axis: ')
x_num_pts = float(result)

result = input('Please enter the number of data points in the Y axis: ')
y_num_pts = float(result)




# Extract data points in img coordinates
DATA = np.nan*np.zeros( (int(x_num_pts), int(y_num_pts), 3) )

for xx in range(int(x_num_pts)):
    for yy in range( int(y_num_pts)):
        x_idx = axes[0] + (xx+0.5) * axes[2] / x_num_pts
        y_idx = axes[1] + (yy+0.5) * axes[3] / y_num_pts

        DATA[xx,yy,:] = frame[ int(y_idx), int(x_idx),: ]



# # Guess the color order
# data_flat = np.array([ DATA[a,b,:] for a in range(DATA.shape[0]) for b in range(DATA.shape[1]) ] )
# pca = PCA(n_components=1).fit(data_flat)
# score = np.matmul(data_flat, pca.components_.T) 
# plt.plot(score,'.')



if input('Is there a colorbar legend? [y/n]: ')=='y':
    print('Please select the colorbar legend with a bounding box as in the previous step.')
    colorbar = cv2.selectROI(frame)
    cv2.destroyAllWindows()
    
    result = input('Please enter colorbar limits as "cmin,cmax": ').split(',')
    cmin = float( result[0] )
    cmax = float( result[1] )
    
    
    c_start = colorbar[1]  ,  colorbar[0]
    c_end   = colorbar[1]+colorbar[3], colorbar[0]+colorbar[2]
    
    npts = 100
    c_xt = np.linspace( c_start[0], c_end[0] , npts )  
    c_yt = np.linspace( c_start[1], c_end[1] , npts )
    cbar = np.zeros((npts, 3))
    for tt in range(npts):
        cbar[tt,:] = frame[ int(c_xt[tt]), int(c_yt[tt]) ,:]
    
    vv = np.linspace(cmin,cmax, npts)
    reg = LinearRegression()
    reg.fit(cbar, vv)
    
        
    print('Thank you, the colorbar range is now calibrated with R2=%1.4f.' %  reg.score(cbar, vv) )
else:
    colorbar = None



# Transform to real values
out = np.zeros( (int(x_num_pts), int(y_num_pts) ) )

for xx in range(int(x_num_pts)):
    for yy in range( int(y_num_pts)):       
        out[xx,yy] = reg.predict( [DATA[xx,yy,:],])



# Prepare data export file
with open(out_filename,'w+') as file:
    numel = out.shape[1]
    out_string = ('%s;'*numel)[:-1]
    
    for jj in range( out.shape[0] ):
        line = np.round( out[jj,:], precision)
        values = [ str(v) for v in line ]
        file.write( ';'.join(values)+'\n' )
    
    
print('Data exported!')    

    


# Prepare figure
plt.figure( dpi=600 )
plt.imshow(out.T, cmap='RdBu_r'); 
plt.colorbar()    
plt.tight_layout()
plt.show()


print('Good bye!')
    
  