import os
import sys
import math
import numpy as np
import sv

# Initialize path names
projDir = "C:\\Websites\\stenosis_research"

modelsDir = os.path.join(projDir, 'model')
pathsDir = os.path.join(projDir, 'paths')
segDir = os.path.join(projDir, 'segments')

# Create paths if does not exist
os.makedirs(modelsDir, exist_ok=True)
os.makedirs(pathsDir, exist_ok=True)
os.makedirs(segDir, exist_ok=True)


sys.path.append(projDir)
import helper_functions
from SV_lofting_solids_meshing import *
sys.path.pop()


# Source: adapted from https://github.com/neilbalch/SimVascular-pythondemos/blob/master/contour_to_lofted_model.py

def radius(radius_inlet, x, A, sigma, mu):
    # Reference: Sun, L., Gao, H., Pan, S. & Wang, J. X. Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data. Computer Methods in Applied Mechanics and Engineering 361, (2020).

    # controls severity of stenosis/aneurysm; positive for stenosis; negative for aneurysm
    return radius_inlet - A/np.sqrt(2.0*np.pi*sigma**0.5)*np.exp(-1.0*((x - mu)**2/2/sigma**0.5))

############################################################
############################################################
################## CHANGE THESE ENTRIES ####################
############################################################
############################################################
model_name = "model01"
x0 = -15.0
xf = 15.0
nx = 31 # make this an odd number, so that we will have a segmentation at the midpoint
distance = 0 # for removal of segmentations surrounding the midsection

midsection_percentage = 15.0 # midsection area as a percentage of the inlet area
radius_inlet = 1.0

sigma = 100.0 # controls the spread of the stenosis/aneurysm
mu = 0.0 # controls the center of the stenosis/aneurysm
path_name = model_name + "_path"
segmentations_name = model_name + "_segmentations"
############################################################
############################################################
############################################################
############################################################
############################################################


def generate_points_list(x0, xf, nx, distance, f=None, g=None, h=None, f_params=[], g_params=[], h_params=[]):
    # make path points list
    # Create z axis
    z = np.linspace(x0, xf, nx)
    
    # Remove segmentations surrounding midsection based on distance provided
    mid_index = math.floor(len(z)/2.0)
    z = np.concatenate((z[0:mid_index - distance], np.array([z[mid_index]]), z[mid_index + 1 + distance:]))

    # Create x and y axis
    y = z.copy()
    x = z.copy()

    # Transform using functions, or default to straight path
    x = np.array([f(x_i, f_params) for x_i in x]) if (f is not None) else (x * 0)
    y = np.array([g(y_i, g_params) for y_i in y]) if (g is not None) else (y * 0)
    z = np.array([h(z_i, h_params) for z_i in z]) if (h is not None) else z

    path_points_array = np.column_stack((x, y, z))
    # print(path_points_array)

    path_points_list = path_points_array.tolist()

    return path_points_list, z

    
def generate_radii_list(midsection_percentage, radius_inlet, sigma, mu, z, radius=radius):
    # make radii list
    area_midsection = (midsection_percentage/100.0)*np.pi*radius_inlet**2.0
    radius_midsection = np.sqrt(area_midsection/np.pi)
    A = -1.0*(radius_midsection - radius_inlet)*np.sqrt(2.0*np.pi*sigma**0.5)/np.exp(-1.0*((0.0 - mu)**2/2/sigma**0.5))  # A = 4.856674471372556
    print("A = ", A)
    radius_array = radius(radius_inlet, z, A, sigma, mu)
    radii_list = radius_array.tolist()
    return radii_list



def generate_model(args):
    model_name, x0, xf, nx, distance, midsection_percentage, radius_inlet, sigma, mu, f, g, radius, f_params, g_params = args
    path_name = model_name + "_path"
    segmentations_name = model_name + "_segmentations"
    
    # make path points list
    path_points_list, z = generate_points_list(x0, xf, nx, distance, f, g, f_params=f_params, g_params=g_params)
    
    # make radii list
    radii_list = generate_radii_list(midsection_percentage, radius_inlet, sigma, mu, z, radius)
    
    if len(radii_list) != len(path_points_list):
        print("Error. Number of points in radius list does not match number of points in path list.")
    
    # create path and segmnetation objects
    path = helper_functions.create_path_from_points_list(path_points_list)
    segmentations = helper_functions.create_segmentations_from_path_and_radii_list(path, radii_list)

    # add path and segmentations to the SimVascular Data Manager (SV DMG) for visualization in GUI
    sv.dmg.add_path(name = path_name, path = path)
    sv.dmg.add_segmentation(name = segmentations_name, path = path_name, segmentations = segmentations)

    segmentPolydata = [obj.get_polydata() for obj in sv.dmg.get_segmentations(segmentations_name)]

    capped_vessels = create_vessels([segmentPolydata,])
    unioned_model = union_all(capped_vessels)
    #unioned_model = capped_vessels[0]
    model = sv.modeling.PolyData()
    tmp = unioned_model.get_polydata()
    print("done unioning")
    sv.dmg.add_model(name = model_name + "_model", model = unioned_model)




# Polynomial parametric equation
def poly(t, x_vec):
    # return 0  # Zero for straight path
    retVal = 0
    mult = 1
    for x in x_vec:
        retVal = mult*x
        mult *= t
    return retVal

# Radius function
def r(radius_inlet, x, A, sigma, mu):
    arr = np.array([(x_i if x_i > 0 else x_i**2) for x_i in x])
    return radius(radius_inlet, arr, A, sigma, mu)


# List of ranges in format (start, end, step), such that range is x in [start, end] and x = start+k*step, k in Z+
# x0 is min z value
# xf is max z value
# nx number of segmentations (must be odd)
# distance - distance around midsection to remove
# midsection_percentage - midsection area as a percentage of the inlet area (Logarithmic scale between 0.1% and 100%)
# radius_inlet
# sigma - controls the spread of the stenosis/aneurysm
# mu - controls the center of the stenosis/aneurysm

ranges = [
    (-15.0, -15.0, 1.0),    # x0
    (15.0, 15.0, 1.0),      # xf
    (31, 31, 2),            # nx - make this an odd number, so that we will have a segmentation at the midpoint
    (0, 0, 1),              # distance - for removal of segmentations surrounding the midsection
    10**np.arange(-1,2.1,0.5),      # midsection_percentage - midsection area as a percentage of the inlet area (Logarithmic scale between 0.1% and 100%)
    (1.0, 1.0, 0.1),        # radius_inlet
    (50, 100, 25),          # sigma - controls the spread of the stenosis/aneurysm
    (0.0, 0.0, 0.25),       # mu - controls the center of the stenosis/aneurysm
    [poly],                    # f - Parametric function of x component
    [poly],                    # g - Parametric function of y component
    [r],                    # r - Parametric function of radius
    [(0, 0, 1/100),],          # inputs to f(t, ...)
    [(0, 0, 0, 1/1000),]          # inputs to g(t, ...)
]

# REMOVEME: Keeps some variables static
ranges[0] = [-15,]
ranges[1] = [15,]
ranges[4] = [1,]
ranges[6] = (50,50,25)


def generate_simulation_inputs(range_index = (len(ranges)-1)):
    # Base case, no variables in list to generate combinations -- return empty list
    if(range_index < 0):
        yield []
    # Recursive case, combinations next variable in list and yield to callee
    else:
        # If a tuple is given, then (from, to, step) is given
        if isinstance(ranges[range_index], tuple):
            # Extract from, to and step of range
            fromVal, toVal, step = ranges[range_index]
            
            # For each prior sub-combinations, create one copy for each value in range
            for arr in generate_simulation_inputs(range_index-1):
                val = fromVal
                while val <= toVal:
                    arr_new = arr.copy()
                    arr_new.append(val)
                    yield arr_new
                    val += step  # Increment value by step
        # Otherwise, a range of values is given directly
        else:
            # For each prior sub-combinations, create one copy for each value in range
            for arr in generate_simulation_inputs(range_index-1):
                for val in ranges[range_index]:  # Extract values from range
                    arr_new = arr.copy()
                    arr_new.append(val)
                    yield arr_new
            
    
def run_simulation():
    i = 1
    fModelList = open(os.path.join(projDir, 'model_list.txt'), 'w')
    # Generate inputs
    for input_arr in generate_simulation_inputs():
        # model_name, x0, xf, nx, distance, midsection_percentage, radius_inlet, sigma, mu, f, g, radius, f_params, g_params = input_arr
        model_name = "model_{0}".format(i)

        # Write params of model to file
        buffStr = "%s : %d, %d\n" % (model_name, x0, xf)
        fModelList.write(buffStr);
        
        args = (model_name, *input_arr)

        generate_model(args)
        
        i += 1


run_simulation()



