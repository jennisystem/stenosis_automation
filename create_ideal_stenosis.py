import os
import sys
import math
import numpy as np
import sv
import pickle

# pathplanning.Path
# sv.pathplanning.Path

# Initialize path names
projDir = "C:/Users/Emmaline/Desktop/Marsden Lab/stenosis_automation"
# projDir = "D:/1_CS/jenn/stenosis"

meshDir = os.path.join(projDir, 'mesh')
modelsDir = os.path.join(projDir, 'model')
pathsDir = os.path.join(projDir, 'paths')
segDir = os.path.join(projDir, 'segments')

# Create paths if does not exist
os.makedirs(meshDir, exist_ok=True)
os.makedirs(modelsDir, exist_ok=True)
os.makedirs(pathsDir, exist_ok=True)
os.makedirs(segDir, exist_ok=True)


sys.path.append(projDir)
import helper_functions
from SV_lofting_solids_meshing import *
sys.path.pop()

sys.path.append('sv')

# Source: adapted from https://github.com/neilbalch/SimVascular-pythondemos/blob/master/contour_to_lofted_model.py
# Credit: adapted from original script by Jonathan Pham

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


#########################################
# DEFINE THE NUMBER OF CAPS YOU EXPECT FOR
# THE TOTAL MODEL (INLETS + OUTLETS)
NUM_CAPS = 2
#########################################


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


    # [=== Create model ===]
    contour_list = [segmentPolydata,]
    capped_vessels = create_vessels(contour_list=contour_list)
    # unioned_model = union_all(capped_vessels)
    unioned_model = capped_vessels[0]
    # model = sv.modeling.PolyData()
    model = clean(unioned_model)
    model = norm(model)
    model = remesh(model, cell_density_mm=[10,1])
    tmp = model.get_polydata()
    sv.dmg.add_model(name = model_name + "_model", model = model)


    # [=== Combine faces ===]
    model.set_surface(tmp)
    model.compute_boundary_faces(45)
    caps = model.identify_caps()
    ids = model.get_face_ids()
    walls = [ids[i] for i,x in enumerate(caps) if not x]
    '''while len(walls) > 1:
        target = walls[0]
        lose = walls[1]
        combined = sv.mesh_utils.remesh_faces(model.get_polydata(),[target],lose)
        model.set_surface(combined)
        ids = model.get_face_ids()
        caps = model.identify_caps()
        walls = [ids[i] for i,x in enumerate(caps) if not x]
        print(walls)
    ids = model.get_face_ids()'''
    cco8_model_name = model_name + '_CC0_8'
    if True:
        sv.dmg.add_model(cco8_model_name, model)

    # Remeshing (uncomment)
    '''if len(ids) > NUM_CAPS:
        face_cells = []
        for idx in ids:
            face = model.get_face_polydata(idx)
            cells = face.GetNumberOfCells()
            face_cells.append(cells)
        data_to_remove = len(ids) - NUM_CAPS
        remove_list = []
        for i in range(data_to_remove):
            remove_list.append(ids[face_cells.index(min(face_cells))])
            face_cells[face_cells.index(min(face_cells))] += 1000
        while len(remove_list) > 0:
            target = walls[0]
            lose = remove_list.pop(-1)
            combined = sv.mesh_utils.remesh_faces(model.get_polydata(),[target],lose)
            model.set_surface(combined)
        print(model.get_face_ids())'''

    ###############################
    # LOCAL SMOOTHING (not included)
    ###############################
    #smoothing_params = {'method':'constrained', 'num_iterations':5, 'constrain_factor':0.2, 'num_cg_solves':30}
    '''smooth_model = model.get_polydata()
    for idx, contour_set in enumerate(contour_list):
         if idx == 0:
              continue
         smoothing_params = {'method':'constrained', 'num_iterations':3, 'constrain_factor':0.1+(0.9*(1-contour_set[0].get_radius()/contour_list[0][0].get_radius())), 'num_cg_solves':30}
         smooth_model = sv.geometry.local_sphere_smooth(smooth_model,contour_set[0].get_radius()*2,contour_set[0].get_center(),smoothing_params)
         print('local sphere smoothing {}'.format(idx))
    model.set_surface(smooth_model)'''

    # [=== Create mesh ===]
    faces = model.get_face_ids()
    mesher = sv.meshing.create_mesher(sv.meshing.Kernel.TETGEN)
    GLOBAL_EDGE_SIZE = 0.1  # 0.01 for production
    tet_options = sv.meshing.TetGenOptions(GLOBAL_EDGE_SIZE,True,True)
    tet_options.no_merge = False
    tet_options.optimization = 5
    tet_options.minimum_dihedral_angle = 18.0
    mesher.set_model(model)
    mesher.set_walls(walls)
    mesher.generate_mesh(tet_options)
    msh = mesher.get_mesh()
    if True:
        sv.dmg.add_mesh(cco8_model_name+'_mesh', msh, cco8_model_name)
        msh = sv.dmg.get_mesh(cco8_model_name+'_mesh')

    return path, segmentations, model, msh




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
    10**np.arange(-1,2.1,1),      # midsection_percentage - midsection area as a percentage of the inlet area (Logarithmic scale between 0.1% and 100%)
    (1.0, 1.0, 0.1),        # radius_inlet
    (50, 100, 50),          # sigma - controls the spread of the stenosis/aneurysm
    (0.0, 0.0, 0.25),       # mu - controls the center of the stenosis/aneurysm
    [poly],                    # f - Parametric function of x component
    [poly],                    # g - Parametric function of y component
    [r],                    # r - Parametric function of radius
    [(0, 0, 1/100),],          # inputs to f(t, ...)
    [(0, 0, 0, 1/1000),]          # inputs to g(t, ...)
]


# test case
'''ranges[4] = [1.0, ]
ranges[6] = [50, ]'''


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

PICKLE_PROTOCOL = 4
    
def generate_models(create=True, addDmg=True):
    if create:
        i = 1
        fModelList = open(os.path.join(projDir, 'model_list.txt'), 'w')
        # Generate inputs
        for input_arr in generate_simulation_inputs():
            # model_name, x0, xf, nx, distance, midsection_percentage, radius_inlet, sigma, mu, f, g, radius, f_params, g_params = input_arr
            model_name = "model_{0}".format(i)

            
            args = (model_name, *input_arr)

            try:
                #  [=== Generate model ===]
                # Create a new model (if possible)
                path, segmentations, model, msh = generate_model(args)
                print(input_arr)
                
                #  [=== Serialize model ===]
                # Path
                '''with open(os.path.join(pathsDir, model_name+'_pth.P'), 'wb') as f_out:
                    pickle.dump(path, f_out, protocol=PICKLE_PROTOCOL)
                # Segment
                with open(os.path.join(segDir, model_name+'_sgmt.P'), 'wb') as f_out:
                    pickle.dump(segmentations, f_out, protocol=PICKLE_PROTOCOL)'''
                # Model
                '''with open(os.path.join(modelsDir, model_name+'_mdl.P'), 'wb') as f_out:
                    pickle.dump(model, f_out, protocol=PICKLE_PROTOCOL)
                # Mesh
                with open(os.path.join(meshDir, model_name+'_msh.P'), 'wb') as f_out:
                    pickle.dump(msh, f_out, protocol=PICKLE_PROTOCOL)'''
                model.write(os.path.join(modelsDir, model_name+'_mdl'), 'vtp')

                # msh.get_polydata.write(os.path.join(meshDir, model_name+'_msh'))
                # msh.write_mesh(os.path.join(meshDir, model_name+'_msh'))

                #  [=== Log model configs in manifest file ===]
                # Write params of model to file
                buffStr = "%s %d %d\n" % (model_name, x0, xf)
                fModelList.write(buffStr)

                # Yield model
                yield path, segmentations, model, msh
            except sv.meshing.Error:
                print("An error occured while creating mesh for %s" % model_name)
            '''except:
                print("An unknown error occured for %s" % model_name)
                return'''
            
            fModelList.flush()
            i += 1
    else:
        fModelList = open(os.path.join(projDir, 'model_list.txt'), 'r')
        for row in fModelList:
            # Read model configs from manifest file
            model_Name, *config_arr = row.strip().split()
            
            # [=== Deserialize model ===]
            # Path
            # with open(os.path.join(pathsDir, model_name+'_pth.P'), 'rb') as f_in:
                #path = pickle.load(f_in)
            # Segment
            # with open(os.path.join(segDir, model_name+'_sgmt.P'), 'rb') as f_in:
                # segmentations = pickle.load(f_in)
            # Model
            model.write(model_name+'_mdl.P')
            # with open(os.path.join(modelsDir, model_name+'_mdl.P'), 'rb') as f_in:
                #model = pickle.load(f_in)
                #model.write(f_in)
            # Mesh
            msh.write(model_name+'_msh.P')
            # with open(os.path.join(meshDir, model_name+'_msh.P'), 'rb') as f_in:
                #msh = pickle.load(f_in)
                #msh.write(f_in)
            
            # [=== Add models to dmg ===]
            if addDmg:
                path_name = model_name + '_path'
                sv.dmg.add_path(name = path_name, path = path)
                segmentations_name = model_name + '_segmentations'
                sv.dmg.add_segmentation(name = segmentations_name, path = path_name, segmentations = segmentations)
                sv.dmg.add_model(name = model_name, model = model)
                sv.dmg.add_mesh(model_name+'_mesh', msh, model_name)
            
            # Yield model
            yield path, segmentations, model, msh

def run_simulation():
    for path, segmentations, model, msh in generate_models():
        # Run simulation on models iteratively
        pass

run_simulation()



