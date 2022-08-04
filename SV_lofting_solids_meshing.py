from sv import *
import vtk
import os
import platform

options = geometry.LoftNurbsOptions()

def bad_edges(model):
    fe = vtk.vtkFeatureEdges()
    fe.FeatureEdgesOff()
    fe.BoundaryEdgesOn()
    fe.NonManifoldEdgesOn()
    fe.SetInputData(model.get_polydata())
    fe.Update()
    return fe.GetOutput().GetNumberOfCells()

def clean(model):
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.PointMergingOn()
    clean_filter.SetInputData(model.get_polydata())
    clean_filter.Update()
    model.set_surface(clean_filter.GetOutput())
    return model

def tri(model):
    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputData(model.get_polydata())
    tri_filter.Update()
    model.set_surface(tri_filter.GetOutput())
    return model

def fill(model):
    poly = vmtk.cap(surface=model.get_polydata(),use_center=False)
    model.set_surface(poly)
    return model

def surf_area(poly):
    mass = vtk.vtkMassProperties()
    mass.SetInputData(poly)
    mass.Update()
    return mass.GetSurfaceArea()

def remesh(model,cell_density_mm=[100,10]):
    """
    PARAMTERS:
    model:        SV solid modeling object
    cell density: number of mesh elements per square
                  mm. Given as an acceptable range.
    """
    poly = model.get_polydata()
    poly_sa = surf_area(poly)*100
    cell_num_hmin = max(5,int(cell_density_mm[0]*poly_sa))
    cell_num_hmax = max(5,int(cell_density_mm[1]*poly_sa))
    hmin = (((poly_sa/100)/cell_num_hmin)*2)**(1/2)
    hmax = (((poly_sa/100)/cell_num_hmax)*2)**(1/2)
    print("Remeshing Model:\nhmin: ----> {}\nhmax ----> {}".format(hmin,hmax))
    remeshed_polydata = mesh_utils.remesh(model.get_polydata(),hmin=hmin,hmax=hmax)
    model.set_surface(remeshed_polydata)
    return model

def remesh_face(model,face_id,cell_density_mm=40):
    face_poly = model.get_face_polydata(face_id)
    face_sa = surf_area(face_poly)*100
    cell_num = max(5,int(cell_density_mm*face_sa))
    edge_size = round((((face_sa/100)/cell_num)*2)**(1/2),5)
    edge_size = max(0.001,edge_size)
    print("Remeshing Face: {} ----> Edge Size: {}".format(face_id,edge_size))
    remeshed_poly = mesh_utils.remesh_faces(model.get_polydata(),[face_id],edge_size)
    model.set_surface(remeshed_poly)
    return model

def remesh_caps(model,cell_density_mm=40):
    cap_ids = model.identify_caps()
    face_ids = model.get_face_ids()
    for i,c in enumerate(cap_ids):
        if c:
            model = remesh_face(model,face_ids[i],cell_density_mm=cell_density_mm)
    return model

def norm(model):
    """
    Determine the normal vectors along the
    polydata surface.

    PARAMETERS
    model:    SV solid modeling object
    """
    norm_filter = vtk.vtkPolyDataNormals()
    norm_filter.AutoOrientNormalsOn()
    norm_filter.ComputeCellNormalsOn()
    norm_filter.ConsistencyOn()
    norm_filter.SplittingOn()
    norm_filter.NonManifoldTraversalOn()
    norm_filter.SetInputData(model.get_polydata())
    norm_filter.Update()
    model.set_surface(norm_filter.GetOutput())
    return model

def loft(contours,num_pts=50,distance=False):
    """
    Generate an open lofted NURBS surface along a given
    vessel contour group.

    PARAMETERS:
    contours (list):  list of contour polydata objects defining one vessel.
    num_pts  (int) :  number of sample points to take along each contour.
    distance (bool):  flag to use distance based method for contour alignment
    """
    for idx in range(len(contours)):
        contours[idx] = geometry.interpolate_closed_curve(polydata=contours[idx],number_of_points=num_pts)
        if idx != 0:
            contours[idx] = geometry.align_profile(contours[idx-1],contours[idx],distance)
    options = geometry.LoftNurbsOptions()
    loft_polydata = geometry.loft_nurbs(polydata_list=contours,loft_options=options)
    loft_solid = modeling.PolyData()
    loft_solid.set_surface(surface=loft_polydata)
    return loft_solid

def loft_all(contour_list):
    """
    Loft all vessels defining the total model that you want to create.

    PARAMETERS
    contour_list: (list): list of lists that contain polydata contour groups
                          Example for two vessels:

                          contour_list -> [[polydataContourObject1,polydataContourObject2],[polydataContourObject1,polydataContourObject2]]

    RETURNS:
    lofts:        (list): list of open sv solid models of the lofted 3D surface. Note that
                          the loft is not yet capped.
    """
    lofts = []
    for group in contour_list:
        lofts.append(loft(group))

    #lofts.append(loft(contour_list))
    return lofts


def cap_all(loft_list):
    """
    Cap all lofted vessels.

    PARAMETERS:
    loft_list  (list): list of sv modeling solid objects that are open lofts generated from
                       the 'loft_all' function.

    RETURNS:
    capped     (list): list of capped solids
    """
    capped = []
    for loft_solid in loft_list:
        capped_solid = modeling.PolyData()
        capped_solid.set_surface(vmtk.cap(surface=loft_solid.get_polydata(),use_center=False))
        capped_solid.compute_boundary_faces(angle=45)
        capped.append(capped_solid)
    return capped

def check_cap_solids(cap_solid_list):
    """
    Check capped solids for bad edges.
    """
    for solid in cap_solid_list:
        if bad_edges(solid) > 0:
            return False
    return True

def create_vessels(contour_list,attempts=5):
    """
    create seperate capped vessels for all contour groups defining a model of interest.

    PARAMETERS:
    contour_list: (list): list of lists of contour polydata objects defining individual vessels
                          within the total model.
    attemps:      (int) : the number of times that bad edges correction will be attemped during loft
                          alignment
    """
    i = 0
    success = False
    while not success and i < attempts:
        lofts = loft_all(contour_list)
        cap_solids = cap_all(lofts)
        success = check_cap_solids(cap_solids)
    if success:
        print('Lofting Passed')
    else:
        print('Lofting Failed')
    return cap_solids

# print("CREATING VESSELS")
# capped_vessels = create_vessels(all_contour_polydata_lists)
# print("CREATED VESSELS")

###############################
# INITIALIZE MODELING KERNEL
###############################
def robust_union(model_1,model_2):
    """
    Union two capped SV solid objects into one sv solid object.

    PARAMETERS:
    model_1: (sv.modeling.solid): first solid model
    model_2: (sv.modeling.solid): second solid model
    """
    modeler = modeling.Modeler(modeling.Kernel.POLYDATA)
    model_1_be = bad_edges(model_1)
    model_2_be = bad_edges(model_2)
    print("Model 1 Bad Edges: {}\n Model 2 Bad Edges: {}".format(model_1_be,model_2_be))
    if model_1_be == 0 and model_2_be == 0:
        unioned_model = modeler.union(model_1,model_2)
        unioned_model = clean(unioned_model)
        unioned_model = norm(unioned_model)
        if bad_edges(unioned_model) > 0:
            print('Unioned Model Bad Edges: {}'.format(bad_edges(unioned_model)))
            print('Filling')
            unioned_model = fill(unioned_model)
            print('Unioned Model Bad Edges: {}'.format(bad_edges(unioned_model)))
            print('Cleaning')
            unioned_model = clean(unioned_model)
            print('Unioned Model Bad Edges: {}'.format(bad_edges(unioned_model)))
            unioned_model = tri(unioned_model)
            print('Unioned Model Bad Edges: {}'.format(bad_edges(unioned_model)))
        print('union successful')
        return unioned_model
    else:
        print('1 or both models have bad edges.')
        unioned_model = modeler.union(model_1,model_2)
        unioned_model = clean(unioned_model)
        unioned_model = norm(unioned_model)
        return unioned_model

def union_all(solids,n_cells=100):
    """
    Union a list of all vessels together.

    PARAMETERS:
    solids:   (list): list of capped sv solid objects

    RETURNS:
    joined  (sv.modeling.solid): sv solid object
    """
    for i in range(len(solids)):
        solids[i] = norm(solids[i])
        solids[i] = remesh(solids[i])
        solids[i] = remesh_caps(solids[i])
    joined = robust_union(solids[0],solids[0])
    # for i in range(2,len(solids)):
    #     print("UNION NUMBER: "+str(i)+"/"+str(len(solids)))
    #     joined = robust_union(joined,solids[i])
    #     if joined is None:
    #         print("unioning failed")
    #         return None
    print("unioning passed")
    return joined
'''
unioned_model = union_all(capped_vessels)
#unioned_model = capped_vessels[0]
model = modeling.PolyData()
tmp = unioned_model.get_polydata()
print("done unioning")
sv.dmg.add_model(name = model_name + "_model", model = unioned_model)
#########################################
# DEFINE THE NUMBER OF CAPS YOU EXPECT FOR
# THE TOTAL MODEL (INLETS + OUTLETS)
NUM_CAPS = 2
#########################################

############################
# COMBINE FACES
############################
"""
if not terminating:
    model.set_surface(tmp)
    model.compute_boundary_faces(45)
    caps = model.identify_caps()
    ids = model.get_face_ids()
    walls = [ids[i] for i,x in enumerate(caps) if not x]
    while len(walls) > 1:
        target = walls[0]
        lose = walls[1]
        combined = mesh_utils.combine_faces(model.get_polydata(),target,lose)
        model.set_surface(combined)
        ids = model.get_face_ids()
        caps = model.identify_caps()
        walls = [ids[i] for i,x in enumerate(caps) if not x]
        print(walls)
    ids = model.get_face_ids()
    if True:
        dmg.add_model(CCO_8,model)
    if len(ids) > NUM_CAPS:
        face_cells = []
        for idx in ids:
            face = model.get_face_polydata(idx)
            cells = face.GetNumberOfCells()
            print(cells)
            face_cells.append(cells)
        data_to_remove = len(ids) - NUM_CAPS
        remove_list = []
        for i in range(data_to_remove):
            remove_list.append(ids[face_cells.index(min(face_cells))])
            face_cells[face_cells.index(min(face_cells))] += 1000
        print(remove_list)
        while len(remove_list) > 0:
            target = walls[0]
            lose = remove_list.pop(-1)
            combined = mesh_utils.combine_faces(model.get_polydata(),target,lose)
            model.set_surface(combined)
            print(remove_list)
        print(model.get_face_ids())
    ###############################
    # LOCAL SMOOTHING (not included)
    ###############################
    #smoothing_params = {'method':'constrained', 'num_iterations':5, 'constrain_factor':0.2, 'num_cg_solves':30}
    smooth_model = model.get_polydata()
    for idx, contour_set in enumerate(contour_list):
         if idx == 0:
              continue
         smoothing_params = {'method':'constrained', 'num_iterations':3, 'constrain_factor':0.1+(0.9*(1-contour_set[0].get_radius()/contour_list[0][0].get_radius())), 'num_cg_solves':30}
         smooth_model = geometry.local_sphere_smooth(smooth_model,contour_set[0].get_radius()*2,contour_set[0].get_center(),smoothing_params)
         print('local sphere smoothing {}'.format(idx))
    model.set_surface(smooth_model)
if not terminating:
    faces = model.get_face_ids()
    mesher = meshing.create_mesher(meshing.Kernel.TETGEN)
    tet_options = meshing.TetGenOptions(0.01,True,True)
    tet_options.no_merge = False
    tet_options.optimization = 5
    tet_options.minimum_dihedral_angle = 18.0
    mesher.set_model(model)
    mesher.set_walls(walls)
    mesher.generate_mesh(tet_options)
    msh = mesher.get_mesh()
    if True:
        dmg.add_mesh('CCO_8',msh,'CCO_8')
    if False:
        os.mkdir('D:\\svcco\\CCO_DATA')
"""
print("FINISHED")
'''