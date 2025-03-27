%aiida
from aiida import orm
from datetime import datetime, timedelta
from aiida.engine import run, submit
from aiida.plugins import DbImporterFactory
CodDbImporter = DbImporterFactory('cod')
from aiida.orm import QueryBuilder
from aiida.orm import load_node, Node, Group, Computer, User, CalcJobNode, Code
from aiida.engine import calcfunction
from aiida_quantumespresso.common.types import SpinType
from aiida_quantumespresso.common.types import RelaxType
from aiida.plugins import CalculationFactory, DataFactory
from aiida.engine import calcfunction, workfunction
from aiida.orm import Code, Float, Str,List,Int
from ase.build import surface, bulk, add_adsorbate,molecule
from random import seed
from random import random
import numpy as np 
import matplotlib.pyplot as plt
from ase.visualize import view
import nglview
import ase.io
from pore import psd 
from porE.io.ase2pore import ase2pore 
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
import io
from pore import psd 
from orix.crystal_map import Phase
from orix.quaternion import Orientation, Rotation, symmetry
from orix.vector import Miller, Vector3d
from diffpy.structure import Lattice, Structure 
from scipy.spatial import KDTree
from aiida.plugins import WorkflowFactory
import nglview
from ase.visualize import view
import time 
PwCalculation = CalculationFactory('quantumespresso.pw')
StructureData = DataFactory('structure')
ArrayData = DataFactory('array')
KpointsData = DataFactory('array.kpoints')
Dict = DataFactory('dict')
UpfData = DataFactory('upf')
PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwBandsWorkChain = WorkflowFactory('quantumespresso.pw.bands')
PwPdosWorkChain = WorkflowFactory('quantumespresso.pdos')
YamboWorkflow = WorkflowFactory('yambo.yambo.yambowf')
YPPWorkflow = WorkflowFactory('yambo.ypp.ypprestart')

PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
import warnings
warnings.filterwarnings('ignore')



@workfunction
def computepores(structure, label):
    """
    we get a list of pores for the structure, (StructureData), and store, using ArrayData, 
    the pore sizes, and their coordinates and distribution, and return the same. The information 
    is stored with labels and descriptions that can be queried from AiiDA later using the user supplied label.
    """
    filz11='zifctmp.cif'
    zif_ase = structure.get_ase()
    ase.io.write(filz11, zif_ase)
    ase2pore(filz11) # creates ziftmp.xyz
    filpz11 = 'zifctmp.xyz'
    # check before compute?
    qb = QueryBuilder()
    FILTER="%{}-poRE%".format(label.value)
    qb.append(ArrayData,filters={'label':{'like': FILTER }})
    ab= qb.all(flat=True)
    if len(ab)>0:
        print("Pore data for {} found, will not recompute pores with poRE".format(label.value))
        return ab[0]
    else:
        print("{} did not yield preexisting pore computations, redoing".format(FILTER))
        no_pores,tmp1,tmp2,tmp3,tmp4 = psd.get_psd(filpz11,200,2000)
        pores_z11           = tmp1[0:no_pores]
        distr_z11           = tmp2[0:no_pores]
        pore_pos_cart_z11   = tmp3[0:no_pores]
        pore_pos_fracz_z11  = tmp4[0:no_pores]
        array = ArrayData()
        array.set_array('pore_sizes', pores_z11)
        array.set_array('pore_size_distribution',distr_z11 )
        array.set_array('pore_cartesian_coordinates',pore_pos_cart_z11  )
        array.set_array('pore_fractional_coordinates', pore_pos_fracz_z11)
        array.label = "{}-poRE".format(label.value)
        array.description="{} pore sizes and distributions from pORE code".format(label.value)
        array.store()
        return array


""" 
# Multiple positions in the void
we get structures multiple positions in the void, every L/24th the L of the cell. We need to compute the 
radius of the molecule, and the diameter of the pore, and place the molecule at every point in a sphere 
thats has a diameter given by: D - (r_of_molecule/2+ 1.7A) to ensure the molecule can be placed at the 
furthest point on the sphere without one of the atoms on it impinging on the wall of the pore.
"""

def molecule_radius(positions):
    "This should compute the radius of a simple molecule, CO2 given the vector coordinate positions of the atoms"
    return np.linalg.norm(np.array([np.max(positions[:,0]) - np.min(positions[:,0]),
        np.max(positions[:,1]) - np.min(positions[:,1]),
        np.max(positions[:,2]) - np.min(positions[:,2])])/2)

def obtain_positions(V, zf,mol, pore_center_coords):
    """
    We get a set of positions every 1/Nth of the unit cell length L, in a sphere with the diameter of
    the void, centered at the void geometric center. Some allowance of the magnitude of the molecule radius
    is made to ensure the molecule at the surface of the sphere does not overlap with the wall of the void.
      V == Pore size (radius)
    this function has to account for pockets in these voids. given a point more than R distance from pore center coords,
    look generally outwards, adding atoms away from the center direction in steps of 1/24, until a barrier is reached?
    should take care of obvious pockets. Occluded pockets will unfortunatelly be unaccounted for.

    """
    R=V-molecule_radius(mol.positions)
    co2 = molecule("CO2")
    r= molecule_radius(mol.positions)
    N= 48
    L= np.max(np.array(zf.cell))
    l= L /N   #  what is up here?  = cell length/24?  #  24 or 8 is steps along each direction of cell,
    num_pts = ((4/3*np.pi*R**3 )// (4/3*np.pi*l**3)).astype(int)  # vol of void/step size vol # we use l to get a unit volume to divide the volume into
    dist1 = np.random.default_rng(2024).standard_normal(size=(num_pts))
    dist2 = np.random.default_rng(2025).standard_normal(size=(num_pts))
    dist3 = np.random.default_rng(2026).standard_normal(size=(num_pts))
    mag = np.sqrt(dist1**2+ dist2**2 + dist3**2) / (np.cbrt( np.linspace(0,1, num_pts) ) *R )
    x, y, z= dist1/mag, dist2/mag, dist3/mag
    coords = np.stack((x,y,z),axis=-1)
    sz = coords.size
    coords_centered = coords+ pore_center_coords # why we are working using pore center, not axis (0,0)
    pocket_coords = get_pocket_position(R, r, num_pts, l, sz, zf, coords_centered, pore_center_coords,l )
    np.savetxt("pocket_coords.csv", pocket_coords, delimiter=",")
    np.savetxt("coords.csv", coords, delimiter=",")
    if pocket_coords.size!=0:
        return np.vstack((coords_centered,  pocket_coords))
    else:
        return coords_centered

def get_pocket_position(R, r, num_pts, step_size, sz, zf, vol_insert_positions , pore_center_coords,l):
    """
    R - available radial space less the molecule's size, to account for overlap elimination.
    r - molecule radius
    num_pts: total volume related to R is chunked up into num_pts pieces, with radii step_size=l,
    step_size:
    sz - num of tentatively available positions?
    zf - ASE object holding ZIF structure
    vol_insert_positions -  coordinate positions in the void.
    pore_center_coords - position of void in actual ZIF cell.
    """
    add_onion = 0 #  1?
    kept=np.array([])
    KR= R
    while add_onion <=3: #  we are adding a whole new onion layer using a normal distro.
        numpts =  ( (4 *np.pi*KR**2 ) // (4*np.pi*(l)**2) ).astype(int) # density uniformity per layer solved.

        dist1 = np.random.default_rng(2024).standard_normal(size=(numpts ))
        dist2 = np.random.default_rng(2025).standard_normal(size=(numpts ))
        dist3 = np.random.default_rng(2026).standard_normal(size=(numpts ))
        mag = np.sqrt(dist1**2+ dist2**2 + dist3**2)
        x, y, z= dist1/mag *KR , dist2/mag *KR, dist3/mag * KR
        pos = np.stack((x,y,z), axis=-1) + pore_center_coords
        total_surface_pos = pos.size
        keep,reject = check_pocket(zf, vol_insert_positions, pos,r) # add check if its inside pore?
        if kept.size==0:
            kept = keep
        elif keep.size!=0:
            kept = np.vstack((kept, keep ))
        time.sleep(0.5)
        if reject >= total_surface_pos   :
            add_onion +=1   # we  retry three step sizes before giving up on pockets.
        else:
            pass
        KR += step_size # this should allow us to reach pockets.
        if KR> R*2:
            print("KR>2R")
            break
    return kept

def check_pocket(zf, vol_insert_positions, pos,r):
    """
    zf - ASE structure object
    vol_insert_positions - coordinates in the void
    pos - coordinates on the new onion layer.
    r - molecule radius
    We have implemented a nearest neighbout search on a KDTree.
    return a list of positions to keep, and those to reject.
    """

    kdtree=KDTree(np.vstack( (zf.get_positions(), vol_insert_positions )))
    reject = 0
    keep = np.array([])
    for position in pos: # loop through all the surface positions
        dist,points=kdtree.query(position,1)  # points are assumed to be idx of closest atoms
        dist = np.linalg.norm(dist)
        nrm = np.linalg.norm(zf.get_positions() - kdtree.data[points] , axis=1) # if points are not idx of closes atoms, this fails
        idx = np.where(nrm==0)
        if idx[0].size:
            ix = idx[0][0]
            zr = covalent_radii[zf.numbers[ix]]
            if ( dist < r+zr):  # reject
                reject +=1
            elif (position[0]> np.max(np.array(zf.cell)[:,0])  or position[1]> np.max(np.array(zf.cell)[:,1]) or\
                position[2]> np.max(np.array(zf.cell)[:,2]) ) or (position[0]< np.min(np.array(zf.cell)[:,0])  or\
                position[1]< np.min(np.array(zf.cell)[:,1]) or position[2] < np.min(np.array(zf.cell)[:,2])  )  :
                reject +=1
            else: # keep position, how to update the kdtree on the fly?
                if keep.size!=0:
                    keep = np.vstack((keep, position))
                else:
                    keep = position
                kdtree = KDTree(np.vstack( (zf.get_positions(),vol_insert_positions , [position])))
        else: # not one of the originals, must be the co2
            if (dist < r+r): # reject
                reject +=1
            elif (position[0]> np.max(np.array(zf.cell)[:,0])  or position[1]> np.max(np.array(zf.cell)[:,1]) or\
                position[2]> np.max(np.array(zf.cell)[:,2]) ) or (position[0] < np.min(np.array(zf.cell)[:,0])  or\
                position[1]< np.min(np.array(zf.cell)[:,1]) or position[2] < np.min(np.array(zf.cell)[:,2]) ):
                reject +=1
            else:
                if keep.size!=0:
                    keep = np.vstack((keep, position))
                else:
                    keep = position
                kdtree = KDTree(np.vstack( (zf.get_positions(), vol_insert_positions, [position])))
    return keep,reject

"""
Multiple directions.
We ensure that no overlap happens on rotation to a different direction.
"""

def check_direction(coo, zif_insert, POS_STR):
    "[-1.    -2.414  0.   ] 12.1303322625.753054897.70107689"
    a = "12.1303322625.753054897.70107689"

    kdtree= KDTree(zif_insert.get_positions() )
    overlap = False
    dist = 1000
    for i in range(coo.positions[0,:].size) :
        di, points= kdtree.query( coo.positions[i],1)
        dit = np.linalg.norm(di)
        dist = min(dist, dit)
        dfz = zif_insert.get_positions() - kdtree.data[points]
        nrm = np.linalg.norm( dfz ,  axis=1) # we cant find this, start  here.
        idx = np.where(nrm==0)
        if idx[0].size:
            ix = idx[0][0]
            zr = covalent_radii[zif_insert.numbers[ix]]
            r =  covalent_radii[coo.numbers[i]]
            if (dit < r+zr): # reject
                overlap = True
                break
        else:
            print("didnt find nrm=0 ")
    return  overlap , dist

def directions(co2, direction, zif_ase,MOLECULE, QUERY_LABEL_ZIF,DIRECTION,pos_cart,POS_STR,pore_size):
    axis = co2.positions[1] - co2.positions[2]
    co2.rotate(axis,direction, center='COM' )
    coo = co2.copy()
    coo.positions[0] = co2.positions[0]+ pos_cart
    coo.positions[1] = co2.positions[1]+ pos_cart
    coo.positions[2] = co2.positions[2]+ pos_cart
    copos = coo.positions
    outside = False
    if (np.max(copos[:,0])> np.max(np.array(zif_ase.cell)[:,0])  or np.max(copos[:,1])> np.max(np.array(zif_ase.cell)[:,1]) or\
                np.max(copos[:,2])> np.max(np.array(zif_ase.cell)[:,2]) ) or (np.min(copos[:,0]) < np.min(np.array(zif_ase.cell)[:,0])  or\
                np.min(copos[:,1])< np.min(np.array(zif_ase.cell)[:,1]) or np.min(copos[:,2]) < np.min(np.array(zif_ase.cell)[:,2])):
        outside = True
    zif_insert = zif_ase.copy()
    overlap, dist = check_direction(coo, zif_insert, POS_STR)
    zif_insert.append('C')
    zif_insert.positions[-1]= coo.positions[0]
    zif_insert.append('O')
    zif_insert.positions[-1]= coo.positions[1]
    zif_insert.append('O')
    zif_insert.positions[-1]= coo.positions[2]

    zif_strucdata = StructureData(ase= zif_insert)
    zif_strucdata.label="ADSORBED-{}-FRAMEWORK-{}-DIRECTION-{}-POS-{}".format(MOLECULE.value,
                                QUERY_LABEL_ZIF.value, DIRECTION,POS_STR)
    zif_strucdata.description = "{} Adsorbed in {} in the direction {}, placed at coordinates {}, with a pore size\
                     {}".format(MOLECULE.value, QUERY_LABEL_ZIF.value,
                                DIRECTION,pos_cart, pore_size)

    return  dist, outside, overlap, zif_strucdata

"""
Generate adsorbed Realizations.
We obtain a set of structures, from the same ZIF, but with a MOLECULE adsorbed in the largest pore of 
the ZIF, treated as a cubic structure, and the molecule rotated over all unique directions in the upper 
hemisphere, one structure for each such unique orientation of the molecule in the pore. 
The MOLECULE is assumed to be linear.
"""

@workfunction
def generate_structures(structure, POREOUTPUT, MOLECULE,QUERY_LABEL_ZIF, ROTATIONS):
    zif_ase = structure.get_ase()
    print(zif_ase.get_cell())
    cubic = Phase(point_group="m3m", structure=Structure(  lattice=Lattice(*zif_ase.cell.cellpar() )))
    pore_pos_cart = POREOUTPUT.get_array('pore_cartesian_coordinates')
    pore_sizes = POREOUTPUT.get_array('pore_sizes')
    families = ROTATIONS.get_array('rotations')
    co2 = molecule(MOLECULE.value)
    ZIF_REALIZATIONS = List()
    positions_in_void = obtain_positions(pore_sizes[-1], zif_ase, co2, pore_pos_cart[-1] )
    np.savetxt("obtained_positions.csv", positions_in_void, delimiter=",")
    #print("positions_in_void",positions_in_void)
    #return
    #print("pos in void and pos_cart", positions_in_void,pore_pos_cart[-1] )
    #print ("positions_in_void {}".format(len(positions_in_void)))
    count = 0 
    omitted= 0 
    for pos_cart in positions_in_void:
        #print("Position in void {}".format(pos_cart))
        for family in  families:
            #print("Family: {}".format(family))
            Z= Miller(uvw=[family], phase=cubic)
            Zf, idx = Z.symmetrise(unique=True, return_index=True)
            #print("Number Unique directions  in the upper hemisphere of {} family  {}".format(
            #       family,len(Zf.coordinates)))
            for direction in Zf.coordinates:
                DIRECTION= np.array2string(direction).replace('[', '').replace(']', '').replace(' ','')
                POS_STR=np.array2string(pos_cart).replace('[', '').replace(']', '').replace(' ','p')# ! how to separate now?
                # check before creating:
                qb = QueryBuilder()
                FILTER = '%ADSORBED-{}-FRAMEWORK-{}-DIRECTION-{}-POS-{}%'.format(MOLECULE.value,
                        QUERY_LABEL_ZIF.value,DIRECTION,POS_STR)
                qb.append(StructureData,filters={'label':{'like': FILTER }})
                exists= qb.all(flat=True)
                if len(exists)>0:
                    print("will not recompute the structures for {} returning stored".format(FILTER))
                    ZIF_REALIZATIONS.extend([e.pk for e in exists ])
                else:
                    min_end, outside, overlap, zif_strucdata = directions(co2, direction, zif_ase,MOLECULE, QUERY_LABEL_ZIF,
                                               DIRECTION,pos_cart, POS_STR,pore_sizes[-1]) 
                    if (min_end>=0.5) and  (not outside) and (not overlap):
                        zif_strucdata.store()
                        stored  = zif_strucdata.pk
                        ZIF_REALIZATIONS.append(stored) 
                        count+=1
                        if count == 401:
                            print("401 =============== {} {}".format(direction, POS_STR))
                    elif min_end < 0.500 or (outside) or (overlap): 
                        omitted+=1
                        #print("not storing structure violating min dist. of 0.5: {},or its outside cell: {}, {}".format(min_end,outside,overlap))
                        pass
    print(" count {}   omitted  {}".format(count, omitted))                    
    ZIF_REALIZATIONS.store()               
    return ZIF_REALIZATIONS


@workfunction
def compose_realizations(structure, families, LABEL, MOLECULE):
    """
    Workfunction that composes the two calcfunctions to preserve provenance
    """
    pore_data = computepores(structure, LABEL)
    realizations = generate_structures(structure, pore_data , MOLECULE,LABEL, families)
    return realizations



