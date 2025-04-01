import readsnap_mod as rs
import numpy as np


header = rs.snapshot_header("data/snapdir_062/snap_062.11") # reads snapshot header
print(header.massarr)
print(header.npart)
print(header.time)
print(header.redshift)
part_type=1
xyz = rs.read_block("data/snapdir_062/snap_062.10","POS ",parttype=part_type,verbose=True) 
print("xyz array of parttype="+str(part_type)+", array shape="+str(xyz.shape)+" from " + str(np.min(xyz)) +" to " + str(np.max(xyz))+"Kpc/h")
Vxyz = rs.read_block("data/snapdir_062/snap_062.10","VEL ",parttype=part_type,verbose=True)
print("Vxyz array of parttype="+str(part_type)+", array shape="+str(Vxyz.shape)+" from " + str(np.min(Vxyz)) +" to " + str(np.max(Vxyz))+"Km/s")