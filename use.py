from . code import *


"""
Compute and generate structures for : ZIF-CO3-1
"""


"""
Import a structure
"""
z11 = ase.io.read("ZIF-CO3-1.cif")
StructureData = DataFactory('structure')
aiida_struc = StructureData(ase = z11)
LABEL='ZIF-CO3-1'
aiida_struc.label = LABEL
aiida_struc.description = "ZIF-CO3-1  https://www.ccdc.cam.ac.uk/structures/Search?Ccdcid=HOYQAP&DatabaseToSearch=Published"
FILTER = '%{}%'.format(LABEL)
qb = QueryBuilder()
qb.append(StructureData,filters={'label':{'==': FILTER }})
exists= qb.all(flat=True)
if len(exists)==0: # save it
    print(aiida_struc.store(), "saved")
else:
    print(exists, "present")
print (exists)

