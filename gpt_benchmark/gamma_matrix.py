# %%
import gpt as g
import numpy as np

g5 = g.gamma[5].tensor()
gT5 = g.gamma["T"].tensor() * g.gamma[5].tensor()
gZ5 = g.gamma["Z"].tensor() * g.gamma[5].tensor()
gplus5 = ( gT5 + 1j * gZ5 ) / np.sqrt(2)


Cg5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma[5].tensor()
CgT5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma["T"].tensor() * g.gamma[5].tensor()
CgZ5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma["Z"].tensor() * g.gamma[5].tensor()
Cgplus5 = ( CgT5 + 1j * CgZ5 ) / np.sqrt(2)

print("gplus5: ")
print(gplus5)

print("Cgplus5: ")
print(Cgplus5)

print("CgT5: ")
print(CgT5)

print("CgZ5: ")
print(CgZ5)

print("Cg5: ")
print(Cg5)




# %%
