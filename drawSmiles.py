from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


def smitosvg(smi, molSize=(400, 200), outFile=None):
	mol = Chem.MolFromSmiles(smi)
	mc = Chem.Mol(mol.ToBinary())
	if not mc.GetNumConformers():
		rdDepictor.Compute2DCoords(mc)
	drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
	opts = drawer.drawOptions()
	drawer.DrawMolecule(mc)
	drawer.FinishDrawing()
	svg = drawer.GetDrawingText()
	svg = svg.replace('svg:', '')

	# bug fix for rendering in chrome
	svg = svg.replace('xmlns:svg=', 'xmlns=')

	if outFile:
		with open(outFile, 'w') as fh:
			fh.write(svg)

	return svg

smiles = 'Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)CO)c2cc1C'
svg = smitosvg(smiles, (150,100), outFile='test.svg')

smiles = '''
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H](C([C@H](O3)CO)=N)O)C)N
C1=CC(=N)C=NNN1[C@]([C@@H]2[C@H]=N[C@@H]([C@H]2CO)OO)C#N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CN)O)O)C)N
C=C1C(=NC=NN1C(=C[C@]2)P[C@H]([C@H](O2)COO)OC#N)N
C=C1C(=NCPN1C(=C)[C@]#N)N
C=C1C(=NC=NP1C(=C=N[C@])[C@]([C@@H])(C([C@H](O)CO)O)C#N)N
C=CC(=NC=NN([C@])P[C@@H][C@@H]=N[C@@H]([C@H]ONC)OOC#N)
C2=C1C(=NC=NN1C(=C2[C@]3([C@@H]([C@@H])))[C@H](O3)COOOC#N)N
C1=NC(=NC=NNC(=C1)[C@]([C@@H]([C@@H]([C@H]))COOO))N
C1=C[C@](=NC=NNC1[C@](=[C@@H]([C@@H](CO)O))C#N)N

'''


smiles = '''
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H](N([C@H](O3)CO)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)C)O)O)C#N)N
C23=C1C(=NC=NN1C(=C2)[C@]([C@@H]([C@@H]([C@H]C3(O))O)O)C#N)N
C2=C1C(=NC=NN1C2[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]OP3[C@@H]([C@@H]([C@H](O3)CO)O)OC#N)N
C23=C1C(=NC=NN1C(=C2)[C@]([C@@H]([C@@H]C3COO)O)C#N)N
C2P1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)
C2=C1C(=NC=NN1C(=C2)N([C@@H]([C@@H]([C@H](O)CO)O)O)C#N)N
C2=C1C(=NN=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)N
C2=CC(=NC1=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O))C#N)N
C2=C1C(=CC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]=NP3[C@@H]([C@@H]([C@H](O3)CO)O)OC#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CN)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H])[C@H](O3)COOO)C#N)N
C1=CC(=NC=NNC(=C1)[C@]2([C@@H]([C@@H]([C@H](O2)CO)O)O)C#N)N
C=1=CC(=NC=1C(=C)[C@]2([C@@H]([C@@H]([C@H](O2)CO)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)C=C)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]([C@@H][C@]3=N[C@@H]([C@H](O3)CO)OO)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O))C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]([C@@H]([C@@H]([C@H](=O))COO)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO))O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H](N([C@H](O3)CO)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CP)O)O)C#N)N
C2=C1C(=NC=NN1[C@@H](C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]([C@@H]([C@@H]([C@H])COO)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)N
C=CC(=NC#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)NO)O)O)C#N)N
C1=CC(=NC=NNN1[C@]2([C@@H]([C@@H]([C@H](O2)CO)O)O)C#N)N
C=C1C(=NC=NN1C(=C[C@])[C@]2([C@@H]([C@@H]([C@H](O2)CO)O)O)C#N)N
C12=CC(N)(N1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C[C@])N
C=CC(=NC=NN=N[C@]1([C@@H]([C@@H]([C@H](O1)CO)O)O)C#N)N
C2=C1C(=N)C=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)NO)O)O)C#N)N
C2=C1C(=NC=NN1P(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)N
C=C1C(=NC=NN1C(=C)[C@]2([C@@H]([C@@H]([C@H](O2)CO)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2[C@]3([C@@H]([C@@H])))[C@H](O3)COOOC#N)N
C23=C1C(=NC=NN1C(=C2)[C@]([C@@H]([C@@H]([C@H][C@]3(O))O)O)C#N)N
C2=C1C(=NC=NN1C(=C2[C@]3([C@@H]([C@@H])))[C@H](O3)COOOC#N)N
C12=CC(N)(N1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]3([C@@H]([C@@H]([C@H](O3)CO)O)O)C#N)
C2=C1C(=NC=NN1C(=C2)[C@]([C@@H]([C@@H]([C@H])COO)O)C#N)N
C2=C1C(=NC=NN1C(=C2)[C@]([C@@H])([C@@H]([C@H](O)CO)O)OC#N)N


'''


smiles = smiles.split('\n')

for n, sm in enumerate(smiles):
	if len(sm)==0: continue
	smitosvg(sm, (300,300), outFile='mols/test_%s.svg'%n)


