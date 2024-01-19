# torch_coulomb_matrix
Coulomb Matrix as Torch Tensor from rdkit molecules

```python
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import torch

def GetTensorCoulombMatrix(molecule, pad_up_to = None, skipConformerGeneration = False):
  """
  Generates a Coulomb Matrix as tensor from a rdkit.Chem.rdchem.Mol or SMILES string.

  Arguments:
    molecule: rdkit.Chem.rdchem.Mol or SMILES-string (str)
    pad_up_to: If not None, the output Tensor is padded up to this number along the 1st dimension.

  Returns:
    A torch.tensor of size (n_atoms x atoms) if pad_up_to = None
    Otherwise returns a tensor of size (n_atoms x pad_up_to)
  
  """

  assert type(molecule) in [str, rdkit.Chem.rdchem.Mol], "Molecule is neither an rdkit mol or string"

  if type(molecule) == str:
    molecule = Chem.MolFromSmiles(molecule)

  if not skipConformerGeneration:
    molecule = Chem.AddHs(molecule)

    ps = AllChem.ETKDGv2()
    AllChem.EmbedMolecule(molecule, ps)

    if molecule.GetNumConformers() == 0: # If conformer generation fails the first time around, try to start from random coordinates
      ps.useRandomCoords = True
      AllChem.EmbedMolecule(molecule, ps) 
  
  try:
    cmat = rdMolDescriptors.CalcCoulombMat(molecule)
    cmat_as_tensor = torch.vstack([torch.tensor(row, dtype = torch.float) for row in cmat])

  except:
    raise ValueError("Failed to generate coulomb matrix :(")

  if pad_up_to is not None:
    width = cmat_as_tensor.shape[1]
    needed_pad = pad_up_to - width
    return torch.nn.functional.pad(input = cmat_as_tensor, pad = (0, needed_pad))

  else:
    return cmat_as_tensor
  
```
