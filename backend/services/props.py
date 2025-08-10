"""
Property calculation utilities.
If RDKit is available, use it. Otherwise provide a lightweight fallback
that estimates a few simple properties.
"""

from typing import Dict
import math

# Try RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False

def calculate_properties(smiles: str) -> Dict:
    """
    Return dict:
      {
        'smiles': str,
        'valid': bool,
        'qed': float,
        'logp': float,
        'sa': float,
        'mw': float
      }
    If RDKit not present, returns heuristic estimates.
    """
    if not HAS_RDKIT:
        # Simple heuristic fallback
        valid = isinstance(smiles, str) and len(smiles) > 0
        length = len(smiles or "")
        qed = max(0.0, min(1.0, 0.5 + (math.log1p(length) - 1.0) * 0.05))
        logp = max(-3.0, min(6.0, (length % 10) * 0.3 - 1.5))
        sa = max(1.0, min(10.0, (length % 8) + 1.0))
        mw = length * 12.0
        return {"smiles": smiles, "valid": valid, "qed": round(qed, 3), "logp": round(logp, 3), "sa": round(sa, 3), "mw": round(mw, 3)}

    # Real RDKit path
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "valid": False, "qed": 0.0, "logp": 0.0, "sa": 999.0, "mw": 0.0}
    try:
        qed = QED.qed(mol)
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        # Here SA (synthetic accessibility) is approximated: use number of rings + HBA as rough proxy
        sa = rdMolDescriptors.CalcNumRings(mol) + rdMolDescriptors.CalcNumHBA(mol)
        return {"smiles": smiles, "valid": True, "qed": round(qed, 3), "logp": round(logp, 3), "sa": float(sa), "mw": round(mw, 3)}
    except Exception as e:
        return {"smiles": smiles, "valid": False, "error": str(e), "qed": 0.0, "logp": 0.0, "sa": 999.0, "mw": 0.0}


# Minimal graph conversion helpers (placeholders).
# If you're using GraphVAE with node/edge logits you must replace graph_to_smiles with proper decoding.

def smiles_to_graph(smiles):
    """Convert SMILES to graph format for model training (placeholder)."""
    if not HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # build minimal node features and edge lists (could be used by torch_geometric)
    atoms = []
    edges = []
    for atom in mol.GetAtoms():
        atoms.append((atom.GetAtomicNum(), int(atom.GetIsAromatic()), atom.GetFormalCharge()))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append((i, j, float(bond.GetBondTypeAsDouble())))
        edges.append((j, i, float(bond.GetBondTypeAsDouble())))
    return {"atoms": atoms, "edges": edges}

def graph_to_smiles(node_logits, edge_logits):
    """
    Placeholder: convert model outputs to a SMILES string.
    In practice, implement a proper graph decoding (atom type selection, bond assignment,
    valence checks, and RDKit molecule construction).
    """
    # If the VAE returns a direct smiles list (DummyVAE), just return it.
    try:
        # Some DummyVAE implementations might return strings directly
        if isinstance(node_logits, str):
            return node_logits
    except Exception:
        pass
    # Fallback placeholder
    return "CCO"
