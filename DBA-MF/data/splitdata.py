import os
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = "./data"

def smiles_to_coords(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
    if result != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol_h)
    except:
        return None
    conf = mol_h.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    return coords


def process_line(line, task_id, name):
    l = line.strip().split(",")
    if len(l) < 2:
        return None

    '''
    toxcast, sider -> smi = l[0]; for i in range(1, size) 
    tox 21 -> smi = l[-1]; for i in range(12):
    muv -> smi = l[-1]; for i in range(17):
    '''
    smi = l[0]
    cur_item = l[task_id].strip()
    if cur_item not in {"0", "0.0", "1", "1.0"}:
        return None

    label = 0 if cur_item in {"0", "0.0"} else 1
    coords = smiles_to_coords(smi)
    if coords is None:
        return None

    return task_id, {"smiles": smi, "label": label, "coords": coords}


if __name__ == "__main__":
    name = "sider"   # muv,tox21,sider
    file_path = '/DBA-MF/data/sider/raw/sider.csv'

    lines = open(file_path, "r").readlines()[1:]
    np.random.shuffle(lines)
    tasks = {}
    max_workers = os.cpu_count() or 4
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for line in lines:
            size = len(line.split(","))
            for task_id in range(1, size):
                futures.append(executor.submit(process_line, line, task_id, name))

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                task_id, item = result
                if task_id not in tasks:
                    tasks[task_id] = []
                tasks[task_id].append(item)

    for task_id, task_data in tasks.items():
        task_dir = os.path.join(BASE_DIR, f"{name}/ll/{task_id}")
        os.makedirs(os.path.join(task_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(task_dir, "processed"), exist_ok=True)

        task_data_sorted = sorted(task_data, key=lambda x: x["label"])

        save_path = os.path.join(task_dir, "raw", f"{name}.json")
        with open(save_path, "w") as f_out:
            json.dump(task_data_sorted, f_out, indent=2)

        print(f"task {task_id}: {len(task_data_sorted)} molecules saved to {save_path}")
