import os
import math
import csv
import requests
import shutil
import pandas as pd
import joblib
import urllib3
from collections import defaultdict, Counter

# Disable HTTPS certificate warnings (RCSB uses an unverified certificate)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Save path to main script directory (so we can load model files reliably)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def download_pdb(pdb_id, dest_folder):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        with open(os.path.join(dest_folder, f"{pdb_id}.pdb"), 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download PDB: {pdb_id}")

def parse_pdb(file):
    protein_residues = defaultdict(lambda: {'residue_name': '', 'atoms': [], 'chain_id': ''})
    nucleotide_residues = defaultdict(lambda: {'residue_name': '', 'atoms': [], 'chain_id': ''})

    model_found = False
    model_count = 0

    with open(file, 'r') as f:
        for line in f:
            if line.startswith("MODEL"):
                if model_found:
                    break
                model_found = True
                model_count += 1
            if line.startswith("ENDMDL") and model_count == 1:
                break
            if line.startswith("ATOM"):
                atom_type = line[76:78].strip()
                residue_name = line[17:20].strip()
                chain_id = line[21:22].strip()
                residue_num = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atom = (atom_type, x, y, z)
                if residue_name in ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", 
                                    "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", 
                                    "SER", "THR", "VAL", "TRP", "TYR"]:
                    protein_residues[(chain_id, residue_num)]['residue_name'] = residue_name
                    protein_residues[(chain_id, residue_num)]['chain_id'] = chain_id
                    protein_residues[(chain_id, residue_num)]['atoms'].append(atom)
                elif residue_name in ["A", "U", "G", "C", "DA", "DT", "DG", "DC"]:
                    nucleotide_residues[(chain_id, residue_num)]['residue_name'] = residue_name
                    nucleotide_residues[(chain_id, residue_num)]['chain_id'] = chain_id
                    nucleotide_residues[(chain_id, residue_num)]['atoms'].append(atom)
    return protein_residues, nucleotide_residues

def calculate_distance(coord1, coord2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

def is_hydrogen_bond(donor_coords, acceptor_coords):
    return calculate_distance(donor_coords, acceptor_coords) <= 3.5

def find_hydrogen_bonds(protein_residues, nucleotide_residues):
    h_bonds = []
    for (p_chain, p_res), p_data in protein_residues.items():
        for p_atom in p_data['atoms']:
            if p_atom[0] in ["N", "O"]:
                for (n_chain, n_res), n_data in nucleotide_residues.items():
                    for n_atom in n_data['atoms']:
                        if is_hydrogen_bond(p_atom[1:], n_atom[1:]):
                            dist = calculate_distance(p_atom[1:], n_atom[1:])
                            h_bonds.append((p_chain, p_res, p_data['residue_name'],
                                            n_chain, n_res, n_data['residue_name'], dist))
    return h_bonds

def count_repeated_hydrogen_bonds(h_bonds):
    return Counter((b[0], b[1], b[2], b[3], b[4], b[5]) for b in h_bonds)

def calculate_center_of_mass(atoms):
    x, y, z = zip(*[(a[1], a[2], a[3]) for a in atoms])
    return (sum(x) / len(atoms), sum(y) / len(atoms), sum(z) / len(atoms))

def find_com_distances(protein_residues, nucleotide_residues):
    results = []
    for (pc, pr), p_data in protein_residues.items():
        p_com = calculate_center_of_mass(p_data['atoms'])
        for (nc, nr), n_data in nucleotide_residues.items():
            n_com = calculate_center_of_mass(n_data['atoms'])
            dist = calculate_distance(p_com, n_com)
            results.append((pc, pr, p_data['residue_name'], nc, nr, n_data['residue_name'], dist))
    return results

def write_merged_output(bond_counts, com_distances, output_file):
    with open(output_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Prot_Chain", "AminoAcid_Number", "AminoAcidID", 
                         "Ligand_Chain", "Ligand_Number", "LigandID", 
                         "Distance", "Hbond(num)"])
        for dist in com_distances:
            key = dist[:6]
            if key in bond_counts:
                writer.writerow(list(dist[:6]) + [f"{dist[6]:.2f}", bond_counts[key]])

def summarize_common_residues(input_file, output_file):
    data = defaultdict(lambda: {"Distance": 0, "Hbond(num)": 0, "Count": 0, "AminoAcidID": "", "LigandID": ""})
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["AminoAcid_Number"], row["LigandID"])
            data[key]["Distance"] += float(row["Distance"])
            data[key]["Hbond(num)"] += int(row["Hbond(num)"])
            data[key]["Count"] += 1
            data[key]["AminoAcidID"] = row["AminoAcidID"]
            data[key]["LigandID"] = row["LigandID"]

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Distance", "AminoAcidID", "Hbond(num)", "LigandID"])
        for (aid, lid), d in data.items():
            avg_dist = d["Distance"] / d["Count"]
            writer.writerow([f"{avg_dist:.2f}", d["AminoAcidID"], d["Hbond(num)"], d["LigandID"]])

def predict_and_average(pdb_id, input_csv, output_csv="pairwise_Results.csv", final_txt="pdbID_score.txt"):
    column_transformer = joblib.load(os.path.join(SCRIPT_DIR, 'RF_column_transformer.pkl'))
    scaler_y = joblib.load(os.path.join(SCRIPT_DIR, 'RF_target_scaler.pkl'))
    model = joblib.load(os.path.join(SCRIPT_DIR, 'Rf_model.pkl'))

    df = pd.read_csv(input_csv)
    for col in ['LigandID', 'AminoAcidID']:
        df[col] = df[col].astype(str)
    X = df[['Distance', 'AminoAcidID', 'Hbond(num)', 'LigandID']]
    X_preprocessed = column_transformer.transform(X)
    y_scaled = model.predict(X_preprocessed)
    y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))
    df['Predicted_Decomp_ENG'] = y_pred
    df.to_csv(output_csv, index=False)

    avg_score = df['Predicted_Decomp_ENG'].mean()
    with open(final_txt, "w") as f:
        f.write(f"{pdb_id} {avg_score:.4f}\n")
    return avg_score

def main():
    with open("list") as f:
        pdb_ids = [line.strip() for line in f if line.strip()]

    scores = []

    for pdb_id in pdb_ids:
        print(f"Processing {pdb_id}")
        os.makedirs(pdb_id, exist_ok=True)
        os.chdir(pdb_id)
        download_pdb(pdb_id, ".")

        protein_residues, nucleotide_residues = parse_pdb(f"{pdb_id}.pdb")
        h_bonds = find_hydrogen_bonds(protein_residues, nucleotide_residues)
        bond_counts = count_repeated_hydrogen_bonds(h_bonds)
        com_distances = find_com_distances(protein_residues, nucleotide_residues)

        write_merged_output(bond_counts, com_distances, "merged_hbonds_com.csv")
        summary_csv = f"{pdb_id}_common_residues_summary.csv"
        summarize_common_residues("merged_hbonds_com.csv", summary_csv)
        avg_score = predict_and_average(pdb_id, summary_csv)
        scores.append((pdb_id, avg_score))

        os.chdir("..")

    with open("all_scores.txt", "w") as f:
        for pdb_id, score in scores:
            f.write(f"{pdb_id} {score:.4f}\n")
    print("All processing complete. Scores written to 'all_scores.txt'.")

if __name__ == "__main__":
    main()

