import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import urllib3
import tempfile
import os
import math
from pathlib import Path
from collections import defaultdict, Counter

# Page config
st.set_page_config(
    page_title="🧬 PANTHER Model",
    page_icon="🧬",
    layout="wide"
)

# Disable warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class PANTHERPipeline:
    def __init__(self):
        self.script_dir = Path.cwd()
        self.verify_models()
    
    def verify_models(self):
        required_files = ["RF_column_transformer.pkl", "RF_target_scaler.pkl", "Rf_model.pkl"]
        missing = [f for f in required_files if not (self.script_dir / f).exists()]
        if missing:
            st.error(f"Missing model files: {missing}")
            st.stop()
    
    def download_pdb(self, pdb_id, dest_folder):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url, verify=False, timeout=30)
        if response.status_code == 200:
            with open(os.path.join(dest_folder, f"{pdb_id}.pdb"), 'wb') as f:
                f.write(response.content)
            return True
        return False
    
    def parse_pdb(self, file):
        protein_residues = defaultdict(lambda: {'residue_name': '', 'atoms': []})
        nucleotide_residues = defaultdict(lambda: {'residue_name': '', 'atoms': []})
        
        with open(file, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom_type = line[76:78].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21:22].strip()
                    residue_num = int(line[22:26].strip())
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    atom = (atom_type, x, y, z)
                    
                    if residue_name in ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", 
                                        "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", 
                                        "SER", "THR", "VAL", "TRP", "TYR"]:
                        protein_residues[(chain_id, residue_num)]['residue_name'] = residue_name
                        protein_residues[(chain_id, residue_num)]['atoms'].append(atom)
                    elif residue_name in ["A", "U", "G", "C", "DA", "DT", "DG", "DC"]:
                        nucleotide_residues[(chain_id, residue_num)]['residue_name'] = residue_name
                        nucleotide_residues[(chain_id, residue_num)]['atoms'].append(atom)
        
        return protein_residues, nucleotide_residues
    
    def calculate_distance(self, coord1, coord2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
    
    def find_hydrogen_bonds(self, protein_residues, nucleotide_residues):
        h_bonds = []
        for (p_chain, p_res), p_data in protein_residues.items():
            for p_atom in p_data['atoms']:
                if p_atom[0] in ["N", "O"]:
                    for (n_chain, n_res), n_data in nucleotide_residues.items():
                        for n_atom in n_data['atoms']:
                            if self.calculate_distance(p_atom[1:], n_atom[1:]) <= 3.5:
                                h_bonds.append((p_chain, p_res, p_data['residue_name'],
                                                n_chain, n_res, n_data['residue_name']))
        return h_bonds
    
    def calculate_center_of_mass(self, atoms):
        x, y, z = zip(*[(a[1], a[2], a[3]) for a in atoms])
        return (sum(x) / len(atoms), sum(y) / len(atoms), sum(z) / len(atoms))
    
    def process_pdb(self, pdb_id, progress_callback=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            if progress_callback:
                progress_callback(f"Downloading {pdb_id}...")
            
            if not self.download_pdb(pdb_id, temp_dir):
                raise Exception(f"Failed to download {pdb_id}")
            
            pdb_file = os.path.join(temp_dir, f"{pdb_id}.pdb")
            
            if progress_callback:
                progress_callback(f"Analyzing {pdb_id}...")
            
            protein_residues, nucleotide_residues = self.parse_pdb(pdb_file)
            
            if not protein_residues or not nucleotide_residues:
                raise Exception(f"No protein-RNA interactions in {pdb_id}")
            
            # Calculate features
            features = []
            h_bonds = self.find_hydrogen_bonds(protein_residues, nucleotide_residues)
            bond_counts = Counter((b[0], b[1], b[2], b[3], b[4], b[5]) for b in h_bonds)
            
            for (pc, pr), p_data in protein_residues.items():
                p_com = self.calculate_center_of_mass(p_data['atoms'])
                for (nc, nr), n_data in nucleotide_residues.items():
                    n_com = self.calculate_center_of_mass(n_data['atoms'])
                    dist = self.calculate_distance(p_com, n_com)
                    
                    key = (pc, pr, p_data['residue_name'], nc, nr, n_data['residue_name'])
                    hbond_count = bond_counts.get(key, 0)
                    
                    features.append({
                        'Distance': dist,
                        'AminoAcidID': p_data['residue_name'],
                        'Hbond(num)': hbond_count,
                        'LigandID': n_data['residue_name']
                    })
            
            return pd.DataFrame(features)
    
    def predict_binding_affinity(self, features_df):
        # Load models
        column_transformer = joblib.load(self.script_dir / 'RF_column_transformer.pkl')
        scaler_y = joblib.load(self.script_dir / 'RF_target_scaler.pkl')
        model = joblib.load(self.script_dir / 'Rf_model.pkl')
        
        # Prepare features
        for col in ['LigandID', 'AminoAcidID']:
            features_df[col] = features_df[col].astype(str)
        
        X = features_df[['Distance', 'AminoAcidID', 'Hbond(num)', 'LigandID']]
        X_preprocessed = column_transformer.transform(X)
        
        # Predict
        y_scaled = model.predict(X_preprocessed)
        y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))
        
        return float(np.mean(y_pred))

def interpret_score(score):
    if score < -3:
        return "🟢 Strong favorable binding"
    elif score < -1:
        return "🟡 Moderate favorable binding"
    elif score < 1:
        return "🟠 Weak/neutral binding"
    else:
        return "🔴 Unfavorable binding"

def main():
    st.title("🧬 PANTHER Model")
    st.markdown("**Protein-Nucleotide Thermodynamic Affinity Predictor**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        PANTHER predicts protein-RNA binding affinities using:
        - Random Forest regression
        - Center-of-mass distances  
        - Hydrogen bond networks
        - Molecular features
        """)
        
        st.header("📊 Test PDB IDs")
        st.code("1A1T\n2F8S\n3P59")
    
    # Input
    st.header("📋 Submit PDB IDs")
    
    col1, col2 = st.columns(2)
    with col1:
        single_pdb = st.text_input("Single PDB ID:", placeholder="e.g., 1A1T", max_chars=4)
    with col2:
        multiple_pdb = st.text_area("Multiple PDB IDs:", placeholder="1A1T\n2F8S\n3P59")
    
    if st.button("🚀 Start Analysis", type="primary"):
        # Get PDB IDs
        pdb_ids = []
        if single_pdb:
            pdb_ids = [single_pdb.upper().strip()]
        elif multiple_pdb:
            pdb_ids = [id.strip().upper() for id in multiple_pdb.split('\n') if id.strip()]
        
        if not pdb_ids:
            st.error("Please enter at least one PDB ID")
            return
        
        # Validate
        invalid_ids = [id for id in pdb_ids if len(id) != 4 or not id.isalnum()]
        if invalid_ids:
            st.error(f"Invalid PDB IDs: {', '.join(invalid_ids)}")
            return
        
        # Run analysis
        pipeline = PANTHERPipeline()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        try:
            for i, pdb_id in enumerate(pdb_ids):
                def update_progress(msg):
                    progress_bar.progress((i + 0.5) / len(pdb_ids))
                    status_text.text(msg)
                
                # Process PDB
                features_df = pipeline.process_pdb(pdb_id, update_progress)
                
                # Predict
                status_text.text(f"Predicting binding affinity for {pdb_id}...")
                score = pipeline.predict_binding_affinity(features_df)
                
                results.append({
                    'PDB_ID': pdb_id,
                    'Score': round(score, 4),
                    'Interpretation': interpret_score(score)
                })
                
                progress_bar.progress((i + 1) / len(pdb_ids))
            
            status_text.text("✅ Analysis complete!")
            
            # Display results
            st.header("📊 Results")
            
            # Summary
            scores = [r['Score'] for r in results]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Analyzed", len(results))
            with col2:
                st.metric("Average Score", f"{np.mean(scores):.3f}")
            with col3:
                favorable = sum(1 for s in scores if s < 0)
                st.metric("Favorable Binding", f"{favorable}/{len(results)}")
            
            # Table
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                f"panther_results_{'-'.join(pdb_ids)}.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
    
    # Methodology
    st.markdown("---")
    st.header("🔬 Methodology")
    st.markdown("""
    **Feature Extraction:** Center-of-mass distances and hydrogen bond analysis
    **ML Model:** Random Forest trained on binding energy data  
    **Interpretation:** Negative = favorable, Positive = unfavorable
    """)

if __name__ == "__main__":
    main()
