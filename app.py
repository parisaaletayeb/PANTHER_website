#!/usr/bin/env python3
"""
PANTHER Model Web Server
========================

A Flask-based web server for the PANTHER (Protein-Nucleotide Thermodynamic 
Affinity Predictor) model. This server provides a RESTful API interface for 
processing PDB structures and predicting binding free energy decomposition scores.

Mathematical Background:
- Feature extraction based on Euclidean distance calculations between 
  center-of-mass coordinates
- Hydrogen bond detection using geometric criteria (d ≤ 3.5 Å)
- Random Forest regression for non-linear relationship modeling
- Standard score normalization for feature scaling

Authors: Integration wrapper for PANTHER pipeline
License: Academic Use
"""

import os
import sys
import json
import time
import uuid
import threading
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import logging

# Configure logging for production deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PANTHERJobManager:
    """
    Job management system for PANTHER analysis pipeline.
    
    Handles concurrent job execution, progress tracking, and result storage
    with thread-safe operations for production deployment.
    """
    
    def __init__(self, work_dir: str = "panther_jobs"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.jobs: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        
    def create_job(self, pdb_ids: List[str]) -> str:
        """
        Initialize a new PANTHER analysis job.
        
        Args:
            pdb_ids: List of 4-character PDB identifiers
            
        Returns:
            Unique job identifier (UUID4)
        """
        job_id = str(uuid.uuid4())
        
        with self.lock:
            self.jobs[job_id] = {
                'id': job_id,
                'pdb_ids': pdb_ids,
                'status': 'queued',
                'progress': 0,
                'current_step': 'Initializing',
                'results': None,
                'error': None,
                'created_at': datetime.now().isoformat(),
                'log': []
            }
            
        logger.info(f"Created job {job_id} for PDB IDs: {', '.join(pdb_ids)}")
        return job_id
    
    def update_job_progress(self, job_id: str, progress: int, step: str, log_entry: str = None):
        """Update job progress with thread-safe locking."""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['progress'] = progress
                self.jobs[job_id]['current_step'] = step
                if log_entry:
                    self.jobs[job_id]['log'].append({
                        'timestamp': datetime.now().isoformat(),
                        'message': log_entry
                    })
                    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Retrieve current job status."""
        with self.lock:
            return self.jobs.get(job_id, None)
    
    def complete_job(self, job_id: str, results: List[Tuple[str, float]]):
        """Mark job as completed with results."""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['status'] = 'completed'
                self.jobs[job_id]['progress'] = 100
                self.jobs[job_id]['current_step'] = 'Analysis complete'
                self.jobs[job_id]['results'] = results
                self.jobs[job_id]['completed_at'] = datetime.now().isoformat()
                
    def fail_job(self, job_id: str, error: str):
        """Mark job as failed with error message."""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['status'] = 'failed'
                self.jobs[job_id]['error'] = error
                self.jobs[job_id]['failed_at'] = datetime.now().isoformat()

class PANTHERPipeline:
    """
    Wrapper for the PANTHER unified pipeline script.
    
    This class provides a clean interface to the existing molecular analysis
    pipeline while maintaining compatibility with the original codebase.
    """
    
    def __init__(self, script_path: str = "unified_pipeline.py"):
        self.script_path = Path(script_path)
        if not self.script_path.exists():
            raise FileNotFoundError(f"Pipeline script not found: {script_path}")
            
        # Verify required model files exist
        required_files = [
            "RF_column_transformer.pkl",
            "RF_target_scaler.pkl", 
            "Rf_model.pkl"
        ]
        
        for file in required_files:
            if not Path(file).exists():
                logger.warning(f"Model file not found: {file}")
    
    def run_analysis(self, job_id: str, pdb_ids: List[str], job_manager: PANTHERJobManager):
        """
        Execute PANTHER analysis pipeline for given PDB structures.
        
        Args:
            job_id: Unique job identifier
            pdb_ids: List of PDB structure identifiers
            job_manager: Job management instance for progress tracking
        """
        
        job_dir = job_manager.work_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the main project directory (where model files and script are)
        main_dir = Path.cwd()
        
        try:
            # Update job status
            job_manager.update_job_progress(
                job_id, 5, "Setting up analysis environment",
                f"Processing {len(pdb_ids)} PDB structure(s)"
            )
            
            # Create PDB list file in the MAIN directory (temporarily)
            list_file = main_dir / "list"
            with open(list_file, 'w') as f:
                for pdb_id in pdb_ids:
                    f.write(f"{pdb_id.upper()}\n")
                    
            job_manager.update_job_progress(
                job_id, 10, "Created PDB input list", 
                f"Input file: {list_file}"
            )
            
            # Verify model files exist in main directory
            model_files = [
                "RF_column_transformer.pkl",
                "RF_target_scaler.pkl", 
                "Rf_model.pkl"
            ]
            
            for model_file in model_files:
                if not (main_dir / model_file).exists():
                    raise FileNotFoundError(f"Model file not found: {main_dir / model_file}")
            
            job_manager.update_job_progress(
                job_id, 15, "Verified model files exist"
            )
            
            # Execute the pipeline script from MAIN directory
            job_manager.update_job_progress(
                job_id, 20, "Starting PANTHER pipeline execution"
            )
            
            # Run the unified pipeline from main directory
            cmd = [sys.executable, str(self.script_path)]
            logger.info(f"Executing command: {' '.join(cmd)} in {main_dir}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=str(main_dir)  # Run from main directory
            )
            
            # Monitor progress through pipeline execution
            step_count = 0
            max_steps = len(pdb_ids) * 10  # Estimate based on pipeline complexity
            
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output:
                    output_lines.append(output.strip())
                    step_count += 1
                    progress = min(20 + (step_count / max_steps) * 70, 90)
                    
                    job_manager.update_job_progress(
                        job_id, int(progress), "Running analysis pipeline",
                        output.strip()
                    )
                    
                    logger.info(f"Job {job_id}: {output.strip()}")
            
            # Check for successful completion
            return_code = process.poll()
            if return_code != 0:
                error_output = '\n'.join(output_lines[-10:])  # Last 10 lines
                raise subprocess.CalledProcessError(
                    return_code, 
                    "unified_pipeline.py",
                    output=error_output
                )
            
            # Move results from main directory to job directory
            main_results_file = main_dir / "all_scores.txt"
            if main_results_file.exists():
                job_results_file = job_dir / "all_scores.txt"
                shutil.move(str(main_results_file), str(job_results_file))
                logger.info(f"Moved results from {main_results_file} to {job_results_file}")
            else:
                # List all files in main directory for debugging
                files_in_main = list(main_dir.glob("*"))
                logger.error(f"Files in main directory: {files_in_main}")
                raise FileNotFoundError(f"Pipeline completed but results file not found. Available files: {files_in_main}")
            
            # Parse results
            results = self._parse_results(job_dir / "all_scores.txt")
            
            job_manager.update_job_progress(
                job_id, 95, "Parsing results",
                f"Successfully analyzed {len(results)} structures"
            )
            
            # Complete the job
            job_manager.complete_job(job_id, results)
            
            logger.info(f"Job {job_id} completed successfully with {len(results)} results")
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            logger.error(f"Job {job_id} failed: {error_msg}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            if hasattr(e, 'output') and e.output:
                logger.error(f"Process output: {e.output}")
            job_manager.fail_job(job_id, error_msg)
            
        finally:
            # Clean up temporary list file
            list_file = main_dir / "list"
            if list_file.exists():
                try:
                    list_file.unlink()
                    logger.info("Cleaned up temporary list file")
                except Exception as e:
                    logger.warning(f"Could not clean up list file: {e}")
                    
            # Also clean up any PDB directories created in main directory
            try:
                for pdb_id in pdb_ids:
                    pdb_dir = main_dir / pdb_id.upper()
                    if pdb_dir.exists() and pdb_dir.is_dir():
                        shutil.rmtree(pdb_dir)
                        logger.info(f"Cleaned up PDB directory: {pdb_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up PDB directories: {e}")
            
    def _parse_results(self, results_file: Path) -> List[Tuple[str, float]]:
        """
        Parse the all_scores.txt file generated by the pipeline.
        
        Args:
            results_file: Path to the results file
            
        Returns:
            List of tuples containing (pdb_id, score)
        """
        results = []
        
        with open(results_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        pdb_id = parts[0]
                        try:
                            score = float(parts[1])
                            results.append((pdb_id, score))
                        except ValueError:
                            logger.warning(f"Could not parse score for {pdb_id}: {parts[1]}")
                            
        return results

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize job manager and pipeline
job_manager = PANTHERJobManager()
pipeline = PANTHERPipeline()

@app.route('/')
def index():
    """Serve the main web interface."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return f"""
        <h1>PANTHER Model Web Server</h1>
        <p>Error loading web interface. Please check that templates/index.html exists.</p>
        <p>Error: {str(e)}</p>
        <p>Current working directory: {os.getcwd()}</p>
        <p>Looking for templates in: {os.path.join(os.getcwd(), 'templates')}</p>
        """

@app.route('/api/submit', methods=['POST'])
def submit_analysis():
    """
    Submit PDB structures for analysis.
    
    Expected JSON payload:
    {
        "pdb_ids": ["1A1T", "2F8S", "3P59"]
    }
    
    Returns:
        JSON response with job_id for tracking
    """
    try:
        data = request.get_json()
        
        if not data or 'pdb_ids' not in data:
            return jsonify({'error': 'Missing pdb_ids in request'}), 400
            
        pdb_ids = data['pdb_ids']
        
        # Validate PDB IDs
        if not isinstance(pdb_ids, list) or len(pdb_ids) == 0:
            return jsonify({'error': 'pdb_ids must be a non-empty list'}), 400
            
        # Validate PDB ID format (4 characters, alphanumeric)
        invalid_ids = []
        valid_ids = []
        
        for pdb_id in pdb_ids:
            pdb_id = str(pdb_id).upper().strip()
            if len(pdb_id) == 4 and pdb_id.isalnum():
                valid_ids.append(pdb_id)
            else:
                invalid_ids.append(pdb_id)
                
        if invalid_ids:
            return jsonify({
                'error': f'Invalid PDB ID format: {", ".join(invalid_ids)}. PDB IDs must be 4 alphanumeric characters.'
            }), 400
            
        # Create and start job
        job_id = job_manager.create_job(valid_ids)
        
        # Start analysis in background thread
        thread = threading.Thread(
            target=pipeline.run_analysis,
            args=(job_id, valid_ids, job_manager)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'pdb_ids': valid_ids,
            'message': f'Analysis started for {len(valid_ids)} PDB structure(s)'
        })
        
    except Exception as e:
        logger.error(f"Error in submit_analysis: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get current status of analysis job."""
    try:
        job = job_manager.get_job_status(job_id)
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
            
        return jsonify(job)
        
    except Exception as e:
        logger.error(f"Error in get_job_status: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/results/<job_id>', methods=['GET'])
def get_results(job_id):
    """Get analysis results for completed job."""
    try:
        job = job_manager.get_job_status(job_id)
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
            
        if job['status'] != 'completed':
            return jsonify({'error': 'Job not completed yet'}), 400
            
        # Format results for frontend
        formatted_results = []
        for pdb_id, score in job['results']:
            interpretation = interpret_score(score)
            formatted_results.append({
                'pdb_id': pdb_id,
                'score': round(score, 4),
                'interpretation': interpretation
            })
            
        return jsonify({
            'job_id': job_id,
            'results': formatted_results,
            'summary': {
                'total_structures': len(formatted_results),
                'average_score': round(sum(r['score'] for r in formatted_results) / len(formatted_results), 4),
                'favorable_count': len([r for r in formatted_results if r['score'] < 0]),
                'unfavorable_count': len([r for r in formatted_results if r['score'] > 0])
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_results: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/download/<job_id>', methods=['GET'])
def download_results(job_id):
    """Download results file for completed job."""
    try:
        job = job_manager.get_job_status(job_id)
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
            
        if job['status'] != 'completed':
            return jsonify({'error': 'Job not completed yet'}), 400
            
        # Create downloadable results file
        results_content = "PDB_ID\tBinding_Score\tInterpretation\n"
        for pdb_id, score in job['results']:
            interpretation = interpret_score(score)
            results_content += f"{pdb_id}\t{score:.4f}\t{interpretation}\n"
            
        # Write to temporary file
        temp_file = job_manager.work_dir / f"{job_id}_results.txt"
        with open(temp_file, 'w') as f:
            f.write(results_content)
            
        return send_file(
            temp_file,
            as_attachment=True,
            download_name=f"panther_results_{job_id[:8]}.txt",
            mimetype='text/plain'
        )
        
    except Exception as e:
        logger.error(f"Error in download_results: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def interpret_score(score: float) -> str:
    """
    Interpret binding affinity score based on thermodynamic principles.
    
    Args:
        score: Predicted binding free energy decomposition score
        
    Returns:
        Human-readable interpretation
    """
    if score < -5:
        return "Strong favorable binding"
    elif score < -2:
        return "Moderate favorable binding"
    elif score < 2:
        return "Weak/neutral binding"
    else:
        return "Unfavorable binding"

if __name__ == '__main__':
    print("🧬 PANTHER Model Web Server")
    print("=" * 50)
    print("Starting Flask server for protein-RNA binding analysis...")
    print(f"Work directory: {job_manager.work_dir.absolute()}")
    print(f"Pipeline script: {pipeline.script_path.absolute()}")
    print()
    
    # In production, use a proper WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
