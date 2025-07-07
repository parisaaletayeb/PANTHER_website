// PANTHER Demo JavaScript
// This creates an interactive demo that simulates the real PANTHER analysis

class PANTHERDemo {
    constructor() {
        this.currentResults = null;
        this.isAnalyzing = false;
        this.initializeEventListeners();
        this.updateCurrentUrl();
    }

    initializeEventListeners() {
        // Form submission
        document.getElementById('pdbForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.startDemoAnalysis();
        });

        // Download button
        document.getElementById('downloadBtn').addEventListener('click', () => {
            this.downloadDemoResults();
        });

        // Reset button
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.resetInterface();
        });

        // Modal close on background click
        document.getElementById('fullAppModal').addEventListener('click', (e) => {
            if (e.target.id === 'fullAppModal') {
                this.closeFullAppInfo();
            }
        });
    }

    updateCurrentUrl() {
        document.getElementById('currentUrl').textContent = window.location.href;
    }

    async startDemoAnalysis() {
        if (this.isAnalyzing) return;

        const singlePDB = document.getElementById('singlePDB').value.trim().toUpperCase();
        const multiplePDB = document.getElementById('multiplePDB').value.trim().toUpperCase();
        
        let pdbIds = [];
        
        if (singlePDB) {
            pdbIds = [singlePDB];
        } else if (multiplePDB) {
            pdbIds = multiplePDB.split('\n')
                .map(id => id.trim())
                .filter(id => id.length > 0);
        } else {
            this.showError('Please enter at least one PDB ID');
            return;
        }

        // Validate PDB IDs
        const invalidIds = pdbIds.filter(id => !/^[A-Z0-9]{4}$/.test(id));
        if (invalidIds.length > 0) {
            this.showError(`Invalid PDB ID format: ${invalidIds.join(', ')}. PDB IDs should be 4 characters long.`);
            return;
        }

        this.isAnalyzing = true;
        this.hideAllSections();
        this.showProgress();
        
        try {
            await this.simulateAnalysis(pdbIds);
        } catch (error) {
            this.showError(`Demo analysis failed: ${error.message}`);
        } finally {
            this.isAnalyzing = false;
        }
    }

    async simulateAnalysis(pdbIds) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const demoLog = document.getElementById('demoLog');
        
        // Demo analysis steps
        const steps = [
            { msg: 'Initializing PANTHER demo pipeline...', delay: 800 },
            { msg: 'Loading demo Random Forest model...', delay: 600 },
            { msg: 'Simulating PDB structure download...', delay: 1000 },
            { msg: 'Parsing molecular structures...', delay: 900 },
            { msg: 'Extracting geometric features...', delay: 700 },
            { msg: 'Computing hydrogen bond networks...', delay: 800 },
            { msg: 'Calculating center-of-mass distances...', delay: 600 },
            { msg: 'Preprocessing feature vectors...', delay: 500 },
            { msg: 'Running ML predictions...', delay: 900 },
            { msg: 'Generating binding scores...', delay: 600 }
        ];

        let currentStep = 0;
        
        for (const pdbId of pdbIds) {
            demoLog.innerHTML += `\n🔄 Processing ${pdbId} (Demo)...`;
            demoLog.scrollTop = demoLog.scrollHeight;
            
            for (let i = 0; i < steps.length; i++) {
                const progress = ((currentStep * steps.length + i + 1) / (pdbIds.length * steps.length)) * 100;
                progressFill.style.width = `${progress}%`;
                progressText.textContent = `${steps[i].msg} (${pdbId})`;
                
                demoLog.innerHTML += `\n  ${steps[i].msg}`;
                demoLog.scrollTop = demoLog.scrollHeight;
                
                await this.delay(steps[i].delay);
            }
            
            // Generate demo score
            const score = this.generateDemoScore(pdbId);
            demoLog.innerHTML += `\n✅ ${pdbId} demo analysis complete. Score: ${score.toFixed(4)}`;
            demoLog.scrollTop = demoLog.scrollHeight;
            
            currentStep++;
        }

        progressText.textContent = 'Demo analysis complete!';
        await this.delay(500);
        
        // Generate and show results
        this.generateDemoResults(pdbIds);
    }

    generateDemoScore(pdbId) {
        // Generate realistic-looking scores based on PDB ID
        const seed = this.hashCode(pdbId) / 2147483647;
        
        // Create different score patterns for demonstration
        if (pdbId.includes('1')) {
            return -4.5 + (seed * 3); // Generally favorable
        } else if (pdbId.includes('2')) {
            return -1.2 + (seed * 4); // Mixed results
        } else if (pdbId.includes('3')) {
            return 2.1 + (seed * 3); // Generally unfavorable
        } else {
            return (seed - 0.5) * 8; // Random
        }
    }

    hashCode(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return Math.abs(hash);
    }

    generateDemoResults(pdbIds) {
        const results = pdbIds.map(pdbId => {
            const score = this.generateDemoScore(pdbId);
            const confidence = this.calculateConfidence(score);
            const interpretation = this.interpretScore(score);
            
            return {
                pdbId: pdbId,
                score: score,
                confidence: confidence,
                interpretation: interpretation
            };
        });

        this.currentResults = results;
        this.displayResults(results);
    }

    calculateConfidence(score) {
        const absScore = Math.abs(score);
        if (absScore > 3) return 'high';
        if (absScore > 1.5) return 'medium';
        return 'low';
    }

    interpretScore(score) {
        if (score < -3) {
            return '🟢 Strong favorable binding';
        } else if (score < -1) {
            return '🟡 Moderate favorable binding';
        } else if (score < 1) {
            return '🟠 Weak/neutral binding';
        } else {
            return '🔴 Unfavorable binding';
        }
    }

    displayResults(results) {
        // Update summary
        this.updateResultsSummary(results);
        
        // Update table
        const resultsBody = document.getElementById('resultsBody');
        resultsBody.innerHTML = '';
        
        results.forEach(result => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td class="pdb-id">${result.pdbId}</td>
                <td class="binding-score">${result.score.toFixed(4)}</td>
                <td><span class="confidence-badge confidence-${result.confidence}">${result.confidence}</span></td>
                <td class="interpretation">${result.interpretation}</td>
            `;
            resultsBody.appendChild(row);
        });

        this.hideProgress();
        this.showResults();
    }

    updateResultsSummary(results) {
        const summaryContainer = document.getElementById('resultsSummary');
        
        const totalStructures = results.length;
        const averageScore = results.reduce((sum, r) => sum + r.score, 0) / totalStructures;
        const favorableCount = results.filter(r => r.score < 0).length;
        const highConfidenceCount = results.filter(r => r.confidence === 'high').length;
        
        summaryContainer.innerHTML = `
            <div class="summary-card">
                <h4>Total Analyzed</h4>
                <div class="value">${totalStructures}</div>
                <div class="label">PDB Structures</div>
            </div>
            <div class="summary-card">
                <h4>Average Score</h4>
                <div class="value">${averageScore.toFixed(3)}</div>
                <div class="label">Binding Affinity</div>
            </div>
            <div class="summary-card">
                <h4>Favorable Binding</h4>
                <div class="value">${favorableCount}/${totalStructures}</div>
                <div class="label">Structures</div>
            </div>
            <div class="summary-card">
                <h4>High Confidence</h4>
                <div class="value">${highConfidenceCount}</div>
                <div class="label">Predictions</div>
            </div>
        `;
    }

    downloadDemoResults() {
        if (!this.currentResults) return;
        
        // Create CSV content
        let csvContent = "PDB_ID,Binding_Score,Confidence,Interpretation\n";
        this.currentResults.forEach(result => {
            const cleanInterpretation = result.interpretation.replace(/[🟢🟡🟠🔴]/g, '').trim();
            csvContent += `${result.pdbId},${result.score.toFixed(4)},${result.confidence},"${cleanInterpretation}"\n`;
        });

        // Create and download file
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `panther_demo_results_${Date.now()}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // Show download notification
        this.showNotification('📥 Demo results downloaded! Note: These are simulated results for demonstration purposes.');
    }

    resetInterface() {
        // Clear form
        document.getElementById('singlePDB').value = '';
        document.getElementById('multiplePDB').value = '';
        
        // Reset state
        this.currentResults = null;
        this.isAnalyzing = false;
        
        // Hide sections
        this.hideAllSections();
        
        // Focus on first input
        document.getElementById('singlePDB').focus();
    }

    showProgress() {
        document.getElementById('progressSection').style.display = 'block';
        document.getElementById('submitBtn').disabled = true;
        
        // Reset progress
        document.getElementById('progressFill').style.width = '0%';
        document.getElementById('progressText').textContent = 'Initializing demo analysis...';
        document.getElementById('demoLog').innerHTML = '🧬 PANTHER Demo Analysis Started';
    }

    hideProgress() {
        document.getElementById('progressSection').style.display = 'none';
        document.getElementById('submitBtn').disabled = false;
    }

    showResults() {
        document.getElementById('resultsSection').style.display = 'block';
        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    }

    hideAllSections() {
        document.getElementById('progressSection').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';
    }

    showError(message) {
        this.showNotification(`❌ ${message}`, 'error');
        this.hideProgress();
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Style notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '15px 20px',
            borderRadius: '10px',
            color: 'white',
            fontWeight: '600',
            zIndex: '1001',
            maxWidth: '400px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            animation: 'slideInRight 0.3s ease'
        });
        
        if (type === 'error') {
            notification.style.background = 'linear-gradient(45deg, #e74c3c, #c0392b)';
        } else {
            notification.style.background = 'linear-gradient(45deg, #27ae60, #229954)';
        }
        
        document.body.appendChild(notification);
        
        // Remove after 4 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 4000);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Modal functions
    showFullAppInfo() {
        document.getElementById('fullAppModal').style.display = 'block';
        document.body.style.overflow = 'hidden';
    }

    closeFullAppInfo() {
        document.getElementById('fullAppModal').style.display = 'none';
        document.body.style.overflow = 'auto';
    }
}

// Global functions for HTML onclick events
function showFullAppInfo() {
    pantherDemo.showFullAppInfo();
}

function closeFullAppInfo() {
    pantherDemo.closeFullAppInfo();
}

// Initialize demo when page loads
let pantherDemo;
document.addEventListener('DOMContentLoaded', function() {
    pantherDemo = new PANTHERDemo();
    
    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOutRight {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
    
    console.log('🧬 PANTHER Demo initialized successfully!');
});
