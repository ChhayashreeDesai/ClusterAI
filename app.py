from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, Response
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
import json
import traceback
from clustering_analyzer import ClusteringAnalyzer

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main landing page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """File upload and initial processing"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Initialize analyzer and load data
                analyzer = ClusteringAnalyzer()
                
                # Load data based on file type
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
                
                load_result = analyzer.load_data(df)
                
                if load_result['success']:
                    # Store analyzer in session (simplified - in production use proper session management)
                    # For now, we'll process immediately
                    
                    # Get complete analysis
                    analysis_result = analyzer.get_complete_analysis()
                    
                    if analysis_result['success']:
                        # Store results in session or database
                        # For simplicity, we'll pass directly to template
                        return render_template('results.html', 
                                            analysis=analysis_result,
                                            filename=filename)
                    else:
                        flash(f'Analysis failed: {analysis_result.get("error", "Unknown error")}', 'error')
                        return redirect(url_for('index'))
                else:
                    flash(f'Data loading failed: {load_result.get("error", "Unknown error")}', 'error')
                    return redirect(url_for('index'))
                    
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(url_for('index'))
        else:
            flash('Invalid file type. Please upload CSV or Excel files only.', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/demo')
def demo():
    """Demo with sample data"""
    try:
        # Create sample mall customer data
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'CustomerID': range(1, 201),
            'Genre': np.random.choice(['Male', 'Female'], 200),
            'Age': np.random.randint(18, 70, 200),
            'Annual Income (k$)': np.random.randint(15, 140, 200),
            'Spending Score (1-100)': np.random.randint(1, 100, 200)
        })
        
        # Make it more realistic with some patterns
        for i in range(200):
            if sample_data.loc[i, 'Annual Income (k$)'] > 70:
                # High income customers might have varied spending
                sample_data.loc[i, 'Spending Score (1-100)'] = np.random.choice([
                    np.random.randint(60, 100),  # High spenders
                    np.random.randint(1, 40)     # Conservative spenders
                ])
            elif sample_data.loc[i, 'Annual Income (k$)'] < 40:
                # Low income customers
                sample_data.loc[i, 'Spending Score (1-100)'] = np.random.choice([
                    np.random.randint(60, 100),  # Impulsive spenders
                    np.random.randint(1, 40)     # Budget conscious
                ])
        
        # Initialize analyzer and process
        analyzer = ClusteringAnalyzer()
        load_result = analyzer.load_data(sample_data)
        
        if load_result['success']:
            analysis_result = analyzer.get_complete_analysis()
            
            if analysis_result['success']:
                return render_template('results.html', 
                                    analysis=analysis_result,
                                    filename='Sample Mall Customer Data')
            else:
                flash(f'Demo analysis failed: {analysis_result.get("error", "Unknown error")}', 'error')
                return redirect(url_for('index'))
        else:
            flash(f'Demo data loading failed: {load_result.get("error", "Unknown error")}', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Demo error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'})
        
        # Process file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Analyze
        analyzer = ClusteringAnalyzer()
        load_result = analyzer.load_data(df)
        
        if not load_result['success']:
            return jsonify(load_result)
        
        analysis_result = analyzer.get_complete_analysis()
        return jsonify(analysis_result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export/csv')
def export_csv():
    """Export cluster assignments as CSV"""
    try:
        # In a real application, you'd retrieve the analyzer from session/database
        # For demo purposes, we'll use the sample data
        analyzer = ClusteringAnalyzer()
        
        # Create sample data for export
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'CustomerID': range(1, 201),
            'Genre': np.random.choice(['Male', 'Female'], 200),
            'Age': np.random.randint(18, 70, 200),
            'Annual Income (k$)': np.random.randint(15, 140, 200),
            'Spending Score (1-100)': np.random.randint(1, 100, 200)
        })
        
        analyzer.load_data(sample_data)
        analyzer.get_complete_analysis()
        
        export_result = analyzer.export_cluster_assignments()
        
        if export_result['success']:
            return Response(
                export_result['csv_string'],
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment; filename=cluster_assignments.csv'}
            )
        else:
            flash(f'Export failed: {export_result.get("error", "Unknown error")}', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Export error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/export/<export_type>')
def api_export(export_type):
    """API endpoint for exports"""
    try:
        if export_type not in ['csv', 'json']:
            return jsonify({'success': False, 'error': 'Invalid export type'})
        
        # For demo - in production, retrieve from session/database
        analyzer = ClusteringAnalyzer()
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'CustomerID': range(1, 201),
            'Genre': np.random.choice(['Male', 'Female'], 200),
            'Age': np.random.randint(18, 70, 200),
            'Annual Income (k$)': np.random.randint(15, 140, 200),
            'Spending Score (1-100)': np.random.randint(1, 100, 200)
        })
        
        analyzer.load_data(sample_data)
        analysis_result = analyzer.get_complete_analysis()
        
        if export_type == 'csv':
            export_result = analyzer.export_cluster_assignments()
            if export_result['success']:
                return Response(
                    export_result['csv_string'],
                    mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=cluster_data.csv'}
                )
        elif export_type == 'json':
            return jsonify(analysis_result)
        
        return jsonify({'success': False, 'error': 'Export failed'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/about')
def about():
    """About page with methodology"""
    return render_template('about.html')

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Please upload a smaller file (max 16MB).', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(error):
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)