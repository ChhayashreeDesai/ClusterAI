from flask import Flask, render_template, request, flash, redirect, url_for, jsonify, session
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator
import os
import io
import base64
import json
from datetime import datetime
import plotly.graph_objs as go
import plotly.express as px
import plotly.utils
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class ClusterAnalyzer:
    def __init__(self, data):
        self.data = data
        self.results = {}
        
    def find_optimal_k(self, max_k=15):
        """Find optimal number of clusters using elbow and silhouette methods"""
        # Prepare data - assuming last two columns are numerical features for clustering
        X = self.data.iloc[:, -2:].values  # Income and Spending Score equivalent
        
        # Elbow Method
        sse = []
        silhouette_scores = []
        k_range = range(1, min(max_k + 1, len(X)))
        
        for k in k_range:
            if k == 1:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                sse.append(kmeans.inertia_)
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                sse.append(kmeans.inertia_)
                
                # Calculate silhouette score
                if len(set(labels)) > 1:  # More than one cluster
                    silh_score = silhouette_score(X, labels)
                    silhouette_scores.append(silh_score)
                else:
                    silhouette_scores.append(0)
        
        # Find optimal k using elbow method
        if len(sse) > 2:
            kl = KneeLocator(list(k_range), sse, curve="convex", direction="decreasing")
            optimal_k_elbow = kl.elbow if kl.elbow else 3
        else:
            optimal_k_elbow = 3
            
        # Find optimal k using silhouette method
        if silhouette_scores:
            optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
        else:
            optimal_k_silhouette = 3
            
        # Combined analysis
        if len(silhouette_scores) > 0:
            normalized_sse = [(max(sse[1:]) - s) / (max(sse[1:]) - min(sse[1:])) for s in sse[1:]]
            normalized_silh = [(s - min(silhouette_scores)) / (max(silhouette_scores) - min(silhouette_scores)) 
                             if max(silhouette_scores) > min(silhouette_scores) else 0 for s in silhouette_scores]
            
            combined_scores = [(normalized_sse[i] + normalized_silh[i]) / 2 for i in range(len(normalized_silh))]
            optimal_k_combined = combined_scores.index(max(combined_scores)) + 2 if combined_scores else 3
        else:
            optimal_k_combined = optimal_k_elbow
            
        self.results['optimal_k'] = optimal_k_combined
        self.results['elbow_k'] = optimal_k_elbow
        self.results['silhouette_k'] = optimal_k_silhouette
        self.results['sse_values'] = sse
        self.results['silhouette_scores'] = silhouette_scores
        
        return optimal_k_combined
    
    def perform_clustering(self, optimal_k):
        """Perform clustering using multiple algorithms"""
        X = self.data.iloc[:, -2:].values
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        
        # Agglomerative Clustering
        agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
        agg_labels = agg_clustering.fit_predict(X)
        
        # DBSCAN Clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find best DBSCAN parameters
        best_dbscan_score = -1
        best_dbscan_labels = None
        
        for eps in [0.3, 0.4, 0.5, 0.6]:
            for min_samples in [3, 4, 5]:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan_labels = dbscan.fit_predict(X_scaled)
                
                n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                
                if n_clusters > 1:
                    try:
                        score = silhouette_score(X_scaled, dbscan_labels)
                        if score > best_dbscan_score:
                            best_dbscan_score = score
                            best_dbscan_labels = dbscan_labels
                    except:
                        continue
        
        # Calculate metrics
        kmeans_silhouette = silhouette_score(X, kmeans_labels)
        agg_silhouette = silhouette_score(X, agg_labels)
        
        # Store results
        self.results['kmeans_labels'] = kmeans_labels
        self.results['agg_labels'] = agg_labels
        self.results['dbscan_labels'] = best_dbscan_labels
        self.results['kmeans_silhouette'] = kmeans_silhouette
        self.results['agg_silhouette'] = agg_silhouette
        self.results['dbscan_silhouette'] = best_dbscan_score
        
        # Advanced metrics
        self.results['calinski_harabasz_kmeans'] = calinski_harabasz_score(X, kmeans_labels)
        self.results['calinski_harabasz_agg'] = calinski_harabasz_score(X, agg_labels)
        self.results['davies_bouldin_kmeans'] = davies_bouldin_score(X, kmeans_labels)
        self.results['davies_bouldin_agg'] = davies_bouldin_score(X, agg_labels)
        self.results['adjusted_rand_index'] = adjusted_rand_score(kmeans_labels, agg_labels)
        
        return kmeans_labels
    
    def analyze_segments(self, labels):
        """Analyze customer segments and provide business insights"""
        # Add cluster labels to dataframe
        df_clustered = self.data.copy()
        df_clustered['Cluster'] = labels
        
        # Analyze each cluster
        segments = {}
        
        for cluster in range(len(set(labels))):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
            
            if len(cluster_data) == 0:
                continue
                
            # Basic demographics
            segment_info = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(df_clustered)) * 100,
            }
            
            # Try to get age and gender if available
            if 'Age' in df_clustered.columns or any('age' in col.lower() for col in df_clustered.columns):
                age_col = 'Age' if 'Age' in df_clustered.columns else next(col for col in df_clustered.columns if 'age' in col.lower())
                segment_info['avg_age'] = cluster_data[age_col].mean()
                segment_info['age_std'] = cluster_data[age_col].std()
            
            # Gender distribution if available
            gender_cols = [col for col in df_clustered.columns if any(term in col.lower() for term in ['gender', 'genre', 'sex'])]
            if gender_cols:
                gender_col = gender_cols[0]
                segment_info['gender_dist'] = cluster_data[gender_col].value_counts().to_dict()
            
            # Financial metrics (assuming last two columns are income and spending)
            income_col = df_clustered.columns[-2]
            spending_col = df_clustered.columns[-1]
            
            segment_info['avg_income'] = cluster_data[income_col].mean()
            segment_info['income_std'] = cluster_data[income_col].std()
            segment_info['avg_spending'] = cluster_data[spending_col].mean()
            segment_info['spending_std'] = cluster_data[spending_col].std()
            
            # Categorize segment
            avg_income = segment_info['avg_income']
            avg_spending = segment_info['avg_spending']
            
            if avg_income >= df_clustered[income_col].quantile(0.7) and avg_spending >= df_clustered[spending_col].quantile(0.7):
                segment_info['category'] = "Premium Customers"
                segment_info['strategy'] = "VIP treatment, premium products, exclusive offers"
                segment_info['priority'] = 1
            elif avg_income >= df_clustered[income_col].quantile(0.7) and avg_spending < df_clustered[spending_col].quantile(0.3):
                segment_info['category'] = "Conservative High-Earners"
                segment_info['strategy'] = "Quality-focused marketing, investment products"
                segment_info['priority'] = 2
            elif avg_income < df_clustered[income_col].quantile(0.3) and avg_spending >= df_clustered[spending_col].quantile(0.7):
                segment_info['category'] = "Young Spenders"
                segment_info['strategy'] = "Trendy products, payment plans, loyalty rewards"
                segment_info['priority'] = 3
            elif avg_income < df_clustered[income_col].quantile(0.3) and avg_spending < df_clustered[spending_col].quantile(0.3):
                segment_info['category'] = "Budget Shoppers"
                segment_info['strategy'] = "Discounts, bulk offers, essential items"
                segment_info['priority'] = 4
            else:
                segment_info['category'] = "Balanced Customers"
                segment_info['strategy'] = "Standard promotions, seasonal offers"
                segment_info['priority'] = 3
            
            # ROI potential
            segment_info['roi_potential'] = (avg_spending / 100) * avg_income * segment_info['size']
            
            segments[f'Cluster {cluster + 1}'] = segment_info
        
        self.results['segments'] = segments
        return segments

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Read CSV file
            df = pd.read_csv(file)
            
            # Basic validation
            if len(df) < 10:
                flash('Dataset too small. Please upload a file with at least 10 records.')
                return redirect(url_for('index'))
            
            if len(df.select_dtypes(include=[np.number]).columns) < 2:
                flash('Dataset needs at least 2 numerical columns for clustering.')
                return redirect(url_for('index'))
            
            # Store data in session (for demo purposes - in production use database)
            session['data'] = df.to_json(orient='records')
            session['columns'] = df.columns.tolist()
            session['shape'] = df.shape
            
            # Perform clustering analysis
            analyzer = ClusterAnalyzer(df)
            optimal_k = analyzer.find_optimal_k()
            labels = analyzer.perform_clustering(optimal_k)
            segments = analyzer.analyze_segments(labels)
            
            # Store results
            session['results'] = analyzer.results
            
            return redirect(url_for('results'))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a CSV file.')
    return redirect(url_for('index'))

@app.route('/results')
def results():
    if 'data' not in session:
        flash('No data found. Please upload a file first.')
        return redirect(url_for('index'))
    
    # Retrieve data and results
    data = json.loads(session['data'])
    df = pd.DataFrame(data)
    results = session['results']
    
    # Create visualizations
    plots = create_visualizations(df, results)
    
    return render_template('results.html', 
                         results=results, 
                         plots=plots,
                         data_shape=session['shape'],
                         columns=session['columns'])

def create_visualizations(df, results):
    """Create interactive visualizations using Plotly"""
    plots = {}
    
    # Elbow method plot
    k_range = list(range(1, len(results['sse_values']) + 1))
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=k_range, y=results['sse_values'], 
                                  mode='lines+markers', name='SSE',
                                  line=dict(color='red', width=3)))
    fig_elbow.add_vline(x=results['elbow_k'], line_dash="dash", 
                       line_color="green", annotation_text=f"Optimal K = {results['elbow_k']}")
    fig_elbow.update_layout(title='Elbow Method for Optimal K',
                           xaxis_title='Number of Clusters (K)',
                           yaxis_title='Sum of Squared Errors (SSE)',
                           template='plotly_white')
    plots['elbow'] = json.dumps(fig_elbow, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Silhouette scores plot
    if results['silhouette_scores']:
        k_range_silh = list(range(2, len(results['silhouette_scores']) + 2))
        fig_silh = go.Figure()
        fig_silh.add_trace(go.Bar(x=k_range_silh, y=results['silhouette_scores'],
                                 name='Silhouette Score', marker_color='blue'))
        fig_silh.add_vline(x=results['silhouette_k'], line_dash="dash",
                          line_color="green", annotation_text=f"Optimal K = {results['silhouette_k']}")
        fig_silh.update_layout(title='Silhouette Analysis for Optimal K',
                              xaxis_title='Number of Clusters (K)',
                              yaxis_title='Silhouette Score',
                              template='plotly_white')
        plots['silhouette'] = json.dumps(fig_silh, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Cluster visualization
    X = df.iloc[:, -2:].values
    labels = results['kmeans_labels']
    
    fig_clusters = go.Figure()
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i in range(len(set(labels))):
        mask = labels == i
        fig_clusters.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1],
                                         mode='markers',
                                         name=f'Cluster {i+1}',
                                         marker=dict(color=colors[i % len(colors)], size=8)))
    
    fig_clusters.update_layout(title='Customer Segmentation Results (K-Means)',
                              xaxis_title=df.columns[-2],
                              yaxis_title=df.columns[-1],
                              template='plotly_white')
    plots['clusters'] = json.dumps(fig_clusters, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Segment size pie chart
    segments = results['segments']
    segment_names = list(segments.keys())
    segment_sizes = [segments[name]['size'] for name in segment_names]
    
    fig_pie = go.Figure(data=[go.Pie(labels=segment_names, values=segment_sizes,
                                    hole=0.3)])
    fig_pie.update_layout(title='Customer Segment Distribution',
                         template='plotly_white')
    plots['pie'] = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)
    
    return plots

@app.route('/download_report')
def download_report():
    # This could generate a PDF report or detailed CSV
    flash('Report download feature coming soon!')
    return redirect(url_for('results'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)