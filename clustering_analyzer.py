"""
Customer Clustering Analysis Backend
Integrates all clustering algorithms and analysis from the notebook
"""

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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

class ClusteringAnalyzer:
    """
    Comprehensive clustering analysis class that integrates all methods from the notebook
    """
    
    def __init__(self):
        self.df = None
        self.x = None
        self.optimal_k = None
        self.kmeans_labels = None
        self.agg_labels = None
        self.best_dbscan_labels = None
        self.results = {}
        
    def load_data(self, file_path_or_dataframe):
        """Load and validate customer data"""
        try:
            if isinstance(file_path_or_dataframe, str):
                self.df = pd.read_csv(file_path_or_dataframe)
            else:
                self.df = file_path_or_dataframe.copy()
            
            # Auto-detect features for clustering
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove ID columns if present
            id_columns = [col for col in numeric_columns if 'id' in col.lower() or 'customer' in col.lower()]
            clustering_columns = [col for col in numeric_columns if col not in id_columns]
            
            # Use the last 2 numeric columns as default (Income and Spending Score pattern)
            if len(clustering_columns) >= 2:
                self.x = self.df[clustering_columns[-2:]].copy()
            else:
                raise ValueError("Need at least 2 numeric columns for clustering")
                
            return {
                'success': True,
                'data_shape': self.df.shape,
                'columns': self.df.columns.tolist(),
                'clustering_features': self.x.columns.tolist(),
                'sample_data': self.df.head().to_dict('records')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def find_optimal_k(self):
        """Find optimal K using elbow and silhouette methods"""
        try:
            # Elbow Method
            sse = []
            silh = []
            k_range = range(1, 16)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.x)
                sse.append(kmeans.inertia_)
                
                if k > 1:  # Silhouette score needs at least 2 clusters
                    score = silhouette_score(self.x, labels)
                    silh.append(score)
            
            # Automatic elbow detection
            kl = KneeLocator(k_range, sse, curve="convex", direction="decreasing")
            optimal_k_elbow = kl.elbow if kl.elbow else 5
            
            # Find optimal k using silhouette method
            optimal_k_silhouette = silh.index(max(silh)) + 2
            
            # Combined analysis
            normalized_sse = [(max(sse) - s) / (max(sse) - min(sse)) for s in sse[1:]]
            normalized_silh = [(s - min(silh)) / (max(silh) - min(silh)) for s in silh]
            
            combined_scores = [(normalized_sse[i] + normalized_silh[i]) / 2 
                             for i in range(len(normalized_silh))]
            optimal_k_combined = combined_scores.index(max(combined_scores)) + 2
            
            self.optimal_k = optimal_k_combined
            
            return {
                'optimal_k_elbow': optimal_k_elbow,
                'optimal_k_silhouette': optimal_k_silhouette,
                'optimal_k_combined': optimal_k_combined,
                'sse_values': sse,
                'silhouette_values': silh,
                'combined_scores': combined_scores,
                'recommended_k': optimal_k_combined
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def perform_clustering(self):
        """Perform clustering with all three algorithms"""
        try:
            if self.optimal_k is None:
                self.find_optimal_k()
            
            # K-Means Clustering
            kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
            self.kmeans_labels = kmeans.fit_predict(self.x)
            kmeans_silhouette = silhouette_score(self.x, self.kmeans_labels)
            
            # Agglomerative Clustering
            agg_clustering = AgglomerativeClustering(n_clusters=self.optimal_k)
            self.agg_labels = agg_clustering.fit_predict(self.x)
            agg_silhouette = silhouette_score(self.x, self.agg_labels)
            
            # DBSCAN Clustering
            dbscan_result = self._optimize_dbscan()
            
            # Advanced metrics comparison
            metrics = self._calculate_advanced_metrics()
            
            # Add cluster labels to dataframe
            self.df['KMeans_Cluster'] = self.kmeans_labels
            self.df['Agg_Cluster'] = self.agg_labels
            
            return {
                'kmeans_silhouette': kmeans_silhouette,
                'agg_silhouette': agg_silhouette,
                'dbscan_result': dbscan_result,
                'metrics_comparison': metrics,
                'optimal_k': self.optimal_k
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _optimize_dbscan(self):
        """Optimize DBSCAN parameters"""
        try:
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(self.x)
            
            eps_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            min_samples_values = [3, 4, 5]
            
            best_score = -1
            best_params = {}
            best_labels = None
            
            for eps in eps_values:
                for min_samples in min_samples_values:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(x_scaled)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    if n_clusters > 1 and n_noise < len(self.x) * 0.9:
                        try:
                            score = silhouette_score(x_scaled, labels)
                            if score > best_score:
                                best_score = score
                                best_params = {'eps': eps, 'min_samples': min_samples}
                                best_labels = labels
                        except:
                            continue
            
            if best_score > -1:
                self.best_dbscan_labels = best_labels
                return {
                    'success': True,
                    'best_params': best_params,
                    'best_score': best_score,
                    'n_clusters': len(set(best_labels)) - (1 if -1 in best_labels else 0)
                }
            else:
                return {'success': False, 'message': 'No suitable DBSCAN parameters found'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_advanced_metrics(self):
        """Calculate advanced clustering evaluation metrics"""
        try:
            metrics = {
                'Silhouette Score': {
                    'K-Means': silhouette_score(self.x, self.kmeans_labels),
                    'Agglomerative': silhouette_score(self.x, self.agg_labels)
                },
                'Calinski-Harabasz Index': {
                    'K-Means': calinski_harabasz_score(self.x, self.kmeans_labels),
                    'Agglomerative': calinski_harabasz_score(self.x, self.agg_labels)
                },
                'Davies-Bouldin Index': {
                    'K-Means': davies_bouldin_score(self.x, self.kmeans_labels),
                    'Agglomerative': davies_bouldin_score(self.x, self.agg_labels)
                }
            }
            
            # Agreement between algorithms
            agreement = adjusted_rand_score(self.kmeans_labels, self.agg_labels)
            
            return {
                'metrics': metrics,
                'algorithm_agreement': agreement
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def analyze_segments(self):
        """Detailed demographic and business analysis"""
        try:
            segments_analysis = []
            
            for cluster in range(self.optimal_k):
                cluster_data = self.df[self.df['KMeans_Cluster'] == cluster]
                
                # Basic stats
                size = len(cluster_data)
                percentage = (size / len(self.df)) * 100
                
                # Financial profile
                numeric_cols = self.x.columns
                avg_values = {}
                for col in numeric_cols:
                    avg_values[col] = cluster_data[col].mean()
                
                # Age analysis if available
                age_analysis = {}
                if 'Age' in self.df.columns:
                    age_analysis = {
                        'avg_age': cluster_data['Age'].mean(),
                        'age_std': cluster_data['Age'].std()
                    }
                
                # Gender analysis if available
                gender_analysis = {}
                gender_col = None
                for col in ['Gender', 'Genre', 'Sex']:
                    if col in self.df.columns:
                        gender_col = col
                        break
                
                if gender_col:
                    gender_dist = cluster_data[gender_col].value_counts().to_dict()
                    gender_analysis = {
                        'distribution': gender_dist,
                        'column_name': gender_col
                    }
                
                # Business categorization
                category_info = self._categorize_cluster(avg_values)
                
                segments_analysis.append({
                    'cluster_id': cluster + 1,
                    'size': size,
                    'percentage': percentage,
                    'avg_values': avg_values,
                    'age_analysis': age_analysis,
                    'gender_analysis': gender_analysis,
                    'category': category_info['category'],
                    'strategy': category_info['strategy'],
                    'roi_potential': category_info['roi_potential']
                })
            
            # Sort by ROI potential
            segments_analysis.sort(key=lambda x: x['roi_potential'], reverse=True)
            
            return {
                'segments': segments_analysis,
                'total_customers': len(self.df),
                'feature_columns': list(self.x.columns)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _categorize_cluster(self, avg_values):
        """Categorize cluster based on average values"""
        try:
            # Get feature names (assuming last two are income and spending-like)
            features = list(avg_values.keys())
            
            # Assume first feature is income-like, second is spending-like
            income_like = avg_values[features[0]] if len(features) > 0 else 50
            spending_like = avg_values[features[1]] if len(features) > 1 else 50
            
            # Categorization logic
            if income_like >= 70 and spending_like >= 70:
                category = "💎 PREMIUM CUSTOMERS"
                strategy = "VIP treatment, premium products, exclusive offers"
                roi_multiplier = 3.0
            elif income_like >= 70 and spending_like < 40:
                category = "💼 CONSERVATIVE HIGH-EARNERS"
                strategy = "Quality-focused marketing, investment products"
                roi_multiplier = 2.0
            elif income_like < 40 and spending_like >= 70:
                category = "🎯 YOUNG SPENDERS"
                strategy = "Trendy products, payment plans, loyalty rewards"
                roi_multiplier = 1.5
            elif income_like < 40 and spending_like < 40:
                category = "🏷️ BUDGET SHOPPERS"
                strategy = "Discounts, bulk offers, essential items"
                roi_multiplier = 1.0
            else:
                category = "⚖️ BALANCED CUSTOMERS"
                strategy = "Standard promotions, seasonal offers"
                roi_multiplier = 1.2
            
            # Calculate ROI potential
            roi_potential = (spending_like / 100) * income_like * roi_multiplier
            
            return {
                'category': category,
                'strategy': strategy,
                'roi_potential': roi_potential
            }
            
        except Exception as e:
            return {
                'category': "⚖️ STANDARD CUSTOMERS",
                'strategy': "Standard marketing approach",
                'roi_potential': 50
            }
    
    def generate_business_recommendations(self, segments_analysis):
        """Generate comprehensive business recommendations"""
        try:
            recommendations = {
                'key_findings': {
                    'total_segments': self.optimal_k,
                    'best_algorithm': 'K-Means',
                    'total_customers': len(self.df)
                },
                'priority_segments': [],
                'action_plan': {},
                'implementation_timeline': [
                    {"period": "Month 1-2", "action": "Deep dive analysis & strategy refinement", "owner": "Data team + Marketing"},
                    {"period": "Month 2-3", "action": "Launch targeted campaigns for top 2 clusters", "owner": "Marketing team"},
                    {"period": "Month 3-4", "action": "A/B test different approaches", "owner": "Product team"},
                    {"period": "Month 4-5", "action": "Scale successful tactics", "owner": "All teams"},
                    {"period": "Month 5-6", "action": "Evaluate results & optimize", "owner": "Management"}
                ],
                'success_metrics': [
                    "Revenue per cluster (target: +15-25% increase)",
                    "Customer lifetime value by segment",
                    "Campaign engagement rates by cluster",
                    "Average order value progression",
                    "Customer retention rates"
                ]
            }
            
            # Priority segments (top 3 by ROI)
            top_segments = segments_analysis['segments'][:3]
            for segment in top_segments:
                recommendations['priority_segments'].append({
                    'cluster_id': segment['cluster_id'],
                    'category': segment['category'],
                    'size': segment['size'],
                    'roi_potential': segment['roi_potential'],
                    'strategy': segment['strategy']
                })
            
            return recommendations
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_visualizations(self):
        """Create comprehensive interactive visualizations using Plotly"""
        try:
            visualizations = {}
            
            if self.kmeans_labels is None:
                return {'error': 'No clustering results available'}
            
            # 1. Enhanced Cluster Scatter Plot with hover information
            scatter_data = self.x.copy()
            scatter_data['Cluster'] = [f'Cluster {i+1}' for i in self.kmeans_labels]
            scatter_data['Customer_ID'] = range(1, len(scatter_data) + 1)
            
            # Add age if available
            if 'Age' in self.df.columns:
                scatter_data['Age'] = self.df['Age'].values
            
            hover_data = ['Customer_ID']
            if 'Age' in scatter_data.columns:
                hover_data.append('Age')
            
            fig_scatter = px.scatter(
                scatter_data,
                x=self.x.columns[0],
                y=self.x.columns[1],
                color='Cluster',
                title=f'🎯 Customer Segments Analysis - {self.optimal_k} Distinct Groups Identified',
                hover_data=hover_data,
                size_max=15,
                template='plotly_white',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            fig_scatter.update_layout(
                height=600,
                showlegend=True,
                title_x=0.5,
                title_font_size=18,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            
            fig_scatter.update_traces(marker=dict(size=10, opacity=0.8))
            
            visualizations['cluster_scatter'] = fig_scatter.to_html(include_plotlyjs=True, div_id="cluster-scatter")
            
            # 2. Enhanced Cluster Size Distribution with percentages
            cluster_sizes = pd.Series(self.kmeans_labels).value_counts().sort_index()
            cluster_names = [f'Cluster {i+1}' for i in cluster_sizes.index]
            cluster_percentages = (cluster_sizes / len(self.kmeans_labels) * 100).round(1)
            
            fig_pie = px.pie(
                values=cluster_sizes.values,
                names=cluster_names,
                title='📊 Customer Distribution Across Segments',
                template='plotly_white',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            fig_pie.update_layout(height=500, title_x=0.5, title_font_size=16)
            
            visualizations['cluster_sizes'] = fig_pie.to_html(include_plotlyjs=True, div_id="cluster-sizes")
            
            # 3. Comprehensive Optimization Analysis
            elbow_data = self.find_optimal_k()
            k_range = range(1, 16)
            silh_k_range = range(2, 16)
            
            fig_optimization = make_subplots(
                rows=1, cols=2,
                subplot_titles=('🔍 Elbow Method Analysis', '📈 Silhouette Score Analysis'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Elbow plot
            fig_optimization.add_trace(
                go.Scatter(
                    x=list(k_range),
                    y=elbow_data['sse_values'],
                    mode='lines+markers',
                    name='SSE',
                    line=dict(color='#FF6B6B', width=3),
                    marker=dict(size=8, color='#FF6B6B'),
                    hovertemplate='K=%{x}<br>SSE=%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Silhouette plot
            fig_optimization.add_trace(
                go.Scatter(
                    x=list(silh_k_range),
                    y=elbow_data['silhouette_values'],
                    mode='lines+markers',
                    name='Silhouette Score',
                    line=dict(color='#4ECDC4', width=3),
                    marker=dict(size=8, color='#4ECDC4'),
                    hovertemplate='K=%{x}<br>Silhouette Score=%{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Add optimal K indicators
            fig_optimization.add_vline(
                x=self.optimal_k, line_dash="dash", line_color="#45B7D1", line_width=3,
                annotation_text=f"✅ Optimal K = {self.optimal_k}",
                annotation_position="top",
                row=1, col=1
            )
            
            fig_optimization.add_vline(
                x=self.optimal_k, line_dash="dash", line_color="#45B7D1", line_width=3,
                row=1, col=2
            )
            
            fig_optimization.update_layout(
                height=500,
                title_text="🎯 Cluster Number Optimization Analysis",
                title_x=0.5,
                title_font_size=16,
                showlegend=True,
                template='plotly_white'
            )
            
            fig_optimization.update_xaxes(title_text="Number of Clusters (K)", row=1, col=1)
            fig_optimization.update_xaxes(title_text="Number of Clusters (K)", row=1, col=2)
            fig_optimization.update_yaxes(title_text="Sum of Squared Errors", row=1, col=1)
            fig_optimization.update_yaxes(title_text="Silhouette Score", row=1, col=2)
            
            visualizations['optimization_analysis'] = fig_optimization.to_html(include_plotlyjs=True, div_id="optimization-analysis")
            
            # 4. Feature Distribution Analysis
            fig_dist = make_subplots(
                rows=1, cols=len(self.x.columns),
                subplot_titles=[f'📊 {col} Distribution by Cluster' for col in self.x.columns]
            )
            
            colors = px.colors.qualitative.Set1[:self.optimal_k]
            
            for i, col in enumerate(self.x.columns):
                for cluster_id in range(self.optimal_k):
                    cluster_data = self.x[self.kmeans_labels == cluster_id][col]
                    
                    fig_dist.add_trace(
                        go.Histogram(
                            x=cluster_data,
                            name=f'Cluster {cluster_id + 1}',
                            opacity=0.7,
                            marker_color=colors[cluster_id],
                            showlegend=(i == 0),  # Only show legend for first subplot
                            nbinsx=20,
                            hovertemplate=f'Cluster {cluster_id + 1}<br>{col}: %{{x}}<br>Count: %{{y}}<extra></extra>'
                        ),
                        row=1, col=i+1
                    )
            
            fig_dist.update_layout(
                height=500,
                title_text="📈 Feature Distribution Analysis Across Clusters",
                title_x=0.5,
                title_font_size=16,
                template='plotly_white',
                barmode='overlay'
            )
            
            visualizations['feature_distribution'] = fig_dist.to_html(include_plotlyjs=True, div_id="feature-distribution")
            
            # 5. Cluster Characteristics Heatmap
            df_with_clusters = self.df.copy()
            df_with_clusters['Cluster'] = self.kmeans_labels
            
            # Include age if available
            analysis_columns = list(self.x.columns)
            if 'Age' in self.df.columns:
                analysis_columns.append('Age')
            
            cluster_stats = df_with_clusters.groupby('Cluster')[analysis_columns].mean().round(2)
            
            fig_heatmap = px.imshow(
                cluster_stats.T,
                labels=dict(x="Cluster", y="Features", color="Average Value"),
                x=[f'Cluster {i+1}' for i in range(self.optimal_k)],
                y=analysis_columns,
                title="🌡️ Cluster Characteristics Heatmap",
                template='plotly_white',
                color_continuous_scale='RdYlBu_r',
                aspect='auto'
            )
            
            fig_heatmap.update_layout(
                height=400,
                title_x=0.5,
                title_font_size=16
            )
            
            # Add text annotations with values
            for i, cluster in enumerate(cluster_stats.columns):
                for j, feature in enumerate(cluster_stats.index):
                    fig_heatmap.add_annotation(
                        x=i, y=j,
                        text=str(cluster_stats.iloc[j, i]),
                        showarrow=False,
                        font=dict(color="white" if cluster_stats.iloc[j, i] > cluster_stats.iloc[j, :].median() else "black")
                    )
            
            visualizations['characteristics_heatmap'] = fig_heatmap.to_html(include_plotlyjs=True, div_id="characteristics-heatmap")
            
            # 6. Algorithm Comparison (if we have multiple algorithm results)
            if hasattr(self, 'agg_labels') and self.agg_labels is not None:
                metrics_data = self._calculate_advanced_metrics()
                
                if 'metrics' in metrics_data:
                    algorithms = ['K-Means', 'Agglomerative']
                    silhouette_scores = [metrics_data['metrics']['Silhouette Score'][alg] for alg in algorithms]
                    calinski_scores = [metrics_data['metrics']['Calinski-Harabasz Index'][alg] for alg in algorithms]
                    davies_scores = [metrics_data['metrics']['Davies-Bouldin Index'][alg] for alg in algorithms]
                    
                    fig_comparison = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=('Silhouette Score', 'Calinski-Harabasz', 'Davies-Bouldin'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    # Silhouette scores (higher is better)
                    fig_comparison.add_trace(
                        go.Bar(x=algorithms, y=silhouette_scores, name='Silhouette', 
                               marker_color=['#FF6B6B', '#4ECDC4'], showlegend=False),
                        row=1, col=1
                    )
                    
                    # Calinski-Harabasz scores (higher is better)
                    fig_comparison.add_trace(
                        go.Bar(x=algorithms, y=calinski_scores, name='Calinski-Harabasz',
                               marker_color=['#FF6B6B', '#4ECDC4'], showlegend=False),
                        row=1, col=2
                    )
                    
                    # Davies-Bouldin scores (lower is better)
                    fig_comparison.add_trace(
                        go.Bar(x=algorithms, y=davies_scores, name='Davies-Bouldin',
                               marker_color=['#FF6B6B', '#4ECDC4'], showlegend=False),
                        row=1, col=3
                    )
                    
                    fig_comparison.update_layout(
                        height=400,
                        title_text="⚖️ Algorithm Performance Comparison",
                        title_x=0.5,
                        title_font_size=16,
                        template='plotly_white'
                    )
                    
                    visualizations['algorithm_comparison'] = fig_comparison.to_html(include_plotlyjs=True, div_id="algorithm-comparison")
            
            return visualizations
            
        except Exception as e:
            return {'error': str(e)}
    
    def export_cluster_assignments(self):
        """Export cluster assignments with original data"""
        try:
            if self.kmeans_labels is None:
                return {'error': 'No clustering results available'}
            
            export_df = self.df.copy()
            export_df['Cluster_ID'] = self.kmeans_labels
            export_df['Cluster_Name'] = [f'Cluster_{i+1}' for i in self.kmeans_labels]
            
            # Add cluster characteristics
            segments_analysis = self.analyze_segments()
            if segments_analysis.get('segments'):
                cluster_info = {seg['cluster_id']-1: seg['category'] for seg in segments_analysis['segments']}
                export_df['Cluster_Category'] = export_df['Cluster_ID'].map(cluster_info)
            
            return {
                'success': True,
                'data': export_df,
                'csv_string': export_df.to_csv(index=False)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_insights_summary(self):
        """Generate executive summary of insights"""
        try:
            segments_analysis = self.analyze_segments()
            if not segments_analysis.get('segments'):
                return {'error': 'No segment analysis available'}
            
            segments = segments_analysis['segments']
            
            # Key statistics
            total_customers = segments_analysis['total_customers']
            largest_segment = max(segments, key=lambda x: x['size'])
            highest_roi_segment = max(segments, key=lambda x: x['roi_potential'])
            
            # Feature analysis
            feature_names = list(self.x.columns)
            
            summary = {
                'executive_summary': {
                    'total_customers_analyzed': total_customers,
                    'segments_identified': len(segments),
                    'largest_segment': {
                        'name': largest_segment['category'],
                        'size': largest_segment['size'],
                        'percentage': largest_segment['percentage']
                    },
                    'highest_value_segment': {
                        'name': highest_roi_segment['category'],
                        'roi_score': highest_roi_segment['roi_potential'],
                        'percentage': highest_roi_segment['percentage']
                    }
                },
                'key_insights': [],
                'strategic_recommendations': [],
                'quick_wins': [],
                'risk_factors': []
            }
            
            # Generate insights based on segment analysis
            for segment in segments:
                avg_vals = segment['avg_values']
                category = segment['category']
                size_pct = segment['percentage']
                
                if size_pct > 25:
                    summary['key_insights'].append(
                        f"{category} represents {size_pct:.1f}% of customers - a major market segment requiring focused attention"
                    )
                
                if segment['roi_potential'] > 150:
                    summary['strategic_recommendations'].append(
                        f"Prioritize {category} segment with premium service offerings and personalized experiences"
                    )
                
                if size_pct < 10 and segment['roi_potential'] > 100:
                    summary['quick_wins'].append(
                        f"Small but valuable {category} segment ({size_pct:.1f}%) offers quick ROI improvement opportunity"
                    )
                
                if size_pct > 20 and segment['roi_potential'] < 80:
                    summary['risk_factors'].append(
                        f"Large {category} segment ({size_pct:.1f}%) shows low engagement - risk of churn"
                    )
            
            # Add general strategic recommendations
            summary['strategic_recommendations'].extend([
                "Implement segment-specific marketing campaigns to improve targeting efficiency",
                "Develop personalized product recommendations based on cluster characteristics",
                "Create loyalty programs tailored to each segment's spending behavior",
                "Monitor segment migration patterns to identify customer lifecycle trends"
            ])
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_cluster_health_metrics(self):
        """Calculate cluster quality and health metrics"""
        try:
            if self.kmeans_labels is None:
                return {'error': 'No clustering results available'}
            
            # Cluster quality metrics
            silhouette_avg = silhouette_score(self.x, self.kmeans_labels)
            
            # Individual cluster silhouette scores
            from sklearn.metrics import silhouette_samples
            sample_silhouette_values = silhouette_samples(self.x, self.kmeans_labels)
            
            cluster_health = {}
            for cluster_id in range(self.optimal_k):
                cluster_silhouette_values = sample_silhouette_values[self.kmeans_labels == cluster_id]
                
                cluster_health[f'cluster_{cluster_id}'] = {
                    'avg_silhouette': float(cluster_silhouette_values.mean()),
                    'silhouette_std': float(cluster_silhouette_values.std()),
                    'size': int(sum(self.kmeans_labels == cluster_id)),
                    'cohesion_score': float(cluster_silhouette_values.mean()),
                    'stability_score': float(1 - cluster_silhouette_values.std() / abs(cluster_silhouette_values.mean()) if cluster_silhouette_values.mean() != 0 else 0)
                }
            
            # Overall clustering health
            overall_health = {
                'overall_silhouette': float(silhouette_avg),
                'cluster_balance': float(1 - np.std([cluster_health[f'cluster_{i}']['size'] for i in range(self.optimal_k)]) / np.mean([cluster_health[f'cluster_{i}']['size'] for i in range(self.optimal_k)])),
                'quality_score': 'Excellent' if silhouette_avg > 0.7 else 'Good' if silhouette_avg > 0.5 else 'Fair' if silhouette_avg > 0.25 else 'Poor',
                'recommended_actions': []
            }
            
            # Generate recommendations based on health metrics
            if silhouette_avg < 0.3:
                overall_health['recommended_actions'].append("Consider different clustering parameters or preprocessing")
            if overall_health['cluster_balance'] < 0.5:
                overall_health['recommended_actions'].append("Address cluster size imbalance - some segments may be too small or large")
            
            weak_clusters = [k for k, v in cluster_health.items() if v['avg_silhouette'] < 0.2]
            if weak_clusters:
                overall_health['recommended_actions'].append(f"Investigate weak clusters: {', '.join(weak_clusters)}")
            
            return {
                'success': True,
                'cluster_health': cluster_health,
                'overall_health': overall_health,
                'recommendation_priority': 'High' if len(overall_health['recommended_actions']) > 2 else 'Medium' if len(overall_health['recommended_actions']) > 0 else 'Low'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_complete_analysis(self):
        """Get complete analysis results with all enhancements"""
        try:
            # Perform all analysis steps
            optimal_k_results = self.find_optimal_k()
            clustering_results = self.perform_clustering()
            segments_analysis = self.analyze_segments()
            business_recommendations = self.generate_business_recommendations(segments_analysis)
            visualizations = self.create_visualizations()
            insights_summary = self.generate_insights_summary()
            health_metrics = self.get_cluster_health_metrics()
            export_data = self.export_cluster_assignments()
            
            return {
                'success': True,
                'optimal_k_analysis': optimal_k_results,
                'clustering_results': clustering_results,
                'segments_analysis': segments_analysis,
                'business_recommendations': business_recommendations,
                'visualizations': visualizations,
                'insights_summary': insights_summary,
                'health_metrics': health_metrics,
                'export_data': export_data,
                'data_info': {
                    'shape': self.df.shape,
                    'features': list(self.x.columns),
                    'sample_data': self.df.head().to_dict('records'),
                    'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Example usage
if __name__ == "__main__":
    analyzer = ClusteringAnalyzer()
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'CustomerID': range(1, 101),
        'Age': np.random.randint(18, 70, 100),
        'Annual_Income': np.random.randint(15, 140, 100),
        'Spending_Score': np.random.randint(1, 100, 100)
    })
    
    result = analyzer.load_data(sample_data)
    if result['success']:
        analysis = analyzer.get_complete_analysis()
        print("Analysis completed successfully!")
        print(f"Optimal K: {analysis['clustering_results']['optimal_k']}")
    else:
        print(f"Error: {result['error']}")