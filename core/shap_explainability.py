"""
Advanced SHAP Explainability Engine for Wildfire Risk Prediction
Provides comprehensive model interpretation and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedSHAPExplainer:
    """Advanced SHAP explainability system with multiple visualization options"""
    
    def __init__(self, model, scaler, feature_names: List[str]):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.background_data = None
        
    def setup_explainer(self, X_background, explainer_type='kernel'):
        """Setup SHAP explainer with background data"""
        print("Setting up SHAP explainer...")
        
        # Scale background data
        X_background_scaled = self.scaler.transform(X_background)
        
        # Choose explainer type based on model
        if hasattr(self.model, 'predict_proba'):
            # For scikit-learn models
            if explainer_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model, X_background_scaled)
            else:
                # Use smaller background dataset for KernelExplainer
                background_summary = shap.kmeans(X_background_scaled, min(100, len(X_background)))
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background_summary)
        else:
            # For custom models like ANFIS
            background_summary = shap.kmeans(X_background_scaled, min(100, len(X_background)))
            self.explainer = shap.KernelExplainer(self.model.predict, background_summary)
        
        self.background_data = X_background_scaled
        print(f"Explainer setup complete using {explainer_type} explainer")
    
    def explain_instance(self, instance, plot=True) -> Dict:
        """Explain a single prediction instance"""
        # Scale the instance
        instance_scaled = self.scaler.transform(instance.reshape(1, -1))
        
        # Calculate SHAP values
        if hasattr(self.explainer, 'shap_values'):
            shap_values = self.explainer.shap_values(instance_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_values = self.explainer(instance_scaled)
        
        # Create explanation dictionary
        explanation = {
            'instance': instance,
            'shap_values': shap_values.flatten(),
            'feature_names': self.feature_names,
            'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            'prediction': self.model.predict(instance_scaled)[0],
            'feature_importance': dict(zip(self.feature_names, shap_values.flatten()))
        }
        
        if plot:
            self.plot_instance_explanation(explanation)
        
        return explanation
    
    def explain_batch(self, X, max_samples=100) -> Dict:
        """Explain multiple instances"""
        print(f"Calculating SHAP values for {min(max_samples, len(X))} samples...")
        
        # Sample data if needed
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            indices = np.arange(len(X))
        
        # Scale data
        X_scaled = self.scaler.transform(X_sample)
        
        # Calculate SHAP values
        if hasattr(self.explainer, 'shap_values'):
            shap_values = self.explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_values = self.explainer(X_scaled)
        
        # Store for later use
        self.shap_values = shap_values
        self.X_explained = X_sample
        
        return {
            'shap_values': shap_values,
            'X_sample': X_sample,
            'indices': indices,
            'feature_names': self.feature_names
        }
    
    def plot_instance_explanation(self, explanation: Dict, save_path=None):
        """Create comprehensive instance explanation plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Wildfire Risk Prediction Explanation\nPredicted Risk: {explanation["prediction"]:.3f}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Waterfall plot (bar chart version)
        ax1 = axes[0, 0]
        shap_vals = explanation['shap_values']
        feature_vals = explanation['instance']
        base_value = explanation['base_value']
        
        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
        
        colors = ['red' if v > 0 else 'blue' for v in shap_vals[sorted_idx]]
        y_pos = np.arange(len(sorted_idx))
        
        bars = ax1.barh(y_pos, shap_vals[sorted_idx], color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([explanation['feature_names'][i] for i in sorted_idx])
        ax1.set_xlabel('SHAP Value')
        ax1.set_title('Feature Impact on Risk Score')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, shap_vals[sorted_idx])):
            ax1.text(val + (0.001 if val > 0 else -0.001), i, f'{val:.3f}', 
                    ha='left' if val > 0 else 'right', va='center')
        
        # 2. Feature values comparison
        ax2 = axes[0, 1]
        feature_impacts = shap_vals[sorted_idx]
        feature_values = feature_vals[sorted_idx]
        
        scatter = ax2.scatter(feature_values, feature_impacts, c=colors, s=100, alpha=0.7)
        ax2.set_xlabel('Feature Value')
        ax2.set_ylabel('SHAP Value (Impact)')
        ax2.set_title('Feature Value vs Impact')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add feature labels for top features
        for i in range(min(5, len(sorted_idx))):
            idx = sorted_idx[i]
            ax2.annotate(explanation['feature_names'][i], 
                        (feature_values[i], feature_impacts[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. Risk decomposition
        ax3 = axes[1, 0]
        
        # Calculate cumulative contribution
        cumulative = np.cumsum(shap_vals[sorted_idx])
        total_risk = base_value + cumulative[-1]
        
        # Create stacked bar chart
        positive_impacts = np.where(shap_vals[sorted_idx] > 0, shap_vals[sorted_idx], 0)
        negative_impacts = np.where(shap_vals[sorted_idx] < 0, shap_vals[sorted_idx], 0)
        
        ax3.barh(y_pos, positive_impacts, color='red', alpha=0.7, label='Increases Risk')
        ax3.barh(y_pos, negative_impacts, color='blue', alpha=0.7, label='Decreases Risk')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([explanation['feature_names'][i] for i in sorted_idx])
        ax3.set_xlabel('Risk Contribution')
        ax3.set_title(f'Risk Decomposition (Total: {total_risk:.3f})')
        ax3.legend()
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = f"""
        PREDICTION SUMMARY
        ==================
        
        Predicted Risk Score: {explanation['prediction']:.3f}
        Base Value: {base_value:.3f}
        Total SHAP Impact: {np.sum(shap_vals):.3f}
        
        TOP 5 RISK FACTORS:
        """
        
        for i in range(min(5, len(sorted_idx))):
            idx = sorted_idx[i]
            feature_name = explanation['feature_names'][idx]
            shap_val = shap_vals[idx]
            feature_val = feature_vals[idx]
            direction = "Increases" if shap_val > 0 else "Decreases"
            summary_text += f"\n{i+1}. {feature_name}: {direction} risk by {abs(shap_val):.3f}"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explanation plot saved to {save_path}")
        
        plt.show()
    
    def plot_global_importance(self, save_path=None):
        """Create global feature importance plot"""
        if self.shap_values is None:
            print("No SHAP values available. Run explain_batch() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Global Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Mean absolute SHAP values
        ax1 = axes[0, 0]
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        bars = ax1.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx], 
                       color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_yticks(range(len(sorted_idx)))
        ax1.set_yticklabels([self.feature_names[i] for i in sorted_idx])
        ax1.set_xlabel('Mean |SHAP Value|')
        ax1.set_title('Global Feature Importance')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.001, i, f'{width:.3f}', ha='left', va='center')
        
        # 2. SHAP summary plot (beeswarm)
        ax2 = axes[0, 1]
        shap.summary_plot(self.shap_values, self.X_explained, 
                          feature_names=self.feature_names, plot_type='dot',
                          show=False, ax=ax2)
        ax2.set_title('SHAP Summary Plot')
        
        # 3. Feature value distribution
        ax3 = axes[1, 0]
        
        # Plot distribution for top 5 features
        top_features = sorted_idx[:5]
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_features)))
        
        for i, (feature_idx, color) in enumerate(zip(top_features, colors)):
            feature_values = self.X_explained[:, feature_idx]
            ax3.hist(feature_values, bins=20, alpha=0.6, color=color, 
                    label=self.feature_names[feature_idx], density=True)
        
        ax3.set_xlabel('Feature Value')
        ax3.set_ylabel('Density')
        ax3.set_title('Feature Value Distribution (Top 5)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. SHAP value distribution
        ax4 = axes[1, 1]
        
        for i, (feature_idx, color) in enumerate(zip(top_features, colors)):
            feature_shap = self.shap_values[:, feature_idx]
            ax4.hist(feature_shap, bins=20, alpha=0.6, color=color,
                    label=self.feature_names[feature_idx], density=True)
        
        ax4.set_xlabel('SHAP Value')
        ax4.set_ylabel('Density')
        ax4.set_title('SHAP Value Distribution (Top 5)')
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Global importance plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, instance_idx=0):
        """Create interactive Plotly dashboard"""
        if self.shap_values is None:
            print("No SHAP values available. Run explain_batch() first.")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Importance', 'SHAP Values', 
                          'Feature Values', 'Risk Breakdown'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Feature importance
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        fig.add_trace(
            go.Bar(x=mean_abs_shap[sorted_idx], 
                   y=[self.feature_names[i] for i in sorted_idx],
                   orientation='h',
                   marker_color='lightblue',
                   name='Importance'),
            row=1, col=1
        )
        
        # 2. SHAP values for specific instance
        instance_shap = self.shap_values[instance_idx]
        colors = ['red' if v > 0 else 'blue' for v in instance_shap[sorted_idx]]
        
        fig.add_trace(
            go.Bar(x=instance_shap[sorted_idx],
                   y=[self.feature_names[i] for i in sorted_idx],
                   orientation='h',
                   marker_color=colors,
                   name='SHAP Values'),
            row=1, col=2
        )
        
        # 3. Feature values for instance
        instance_values = self.X_explained[instance_idx]
        
        fig.add_trace(
            go.Bar(x=instance_values[sorted_idx],
                   y=[self.feature_names[i] for i in sorted_idx],
                   orientation='h',
                   marker_color='lightgreen',
                   name='Feature Values'),
            row=2, col=1
        )
        
        # 4. Risk breakdown (cumulative)
        cumulative = np.cumsum(instance_shap[sorted_idx])
        base_value = self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0
        
        fig.add_trace(
            go.Bar(x=cumulative,
                   y=[self.feature_names[i] for i in sorted_idx],
                   orientation='h',
                   marker_color='orange',
                   name='Cumulative Impact'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f'Interactive SHAP Dashboard - Instance {instance_idx}',
            showlegend=False
        )
        
        return fig
    
    def generate_explanation_report(self, instance, save_path='explanation_report.html'):
        """Generate comprehensive HTML explanation report"""
        explanation = self.explain_instance(instance, plot=False)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Wildfire Risk Prediction Explanation</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .risk-score {{ font-size: 48px; font-weight: bold; color: #d32f2f; }}
                .feature-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .feature-table th, .feature-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .feature-table th {{ background-color: #f2f2f2; }}
                .positive {{ color: #d32f2f; }}
                .negative {{ color: #388e3c; }}
                .summary {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Wildfire Risk Prediction Explanation</h1>
                <div class="risk-score">Risk Score: {explanation['prediction']:.3f}</div>
                <p>Base Value: {explanation['base_value']:.3f} | Total SHAP Impact: {np.sum(explanation['shap_values']):.3f}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>This prediction was made based on {len(explanation['feature_names'])} environmental features. 
                The model identified several key factors contributing to the wildfire risk.</p>
            </div>
            
            <h2>Feature Impact Analysis</h2>
            <table class="feature-table">
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>SHAP Impact</th>
                    <th>Contribution</th>
                </tr>
        """
        
        # Sort features by impact
        sorted_features = sorted(explanation['feature_importance'].items(), 
                                key=lambda x: abs(x[1]), reverse=True)
        
        for feature_name, shap_val in sorted_features:
            feature_idx = explanation['feature_names'].index(feature_name)
            feature_val = explanation['instance'][feature_idx]
            contribution_class = 'positive' if shap_val > 0 else 'negative'
            contribution_text = 'Increases Risk' if shap_val > 0 else 'Decreases Risk'
            
            html_content += f"""
                <tr>
                    <td>{feature_name}</td>
                    <td>{feature_val:.3f}</td>
                    <td class="{contribution_class}">{shap_val:.4f}</td>
                    <td>{contribution_text}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Key Insights</h2>
            <ul>
        """
        
        # Add key insights
        top_positive = [f for f in sorted_features if f[1] > 0][:3]
        top_negative = [f for f in sorted_features if f[1] < 0][:3]
        
        if top_positive:
            html_content += "<li><strong>Main Risk Factors:</strong><ul>"
            for feature_name, shap_val in top_positive:
                html_content += f"<li>{feature_name} (impact: {shap_val:.3f})</li>"
            html_content += "</ul></li>"
        
        if top_negative:
            html_content += "<li><strong>Protective Factors:</strong><ul>"
            for feature_name, shap_val in top_negative:
                html_content += f"<li>{feature_name} (impact: {shap_val:.3f})</li>"
            html_content += "</ul></li>"
        
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        # Save HTML file
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Explanation report saved to {save_path}")
        return save_path

def demo_shap_explainability():
    """Demonstration of advanced SHAP explainability"""
    print("="*60)
    print("ADVANCED SHAP EXPLAINABILITY ENGINE")
    print("="*60)
    
    # Load sample data and model
    try:
        X_test = np.load('models/X_test.npy')
        model = joblib.load('models/mlp_pso.pkl')
        scaler = joblib.load('models/scaler.pkl')
    except:
        print("Error loading model/data. Please run the pipeline first.")
        return
    
    feature_names = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Temp_RH_Index', 'Wind_Rain_Interaction']
    
    # Initialize explainer
    explainer = AdvancedSHAPExplainer(model, scaler, feature_names)
    
    # Setup explainer with background data
    background_data = X_test[:100]  # Use first 100 samples as background
    explainer.setup_explainer(background_data)
    
    # Explain a single instance
    print("\nExplaining single prediction...")
    test_instance = X_test[0]
    explanation = explainer.explain_instance(test_instance)
    
    print(f"Prediction: {explanation['prediction']:.3f}")
    print(f"Top risk factor: {max(explanation['feature_importance'].items(), key=lambda x: x[1])}")
    
    # Explain batch of instances
    print("\nCalculating batch explanations...")
    batch_results = explainer.explain_batch(X_test[:50])
    
    # Generate plots
    explainer.plot_global_importance(save_path='outputs/shap_global_importance.png')
    
    # Create interactive dashboard
    fig = explainer.create_interactive_dashboard(instance_idx=0)
    fig.write_html('outputs/shap_interactive_dashboard.html')
    
    # Generate explanation report
    explainer.generate_explanation_report(test_instance, 'outputs/explanation_report.html')
    
    print("\nSHAP analysis complete!")
    print("- Global importance plot: outputs/shap_global_importance.png")
    print("- Interactive dashboard: outputs/shap_interactive_dashboard.html")
    print("- Explanation report: outputs/explanation_report.html")

if __name__ == "__main__":
    demo_shap_explainability()
