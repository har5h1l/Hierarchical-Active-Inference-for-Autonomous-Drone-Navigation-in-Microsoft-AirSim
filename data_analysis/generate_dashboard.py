"""
PLOT VIEWER GENERATOR

Creates an HTML dashboard to view all generated plots and analysis results.
"""

import os
import glob
from datetime import datetime

def generate_html_dashboard(results_dir):
    """Generate an HTML dashboard for viewing analysis results"""
    
    # Find all plot directories
    plot_dirs = glob.glob(os.path.join(results_dir, "plots_*"))
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Active Inference Experiment Analysis Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .environment-section {{
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .environment-title {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
        }}
        .plot-container {{
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            background: white;
        }}
        .plot-title {{
            background: #f8f9fa;
            padding: 10px;
            font-weight: bold;
            color: #495057;
            border-bottom: 1px solid #ddd;
        }}
        .plot-image {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .summary {{
            background: #e8f4fd;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #667eea;
        }}
        .report-link {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 6px;
            margin: 10px 0;
            font-weight: bold;
        }}
        .report-link:hover {{
            background: #5a6fd8;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>[BRAIN] Active Inference Experiment Analysis Dashboard</h1>
        <p>Comprehensive analysis of drone navigation experiments</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h3>[CHART] Analysis Summary</h3>
        <p>This dashboard presents the results of analyzing {len(plot_dirs)} environment(s) from your active inference drone navigation experiments.</p>
        <a href="experiment_analysis_report.md" class="report-link">[FILE] View Detailed Report</a>
    </div>
"""
    
    # Add sections for each environment
    for plot_dir in plot_dirs:
        env_name = os.path.basename(plot_dir).replace('plots_', '')
        
        html_content += f"""
    <div class="environment-section">
        <h2 class="environment-title">🌍 Environment: {env_name.title()}</h2>
        <div class="plot-grid">
"""
        
        # Find all plots in this environment
        plot_files = glob.glob(os.path.join(plot_dir, "*.png"))
        
        plot_descriptions = {
            'timeseries_analysis.png': 'Time Series Analysis - Evolution of key metrics over time',
            'planning_behavior.png': 'Planning Behavior - Analysis of when and how planning occurs',
            'correlation_matrix.png': 'Correlation Matrix - Relationships between variables',
            'lagged_correlations.png': 'Lagged Correlations - Temporal relationships',
            'performance_analysis.png': 'Performance Analysis - Success rates and efficiency metrics',
            'uncertainty_analysis.png': 'Uncertainty Analysis - Entropy and volatility patterns',
            'behavioral_clustering.png': 'Behavioral Clustering - Grouping similar behaviors'
        }
        
        for plot_file in sorted(plot_files):
            plot_name = os.path.basename(plot_file)
            plot_title = plot_descriptions.get(plot_name, plot_name.replace('_', ' ').title())
            
            # Use relative path for the image
            rel_path = os.path.relpath(plot_file, results_dir).replace('\\', '/')
            
            html_content += f"""
            <div class="plot-container">
                <div class="plot-title">{plot_title}</div>
                <img src="{rel_path}" alt="{plot_title}" class="plot-image">
            </div>
"""
        
        html_content += """
        </div>
    </div>
"""
    
    html_content += """
    <div class="summary">
        <h3>🔬 Key Insights</h3>
        <ul>
            <li><strong>Planning Frequency:</strong> Analysis shows how often the drone triggers replanning</li>
            <li><strong>Performance Correlation:</strong> Relationship between planning behavior and task success</li>
            <li><strong>Uncertainty Patterns:</strong> How entropy and variational free energy evolve</li>
            <li><strong>Behavioral Clusters:</strong> Different navigation strategies identified</li>
        </ul>
        
        <h3>[FOLDER] Files Generated</h3>
        <ul>
            <li><code>experiment_analysis_report.md</code> - Comprehensive statistical report</li>
            <li><code>plots_[environment]/</code> - Environment-specific visualizations</li>
            <li><code>dashboard.html</code> - This interactive dashboard</li>
        </ul>
    </div>
    
    <footer style="text-align: center; margin-top: 40px; color: #666;">
        <p>Generated by Active Inference Experiment Analysis Pipeline</p>
    </footer>
</body>
</html>
"""
    
    # Save the HTML file
    dashboard_path = os.path.join(results_dir, "dashboard.html")
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return dashboard_path

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    dashboard_path = generate_html_dashboard(results_dir)
    print(f"[CHART] Dashboard generated: {dashboard_path}")
    print("[TIP] Open this file in your browser to view the analysis results!")
