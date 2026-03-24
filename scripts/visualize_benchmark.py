#!/usr/bin/env python3
import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Gathering logs from 'logs/'...")
    log_files = glob.glob("logs/*.json")
    if not log_files:
        print("No log files found in logs/")
        return
        
    # Sort log files alphabetically. Since timestamps are formatted as %Y%m%d_%H%M%S, 
    # the latest files will naturally appear last.
    log_files.sort()
    
    latest_files = {}
    for file in log_files:
        filename = os.path.basename(file)
        parts = filename.replace(".json", "").split("_")
        run_type = parts[0] if len(parts) > 0 else "unknown"
        algo_name = parts[1] if len(parts) > 1 else "unknown"
        # Overwrite with the latest file
        latest_files[(run_type, algo_name)] = (file, filename, run_type, algo_name)
        
    all_data = []
    for key, (file, filename, run_type, algo_name) in latest_files.items():
        with open(file, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {filename}")
                continue
            
        for r in results:
            r['run_type'] = run_type
            r['algo_name'] = algo_name
            r['source_file'] = filename
            all_data.append(r)
            
    if not all_data:
        print("No valid data found in log files.")
        return
        
    df = pd.DataFrame(all_data)
    
    # Extract robust R² column dynamically natively from 'test_r2' or 'val_r2'
    if 'test_r2' in df.columns and 'val_r2' in df.columns:
        df['r2'] = df['test_r2'].fillna(df['val_r2'])
    elif 'test_r2' in df.columns:
        df['r2'] = df['test_r2']
    elif 'val_r2' in df.columns:
        df['r2'] = df['val_r2']
    else:
        print("No R2 column ('test_r2' or 'val_r2') found in the dataset.")
        return
    
    os.makedirs("results", exist_ok=True)
    
    print(f"Compiling {len(df)} entries into a beautiful Matplotlib graph...")
    
    # Set the style for seaborn
    sns.set_theme(style="darkgrid", palette="pastel")
    
    plt.figure(figsize=(10, 6))
    
    # Title and labels
    plt.title("Symbolic Regression Benchmark: R² Scores", fontsize=16, pad=15)
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("R² Score", fontsize=12)
    
    # Box plot
    ax = sns.boxplot(x="algo_name", y="r2", hue="run_type", data=df, width=0.6)
    
    # Strip plot to show individual points (fixed seaborn deprecation warning)
    sns.stripplot(x="algo_name", y="r2", hue="run_type", data=df, size=4, palette="dark:.3", linewidth=0, dodge=True, alpha=0.7)
    
    # Fix the legend (remove duplicate entries from stripplot)
    handles, labels = ax.get_legend_handles_labels()
    n_colors = len(df['run_type'].unique())
    plt.legend(handles[:n_colors], labels[:n_colors], title="Evaluation Stage", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    out_path = "results/benchmark_r2.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Dashboard Image Saved! -> {out_path}")

if __name__ == "__main__":
    main()
