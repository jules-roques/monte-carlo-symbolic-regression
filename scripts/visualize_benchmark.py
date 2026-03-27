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
    
    # Add 'method' column for consistency with instructions
    df['method'] = df['algo_name']
    
    # Define a consistent palette for 'method' to ensure consistency across plots
    unique_methods = sorted(df['method'].unique())
    method_palette = dict(zip(unique_methods, sns.color_palette("husl", len(unique_methods))))
    
    os.makedirs("results", exist_ok=True)
    
    print(f"Compiling {len(df)} entries into beautiful Matplotlib graphs...")
    
    # Set the style for seaborn
    sns.set_theme(style="darkgrid", palette="pastel")
    
    # --- Plot 1: R2 Boxplot (Consistent Coloring) ---
    plt.figure(figsize=(10, 6))
    plt.title("Symbolic Regression Benchmark: R² Scores by Algorithm", fontsize=16, pad=15)
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("R² Score", fontsize=12)
    
    # Using method_palette to match other plots
    ax = sns.boxplot(x="method", y="r2", hue="method", data=df, palette=method_palette, width=0.6, legend=False)
    sns.stripplot(x="method", y="r2", hue="method", data=df, size=4, palette="dark:.3", linewidth=0, alpha=0.5, legend=False)
    
    plt.tight_layout()
    out_path = "results/benchmark_r2.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ R2 Boxplot Saved! -> {out_path}")

    # --- Plot 2: Heatmap (Equation × Method) ---
    try:
        pivot = df.pivot_table(index="name", columns="method", values="r2", aggfunc='mean')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".3f")
        plt.title("Heatmap: Mean R² Scores by Equation and Method")
        plt.xlabel("Method")
        plt.ylabel("Equation")
        plt.tight_layout()
        heatmap_path = "results/heatmap.png"
        plt.savefig(heatmap_path)
        plt.close()
        print(f"✅ Heatmap Saved! -> {heatmap_path}")
    except Exception as e:
        print(f"⚠️ Could not generate heatmap: {e}")

    # --- Plot 3: Grouped Bar Chart (by Equation) ---
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="name", y="r2", hue="method", palette=method_palette)
    plt.xticks(rotation=45, ha='right')
    plt.title("R² Score per Equation")
    plt.xlabel("Equation Name")
    plt.ylabel("R² Score")
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    barplot_path = "results/barplot_grouped.png"
    plt.savefig(barplot_path)
    plt.close()
    print(f"✅ Grouped Bar Chart Saved! -> {barplot_path}")

    # --- Plot 4: Scatter Plot (R² vs Time) ---
    if 'elapsed_seconds' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="elapsed_seconds", y="r2", hue="method", style="run_type", s=100, palette=method_palette)
        plt.title("Performance (R²) vs Execution Time")
        plt.xlabel("Elapsed Seconds")
        plt.ylabel("R² Score")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        scatter_path = "results/scatter_time_vs_r2.png"
        plt.savefig(scatter_path)
        plt.close()
        print(f"✅ Scatter Plot Saved! -> {scatter_path}")
    else:
        print("⚠️ 'elapsed_seconds' not found, skipping scatter plot.")
    
    print(f"✅ Dashboard Image Saved! -> {out_path}")

if __name__ == "__main__":
    main()
