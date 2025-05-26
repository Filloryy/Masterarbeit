import os
import matplotlib.pyplot as plt
import csv

def plot_csv(filepath, title, ax):
    x = []
    y = []
    try:
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            prev_step = None
            for row in reader:
                if len(row) < 2 or not row[0].strip() or not row[1].strip():
                    continue
                try:
                    step = int(row[0])
                    value = float(row[1])
                except ValueError:
                    continue
                # If step decreases, plot current segment and start a new one
                if prev_step is not None and step < prev_step:
                    if len(x) > 1:
                        ax.plot(x, y)
                    x, y = [], []
                x.append(step)
                y.append(value)
                prev_step = step
            # Plot the last segment
            if len(x) > 1:
                ax.plot(x, y)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(title)
        return True
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return False

def get_min_max(filepath):
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    try:
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            prev_step = None
            for row in reader:
                if len(row) < 2 or not row[0].strip() or not row[1].strip():
                    continue
                try:
                    step = int(row[0])
                    value = float(row[1])
                except ValueError:
                    continue
                x_min = min(x_min, step)
                x_max = max(x_max, step)
                y_min = min(y_min, value)
                y_max = max(y_max, value)
        return x_min, x_max, y_min, y_max
    except FileNotFoundError:
        return None

def comparison(exp1, exp2, same_scale=False):
    csv_dir1 = f'Logs/{exp1}/scalars'
    csv_dir2 = f'Logs/{exp2}/scalars'
    csv_files = [f for f in os.listdir(csv_dir1) if f.endswith('.csv')]
    for fname in csv_files:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        exp1_path = os.path.join(csv_dir1, fname)
        exp2_path = os.path.join(csv_dir2, fname)
        plot_csv(exp1_path, f'{exp1}: {fname[:-4]}', ax1)
        plot_csv(exp2_path, f'{exp2}: {fname[:-4]}', ax2)
        if same_scale:
            minmax1 = get_min_max(exp1_path)
            minmax2 = get_min_max(exp2_path)
            if minmax1 and minmax2:
                x_min = min(minmax1[0], minmax2[0])
                x_max = max(minmax1[1], minmax2[1])
                y_min = min(minmax1[2], minmax2[2])
                y_max = max(minmax1[3], minmax2[3])
                ax1.set_xlim(x_min, x_max)
                ax2.set_xlim(x_min, x_max)
                ax1.set_ylim(y_min, y_max)
                ax2.set_ylim(y_min, y_max)
        plt.suptitle(f'Comparison for {fname[:-4]}')
        plt.tight_layout()
        plt.show()

def plot_average_runs_csv(filepath, title, ax):
    runs = []
    current_run = []
    prev_step = None
    try:
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 2 or not row[0].strip() or not row[1].strip():
                    continue
                try:
                    step = int(row[0])
                    value = float(row[1])
                except ValueError:
                    continue
                if prev_step is not None and step < prev_step:
                    if current_run:
                        runs.append(current_run)
                    current_run = []
                current_run.append((step, value))
                prev_step = step
            if current_run:
                runs.append(current_run)
        # Aggregate by step
        from collections import defaultdict
        step_values = defaultdict(list)
        for run in runs:
            for step, value in run:
                step_values[step].append(value)
        # Average values for each step
        avg_steps = sorted(step_values.keys())
        avg_values = [sum(step_values[step]) / len(step_values[step]) for step in avg_steps]
        ax.plot(avg_steps, avg_values, label=filepath)
        ax.set_xlabel('Step')
        ax.set_ylabel('Average Value')
        ax.set_title(title)
        ax.legend()
        return True
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return False

def comparison_average(exp1, exp2):
    csv_dir1 = f'Logs/{exp1}/scalars'
    csv_dir2 = f'Logs/{exp2}/scalars'
    csv_files = [f for f in os.listdir(csv_dir1) if f.endswith('.csv')]
    for fname in csv_files:
        fig, ax = plt.subplots(figsize=(8, 5))
        csv_path1 = os.path.join(csv_dir1, fname)
        csv_path2 = os.path.join(csv_dir2, fname)
        plot_average_runs_csv(csv_path1, f'{exp1} avg: {fname[:-4]}', ax)
        plot_average_runs_csv(csv_path2, f'{exp2} avg: {fname[:-4]}', ax)
        ax.set_title(f'Average comparison: {fname[:-4]}')
        ax.legend()
        plt.tight_layout()
        plt.show()

#comparison_average('fullbodygraph', 'single_node')
comparison('hetero', 'single_node', same_scale=False)
