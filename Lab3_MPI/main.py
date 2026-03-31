import matplotlib.pyplot as plt
import numpy as np

def read_timing_data(filename='timing_results.txt'):
    data = {}
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) != 3:
                continue
            
            size_part = parts[0].strip()
            matrix_size = int(size_part.split(':')[1].strip())
            
            process_part = parts[1].strip()
            num_processes = int(process_part.split(':')[1].strip())
            
            time_part = parts[2].strip()
            exec_time = float(time_part.split(':')[1].strip().split()[0])
            
            if num_processes not in data:
                data[num_processes] = {'sizes': [], 'times': []}
            
            data[num_processes]['sizes'].append(matrix_size)
            data[num_processes]['times'].append(exec_time)
    
    return data

data = read_timing_data('timing_results.txt')

fig, ax = plt.subplots(figsize=(10, 6))

colors = {1: 'blue', 2: 'orange', 4: 'green', 8: 'red'}
labels = {1: '1 process', 2: '2 processes', 4: '4 processes', 8: '8 processes'}

for num_processes in sorted(data.keys()):
    sizes = data[num_processes]['sizes']
    times = data[num_processes]['times']
    
    sorted_indices = np.argsort(sizes)
    sizes_sorted = [sizes[i] for i in sorted_indices]
    times_sorted = [times[i] for i in sorted_indices]
    
    ax.plot(sizes_sorted, times_sorted, 
            marker='o', 
            linestyle='-', 
            linewidth=1.5, 
            markersize=6,
            color=colors.get(num_processes, 'gray'),
            label=labels.get(num_processes, f'{num_processes} processes'))

ax.set_xlabel('Matrix Size (N x N)', fontsize=12, fontweight='bold')
ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('MPI Performance by Matrix Size', fontsize=14, fontweight='bold', pad=20)

ax.set_ylim(-0.5, 40)
ax.set_yticks(np.arange(0, 41, 5))

ax.grid(True, alpha=0.3, linestyle='--')

ax.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)

all_sizes = sorted(set([size for p in data.values() for size in p['sizes']]))
ax.set_xticks(all_sizes)
ax.set_xticklabels(all_sizes)

ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig("graphic_time.png")
plt.show()