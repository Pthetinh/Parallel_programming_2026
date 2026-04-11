import pandas as pd
import matplotlib.pyplot as plt

configs = [(8, 8), (16, 16), (32, 32)]
colors = ['blue', 'green', 'red']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (bx, by) in enumerate(configs):
    filename = f'results_{bx}x{by}.csv'
    df = pd.read_csv(filename)

    ax = axes[idx]

    ax.plot(df['N'], df['GPU_ms'],
            marker='o', linewidth=2, markersize=8,
            color=colors[idx], label=f'GPU ({bx}×{by})')

    for _, row in df.iterrows():
        ax.annotate(f'{row["GPU_ms"]:.2f}',
                   (row['N'], row['GPU_ms']),
                   textcoords="offset points",
                   xytext=(0, 10),
                   ha='center', fontsize=9)

    ax.set_xlabel('Matrix Size (N)', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title(f'Block: {bx}×{by}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cuda_performance_3plots.png', dpi=300, bbox_inches='tight')
plt.show()
