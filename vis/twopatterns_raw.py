import sys
sys.path.insert(0, '/path/to/project')

import matplotlib.pyplot as plt
from TwoPatternsDataset import TwoPatternsDataset

def main():
    train_data = TwoPatternsDataset("/path/to/project/TwoPatterns/TwoPatterns_TRAIN.tsv")
    
    unique_classes = sorted(set(y.item() if hasattr(y, 'item') else y for _, y in train_data))
    high_contrast_colors = ['#FF0000', '#0000FF', '#00CC00', '#FF8000']
    
    samples_by_class = {c: None for c in unique_classes}
    for x, y in train_data:
        label = y.item() if hasattr(y, 'item') else y
        if samples_by_class[label] is None:
            samples_by_class[label] = x
        if all(v is not None for v in samples_by_class.values()):
            break
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for i, c in enumerate(unique_classes):
        row, col = i // 2, i % 2
        x = samples_by_class[c].squeeze().numpy()
        axes[row, col].plot(x, color=high_contrast_colors[i], linewidth=0.8)
        axes[row, col].set_title(f'Class {int(c)}', color=high_contrast_colors[i], fontweight='bold')
        axes[row, col].set_xlabel('Time Step')
        axes[row, col].set_ylabel('Value')
    
    plt.suptitle('TwoPatterns Raw Data Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/path/to/project/vis/twopatterns_raw.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
