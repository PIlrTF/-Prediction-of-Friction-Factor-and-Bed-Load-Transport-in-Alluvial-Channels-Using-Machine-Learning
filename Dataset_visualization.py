# HEATMAP

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

filepath="/content/preprocessed_dataset.csv"
df=pd.read_csv(filepath)
plt.figure(figsize=(14, 12))

# Calculate correlations
corr_matrix = df[numeric_cols].corr()

# Mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)


# Plot heatmap
heatmap = sns.heatmap(
    corr_matrix,
    center=0,
    annot=True,
)

# Fix label rotations and font sizes
heatmap.set_xticklabels(
    heatmap.get_xticklabels(),
    rotation=45,
    horizontalalignment='right',
    fontsize=11
)

heatmap.set_yticklabels(
    heatmap.get_yticklabels(),
    rotation=0,
    fontsize=11
)

# Better title
plt.title('Correlation Matrix of Hydraulic Parameters', fontsize=16, pad=20)

plt.tight_layout()
plt.savefig('improved_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
