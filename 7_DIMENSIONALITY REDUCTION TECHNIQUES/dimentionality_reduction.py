#pip install -r requirements.txt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA, FastICA, NMF, FactorAnalysis


penguins = sns.load_dataset("penguins")
penguins = (penguins.dropna())
penguins.head()
data = (penguins.select_dtypes(np.number))
data.head()
random_state = 0
pca_pl = make_pipeline(StandardScaler(),PCA(n_components=2,random_state=random_state))
pcs = pca_pl.fit_transform(data)
pcs[0:5,:]
pcs_df = pd.DataFrame(data = pcs ,columns = ['PC1', 'PC2'])
pcs_df['Species'] = penguins.species.values
pcs_df['Sex'] = penguins.sex.values
pcs_df.head()
plt.figure(figsize=(12,10))
with sns.plotting_context("talk",font_scale=1.25):
    # The following line was not indented correctly, causing the error.
    sns.scatterplot(x="PC1", y="PC2",data=pcs_df,hue="Species",style="Sex",s=100)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA", size=24)
    plt.savefig("PCA_Example_in_Python.png",format='png',dpi=75)
    plt.show()