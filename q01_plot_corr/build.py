# Default imports
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#from matplotlib.pyplot import yticks, xticks, subplots, set_cmap

data = pd.read_csv('data/house_prices_multivariate.csv')
def plot_corr(data,size=11):

    corr = data.corr()
    plt.figure(figsize=(11,6))

    sns.heatmap(corr,cmap='YlOrRd')

# Write your solution here:
