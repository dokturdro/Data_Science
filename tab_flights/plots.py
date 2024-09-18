def plot_heatmap(data):
    matrix = np.triu(data.corr())
    plt.figure(figsize=(8, 8))
    heatmap = sns.heatmap(data.corr(), annot=True, annot_kws={"fontsize": 8}, mask=matrix, linewidths=0.2, square=True)
    figure = heatmap.get_figure()

    return figure