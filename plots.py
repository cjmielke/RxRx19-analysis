import matplotlib.pyplot as plt
import pandas
from matplotlib import colors
import matplotlib.cm as cmx

df = pandas.read_hdf('DF.hdf','df')

print(df.columns)
print(df.treatment_conc.value_counts())

print()

def scatter(df, colname, title=None):
    '''
    color scatterplot with colors determined from categorical values
    based on https://stackoverflow.com/questions/28033046/matplotlib-scatter-color-by-categorical-factors
    :type df: pandas.DataFrame
    '''

    title = title or colname

    numPoints = len(df)
    if numPoints>100000: pointSize = 0.005
    else: pointSize = 0.1

    uniq = sorted(list(set(df[colname].dropna())))

    if len(uniq) > 50:
        print('skipping ', colname)
        return

    print('\n\n')
    print(df[colname].value_counts())

    # Set the color map to match the number of species

    hot = plt.get_cmap('rainbow')
    cNorm  = colors.Normalize(vmin=0, vmax=len(uniq))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)

    for i, colVal in enumerate(uniq):
        sdf = df[df[colname]==colVal]
        plt.scatter(sdf.x, sdf.y, s=pointSize, color=scalarMap.to_rgba(i), label=colVal)

    plt.title(title)
    if len(uniq) < 50:
        lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=5)
        for lh in lgnd.legendHandles: lh._sizes = [50]
    #plt.show()

    fname = colname
    plt.savefig('tsne-%s.png' % fname, dpi=300)
    plt.clf()



scatter(df, 'treatment_conc')
#scatter(df, 'cell_type')
#scatter(df, 'experiment')

plotcols = [c for c in df.columns if c not in ['x', 'y', 'SMILES']]
print(plotcols)

for col in plotcols: scatter(df, col)
