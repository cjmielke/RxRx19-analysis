import matplotlib.pyplot as plt
import pandas
from matplotlib import colors
import matplotlib.cm as cmx

import tables

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



#scatter(df, 'treatment_conc')
#scatter(df, 'cell_type')
#scatter(df, 'experiment')

plotcols = [c for c in df.columns if c not in ['x', 'y', 'SMILES']]
print(plotcols)

#for col in plotcols: scatter(df, col)




# crazy idea! Now that I have these embeddings, can we find neighboring points of the healthy cells?

from scipy import spatial

# construct a KD tree on all of the points corresponding to drugs
drugs = df[df.disease_condition=='Active SARS-CoV-2']
drugPoints = zip(drugs.x.values, drugs.y.values)

drugPoints = drugs[['x','y']].values

print(drugPoints.shape)

tree = spatial.KDTree(drugPoints)

normalCells = df[df.disease_condition != 'Active SARS-CoV-2']
normalPoints = normalCells[['x','y']].values

hits = tree.query_ball_point(normalPoints, 0.1)

print(hits)

allHits = set()
for row in hits:
    for index in row: allHits.add(index)


print('----------hits------------')
print(len(allHits))
print('----------hits------------')

hits = df.iloc[sorted(list(allHits))]

print(hits)

#df['hit'] = df[df.iloc.isin(allHits)]

df['hit']=0
df.loc[df.index.isin(hits.index), 'hit']=1
df.loc[df.index.isin(hits.index), 'disease_condition']='ZHit'


scatter(df, 'hit')
scatter(df, 'disease_condition', 'drughit')


hits.to_csv('hits.tsv', sep='\t')


#[4, 8, 9, 12]






