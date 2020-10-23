import argparse

import matplotlib.pyplot as plt
import pandas
from matplotlib import colors
import matplotlib.cm as cmx

import tables



def scatter(df, colname, title=None, pointSize=None):
    '''
    color scatterplot with colors determined from categorical values
    based on https://stackoverflow.com/questions/28033046/matplotlib-scatter-color-by-categorical-factors
    :type df: pandas.DataFrame
    '''

    title = title or colname

    numPoints = len(df)

    if not pointSize:
        if numPoints<5000: pointSize = 10.0
        elif numPoints>100000: pointSize = 0.005
        else: pointSize = 1.0

    uniq = sorted(list(set(df[colname].dropna())))

    if len(uniq) > 50:
        return


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


# code for producing tSNE plots on image embeddings
# TODO - encapsulate this, or move it to those image-embedding specfic parts of the pipeline
if __name__ == '__main__':

    df = pandas.read_hdf('tSNE.hdf', 'df')

    # Make plots of the tSNE embeddings, colorizing for each of the categorical variables

    plotcols = [c for c in df.columns if c not in ['x', 'y', 'SMILES']]
    for col in plotcols: scatter(df, col)




    ###################################
    ##########  FIND SOME DRUG HITS!



    parser = argparse.ArgumentParser(description='Plot tSNE embeddings and search for drug hits')
    parser.add_argument('-radius', type=float, default=0.05, help='Set the search radius for drug-treatment points neighboring the homeostatic cells.')
    args = parser.parse_args()


    # crazy idea! Now that I have these embeddings, can we find neighboring points of the healthy cells?

    from scipy import spatial

    # construct a KD tree on all of the points corresponding to drugs
    drugs = df[df.disease_condition=='Active SARS-CoV-2']
    drugPoints = zip(drugs.x.values, drugs.y.values)

    drugPoints = drugs[['x','y']].values


    tree = spatial.KDTree(drugPoints)

    normalCells = df[df.disease_condition != 'Active SARS-CoV-2']
    normalPoints = normalCells[['x','y']].values

    hits = tree.query_ball_point(normalPoints, args.radius)


    # instead, what about just querying for points contained within the dominant clusters?
    normalPoints = tables.numpy.array([[-6,4], [-10, -7], [4, 5]])

    hits = tree.query_ball_point(normalPoints, 2)



    allHits = set()
    for row in hits:
        for index in row: allHits.add(index)



    hits = df.iloc[sorted(list(allHits))]


    #df['hit'] = df[df.iloc.isin(allHits)]

    df['hit']=0
    df.loc[df.index.isin(hits.index), 'hit']=1
    df.loc[df.index.isin(hits.index), 'disease_condition']='ZHit'


    scatter(df, 'hit')
    scatter(df, 'disease_condition', 'drughit')


    hits.to_csv('hits.tsv', sep='\t')


    print('\n\n\n=========== Drug Hits ===========')
    print('Total found : ', len(allHits))

    print('\n\n\n=========   Most common drugs found in search radius    ==============')
    print(hits.treatment.value_counts().head(30))


    #[4, 8, 9, 12]






