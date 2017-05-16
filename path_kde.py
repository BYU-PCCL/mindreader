import numpy as np

'''
Support functions to make a heatmap of occupancy probabilities over time

Main function is 'path_kde'

'''

def make_gauss( x, y, ss=100 ):
        X, Y = np.meshgrid( range(500), range(500) )
        return np.exp( -(1.0/(2.0*ss)) * ( (X.astype('float32')-x)**2.0 + (Y.astype('float32')-y)**2.0 ) )

def make_heatmap( pts, ss=100 ):
        # pts is a list of 2-tuples.  build a simple kde.
        tmp = np.zeros((500,500))
        for p in pts:
                tmp += make_gauss( 500.0*p[0], 500.0*p[1], ss=ss )
#                tmp[ int(500.0*p[1]), int(500.0*p[0]) ] = 2.0
        tmp = tmp / float( len( pts ) )
        return tmp

def path_kde( set_of_rrts, cnt=300, ss=100 ):
        # construct a heat map of occupancy probabilities for each time
        heatmap = np.zeros( (500,500,cnt) )
        for t in range( heatmap.shape[2] ):
                pts = []
                for k in set_of_rrts:
                        if t < len( k ):
                                pts.append( k[t] )
                heatmap[:,:,t] = make_heatmap( pts, ss=ss )
        return heatmap
