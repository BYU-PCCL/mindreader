
import numpy as np
from methods import * 
from rrt_smooth import *

from path_kde import *

cand_locs = (
	(0.042, 0.038) , (0.098, 0.124) , (0.27, 0.03) ,
	(0.454, 0.114) , (0.614, 0.244) , (0.72, 0.026) ,
	(0.754, 0.106) , (0.79, 0.264) , (0.84, 0.33) ,
	(0.842, 0.534) , (0.682, 0.428) , (0.544, 0.608) ,
	(0.296, 0.382) , (0.234, 0.472) , (0.056, 0.508) ,
	(0.086, 0.238) , (0.046, 0.704) , (0.026, 0.872) ,
	(0.314, 0.974) , (0.372, 0.77) , (0.25, 0.636) ,
	(0.54, 0.818) , (0.628, 0.88) , (0.736, 0.762) ,
	(0.9, 0.736) , (0.912, 0.948))

polygonSegments = load_polygons()
isovist = iso.Isovist( polygonSegments )
X1, Y1, X2, Y2 = polygons_to_segments(load_polygons_here())

# ==================================================================================

types = [ 'alley', 'grid', 'random', 'swirl_in', 'swirl_out' ]

results = []
for test_name in types:
    paths = load_data("NaiveAgentPaths/" + test_name + "_paths")
    paths = combine_paths(paths)
    paths = smooth(paths)

    results.append( path_to_heatmap(paths) )

tmarg = []
for r in results:
    tmarg.append( np.mean( r, axis=2 ) )

plt.imshow( tmarg[0] ); plt.show()
    
ttmarg = np.mean( np.stack( tmarg, axis=2 ), axis=2 )

trim_results = [ x[:,:,0:1000] for x in results ]

heatcube = np.stack( trim_results, axis=3 )

np.save( './heatcube.npy', heatcube )
