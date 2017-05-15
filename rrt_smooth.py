

from methods import *


def remove_duplicates(path):

	new_path = []
	count = 0
	for i in xrange(1, len(path)):
		space = dist(scale_up(path[i-1]), scale_up(path[i]))
		if space == 0:
			count += 1
		if space != 0:
			new_path.append(path[i-1])
			if (i == len(path) - 1):
				new_path.append(path[i])
	return new_path, count

'''
goosh adds in between points between pt1 and pt2
makes steps of about 5
'''
def goosh(pt1, pt2):
	#print pt1, pt2
	_dir = direction(pt2, pt1)
	mag = dist(pt1, pt2)
	way_pt_amt = mag/5.0

	short_dir = (_dir[0]/way_pt_amt, _dir[1]/way_pt_amt)
	#print _dir, mag, way_pt_amt, short_dir

	pts = []

	curr_pt = pt1
	pts.append((curr_pt[0]/500.0, curr_pt[1]/500.0))
	for i in xrange(int(way_pt_amt)):
		curr_pt = (curr_pt[0] + short_dir[0], curr_pt[1] + short_dir[1])
		pts.append((curr_pt[0]/500.0, curr_pt[1]/500.0))
		

	return pts

def divide_long_steps(path):
	div_path = []
	for i in xrange(1, len(path)):
		space = dist(scale_up(path[i-1]), scale_up(path[i]))
		if space >= 10:
			way_pts = goosh(scale_up(path[i-1]) , scale_up(path[i]))
			div_path.extend(way_pts)
		else:
			div_path.append(path[i-1])
		if i == len(path) - 1:
			div_path.append(path[i])

	return div_path

def show_spacing_btwn_pts(path_):
	for i in xrange(1, len(path_)):
		space = dist(scale_up(path_[i-1]), scale_up(path_[i]))
		print scale_up(path_[i-1]), scale_up(path_[i]), ": ", space



def smooth(path):
	new_path,_ = remove_duplicates(path)
	div_path = divide_long_steps(new_path)
	return div_path



def main(test_name = "grid"):

	# TEST REGULAR RRT without way points (can have 0 step sizes)
	X1, Y1, X2, Y2 = polygons_to_segments(load_polygons_here())
	path = run_rrt( scale_down(scale_up((0.042, 0.038))), scale_down(scale_up((0.736, 0.762))), X1, Y1, X2, Y2)

	# TEST RRT with way points (huge steps)
	paths = load_data(test_name + "_paths")
	path = combine_paths(paths)

	# DOES THE WORK 
	div_path = smooth(path)
	
	# SHOW SPACING
	show_spacing_btwn_pts(div_path)
	


if __name__ == '__main__':
	main(test_name = "grid")


