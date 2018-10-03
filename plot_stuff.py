

import pickle
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def read_store_1024_weights():
	file_ids = [1538256755, 1538260860, 1538265457, 1538268999, 1538273112, 1538278544]


	all_scores = []
	for file_id in file_ids:
		directory = "PO_forward_runs/conditioned/SMC_simulations/sim-"+str(file_id)+"/"
		filename = str(file_id)+"_t-"

		chaser_weights = []

		for t in tqdm(xrange(1,29)):
			file_t = pickle.load( open( directory+filename+str(t)+".p", "rb" ) )
			# [K, t, sampled_Q_ks, Q_k_scores]
			scores = file_t[3]
			print len(scores)
			chaser_weights.append(scores)
			file_t = None


		all_scores.append(chaser_weights)	

	pickle.dump( all_scores, open( "6-1024-chaser_weights.p", "wb" ))


def cheat_read_store_2048_weights(i):
	file_ids = [1538296170, 1538304215, 1538311513, 1538318557, 1538325638, 1538333434, 1538352234, 1538387163, 1538397571, 1538405094, 1538412127, 1538419405]
	
	file_id = file_ids[i]
	directory = "PO_forward_runs/conditioned/SMC_simulations/sim-"+str(file_id)+"/"
	filename = str(file_id)+"_t-"

	chaser_weights = []

	for t in tqdm(xrange(1,29)):
		file_t = pickle.load( open( directory+filename+str(t)+".p", "rb" ) )
		# [K, t, sampled_Q_ks, Q_k_scores]
		scores = file_t[3]
		print len(scores)
		chaser_weights.append(scores)
		file_t = None

	pickle.dump( chaser_weights, open( "12-2048-chaser_weights-"+str(i)+"-.p", "wb" ))

def read_store_2048_weights():
	file_ids = [1538296170, 1538304215, 1538311513, 1538318557, 1538325638, 1538333434, 1538352234, 1538387163, 1538397571, 1538405094, 1538412127, 1538419405]
	
	all_scores = []
	for file_id in file_ids:
		directory = "PO_forward_runs/conditioned/SMC_simulations/sim-"+str(file_id)+"/"
		filename = str(file_id)+"_t-"

		chaser_weights = []

		for t in tqdm(xrange(1,29)):
			file_t = pickle.load( open( directory+filename+str(t)+".p", "rb" ) )
			# [K, t, sampled_Q_ks, Q_k_scores]
			scores = file_t[3]
			print len(scores)
			chaser_weights.append(scores)
			file_t = None

		all_scores.append(chaser_weights)

	pickle.dump( all_scores, open( "12-2048-chaser_weights.p", "wb" ))

from scipy.special import logsumexp
def read_plot_weights(filename):
	weights = pickle.load( open( filename, "rb" ) )
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	for i in xrange(len(weights)):
		simulation = weights[i]
		sim = np.array(simulation).T
		print "sim shape:", sim.shape


		# maybe normalize?

		log_normalizer = logsumexp(sim, axis=0) - np.log(sim.shape[0]) 
		w = np.exp(sim - log_normalizer - np.log(sim.shape[0]))

		ess = 1.0/np.sum(w*w,axis=0)		

		# plot probability
		
		ax.plot(list(xrange(1, 29)), ess,label="K="+str(sim.shape[0]))


	ax.set_xlabel("time step")
	ax.set_ylabel("ESS")
	ax.legend(bbox_to_anchor=(1.3, 1.05))
	fig.savefig("1024-ESS.eps", bbox_inches='tight')

from matplotlib.font_manager import FontProperties
def read_plot_weights_per_file():

	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	#[[2048,1],[512,4],[128,16],[32,64], [8,256], [2,1024], [1,2048], [4,512], [16,128], [64,32], [256,8], [1024,2]]
	samps = [0, 11, 1, 10, 2, 9, 3, 8, 4, 7, 5, 6]
	for i in samps[:6]:
		filename = "12-2048-chaser_weights-"+str(i)+"-.p"
		simulation = pickle.load( open( filename, "rb" ) )
		sim = np.array(simulation).T
		print "sim shape:", sim.shape
		log_normalizer = logsumexp(sim, axis=0) - np.log(sim.shape[0]) 
		w = np.exp(sim - log_normalizer - np.log(sim.shape[0]))
		ess = (1.0/np.sum(w*w,axis=0)) / sim.shape[0]
		ax.plot(list(xrange(1, 29)), ess,label="K="+str(sim.shape[0]))

	ax.set_xlabel("time step")
	ax.set_ylabel("ESS")


	ax.legend(bbox_to_anchor=(1.1, 1.05))
	fig.savefig("2048-ESS.eps", bbox_inches='tight')



# def cheat_read_store_2048_kl_weights(i):
# 	file_ids = [1538296170, 1538304215, 1538311513, 1538318557, 1538325638, 1538333434, 1538352234, 1538387163, 1538397571, 1538405094, 1538412127, 1538419405]
	
# 	file_id = file_ids[i]
# 	directory = "PO_forward_runs/conditioned/SMC_simulations/sim-"+str(file_id)+"/"
# 	filename = str(file_id)+"_t-"

# 	kl_weights = []

# 	for t in tqdm(xrange(1,29)):
# 		file_t = pickle.load( open( directory+filename+str(t)+".p", "rb" ) )
# 		# [K, t, sampled_Q_ks, Q_k_scores]
# 		trace = file_t[2]
# 		print trace["all_Q_"]
# 		# need to keep track of all the times Qkl detected runner
# 		# for all K Qs:

# 		#kl_weights.append(-)
# 		file_t = None

# 	pickle.dump( chaser_weights, open( "12-2048-chaser_kl-weights-"+str(i)+".p", "wb" ))


def store_resampled_2048_weights():
	file_ids = [1538440198, 1538448489, 1538499742, 1538462807, 1538470154, 1538477199, 1538484792, 1538496244]
	file_ids = [1538532620, 1538539386, 1538545347, 1538551100, 1538556818, 1538562473, 1538568646]
	# fig = plt.figure(1)
	# fig.clf()
	# ax = fig.add_subplot(1, 1, 1)
	
	for file_id in file_ids[:7]:
		directory = "PO_forward_runs/conditioned/SMC_simulations/resamp-sim-"+str(file_id)+"/"
		filename = str(file_id)+"_t-"

		weights_in_time = []
		log_norms = []
		for i in xrange(1,29):
			KQ_info = pickle.load( open( directory + filename+str(i)+".p", "rb" ) )
			w, log_normalizer = KQ_info["norm_weights"], KQ_info["log_normalizer"]
			weights_in_time.append(w)
			log_norms.append(log_normalizer)
			# log_normalizer = logsumexp(sim, axis=0) - np.log(sim.shape[0]) 
			# w = np.exp(sim - log_normalizer - np.log(sim.shape[0]))
			# ess = (1.0/np.sum(w*w,axis=0)) / sim.shape[0]
			# ax.plot(list(xrange(1, 29)), ess,label="K="+str(sim.shape[0]))
		print np.array(weights_in_time).T.shape
		pickle.dump( [np.array(weights_in_time).T, np.array(log_norms)] , open( "resamp-weights-"+str(file_id)+"_1.p", "wb" ))

	# ax.set_xlabel("time step")
	# ax.set_ylabel("ESS")


	# ax.legend(bbox_to_anchor=(1.1, 1.05))
	#fig.savefig("resamp-2048-ESS.eps", bbox_inches='tight')

def read_plot_resampled_2048_weights():
	file_ids = [1538440198, 1538448489, 1538499742, 1538462807, 1538470154, 1538477199, 1538484792, 1538496244]
	file_ids = [1538532620, 1538539386, 1538545347, 1538551100, 1538556818, 1538562473, 1538568646]
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	for file_id in file_ids[:7]:
		filename = "resamp-weights-"+str(file_id)+"_1.p"

		weights_in_time = []
		
		w  = pickle.load( open(filename, "rb" ) )[0]
		print w.shape
		ess = (1.0/np.sum(w*w,axis=0)) / w.shape[0]
		ax.plot(list(xrange(1, 29)), ess, label="K="+str(w.shape[0]))
	ax.set_xlabel("time step")
	ax.set_ylabel("ESS")


	ax.legend(bbox_to_anchor=(1.1, 1.05))
	fig.savefig("resamp-2048-ESS-fraction_1.eps", bbox_inches='tight')
	

def plot_chaser_means():
	file_ids = [1538440198, 1538448489, 1538499742, 1538462807, 1538470154, 1538477199, 1538484792, 1538496244]
	file_ids = [1538532620, 1538539386, 1538545347, 1538551100, 1538556818, 1538562473, 1538568646]
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	for file_id in file_ids[:7]:
		filename = "resamp-weights-"+str(file_id)+"_1.p"
		
		info = pickle.load( open(filename, "rb" ) )
		log_norms = info[1]
		print log_norms.shape
		#ess = (1.0/np.sum(w*w,axis=0)) #/ w.shape[0]
		ax.plot(list(xrange(1, 29)), log_norms, label="K="+str(info[0].shape[0]))
	ax.set_xlabel("time step")
	ax.set_ylabel("log average weight")


	ax.legend(bbox_to_anchor=(1.1, 1.05))
	fig.savefig("resamp-2048-LOGNORMS_2.eps", bbox_inches='tight')


def store_resampled_2048_kl_weights():
	#file_ids = [1538440198, 1538448489, 1538499742, 1538462807, 1538470154, 1538477199, 1538484792, 1538496244]
	file_ids = [1538532620, 1538539386, 1538545347, 1538551100, 1538556818, 1538562473, 1538568646]
	shapes = [[2048,1], [512,4], [128,16], [64,32], [32,64], [16,128], [4,512], [1,2048]]
	j = 0
	for file_id in file_ids[:7]:
		shape = shapes[j]
		print ("SHAPE:", shape)
		j+=1
		directory = "PO_forward_runs/conditioned/SMC_simulations/resamp-sim-"+str(file_id)+"/"
		filename = str(file_id)+"_t-"

		all_kl_scores = []
		all_q_scores = []
		for i in xrange(1,29):
			KQ_info = pickle.load( open( directory + filename+str(i)+".p", "rb" ) )
			
			Qq_scores_t = []
			q_scores_t = []
			# for each K sample
			for k in xrange(1,shape[0]+1):
				small_trace = KQ_info[str(k)]
				Qq_scores_t.append(small_trace["all_Qls_scores"])
				q_scores_t.append(small_trace["all_ql_scores"])
				#print (k, small_trace["all_Qls_scores"][0], small_trace["all_ql_scores"][0])

			all_kl_scores.append(Qq_scores_t)
			all_q_scores.append(q_scores_t)

		all_kl_scores = np.array(all_kl_scores)
		all_q_scores = np.array(all_q_scores)
		print (all_kl_scores.shape)
		print (all_q_scores.shape)	

		pickle.dump( all_kl_scores, open( "Qkl-scores-resamp-weights-"+str(file_id)+".p", "wb" ))
		pickle.dump( all_q_scores, open( "ql-scores-resamp-weights-"+str(file_id)+".p", "wb" ))

def plot_resamped_2048_kl_weights():
	#file_ids = [1538440198, 1538448489, 1538499742, 1538462807, 1538470154, 1538477199, 1538484792, 1538496244]
	file_ids = [1538532620, 1538539386, 1538545347, 1538551100, 1538556818, 1538562473, 1538568646]
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	for file_id in file_ids[:7]:
		#filename = "Qkl-scores-resamp-weights-"+str(file_id)+".p"
		filename = "ql-scores-resamp-weights-"+str(file_id)+".p"
		
		all_qkl_weights = pickle.load( open(filename, "rb" ) )
		print (all_qkl_weights.shape)
		# average across L
		#sim = (logsumexp(all_qkl_weights, axis=2) - np.log(all_qkl_weights.shape[2])).T
		print "all the runner weights:", all_qkl_weights.shape,np.unique(all_qkl_weights)
		print ("weights:", all_qkl_weights)
		raw_input()
		sim = np.mean(all_qkl_weights, axis=2).T
		print sim.shape
		
		log_normalizer = logsumexp(sim, axis=0) - np.log(sim.shape[0]) 
		w = np.exp(sim - log_normalizer - np.log(sim.shape[0]))
		print "norm weights:", w.shape
		#print (np.sum(w, axis=0))
		
		ess = (1.0/np.sum(w*w,axis=0)) / sim.shape[0]
		print "ess:", ess
		raw_input()
		ax.plot(list(xrange(1, 29)), ess,label="K="+str(sim.shape[0]))

	ax.set_xlabel("time step")
	ax.set_ylabel("ESS")


	ax.legend(bbox_to_anchor=(1.1, 1.05))
	#fig.savefig("resamp-QKL-2048-ESS_1.eps", bbox_inches='tight')
	fig.savefig("resamp-ql-2048-ESS_1.eps", bbox_inches='tight')



if __name__ == '__main__':
	# method to read and pickle all the chaser scores at every time step in one file
	#read_store_1024_weights()
	# method to read and pickle all the chaser scores at every time step in one file
	#read_store_2048_weights()

	#read_plot_weights("6-1024-chaser_weights.p")
	#cheat_read_store_2048_weights(11)
	#diff_samples_1 = [[2048,1],[512,4],[128,16],[32,64], [8,256], [2,1024], [1,2048], [4,512], [16,128], [64,32], [256,8], [1024,2]]

	#read_plot_weights_per_file()

	#cheat_read_store_2048_kl_weights(0)

	#store_resampled_2048_weights()

	#read_plot_resampled_2048_weights()
	#store_resampled_2048_weights()
	#plot_chaser_means()
	#read_plot_resampled_2048_weights()
	#store_resampled_2048_kl_weights()
	plot_resamped_2048_kl_weights()
	#read_plot_weights_per_file()