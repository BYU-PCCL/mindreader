

import pickle
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=0.8)
sns.set_style("white")


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


def plot_resamped_2048_kl_weights():
	#file_ids = [1538440198, 1538448489, 1538499742, 1538462807, 1538470154, 1538477199, 1538484792, 1538496244]
	file_ids = [1538532620, 1538539386, 1538545347, 1538551100, 1538556818, 1538562473, 1538568646]
	#file_ids = [1538590560]
	
	for file_id in file_ids[:]:
		#filename = "Qkl-scores-resamp-weights-"+str(file_id)+".p"
		filename = "ql-scores-resamp-weights-"+str(file_id)+".p"
		all_ql_weights = pickle.load( open(filename, "rb" ) )
		print ("all_ql_weights:", all_ql_weights.shape)

		ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T
		print ql_means.shape
		ql_means = (logsumexp(ql_means, axis=0) - np.log(ql_means.shape[0]))
		print ql_means.shape

		filename = "Qkl-scores-resamp-weights-"+str(file_id)+".p"
		all_Qkl_weights = pickle.load( open(filename, "rb" ) )
		print ("all_Qkl_weights:", all_Qkl_weights.shape)

		QKL_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T
		print QKL_means.shape
		QKL_means = (logsumexp(QKL_means, axis=0) - np.log(QKL_means.shape[0]))
		print QKL_means.shape
		
		
		fig = plt.figure(1)
		fig.clf()
		ax = fig.add_subplot(1, 1, 1)
		ax.plot(list(xrange(1, 29)), QKL_means/ql_means,label="K="+str(all_ql_weights.shape[1]))

		ax.set_xlabel("time step")
		ax.set_ylabel("ratio")


		ax.legend(bbox_to_anchor=(1.1, 1.05))
		#fig.savefig("resamp-QKL-2048-ESS_1.eps", bbox_inches='tight')
		fig.savefig("resamp-ql-2048-probs_"+str(all_ql_weights.shape[1])+".eps", bbox_inches='tight')


def store_resampled_2048_kl_weights():
	#file_ids = [1538440198, 1538448489, 1538499742, 1538462807, 1538470154, 1538477199, 1538484792, 1538496244]
	#file_ids = [1538532620, 1538539386, 1538545347, 1538551100, 1538556818, 1538562473, 1538568646]
	#shapes = [[2048,1], [512,4], [128,16], [64,32], [32,64], [16,128], [4,512], [1,2048]]
	#file_ids = [1538590560]

	#shapes = [[2048,1],[512,4], [256,8], [128,16], [64,32],[32,64] , [8,256]]
	#file_ids = [1538609735, 1538609635,1538615745, 1538616566] unordered
	#file_ids = [1538609735, 1538616566, 1538609635,1538622239, 1538618520, 1538615745, 1538622917]
	#shapes = [[2048,1],[512,4], [4,512],[1,1]]
	file_ids =  [1538639858, 1538674801, 1538675256, 1538680844]
	file_ids = [1538639858,1538674801,1538675256,1538680844,1538681038,1538687390,1539703235,1539703509,
				1539709650,1539715775,1539715788,1539715836,1539721850,1539728018,1539728576,1539728955,
				1539734254,1539740703,1539740894,1539741096,1539748118,1539752703,1539753150,1539755226,
				1539762160,1539768861,1539775547,1539787505,1539788123,1539793999,1539794919,1539799832,
				1539801294,1539805777,1539807871,1539811558,1539814586,1539817295,1539820829,1539822995,
				1539826901,1539828809,1539832722,1539834533,1539838491,1539844333,1539849974,1539855835,
				1539861654,1539867443,1539873331,1539889386,1539895441,1539898338,1539901461,1539905170,
				1539907884,1539911988,1539913939,1539918800,1539920274,1539925648,1539926235,1539932183,
				1539932581,1539938445,1539939465,1539946246,1539953057,1538687079]
	file_ids = [1538687079]
	j = 0
	for file_id in file_ids[63:]:
		
		j+=1
		directory = "PO_forward_runs/conditioned/SMC_simulations/resamp-sim-"+str(file_id)+"/"
		filename = str(file_id)+"_t-"

		all_kl_scores = []
		all_q_scores = []
		for i in xrange(1,29):
			KQ_info = pickle.load( open( directory + filename+str(i)+".p", "rb" ) )
			K = KQ_info["K"]
			Qq_scores_t = []
			q_scores_t = []
			# for each K sample
			for k in xrange(1,K+1):
				small_trace = KQ_info["orig-"+str(k)]
				Qq_scores_t.append(small_trace["all_Qls_scores"])
				q_scores_t.append(small_trace["all_ql_scores"])
				#print (k, small_trace["all_Qls_scores"][0], small_trace["all_ql_scores"][0])

			all_kl_scores.append(Qq_scores_t)
			all_q_scores.append(q_scores_t)

		all_kl_scores = np.array(all_kl_scores)
		all_q_scores = np.array(all_q_scores)
		# remove baselines
		all_kl_scores = all_kl_scores + 5.9914
		all_q_scores = all_q_scores + 8.294
		print (all_kl_scores.shape)
		print (all_q_scores.shape)	

		pickle.dump( all_kl_scores, open( "PO_forward_runs/conditioned/SMC_simulations/data/Qkl-scores-resamp-weights-"+str(file_id)+".p", "wb" ))
		pickle.dump( all_q_scores, open( "PO_forward_runs/conditioned/SMC_simulations//data/ql-scores-resamp-weights-"+str(file_id)+".p", "wb" ))



def collaps_data():

	q_barn = {}
	Q_barn = {}
	file_ids =[1538639858,1538674801,1538675256,1538680844,1538681038,1538687390,1539703235,1539703509,
				1539709650,1539715775,1539715788,1539715836,1539721850,1539728018,1539728576,1539728955,
				1539734254,1539740703,1539740894,1539741096,1539748118,1539752703,1539753150,1539755226,
				1539762160,1539768861,1539775547,1539787505,1539788123,1539793999,1539794919,1539799832,
				1539801294,1539805777,1539807871,1539811558,1539814586,1539817295,1539820829,1539822995,
				1539826901,1539828809,1539832722,1539834533,1539838491,1539844333,1539849974,1539855835,
				1539861654,1539867443,1539873331,1539889386,1539895441,1539898338,1539901461,1539905170,
				1539907884,1539911988,1539913939,1539918800,1539920274,1539925648,1539926235,1539932183,
				1539932581,1539938445,1539939465,1539946246,1539953057,1538687079]


	for file_id in file_ids[:]:
		filename = "PO_forward_runs/conditioned/SMC_simulations/data/ql-scores-resamp-weights-"+str(file_id)+".p"
		all_ql_weights = pickle.load( open(filename, "rb" ) )

		filename = "PO_forward_runs/conditioned/SMC_simulations/data/Qkl-scores-resamp-weights-"+str(file_id)+".p"
		all_Qkl_weights = pickle.load( open(filename, "rb" ) )

		s_ = all_ql_weights.shape
		if s_ not in q_barn:
			q_barn[s_] = []
			Q_barn[s_] = []

		q_barn[s_].append(all_ql_weights)
		Q_barn[s_].append(all_Qkl_weights)

	for key in q_barn.keys():
		nparr = np.array(q_barn[key])
		q_barn[key] = nparr
		Q_barn[key] = np.array(Q_barn[key])
		#print q_barn[key].shape
		#print Q_barn[key].shape

		#q_barn[key] = (logsumexp(q_barn[key], axis=0) - np.log(key[2]))
		#Q_barn[key] = (logsumexp(Q_barn[key], axis=0) - np.log(key[2]))

		#print q_barn[key].shape
		#print Q_barn[key].shape

	return q_barn, Q_barn

def EXPECTATION_plot_resamped_2048_kl_weights_mean_ratio(all_ql_data, all_Qkl_data):
	
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'c']
	i = 0

	for key in all_ql_data.keys():
		all_ql_weights = all_ql_data[key]

		all_Qkl_weights = all_Qkl_data[key]

		ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T
		ql_means = (logsumexp(ql_means, axis=0) - np.log(ql_means.shape[0]))


		qk_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T
		qk_means = (logsumexp(qk_means, axis=0) - np.log(qk_means.shape[0]))
		
		
		ax.plot(list(xrange(1, 29)),qk_means/ql_means,color=colors[i],label="("+str(all_Qkl_weights.shape[1])+ ","+str(all_Qkl_weights.shape[2])+str(")"))
		i +=1

	
	ax.set_xlabel("time step")
	ax.set_ylabel("mean ratio: Cw/Rw")
	ax.set_title("Mean Weight Ratio")
	ax.legend(bbox_to_anchor=(1.1, 1.05))
		
	fig.savefig("EXP-resamp-Qkl-2048-qk-mean-ratio.eps", bbox_inches='tight')



def plot_resamped_2048_kl_weights_mean_ratio():
	file_ids =  [1538639858, 1538674801, 1538675256, 1538680844, 1538681038, 1538687079, 1538687390]
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'c']
	i = 0

		
	for file_id in file_ids[:]:
		filename = "PO_forward_runs/conditioned/SMC_simulations/data/ql-scores-resamp-weights-"+str(file_id)+".p"
		all_ql_weights = pickle.load( open(filename, "rb" ) )
		s_ = all_ql_weights.shape

		ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T
		ql_means = (logsumexp(ql_means, axis=0) - np.log(ql_means.shape[0]))


		filename = "PO_forward_runs/conditioned/SMC_simulations/data/Qkl-scores-resamp-weights-"+str(file_id)+".p"
		all_Qkl_weights = pickle.load( open(filename, "rb" ) )
		qk_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T
		qk_means = (logsumexp(qk_means, axis=0) - np.log(qk_means.shape[0]))
		
		
		
		
		ax.plot(list(xrange(1, 29)),qk_means/ql_means,color=colors[i],label="("+str(all_Qkl_weights.shape[1])+ ","+str(all_Qkl_weights.shape[2])+str(")"))
		i +=1

	
	ax.set_xlabel("time step")
	ax.set_ylabel("mean ratio: Cw/Rw")
	ax.set_title("Mean Weight Ratio")
	ax.legend(bbox_to_anchor=(1.1, 1.05))
		
	fig.savefig("resamp-Qkl-2048-qk-mean-ratio.eps", bbox_inches='tight')


def EXPECTATION_plot_resamped_2048_kl_weights_ess(all_ql_data, all_Qkl_data):
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	colors = ['b', 'g', 'r', 'c', 'm', 'y',  'c']
	i = 0
	for key in all_ql_data.keys():
		all_ql_weights = all_ql_data[key]

		all_Qkl_weights = all_Qkl_data[key]

		
		ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T

		sim = ql_means
		log_normalizer = logsumexp(sim, axis=0) - np.log(sim.shape[0]) 
		w = np.exp(sim - log_normalizer - np.log(sim.shape[0]))
		ess = (1.0/np.sum(w*w,axis=0)) / sim.shape[0]
		ax.plot(list(xrange(1, 29)), ess, colors[i]+str("--"), label="("+str(sim.shape[0])+","+str(all_ql_weights.shape[2])+str(")"))

		qk_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T
		#print qk_means.shape

		#print np.mean(qk_means, axis=0)
		#print qk_means.shape

		sim = qk_means
		log_normalizer = logsumexp(sim, axis=0) - np.log(sim.shape[0]) 
		w = np.exp(sim - log_normalizer - np.log(sim.shape[0]))
		ess = (1.0/np.sum(w*w,axis=0)) / sim.shape[0]
		ax.plot(list(xrange(1, 29)), ess, colors[i]+str("-"), label="("+str(sim.shape[0])+","+str(all_Qkl_weights.shape[2])+str(")"))
		i+=1
		
		#ax.plot(list(xrange(1, 29)),np.mean(qk_means, axis=0)/np.mean(ql_means, axis=0),label="K="+str(all_Qkl_weights.shape[1]))
		#ax.plot(list(xrange(1, 29)),np.mean(qk_means, axis=0),label="K="+str(all_Qkl_weights.shape[1]))
	ax.set_title("Effective Sample Size")
	ax.set_xlabel("time step")
	ax.set_ylabel("ESS (fraction)")
	ax.legend(bbox_to_anchor=(1.1, 1.05))
		#fig.savefig("resamp-QKL-2048-ESS_1.eps", bbox_inches='tight')
	#ax.set_ylim(-10, -5)
	fig.savefig("EXP-resamp-Qkl-2048-ql-ess.eps", bbox_inches='tight')




def plot_resamped_2048_kl_weights_ess():
	#file_ids = [1538440198, 1538448489, 1538499742, 1538462807, 1538470154, 1538477199, 1538484792, 1538496244]
	file_ids = [1538532620, 1538539386, 1538545347, 1538551100, 1538556818, 1538562473, 1538568646]
	file_ids = [1538609735, 1538616566, 1538609635,1538622239, 1538615745]# 1538622917] #1538618520 64x32 miss
	file_ids =  [1538639858, 1538674801, 1538675256, 1538680844, 1538681038, 1538687079, 1538687390]
	#file_ids =  [1538674801, 1538675256, 1538680844, 1538681038]
	
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	
	colors = ['b', 'g', 'r', 'c', 'm', 'y',  'c']
	i = 0
	for file_id in file_ids[:]:
		
		filename = "ql-scores-resamp-weights-"+str(file_id)+".p"
		all_ql_weights = pickle.load( open(filename, "rb" ) )
		ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T

		sim = ql_means
		log_normalizer = logsumexp(sim, axis=0) - np.log(sim.shape[0]) 
		w = np.exp(sim - log_normalizer - np.log(sim.shape[0]))
		ess = (1.0/np.sum(w*w,axis=0)) / sim.shape[0]
		ax.plot(list(xrange(1, 29)), ess, colors[i]+str("--"), label="("+str(sim.shape[0])+","+str(all_ql_weights.shape[2])+str(")"))

		filename = "Qkl-scores-resamp-weights-"+str(file_id)+".p"
		all_Qkl_weights = pickle.load( open(filename, "rb" ) )
		print all_Qkl_weights.shape
		qk_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T
		print qk_means.shape

		print np.mean(qk_means, axis=0)
		print qk_means.shape

		sim = qk_means
		log_normalizer = logsumexp(sim, axis=0) - np.log(sim.shape[0]) 
		w = np.exp(sim - log_normalizer - np.log(sim.shape[0]))
		ess = (1.0/np.sum(w*w,axis=0)) / sim.shape[0]
		ax.plot(list(xrange(1, 29)), ess, colors[i]+str("-"), label="("+str(sim.shape[0])+","+str(all_Qkl_weights.shape[2])+str(")"))
		i+=1
		
		#ax.plot(list(xrange(1, 29)),np.mean(qk_means, axis=0)/np.mean(ql_means, axis=0),label="K="+str(all_Qkl_weights.shape[1]))
		#ax.plot(list(xrange(1, 29)),np.mean(qk_means, axis=0),label="K="+str(all_Qkl_weights.shape[1]))
	ax.set_title("Effective Sample Size")
	ax.set_xlabel("time step")
	ax.set_ylabel("ESS (fraction)")
	ax.legend(bbox_to_anchor=(1.1, 1.05))
		#fig.savefig("resamp-QKL-2048-ESS_1.eps", bbox_inches='tight')
	#ax.set_ylim(-10, -5)
	fig.savefig("resamp-Qkl-2048-ql-ess.eps", bbox_inches='tight')

def EXPECTATION_plot_resamped_2048_kl_weights_var(all_ql_data, all_Qkl_data):
	
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'c']
	i = 0

	for key in all_ql_data.keys():
		all_ql_weights = all_ql_data[key]

		all_Qkl_weights = all_Qkl_data[key]

		qk_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T
		

		fig = plt.figure(figsize=(4,4))
		fig.clf()
		ax = fig.add_subplot(1, 1, 1)

		x = list(xrange(1, 29))
		y = (logsumexp(qk_means, axis=0) - np.log(qk_means.shape[0])) # np.mean(qk_means, axis=0)
		e = np.var(qk_means, axis=0)
		#e = [1]*28
		ax = sns.boxplot(data=qk_means)
		#ax.errorbar(x, y, e, linestyle='None', marker='^', color=colors[i],label="("+str(all_Qkl_weights.shape[1])+ ","+str(all_Qkl_weights.shape[2])+str(")"))
		i=i+1

		ax.set_xlabel("time step")
		ax.set_ylabel("weight variance}")
		ax.legend(bbox_to_anchor=(1.1, 1.05))
		ax.set_title("("+str(all_Qkl_weights.shape[1])+ ","+str(all_Qkl_weights.shape[2])+str(") ")+" Cw Mean and Variances ")
		ax.set_ylim(-30, 10)
			#fig.savefig("resamp-QKL-2048-ESS_1.eps", bbox_inches='tight')
		#ax.set_ylim(-10, -5)
		fig.savefig("EXP-resamp-Qkl-2048-qk-var"+str(all_Qkl_weights.shape[1])+".eps", bbox_inches='tight')

def plot_resamped_2048_kl_weights_var():
	#file_ids = [1538440198, 1538448489, 1538499742, 1538462807, 1538470154, 1538477199, 1538484792, 1538496244]
	file_ids = [1538532620, 1538539386, 1538545347, 1538551100, 1538556818, 1538562473, 1538568646]
	file_ids = [1538609735, 1538616566, 1538609635,1538622239, 1538615745]#, 1538622917] #1538618520 64x32 miss
	file_ids =  [1538639858, 1538674801, 1538675256, 1538680844, 1538681038, 1538687079, 1538687390]
	
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	colors = ['b', 'g', 'r', 'c', 'm', 'y',  'c']
	i = 0

	for file_id in file_ids[:]:
		#--------------------------------
		filename = "Qkl-scores-resamp-weights-"+str(file_id)+".p"
		all_Qkl_weights = pickle.load( open(filename, "rb" ) )
		qk_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T
		print qk_means.shape

		fig = plt.figure(1)
		fig.clf()
		ax = fig.add_subplot(1, 1, 1)

		x = list(xrange(1, 29))
		y = (logsumexp(qk_means, axis=0) - np.log(qk_means.shape[0])) # np.mean(qk_means, axis=0)
		e = np.var(qk_means, axis=0)
		#e = [1]*28
		ax = sns.boxplot(data=qk_means)
		#ax.errorbar(x, y, e, linestyle='None', marker='^', color=colors[i],label="("+str(all_Qkl_weights.shape[1])+ ","+str(all_Qkl_weights.shape[2])+str(")"))
		i=i+1

		ax.set_xlabel("time step")
		ax.set_ylabel("weight variance}")
		ax.legend(bbox_to_anchor=(1.1, 1.05))
		ax.set_title("("+str(all_Qkl_weights.shape[1])+ ","+str(all_Qkl_weights.shape[2])+str(") ")+" Cw Mean and Variances ")
		ax.set_ylim(-30, 10)
			#fig.savefig("resamp-QKL-2048-ESS_1.eps", bbox_inches='tight')
		#ax.set_ylim(-10, -5)
		fig.savefig("resamp-Qkl-2048-qk-var"+str(all_Qkl_weights.shape[1])+".eps", bbox_inches='tight')


def plot_resamped_2048_l_weights_var():
	#file_ids = [1538440198, 1538448489, 1538499742, 1538462807, 1538470154, 1538477199, 1538484792, 1538496244]
	file_ids = [1538532620, 1538539386, 1538545347, 1538551100, 1538556818, 1538562473, 1538568646]
	file_ids = [1538609735, 1538616566, 1538609635,1538622239, 1538615745]#, 1538622917] #1538618520 64x32 miss
	file_ids =  [1538639858, 1538674801, 1538675256, 1538680844, 1538681038, 1538687079, 1538687390]
	
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'c']
	#colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
	i = 0

	for file_id in file_ids[:]:
		
		
		filename = "ql-scores-resamp-weights-"+str(file_id)+".p"
		all_ql_weights = pickle.load( open(filename, "rb" ) )
		ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T
		#ql_means = (logsumexp(ql_means, axis=0) - np.log(ql_means.shape[0]))
		
		fig = plt.figure(1)
		fig.clf()
		ax = fig.add_subplot(1, 1, 1)

		x = list(xrange(1, 29))
		y = (logsumexp(ql_means, axis=0) - np.log(ql_means.shape[0])) # np.mean(qk_means, axis=0)
		e = np.var(ql_means, axis=0)
		#e = [1]*28

		#ax.errorbar(x, y, e, linestyle='None', marker='^', color=colors[i],label="("+str(all_ql_weights.shape[1])+ ","+str(all_ql_weights.shape[2])+str(")"))
		sns.set(font_scale=0.8)
		ax = sns.boxplot(data=ql_means)
		
		#ax.add_subplot(x, y, e)
		i=i+1
		
		ax.set_xlabel("time step")
		ax.set_ylabel("weight variance")
		ax.legend(bbox_to_anchor=(1.1, 1.05))
		#ax.set_title("Rw Mean and Variances ")
		ax.set_title("("+str(all_ql_weights.shape[1])+ ","+str(all_ql_weights.shape[2])+str(") ")+" Rw Mean and Variances ")
		
			#fig.savefig("resamp-QKL-2048-ESS_1.eps", bbox_inches='tight')
		ax.set_ylim(-30, 10)
		fig.savefig("resamp-Qkl-2048-q-var"+str(all_ql_weights.shape[1])+".eps", bbox_inches='tight')


def plot_resamped_2048_kl_weights_mean():
	#file_ids = [1538440198, 1538448489, 1538499742, 1538462807, 1538470154, 1538477199, 1538484792, 1538496244]
	file_ids = [1538532620, 1538539386, 1538545347, 1538551100, 1538556818, 1538562473, 1538568646]
	file_ids = [1538609735, 1538616566, 1538609635,1538622239, 1538615745]#, 1538622917] #1538618520 64x32 miss
	file_ids = [1538639858, 1538674801, 1538675256, 1538680844, 1538681038, 1538687079, 1538687390]
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)

	colors = ['b', 'g', 'r', 'c', 'm', 'y',  'c']
	i = 0
	for file_id in file_ids[:]:
		#filename = "Qkl-scores-resamp-weights-"+str(file_id)+".p"
		#-----------------
		filename = "ql-scores-resamp-weights-"+str(file_id)+".p"
		all_ql_weights = pickle.load( open(filename, "rb" ) )
		print ("all_ql_weights:", all_ql_weights.shape)
		ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T


		filename = "Qkl-scores-resamp-weights-"+str(file_id)+".p"
		all_Qkl_weights = pickle.load( open(filename, "rb" ) )
		qk_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T
		print qk_means.shape

		
		ax.plot(list(xrange(1, 29)),(logsumexp(qk_means, axis=0) - np.log(qk_means.shape[0])) , colors[i]+str("-"), label="Cw ("+str(all_Qkl_weights.shape[1])+","+str(all_Qkl_weights.shape[2])+str(")"))
		ax.plot(list(xrange(1, 29)),(logsumexp(ql_means, axis=0) - np.log(ql_means.shape[0])) , colors[i]+str("--"), label="Rw ("+str(all_Qkl_weights.shape[1])+","+str(all_Qkl_weights.shape[2])+str(")"))
		i = i+1

	ax.set_xlabel("time step")
	ax.set_ylabel("mean weights")
	ax.legend(bbox_to_anchor=(1.1, 1.05))
	fig.savefig("resamp-Qkl-2048-qk-means.eps", bbox_inches='tight')

import matplotlib.pylab as pl


def EXP_grid_plot_lines_vars(all_ql_data, all_Qkl_data):
	fig = plt.figure(figsize=(6,12))
	fig.clf()
	
	rows = 7
	cols = 2
	colors = ['b', 'g', 'r', 'c', 'm', 'y',  'c']
	colors = ['darkblue','dodgerblue', 'darkorchid', 'mediumseagreen','indianred', 'orange','lightgrey']
	#ordered_keys= [(28, 2048, 1), (28, 512, 4), (28, 32, 64), (28, 4, 512), (28, 16, 128), (28, 64, 32), (28, 128, 16)]
	ordered_keys= [(28, 2048, 1), (28, 512, 4), (28, 128, 16), (28, 64, 32), (28, 32, 64), (28, 16, 128), (28, 4, 512)]

	lw =0.8
	blw = 1.5
	#colors = pl.cm.rainbow(np.linspace(0,1,len(file_ids)))
	print colors
	print all_ql_data.keys()
	ax = fig.add_subplot(rows, cols, 1)
	
	spot = 1
	i=0
	S = 2
	for key in ordered_keys:

		ax = fig.add_subplot(rows, cols, spot)

		all_Qkl_weights = all_Qkl_data[key]
		qk_means = (logsumexp(all_Qkl_weights, axis=3) - np.log(all_Qkl_weights.shape[3])) #(10, 28, K)
		qk_means = qk_means.swapaxes(0,1) # (28, 10, 512)
		qk_means = qk_means.reshape((28, 10 * all_Qkl_weights.shape[2])) #(28, 10 * K)
		data = qk_means
		print data.shape
		qk_means = (logsumexp(qk_means, axis=1) - np.log(qk_means.shape[1])) #( 28,)
		
		qk_means_2 = (logsumexp(2*data, axis=1) - np.log(data.shape[1])) #( 28,)
		qk_var = np.log(np.exp(qk_means_2) - np.exp(qk_means)**2)


		#ax.set_ylim(-10, 10) # must uncomment
		ax.set_title("Chaser")
		ax.get_xaxis().set_ticks([])
		ax.plot(list(xrange(1, 29)),qk_means, color=colors[i], linewidth=blw)  # must uncomment
		
		print "var-------------"
		print np.absolute(qk_var)
		#print np.sqrt(np.absolute(qk_var))
		sd = np.sqrt(np.absolute(qk_var))
		ax.set_ylabel("K="+str(key[1])+" L="+str(key[2]))


		ax.fill_between(list(xrange(1,29)), qk_means - S*sd, qk_means + S*sd, facecolor=colors[i], alpha=0.6)# facecolor=colors[i], alpha=0.1)

		
		spot +=1
		#---------------------------------------------------------------------------------

		ax = fig.add_subplot(rows, cols, spot)
		all_ql_weights = all_ql_data[key]

		ql_means = (logsumexp(all_ql_weights, axis=3) - np.log(all_ql_weights.shape[3])) #(10, 28, K)
		ql_means = ql_means.swapaxes(0,1) # (28, 10, 512)
		ql_means = ql_means.reshape((28, 10 * all_ql_weights.shape[2])) #(28, 10 * K)
		data = ql_means

		ql_means = (logsumexp(ql_means, axis=1) - np.log(ql_means.shape[1]))

		ql_means_2 = (logsumexp(2*data, axis=1) - np.log(data.shape[1])) #( 28,)
		ql_var = np.log(np.exp(ql_means_2) - np.exp(ql_means)**2)
		sd = np.sqrt(np.absolute(qk_var))
		ax.plot(list(xrange(1, 29)), ql_means, color=colors[i], linewidth=blw)
		ax.set_title("Runner")
		ax.set_ylabel("$\mathrm{\mathbb{E}}$(Log Mean Weight)")
		ax.set_ylim(-10, 1) 
		#ax.fill_between(list(xrange(1,29)), ql_means - S*sd, ql_means + S*sd, facecolor=colors[i], alpha=0.6)# facecolor=colors[i], alpha=0.1)

		ax.fill_between(list(xrange(1,29)), ql_means - S*sd, ql_means , facecolor=colors[i], alpha=0.6)# facecolor=colors[i], alpha=0.1)


		#ax.set_ylim(-1.3, -0.25)



		i+=1
		spot +=1

	plt.tight_layout()
	#fig.savefig("EXP-particle_exp_updated.eps", bbox_inches='tight')
	fig.savefig("EXP-particle_exp_with_vars.SVG", bbox_inches='tight')



def EXP_grid_plot_lines(all_ql_data, all_Qkl_data):
	fig = plt.figure(figsize=(6,6))
	fig.clf()
	
	rows = 3
	cols = 2
	colors = ['b', 'g', 'r', 'c', 'm', 'y',  'c']
	colors = ['darkblue','dodgerblue', 'darkorchid', 'mediumseagreen','indianred', 'orange','lightgrey']
	#ordered_keys= [(28, 2048, 1), (28, 512, 4), (28, 32, 64), (28, 4, 512), (28, 16, 128), (28, 64, 32), (28, 128, 16)]
	ordered_keys= [(28, 2048, 1), (28, 512, 4), (28, 128, 16), (28, 64, 32), (28, 32, 64), (28, 16, 128), (28, 4, 512)]

	lw =0.8
	blw = 1.5
	#colors = pl.cm.rainbow(np.linspace(0,1,len(file_ids)))
	print colors
	print all_ql_data.keys()
	ax = fig.add_subplot(rows, cols, 1)
	
	i=0
	for key in ordered_keys:

		all_Qkl_weights = all_Qkl_data[key]
		qk_means = (logsumexp(all_Qkl_weights, axis=3) - np.log(all_Qkl_weights.shape[3])) #(10, 28, K)
		qk_means = qk_means.swapaxes(0,1) # (28, 10, 512)
		qk_means = qk_means.reshape((28, 10 * all_Qkl_weights.shape[2])) #(28, 10 * K)
		data = qk_means
		print data.shape
		qk_means = (logsumexp(qk_means, axis=1) - np.log(qk_means.shape[1])) #( 28,)
		
		qk_means_2 = (logsumexp(2*data, axis=1) - np.log(data.shape[1])) #( 28,)
		qk_var = np.log(np.exp(qk_means_2) - np.exp(qk_means)**2)


		#ax.set_ylim(-10, 10) # must uncomment
		ax.set_title("Chaser")
		ax.get_xaxis().set_ticks([])
		ax.plot(list(xrange(1, 29)),qk_means, color=colors[i], linewidth=blw)  # must uncomment

		#print "var-------------"
		#print np.absolute(qk_var)
		#print np.sqrt(np.absolute(qk_var))
		#sd = np.sqrt(np.absolute(qk_var))


		#ax.fill_between(list(xrange(1,29)), qk_means - 2*sd, qk_means + 2*sd, facecolor=colors[i], alpha=0.9)# facecolor=colors[i], alpha=0.1)


		i+=1

	ax = fig.add_subplot(rows, cols, 2)	
	i = 0
	for key in ordered_keys:
		all_ql_weights = all_ql_data[key]
		#ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T
		
		ql_means = (logsumexp(all_ql_weights, axis=3) - np.log(all_ql_weights.shape[3])) #(10, 28, 512)
		ql_means = ql_means.swapaxes(0,1) # (28, 10, 512)
		ql_means = ql_means.reshape((28, 10 * all_ql_weights.shape[2])) #(10 * 512)

		ql_means = (logsumexp(ql_means, axis=1) - np.log(ql_means.shape[1]))

		#print "--ql_means.shape", ql_means.shape
		ax.set_ylim(-1.3, -0.25)
		
		#ax.set_ylabel("E(Log Mean Weight)")
		ax.set_ylabel("$\mathrm{\mathbb{E}}$(Log Mean Weight)")
		ax.set_title("Runner")
		ax.get_xaxis().set_ticks([])
		ax.plot(list(xrange(1, 29)), ql_means, color=colors[i], linewidth=blw)
		i+=1
	

	
	
	ax = fig.add_subplot(rows, cols, 3)
	i=0
	line_list = []
	for key in ordered_keys:
		all_Qkl_weights = all_Qkl_data[key]
		
		qk_means = (logsumexp(all_Qkl_weights, axis=3) - np.log(all_Qkl_weights.shape[3])) #(10, 28, 512)
		qk_means = qk_means.swapaxes(0,1) # (28, 10, 512)
		qk_means = qk_means.reshape((28, 10 * all_Qkl_weights.shape[2])).T #(28, 10 * 512)
		



		sim = qk_means
		log_normalizer = logsumexp(sim, axis=0) - np.log(sim.shape[0]) 
		w = np.exp(sim - log_normalizer - np.log(sim.shape[0]))
		ess = (1.0/np.sum(w*w,axis=0)) / sim.shape[0]
		#ax.set_xlabel("Time Step")
		#ax.get_yaxis().set_ticks([])
		ax.get_xaxis().set_ticks([])
		ax.set_ylim(0, 1)
		line = ax.plot(list(xrange(1, 29)), ess, linewidth=blw, color=colors[i])
		line_list.append(line)
		i+=1



	ax = fig.add_subplot(rows, cols, 4)
	i=0
	for key in ordered_keys:
		all_ql_weights = all_ql_data[key] #(10, 28, 512, 4)

		ql_means = (logsumexp(all_ql_weights, axis=3) - np.log(all_ql_weights.shape[3])) #(10, 28, 512)
		ql_means = ql_means.swapaxes(0,1) # (28, 10, 512)
		ql_means = ql_means.reshape((28, 10 * all_ql_weights.shape[2])).T #(28, 10 * 512)
		

		sim = ql_means
		log_normalizer = logsumexp(sim, axis=0) - np.log(sim.shape[0]) 
		w = np.exp(sim - log_normalizer - np.log(sim.shape[0]))
		ess = (1.0/np.sum(w*w,axis=0)) / sim.shape[0]

		ax.plot(list(xrange(1, 29)), ess, linewidth=blw, color=colors[i])
		ax.set_ylabel("ESS/K")
		#ax.set_xlabel("Time Step")
		ax.set_ylim(0, 1)
		i+=1

	ax = fig.add_subplot(rows, cols, 5)
	i=0
	line_list = []
	for key in ordered_keys:
		all_Qkl_weights = all_Qkl_data[key]
		
		qk_means = (logsumexp(all_Qkl_weights, axis=3) - np.log(all_Qkl_weights.shape[3])) #(10, 28, 512)
		qk_means = qk_means.swapaxes(0,1) # (28, 10, 512)
		qk_means = qk_means.reshape((28, 10 * all_Qkl_weights.shape[2])).T #(28, 10 * 512)
		



		sim = qk_means
		log_normalizer = logsumexp(sim, axis=0) - np.log(sim.shape[0]) 
		w = np.exp(sim - log_normalizer - np.log(sim.shape[0]))
		ess = (1.0/np.sum(w*w,axis=0)) 
		ax.set_xlabel("Time Step")
		#ax.get_yaxis().set_ticks([])
		#ax.set_ylim(0, 1)
		line = ax.plot(list(xrange(1, 29)), ess, linewidth=blw, color=colors[i])


    
		line_list.append(line)
		i+=1



	ax = fig.add_subplot(rows, cols, 6)
	i=0
	for key in ordered_keys:
		all_ql_weights = all_ql_data[key] #(10, 28, 512, 4)

		ql_means = (logsumexp(all_ql_weights, axis=3) - np.log(all_ql_weights.shape[3])) #(10, 28, 512)
		ql_means = ql_means.swapaxes(0,1) # (28, 10, 512)
		ql_means = ql_means.reshape((28, 10 * all_ql_weights.shape[2])).T #(28, 10 * 512)
		

		#ql_means = (logsumexp(ql_means, axis=1) - np.log(ql_means.shape[1])) # (28,)
		print "** ql_means", ql_means.shape


		#ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T

		sim = ql_means
		log_normalizer = logsumexp(sim, axis=0) - np.log(sim.shape[0]) 
		w = np.exp(sim - log_normalizer - np.log(sim.shape[0]))
		ess = (1.0/np.sum(w*w,axis=0)) 

		ax.plot(list(xrange(1, 29)), ess, linewidth=blw, color=colors[i])
		ax.set_ylabel("ESS")
		ax.set_xlabel("Time Step")
		#ax.set_ylim(0, 1)
		i+=1


	


	plt.tight_layout()
	#fig.savefig("EXP-particle_exp_updated.eps", bbox_inches='tight')
	fig.savefig("EXP-particle_exp_updated.SVG", bbox_inches='tight')
	
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def grid_plot_lines():
	file_ids = [1538639858, 1538674801,  1538680844, 1538687079, 1538675256, 1538681038, 1538687390]
	
	fig = plt.figure(figsize=(6,4.5))
	fig.clf()
	
	rows = 2
	cols = 2
	colors = ['b', 'g', 'r', 'c', 'm', 'y',  'c']
	colors = ['darkblue','dodgerblue', 'darkorchid', 'mediumseagreen','indianred', 'orange','lightgrey']
	lw =0.8
	blw = 1.5
	#colors = pl.cm.rainbow(np.linspace(0,1,len(file_ids)))
	print colors
	ax = fig.add_subplot(rows, cols, 1)
	i = 0
	for file_id in file_ids[:]:
		filename = "ql-scores-resamp-weights-"+str(file_id)+".p"
		all_ql_weights = pickle.load( open(filename, "rb" ) )
		ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T
		ax.set_ylim(-10, 10)
		ax.set_ylabel("(Log Mean Weight)")
		ax.set_title("Runner")
		ax.get_xaxis().set_ticks([])
		ax.plot(list(xrange(1, 29)),(logsumexp(ql_means, axis=0) - np.log(ql_means.shape[0])), color=colors[i], linewidth=blw)
		i+=1
	

	ax = fig.add_subplot(rows, cols, 2)
	i=0
	for file_id in file_ids[:]:
		filename = "Qkl-scores-resamp-weights-"+str(file_id)+".p"
		all_Qkl_weights = pickle.load( open(filename, "rb" ) )
		qk_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T
		ax.set_ylim(-10, 10)
		ax.set_title("Chaser")
		ax.get_xaxis().set_ticks([])
		ax.get_yaxis().set_ticks([])
		ax.plot(list(xrange(1, 29)),(logsumexp(qk_means, axis=0) - np.log(qk_means.shape[0])), color=colors[i], linewidth=blw)
		i+=1
	

	ax = fig.add_subplot(rows, cols, 3)
	i=0
	for file_id in file_ids[:]:
		
		filename = "ql-scores-resamp-weights-"+str(file_id)+".p"
		all_ql_weights = pickle.load( open(filename, "rb" ) )
		ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T

		sim = ql_means
		log_normalizer = logsumexp(sim, axis=0) - np.log(sim.shape[0]) 
		w = np.exp(sim - log_normalizer - np.log(sim.shape[0]))
		ess = (1.0/np.sum(w*w,axis=0)) / sim.shape[0]
		ax.plot(list(xrange(1, 29)), ess, linewidth=blw, color=colors[i])
		ax.set_ylabel("ESS/K")
		ax.set_xlabel("Time Step")
		ax.set_ylim(0, 1)
		i+=1

	ax = fig.add_subplot(rows, cols, 4)
	i=0
	line_list = []
	for file_id in file_ids[:]:
		
		filename = "Qkl-scores-resamp-weights-"+str(file_id)+".p"
		all_Qkl_weights = pickle.load( open(filename, "rb" ) )
		qk_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T

		sim = qk_means
		log_normalizer = logsumexp(sim, axis=0) - np.log(sim.shape[0]) 
		w = np.exp(sim - log_normalizer - np.log(sim.shape[0]))
		ess = (1.0/np.sum(w*w,axis=0)) / sim.shape[0]
		ax.set_xlabel("Time Step")
		ax.get_yaxis().set_ticks([])
		ax.set_ylim(0, 1)
		line = ax.plot(list(xrange(1, 29)), ess, linewidth=blw, color=colors[i])
		line_list.append(line)
		i+=1

# l1, l2 = ax1.plot(x, y1, 'rs-', x, y2, 'go')

# y3 = np.sin(4*np.pi*x)
# y4 = np.exp(-2*x)
# l3, l4 = ax2.plot(x, y3, 'yd-', x, y4, 'k^')

	#fig.legend(line_list, ('Line 1', 'Line 2', 'Line 2', 'Line 2', 'Line 2', 'Line 2', 'Line 2'), 'upper left')
# fig.legend((l3, l4), ('Line 3', 'Line 4'), 'upper right')
# plt.show()



	plt.tight_layout()
	#plt.figlegend( line_list, ('Line 1', 'Line 2', 'Line 2', 'Line 2', 'Line 2', 'Line 2', 'Line 2'), loc = 'lower center' )
	#ax.set_ylabel("mean weights")
	#ax.legend(bbox_to_anchor=(1.1, 1.05))
	#ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
	fig.savefig("particle_exp.eps", bbox_inches='tight')
	#plt.show()

		
# Approach 1:
# 10 x 28 x K x L  -> 10 x ( 10*K*L )
#
def EXP_A1_grid_plot_box(all_ql_data, all_Qkl_data):

	fig = plt.figure(figsize=(6,12))
	fig.clf()
	
	rows = 7
	cols = 2
	colors = ['b', 'g', 'r', 'c', 'm', 'y',  'c']
	colors = ['darkblue','dodgerblue', 'darkorchid', 'mediumseagreen','indianred', 'orange','lightgrey']
	lw = 0.8
	blw = 1.5
	#colors = pl.cm.rainbow(np.linspace(0,1,len(file_ids)))

	r = 1
	c = 2
	ordered_keys = [(28, 2048, 1), (28, 512, 4), (28, 128, 16), (28, 64, 32), (28, 32, 64), (28, 16, 128), (28, 4, 512)]
	for key in ordered_keys:#all_ql_data.keys():
		
		ax = fig.add_subplot(rows, cols, r)
		
		all_Qkl_weights = all_Qkl_data[key]
		print "all_qk_weights", all_Qkl_weights.shape #(10, 28, 512, 4)
		qk_means = all_Qkl_weights.swapaxes(0,1)
		qk_means = qk_means.reshape((28, 10 * 2048)).T #(10 * 2018, 28)
		print "qk_means", qk_means.shape
		ax = sns.boxplot(data=qk_means, linewidth=lw)
		ax.get_xaxis().set_ticks([])
		ax.get_yaxis().set_ticks([])
		ax.set_ylabel("Log Weight")
		ax.set_title("K = " + str(key[1]))
		ax.set_ylim(-17, 17)



		r += 1
		ax = fig.add_subplot(rows, cols, r)

		all_ql_weights = all_ql_data[key] # (10, 28, 512, 4)
		print "all_ql_weights", all_ql_weights.shape
		ql_means = all_ql_weights.swapaxes(0,1)
		ql_means = ql_means.reshape((28, 10 * 2048)).T 
		print "ql_means", ql_means.shape
		ax = sns.boxplot(linewidth=lw, data=ql_means)
		ax.set_title("L = " + str(key[2]))
		ax.get_xaxis().set_ticks([])
		ax.set_ylim(-17, 17)

		r += 1


	plt.tight_layout()
	fig.savefig("EXP-A1-particle_var.eps", bbox_inches='tight')


def EXP_A2_grid_plot_box(all_ql_data, all_Qkl_data):

	fig = plt.figure(figsize=(6,12))
	fig.clf()
	
	rows = 7
	cols = 2
	colors = ['b', 'g', 'r', 'c', 'm', 'y',  'c']
	colors = ['darkblue','dodgerblue', 'darkorchid', 'mediumseagreen','indianred', 'orange','lightgrey']
	lw = 0.8
	blw = 1.5
	#colors = pl.cm.rainbow(np.linspace(0,1,len(file_ids)))

	r = 1
	c = 2
	ordered_keys = [(28, 2048, 1), (28, 512, 4), (28, 128, 16), (28, 64, 32), (28, 32, 64), (28, 16, 128), (28, 4, 512)]
	for key in ordered_keys:#all_ql_data.keys():
		print r, c
		

		
		ax = fig.add_subplot(rows, cols, r)
		
		all_Qkl_weights = all_Qkl_data[key]
		print "all_qk_weights", all_Qkl_weights.shape #(10, 28, 512, 4)
		qk_means = (logsumexp(all_Qkl_weights, axis=3) - np.log(all_Qkl_weights.shape[3])) #(10, 28, 512)
		qk_means = qk_means.swapaxes(0,1)
		qk_means = qk_means.reshape((28, 10 * all_Qkl_weights.shape[2])).T #(10 * 512)
		print "qk_means", qk_means.shape
		ax = sns.boxplot(data=qk_means, linewidth=lw, fliersize=2)
		ax.get_xaxis().set_ticks([])
		ax.get_yaxis().set_ticks([])
		ax.set_ylabel("Log Weight")
		ax.set_title("K = " + str(key[1]))
		ax.set_ylim(-17, 17)

		r += 1

		ax = fig.add_subplot(rows, cols, r)

		all_ql_weights = all_ql_data[key] # (10, 28, 512, 4)
		print "all_ql_weights", all_ql_weights.shape
		ql_means = (logsumexp(all_ql_weights, axis=3) - np.log(all_ql_weights.shape[3])) #(10, 28, 512)
		#ql_means = ql_means.reshape((28, all_ql_weights.shape[2], 10))
		ql_means = ql_means.swapaxes(0,1)
		ql_means = ql_means.reshape((28, 10 * all_ql_weights.shape[2])).T #ql_means (5120, 28)
		print "ql_means", ql_means.shape
		ax = sns.boxplot(linewidth=lw, data=ql_means, fliersize=2)
		
		ax.set_title("L = " + str(key[2]))
		ax.get_xaxis().set_ticks([])
		ax.set_ylim(-10, 1)

		r += 1


	plt.tight_layout()
	fig.savefig("EXP-A2-particle_var.SVG", bbox_inches='tight')


def EXP_A3_grid_plot_box(all_ql_data, all_Qkl_data):

	fig = plt.figure(figsize=(6,18))
	fig.clf()
	
	rows = 7
	cols = 2
	colors = ['b', 'g', 'r', 'c', 'm', 'y',  'c']
	colors = ['darkblue','dodgerblue', 'darkorchid', 'mediumseagreen','indianred', 'orange','lightgrey']
	lw = 0.8
	blw = 1.5
	#colors = pl.cm.rainbow(np.linspace(0,1,len(file_ids)))

	r = 1
	c = 2
	ordered_keys = [(28, 2048, 1), (28, 512, 4), (28, 128, 16), (28, 64, 32), (28, 32, 64), (28, 16, 128), (28, 4, 512)]
	for key in ordered_keys:#all_ql_data.keys():
		print r, c
		

		
		ax = fig.add_subplot(rows, cols, r)
		
		all_Qkl_weights = all_Qkl_data[key]
		print "all_qk_weights", all_Qkl_weights.shape #(10, 28, 512, 4)
		qk_means = (logsumexp(all_Qkl_weights, axis=3) - np.log(all_Qkl_weights.shape[3])) #(10, 28, 512)
		qk_means = (logsumexp(qk_means, axis=2) - np.log(all_Qkl_weights.shape[2])) #(10, 28)
		ax = sns.boxplot(data=qk_means, linewidth=lw)
		ax.get_xaxis().set_ticks([])
		ax.get_yaxis().set_ticks([])
		ax.set_ylabel("Log Weight")
		ax.set_title("K = " + str(key[1]))
		ax.set_ylim(-17, 17)

		r += 1

		ax = fig.add_subplot(rows, cols, r)

		all_ql_weights = all_ql_data[key] # (10, 28, 512, 4)
		print "all_ql_weights", all_ql_weights.shape
		ql_means = (logsumexp(all_ql_weights, axis=3) - np.log(all_ql_weights.shape[3])) #(10, 28, 512)
		ql_means = (logsumexp(ql_means, axis=2) - np.log(all_ql_weights.shape[2]))
		print "ql_means", ql_means.shape
		ax = sns.boxplot(linewidth=lw, data=ql_means)
		
		ax.set_title("L = " + str(key[2]))
		ax.get_xaxis().set_ticks([])
		ax.set_ylim(-1.75, 0.25)

		r += 1


	plt.tight_layout()
	fig.savefig("EXP-A3-particle_var.eps", bbox_inches='tight')


def grid_plot_box():
	file_ids = [1538639858, 1538674801,  1538680844, 1538687079, 1538675256, 1538681038, 1538687390]
	
	fig = plt.figure(figsize=(6,6))
	fig.clf()
	
	rows = 3
	cols = 2
	colors = ['b', 'g', 'r', 'c', 'm', 'y',  'c']
	colors = ['darkblue','dodgerblue', 'darkorchid', 'mediumseagreen','indianred', 'orange','lightgrey']
	lw =0.8
	blw = 1.5
	#colors = pl.cm.rainbow(np.linspace(0,1,len(file_ids)))
	file_id = 1538674801
	ax = fig.add_subplot(rows, cols, 1)
	
	filename = "ql-scores-resamp-weights-"+str(file_id)+".p"
	all_ql_weights = pickle.load( open(filename, "rb" ) )
	#print (all_ql_weights.shape)
	ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T
	#print (ql_means.shape)
	#ql_mean = all_ql_weights.reshape((28,2048))
	ax = sns.boxplot(data=ql_means, linewidth=lw)
	ax.set_ylabel("Log Weight")
	ax.set_title("L = 4")
	ax.get_xaxis().set_ticks([])
	ax.set_ylim(-17, 17)


	ax = fig.add_subplot(rows, cols, 2)
	
	filename = "Qkl-scores-resamp-weights-"+str(file_id)+".p"
	all_Qkl_weights = pickle.load( open(filename, "rb" ) )
	qk_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T
	ax = sns.boxplot(data=qk_means, linewidth=lw)
	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().set_ticks([])
	ax.set_title("K = 512")
	ax.set_ylim(-17, 17)

	

	#-------------------------------------------------
	file_id = 1538680844
	ax = fig.add_subplot(rows, cols, 3)
	
	filename = "ql-scores-resamp-weights-"+str(file_id)+".p"
	all_ql_weights = pickle.load( open(filename, "rb" ) )
	ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T
	#ql_mean = all_ql_weights.reshape((28,2048))
	ax = sns.boxplot(data=ql_means, linewidth=lw)
	ax.set_ylabel("Log Weight")
	ax.set_title("L = 16")
	ax.get_xaxis().set_ticks([])
	ax.set_ylim(-17, 17)



	ax = fig.add_subplot(rows, cols, 4)
	
	filename = "Qkl-scores-resamp-weights-"+str(file_id)+".p"
	all_Qkl_weights = pickle.load( open(filename, "rb" ) )
	qk_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T
	ax = sns.boxplot(data=qk_means, linewidth=lw)
	ax.set_title("K = 128")
	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().set_ticks([])
	ax.set_ylim(-17, 17)

	#----------------------------------------------------
	file_id = 1538681038
	ax = fig.add_subplot(rows, cols, 5)
	
	filename = "ql-scores-resamp-weights-"+str(file_id)+".p"
	all_ql_weights = pickle.load( open(filename, "rb" ) )
	ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T
	#ql_mean = all_ql_weights.reshape((28,2048))
	ax = sns.boxplot(data=ql_means, linewidth=lw)
	ax.set_ylabel("Log Weight")
	ax.set_title("L = 128")
	ax.get_xaxis().set_ticks([])
	ax.set_xlabel("Time Step")
	ax.set_ylim(-17, 17)


	ax = fig.add_subplot(rows, cols, 6)
	
	filename = "Qkl-scores-resamp-weights-"+str(file_id)+".p"
	all_Qkl_weights = pickle.load( open(filename, "rb" ) )
	qk_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T
	ax = sns.boxplot(data=qk_means, linewidth=lw)
	ax.set_title("K = 16")
	ax.get_xaxis().set_ticks([])
	ax.set_xlabel("Time Step")
	ax.get_yaxis().set_ticks([])
	ax.set_ylim(-17, 17)



	plt.tight_layout()
	#ax.set_ylabel("mean weights")
	#ax.legend(bbox_to_anchor=(1.1, 1.05))
	fig.savefig("particle_var.eps", bbox_inches='tight')
	#plt.show()

def plot_for_legend():
	#file_ids = [1538440198, 1538448489, 1538499742, 1538462807, 1538470154, 1538477199, 1538484792, 1538496244]
	#file_ids = [1538532620, 1538539386, 1538545347, 1538551100, 1538556818, 1538562473, 1538568646]
	#file_ids = [1538609735, 1538616566, 1538609635,1538622239, 1538615745]#, 1538622917] #1538618520 64x32 miss
	file_ids = [1538639858, 1538674801, 1538680844, 1538687079, 1538675256, 1538681038, 1538687390]
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	colors = ['darkblue','dodgerblue', 'darkorchid', 'mediumseagreen','indianred', 'orange','lightgrey']
	i = 0
	for file_id in file_ids[:]:
		filename = "ql-scores-resamp-weights-"+str(file_id)+".p"
		all_ql_weights = pickle.load( open(filename, "rb" ) )
		ql_means = (logsumexp(all_ql_weights, axis=2) - np.log(all_ql_weights.shape[2])).T
		ql_means = (logsumexp(ql_means, axis=0) - np.log(ql_means.shape[0]))


		filename = "Qkl-scores-resamp-weights-"+str(file_id)+".p"
		all_Qkl_weights = pickle.load( open(filename, "rb" ) )
		qk_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T
		qk_means = (logsumexp(qk_means, axis=0) - np.log(qk_means.shape[0]))
		

		# QKL_means = (logsumexp(all_Qkl_weights, axis=2) - np.log(all_Qkl_weights.shape[2])).T
		# print QKL_means.shape
		# QKL_means = (logsumexp(QKL_means, axis=0) - np.log(QKL_means.shape[0]))
		# print QKL_means.shape
		
		
		
		ax.plot(list(xrange(1, 29)),qk_means/ql_means,color=colors[i],label="("+str(all_Qkl_weights.shape[1])+ ","+str(all_Qkl_weights.shape[2])+str(")"))
		#ax.plot(list(xrange(1, 29)),np.mean(qk_means, axis=0),label="K="+str(all_Qkl_weights.shape[1]))
		i +=1
	ax.set_xlabel("time step")
	ax.set_ylabel("mean ratio: Cw/Rw")
	ax.set_title("Mean Weight Ratio")
	#ax.legend(bbox_to_anchor=(1.1, 1.05))
	# Shrink current axis's height by 10% on the bottom
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1,
	                 box.width, box.height * 0.9])

	# Put a legend below current axis
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
	          fancybox=True, shadow=False, ncol=4)

		#fig.savefig("resamp-QKL-2048-ESS_1.eps", bbox_inches='tight')
	#ax.set_ylim(-10, -5)
	fig.savefig("resamp-Qkl-2048-qk-mean-ratio.eps", bbox_inches='tight')

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
	# plot_resamped_2048_kl_weights_means()
	# plot_resamped_2048_kl_weights_ess()
	# read_plot_weights_per_file()

	sns.set(font_scale=0.9)
	sns.set_style("white")
	sns.palplot(sns.color_palette("cubehelix", 8))
	#grid_plot_lines()
	#grid_plot_box()
	#plot_for_legend()

	#sns.set(font_scale=0.8)
	#-------------
	#store_resampled_2048_kl_weights()

	all_ql_data, all_Qkl_data = collaps_data()
	#EXP_A1_grid_plot_box(all_ql_data, all_Qkl_data)

	#---------------------------------------------
	#EXP_A2_grid_plot_box(all_ql_data, all_Qkl_data)

	#EXP_grid_plot_lines(all_ql_data, all_Qkl_data)
	EXP_grid_plot_lines_vars(all_ql_data, all_Qkl_data)
	#---------------------------------------------



	#EXP_A3_grid_plot_box(all_ql_data, all_Qkl_data)

	# for when we want to  plot an expectation over 10 samples for each 
	#all_ql_data, all_Qkl_data = collaps_data()
	#EXPECTATION_plot_resamped_2048_kl_weights_mean_ratio(all_ql_data, all_Qkl_data)
	#EXPECTATION_plot_resamped_2048_kl_weights_ess(all_ql_data, all_Qkl_data)
	#EXPECTATION_plot_resamped_2048_kl_weights_var(all_ql_data, all_Qkl_data)





	#EXP_grid_plot_box(all_ql_data, all_Qkl_data)
	#keys = plot_resamped_2048_kl_weights_mean_ratio(E=True)


	# for when we just want to plot initial plots (first 10 , 1 each)
	# plot_resamped_2048_kl_weights_ess()
	# plot_resamped_2048_kl_weights_mean_ratio
	# plot_resamped_2048_kl_weights_var()
	# plot_resamped_2048_l_weights_var()
	# plot_resamped_2048_kl_weights_mean()