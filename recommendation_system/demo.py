import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data and format
data = fetch_movielens(min_rating=4.0)

#print training and testing data
#print(repr(data['train']))
#print(repr(data['test']))

#create model warp = weighted approxiate-rank pairwise
model = LightFM(loss='warp')

#train model
model.fit(data['train'], epochs=30, num_threads=4)

def sample_recommendation(model, data, user_ids):

	#number of users and movies in training data
	n_users, n_items = data['train'].shape

	#generate recommendations for each user we input
	for user_id in user_ids:

		#movies they like
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		#movies for our model predicting
		scores = model.predict(user_id, np.arange(n_items))
		#rank them from most liked to least
		top_items = data['item_labels'][np.argsort(-scores)]

		#print out the results
		print("User %s" % user_id)
		print("     Known positives:")

		for x in known_positives[:3]:
			print("       %s" % x)

		print("     Recommended")

		for x in top_items[:3]:
			print("      %s" % x)

sample_recommendation(model, data, [69,420,666])