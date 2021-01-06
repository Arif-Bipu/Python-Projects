import numpy as np
from lightfm import LightFM
from fetch_lastfm import fetch_lastfm

#fetch data and format
data = fetch_lastfm()

#create model warp = weighted approxiate-rank pairwise
model = LightFM(loss='warp')

#train model
model.fit(data['matrix'], epochs=30, num_threads=2)

def get_recommendations(model, coo_mtrx, user_ids):

	#number of users in matrix shape
	n_items = coo_mtrx.shape[1]

	#generate recommendations for each user we input
	for user in user_ids:

		# TODO creates known positives
		# sorts the model predictions
		scores = model.predict(user, np.arange(n_items))
		top_scores = np.argsort(-scores)[:3]

		print("Recommendations for user %s" % user)

		for x in top_scores.tolist():
			for artist, values in data['artists'].items():
				if int(x) == values['id']:
					print("    - %s" % values['name'])

		print("\n")


get_recommendations(model, data['matrix'], [420, 69, 666])




