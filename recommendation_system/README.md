# Overview

Followed tutorial challenge Recommender system by @Sirajology on Youtube. This uses lightfm to train a model from a data set of movies and artists to recommend based off the users likes and dislikes.  

## What I have added

There are three files to this, demo.py, demo_challenge.py, and fetch_lastfm.py, with the challenge having my changes. I also created the fetch_lastfm based off user ciurana2016 on github where it sparses through a data set to create a matrix of artists, users, and songs that they like. The demo challenge uses this data to train a recommendation system on specific users of the code runners choice to see what recommended artists are shown for them. 

## Dependencies 

- numpy (http://www.numpy.org/)
- scipy (https://www.scipy.org/)
- lightfm (https://github.com/lyst/lightfm)


## Usage

Run script in cmd or terminal via

`python demo.py`
`python demo_challenge.py`
