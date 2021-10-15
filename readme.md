## Welcome! <img src="https://raw.githubusercontent.com/MartinHeinz/MartinHeinz/master/wave.gif" width="30px">
## :musical_keyboard: Predicting a Hit or a Flop with Spotify Data :musical_keyboard:
This project utilized data acquired from Spotify that categorized songs from 1960 - 2019 as a hit or a flop.  I used this data set to analyze the drivers of hits and flops and to create a supervised machine learning model that can predict if a song will be a hit or a flop.  I was able to improve baseline performance by 23% using a random forest model classification model.   The model can be used by artists, producers, and record labels when selecting songs for albums or as singles.  The key drivers of hit songs can be used by artists, producers and writers in the creative process to ensure a song will be a hit.

## :clipboard: Plan - Supervised Learning Classification Model
- Complete the entire Data Science Pipleline in order to create a classification model to predict whether a song will be a hit or a flop.  
- Explore data to gain insight and provide recommendations related to drivers of a hit song.  
= Create insights from data analysis and the model that will create actionable items for artists, producers or record labels to use in the creative process or when deciding if a song will be a hit or not.
My initial hypotheses are that danceability, instrumentalness, and duration are drivers of a song being a hit.
1.  **Acquire** data from kaggle database.
2.  **Prepare** - data by handling outliers, nulls, and determining which features to keep.
    - Create functions to clean data, split and scale before moving into explore and model.
3.  **Explore** - Univariate and Bivariate exploration and statistical testing done to determine features for model and gain insight.
4.  **Model** - develop a classification model that performs better than baseline & evaluate the model through numeric measures and visually.

## :bar_chart: Model
The project goal is to develop a classification model that preforms better than baseline & evaluate the model through numeric measures and visually.
## :green_book: Data Dictionary


Target | Description | Data Type
---------|----------|---------
target| The target variable for the track. It can be either '0' or '1'. '1' implies that this song has featured in the weekly list (Issued by Billboards) of Hot-100 tracks in that decade at least once and is therefore a 'hit'. '0' Implies that the track is a 'flop'. | object


Feature| Description | Data Type
---------|----------|---------
  track |  The name of the track | object
  artist |  name of artist  | object
  uri|  The resource identifier for the track | object
  danceability | Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. | float64
 energy |  Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.  | float64
  key | The estimated overall key of the track. | float64
  mode | Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0. | int64
  speechiness |  Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. | float64
  acousticness |  A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. | float64
  instrumentalness |  Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. | float64
  liveness |  Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. | float64
  valance |  A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). | float64
  tempo |  The overall estimated tempo of a track in beats per minute (BPM). | float64
  duration_ms |  duration of the track in milliseconds | int64
  time_signature |  An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).| int64
  chourus_hit |  This the the author's best estimate of when the chorus would start for the track. Its the timestamp of the start of the third section of the track. This feature was extracted from the data received by the API call for Audio Analysis of that particular track. | float64
  sections |  The number of sections the particular track has. | int64
  decade |  the decade the track was released | object
  loudness |  The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. | float64

## Final Project Conclusions:
The Random Forest Model preformed best and was the most well rounded model for the job
Accuracy on validate was 77%
Validate Recall was 93%
Accuracy on Test 76% accuracy which was only slighly lower than our accuracy on validate set
Recall stayed almost the same high as well 92%
The model had a 23% improvement on baseline preformance 🥳
### 📄 Recommendations:
Use model to optimize song selection for singles or inclusion in albums or concert setlists.
Use model feature importance paired with correlation found in exploratory data analysis to determine song features during creative process.
Hit songs tend to be less instrumental meaning songs with words tend to be hits vs songs without words.
The more danceable a song, the more likely it is to be a hit
the higher the energy of a song, the more likely it is to be a hit.

## :briefcase: Modules:
- Wrangle.py - contains all functions to acquire and clean data for project.
- explore.py - contains functions used when exploring data.

 ## :pencil: How to Duplicate this Project:

1.  Read the README.md
2.  Download the wrangle.py, explore.py, and spotify_final_notebook.ipynb files into your working directory, or clone this repository
3.  Run the spotify_hit_or_flop_final.ipynb notebook.

You can also download the spotify_hit_or_miss.ipynb to see more of the initial exploratory analysis, modeling, and data cleaning.