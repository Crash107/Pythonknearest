# K-nearest using Python

## I used k-nearest neighbors as my algorithm for the data set. 
## For my problem, I wanted to see if danceability was a good way to predict the number of streams a song gets. My target variable is streams since its the value I'm trying to predict. My input value is danceability since it's the value I want to use to make the prediction. I decided k=10. I choose ten as my k value because a large number like ten provide a smoother and more generalized estimation of the target variable. I used python to solve this problem and this is the scatter plot that was generated after running the code

![A description of the image](../Pictures/graph.png)
## In my scatter plot I have two types of points, actual streams and predicted streams. Actual streams are the actual number of streams from the database. Predicted streams is what the number of streams my knn model predicted. 
## After running this code, I also received a Mean Squared Error value of 2.852326722269527e+17 which is considered a high amount. The high Mean Squared Error in combination with the fact that the actual streams have more points than the predicted streams, suggest that danceability is not a strong indicator of the number of streams a song will receive. This suggest that their are other factors in play when it comes to predicting the number of streams a song gets
