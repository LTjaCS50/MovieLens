rm(list = ls())
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



################################################################
# Create train and test sets from edx set
################################################################

# Test will be 10% of edx data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temporary <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- temporary %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed2 <- anti_join(temporary, test_set)
train_set <- rbind(train_set, removed2)

################################################################
# Understanding the Data Sets
################################################################
dim(edx)
dim(validation)
head(edx)
class(edx)
head(validation)

dim(train_set)
dim(test_set)

################################################################
# Basic Model + Movie Effect + User Effect
###############################################################

# The section below develops the basic model, which code is derived from the text book. 
# To meet the project's requirement, I built the basic model by including an extra bias (genre)
# and extra technique (regularization). The build-up is in the subsequent sections following 
# the one below.

# Note: Although the following code and the build up code uses the test set to find predicted
# rating, in running the final model, only the edx set is used (validation set is not used until
# the very end)

# the RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Basic Model RMSE
mu_hat <- mean(train_set$rating)
mu_hat

naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

model_0 <- tibble(method = "Basic Model", RMSE = naive_rmse)

rmse_results <- model_0 

print.data.frame(rmse_results)

# Basic Model + Movie Effects RMSE
mu <- mean(train_set$rating) 

movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
movie_avgs

predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

model_1 <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results,
                data_frame(method="Basic+Movie", RMSE = model_1 ))

print.data.frame(rmse_results)

# Basic Model + Movie Effects + User Effects RMSE

user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

model_2 <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Basic+Movie+User", RMSE = model_2))

print.data.frame(rmse_results)

#################################################################
# Basic Model + Movie Effect + User Effect + Genre Effect
# The following is the build-up from the given code.
#################################################################

# Addition 1: Genre Effects

# Review: Does genre have effects on the ratings?
train_set %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 10000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 10*se, ymax = avg + 10*se)) + 
  geom_point() +
  scale_x_discrete(breaks = train_set$genres[c(T,rep(F,1000000))]) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size=7)) 
  
# Basic Model + Movie Effects + User + Genre Effects RMSE
genre_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by= 'genres') %>%
  mutate(pred = mu + b_i + b_u +b_g) %>%
  pull(pred)

model_3 <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Basic+Movie+User+Genre", RMSE = model_3))

print.data.frame(rmse_results)

######################################################################
# Regularized Basic Model + Movie Effects + User Effects
# The following is another build-up.
######################################################################

# Addition 2: Regularized Basic Model + Movie Effects + User Effects

# Note: Although the following code uses the test set to find lambda,
# in running the final model, only the edx set is used (validation set is not used until
# the very end)

# Find the most effective lambda for the model

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

rmses

lambda <- lambdas[which.min(rmses)]

lambda

# Find the RMSE for the model using the lambda derived above

lambda <- 5

mu <- mean(train_set$rating)

movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

user_reg_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda), n_u = n())

predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by="movieId") %>%
  left_join(user_reg_avgs, by="userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

model_4 <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Basic+Movie+User", RMSE = model_3))

print.data.frame(rmse_results)

######################################################################
# Regularized Basic Model + Movie Effects + User Effects + Genre Effects
# The following is another build-up.
######################################################################

# Addition 3: Regularized Basic Model + Movie Effects + User Effects + Genre Effects

# Note: Although the following code uses the train and test sets to find lambda,
# in running the final model, only the edx set is used (validation set is not used until
# the very end)

# Find the most effective lambda for the model

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu )/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

rmses

lambda <- lambdas[which.min(rmses)]

lambda

# Find the RMSE for the model using the lambda derived above

lambda <- 4.75

mu <- mean(train_set$rating)

movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

user_reg_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda), n_u = n())

genre_reg_avgs <- train_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+lambda), n_g = n())

predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by="movieId") %>%
  left_join(user_reg_avgs, by="userId") %>%
  left_join(genre_reg_avgs, by="genres") %>%
  mutate(pred = mu + b_i + b_u +b_g) %>%
  pull(pred)

model_5 <- RMSE(predicted_ratings, test_set$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Basic+Movie+User+Genre", RMSE = model_5))

print.data.frame(rmse_results)

#######################################################################
# Final Model Against Validation Set
#######################################################################

# From the above, the best model is Regularized Model that takes into 
# account the Movie + User + Genre Effects (model_5). So, that's the model
# that'll we'll use to test against the validation set.

# Note: In the following instead of using train and test sets, we'll instead
# use the edx to come up with the predicted ratings against the validation set. 
# We'll use the lambda derived from cross validating the training + test sets 
# as both sets make up the edx set.

# Find the RMSE for the model using the lambda derived above

lambda <- 4.75

mu <- mean(edx$rating)

movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

user_reg_avgs <- edx %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda), n_u = n())

genre_reg_avgs <- edx %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  left_join(user_reg_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+lambda), n_g = n())

predicted_ratings <- validation %>% 
  left_join(movie_reg_avgs, by="movieId") %>%
  left_join(user_reg_avgs, by="userId") %>%
  left_join(genre_reg_avgs, by="genres") %>%
  mutate(pred = mu + b_i + b_u +b_g) %>%
  pull(pred)

model_6 <- RMSE(predicted_ratings, validation$rating)
model_6

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Final Model and Validation", RMSE = model_6))

print.data.frame(rmse_results)
