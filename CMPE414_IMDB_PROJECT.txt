# 1) web crawling : web crawling/ scraping done by using R studio librarires (rvest)

install.packages("rvest")
install.packages("magrittr")
install.packages("jsonlite")
install.packages("R2HTML")
install.packages("xml2")

#needed libraries
library("xml2")
library("R2HTML")
library("rvest")
library("magrittr")
library("jsonlite")
library("dplyr")

# you need to create a folder named as scraped_html and locate it in main pc 
folder <- "C:/scraped_html"
setwd(folder) # it sets working directory as folder


#website link
link <- "https://www.imdb.com/search/title/?groups=top_100&sort=user_rating,desc.html"
link %>%
  read_html() -> myHTML
myHTML %>%
  write_html("IMDB_top100.html")

# name,year,cast,genre and rating attributes of the website
name <- myHTML %>% html_nodes(".lister-item-header a") %>% html_text()
year <- myHTML %>% html_nodes(".text-muted.unbold") %>% html_text()
cast <- myHTML %>% html_nodes(".text-muted+ p") %>% html_text()
genre <- myHTML %>% html_nodes(".genre") %>% html_text()
rating <- myHTML %>% html_nodes(".ratings-imdb-rating strong") %>% html_text()

#it collabs them into a data frame called movies
movies <- data.frame(name,year,cast,genre,rating, stringsAsFactors = FALSE)

View(movies)

length(which(!complete.cases(movies))) # there is no missing data


# 2) Data Preprocessing
movies$name
library("caret")
set.seed(44186) # for reproducability 


sample.size <- floor(0.80 * nrow(movies)) # %80 for training, %20 for testing
train.index <- sample(seq_len(nrow(movies)), size = sample.size)
train <- movies[train.index, ]
test <- movies[-train.index, ]
View(train)
View(test)

# verify proportions
prop.table(table(train$name))
prop.table(table(test$name))

install.packages("quanteda")
library("quanteda")

# making tokenization on names
train.tokens <- tokens(train$name, what = "word", remove_numbers = TRUE,remove_punct = TRUE,remove_symbols = TRUE, remove_separators = TRUE)

# see the change of an example
train.tokens[[1]]

#lowercase the tokens
train.tokens <- tokens_tolower(train.tokens)
train.tokens[[1]]

#stopwords
train.tokens <- tokens_select(train.tokens,stopwords(),selection = "remove")
train.tokens[[1]]
stopwords()

#stemming
train.tokens <- tokens_wordstem(train.tokens,language = "english")
train.tokens[[9]] # real one is saving private ryan

# bag of words model creation
train.tokens.dfm <- dfm(train.tokens,tolower = FALSE)# document feature matrix

#transfroming to a matrix
train.tokens.matrix <- as.matrix(train.tokens.dfm)
View(train.tokens.matrix[1:20,1:40])

# dimension
dim(train.tokens.matrix) # 40 doc 77 column

#observing the effects of stemming
colnames(train.tokens.matrix)[1:40]

#cross validation which is used for useing the train data most efficient way

# making feature based data frame 
x <- convert(train.tokens.dfm, to = "data.frame")
train.tokens.df <- cbind(name = train$name, x)

#Clean up the column names
names(train.tokens.df) <- make.names(names(train.tokens.df))

#creating stratified folds for 10-fold cross validation repeated
set.seed(44186)
cross_validation.folds <- createMultiFolds(train$name, k = 10, times = 3)
cross_validation.cntrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, index = cross_validation.folds)

# creating a function for calculating the term frequency
term.frequency <- function(row){
  row / sum(row)
}

# creating a function for calculating inverse document frequency
inverse.document.frequency <- function(col){
  corpus.size <- length(col)
  document.count <- length(which(col > 0))
  log10(corpus.size / document.count)
}

# creating a function for calculating term frequency and inverse document frequency

tf.idf <- function(t,idf){
  t * idf
}

# normalize all docs by term frequency , apply: function that applies functions to matrixes
train.tokens.df <- apply(train.tokens.matrix, 1 , term.frequency)
dim(train.tokens.df)
View(train.tokens.df[1:20,1:40]) # in last view function, the values are 1 but now values are 0.5

# calculating inverse document frequency vector for train and test
train.tokens.idf <- apply(train.tokens.matrix,2,inverse.document.frequency)
str(train.tokens.idf)

#calculating term frequency and inverse document frequency for training set
train.tokens.term_freq_inverse_document_freq <- apply(train.tokens.df,2, tf.idf, idf = train.tokens.idf)
dim(train.tokens.term_freq_inverse_document_freq)
View(train.tokens.term_freq_inverse_document_freq[1:20,1:20])
# notice that thew values for the text 1 used to be same because they were normalized to that individual document vector
# but now they are rationalized and have been combined with the term frequency in the inverse doc freq values to achieve our two goals
# first goal is normalizing documents based on their length so that we can compare documents on equal footing and then rationalize individual
# terms in our training corpus to observe the terms are that appear more frequently are going to be less useful then those terms that appear rarely

# we need to transfer to the matix
train.tokens.term_freq_inverse_document_freq <- t(train.tokens.term_freq_inverse_document_freq)
dim(train.tokens.term_freq_inverse_document_freq) # 40 docs and each one has 80 columns
View(train.tokens.term_freq_inverse_document_freq[1:20,1:20])

# making a quick check of incomplete cases 
incomp.cases <- which(!complete.cases(train.tokens.term_freq_inverse_document_freq))
train$name[incomp.cases] # 0 incomplete cases

# n-gram : allows us to extend bag of words model to include word ordering

train.tokens[[4]]
train.tokens <- tokens_ngrams(train.tokens, n = 1:2) # bigram
train.tokens[[4]]

# transform document feature matrix and then a matrix
train.tokens.dfm <- dfm(train.tokens,tolower = FALSE)
train.tokens.matrix <- as.matrix(train.tokens.dfm)
train.tokens.dfm # by adding bi-grams, the number of columns in our matrix has has increased almost 2x 

#normalizing the docs by term freq
train.tokens.df <- apply(train.tokens.matrix,1,term.frequency)

# calculating the inverse document freq 
train.tokens.idf <- apply(train.tokens.matrix,2,inverse.document.frequency)

#tf-idf calculation
train.tokens.term_freq_inverse_document_freq <- apply(train.tokens.df,2, tf.idf, idf = train.tokens.idf)

#transposing the matrix
train.tokens.term_freq_inverse_document_freq <- t(train.tokens.term_freq_inverse_document_freq)
# garbage collection for cleaning unused obj
gc()


# Vector Space Model
# representing documents as vectors of numbers
# we assume that all document vectors originate from origin
# The ways of generating vectors for documents are Bag of Words, TF-IDF (this means that the vector space model creation process started with that processes)
# We need to find the similarity between vectors via cosine similarity

# Let's make decompose a matrix into a successive aproximation = SVD(Singular Value Decomposition)

install.packages("irlba")
library("irlba") # Find a few approximate singular values and corresponding singular vectors of a matrix.

#Reducing dimentionality to 40 columns for latent semantic analysis

# start.time <- Sys.time()
train.irlba <- irlba(t(train.tokens.term_freq_inverse_document_freq), nv = 20 , maxit = 76 )
# total.time <- Sys.time() - start.time
#total.time

# observing new feature of data
View(train.irlba$v) # v = approximate right singular vectors

#mapping tf-idf into the singular value decomposition at a semantic space
sigma.inverse <- 1/train.irlba$d # d = approximate singular values
u.transpose <- t(train.irlba$u) # approximate left singular values
document <- train.tokens.term_freq_inverse_document_freq[1,]
document.hat <- sigma.inverse*u.transpose %*% document # %*% matrix multiplication

document.hat[1:10]
train.irlba$v[1,1:10]
# some values are same and the ones that are not same are very small numbers, after calculating the cosine similarity between these vectors 
# the value that we may get back is 1 because the value are small

# creating a new feature based data fram including irlba for document semantic space

#clustering part

#install.packages("parallel")
#library("parallel")
#install.packages("doSNOW")
#library("doSNOW")
#start.time <- Sys.time()
# Create a cluster to work on 10 logical cores.
#cl <- makeCluster(3, type = "SOCK")
#registerDoSNOW(cl)
# As our data is non-trivial in size at this point, use a single decision
# tree alogrithm 
#r.cv.4 <- train(name ~ ., data = train.singular_value_decomposition, method = "rpart", 
#                    trControl = cross_validation.cntrl, tuneLength = 4)
# Processing is done, stop cluster.
#stopCluster(cl)
# Total time of execution on workstation was approximately 4 minutes. 
#total.time <- Sys.time() - start.time
#total.time
# Check out our results.
#r.cv.4
library("caret")
install.packages("randomForest")
library("randomForest")
library("mlbench")

#Confusion Matrix
ifelse( year < 2000, "TRUE", "FALSE")
set.seed(123) # used for reproduce random numbers 
data <- data.frame(Actual = sample(c("True","False"), 100, replace = TRUE),
                   Prediction = sample(c("True","False"), 100, replace = TRUE))
library(caret)
confusionMatrix(as.factor(data$Prediction), as.factor(data$Actual), positive = "True")
# recall = 23/43 = 0.534, precission = 23/49 = 0.469, accuracy = 0.49, f1 score = 23/48.5  = 0.474


#cosine similarity ve conf matrix kismi daha doldurulmadi
install.packages("lsa")
library("lsa")

#train.similarity.finder <- cosine(t(as.matrix(train.singular_value_decomposition[,-c(1,ncol(train.singular_value_decomposition))])))



# TEST DATA PREPROCESSING(TOKENIZATION, STEMMING, STOPWORD, NGRAM, DFM)

#tokenization
test.tokens <- tokens(test$name, what = "word", remove_numbers = TRUE, remove_punct = TRUE, remove_symbols = TRUE, remove_separators = TRUE)
#lower case
test.tokens <- tokens_tolower(test.tokens)
#removing stopwords
test.tokens <- tokens_select(test.tokens,stopwords(),selection = "remove")
#stemming
test.tokens <- tokens_wordstem(test.tokens, language = "english")
#ngram (n = 2)
test.tokens <- tokens_ngrams(test.tokens, n = 1:2)

# bag of words model creation
test.tokens.dfm <- dfm(test.tokens,tolower = FALSE)
#transfroming to a matrix
test.tokens.matrix <- as.matrix(test.tokens.dfm)
View(test.tokens.matrix[1:5,1:10])
# dimension
dim(test.tokens.matrix) # 10 doc 34 column
#observing the effects of stemming
colnames(test.tokens.matrix)[1:10]
#cross validation which is used for useing the train data most efficient way

# making feature based data frame 
test_converter <- convert(test.tokens.dfm, to = "data.frame")
test.tokens.df <- cbind(name = test$name, test_converter)
#Clean up the column names

names(test.tokens.df) <- make.names(names(test.tokens.df))


# train and test dfm object observation
test.tokens.dfm
train.tokens.dfm


# provide the test dfm has the same ngram as the train dfm
test.tokens.dfm <- dfm_select(test.tokens.dfm, featnames(train.tokens.dfm))
test.tokens.matrix <- as.matrix(test.tokens.dfm)
test.tokens.dfm



# normalizing the data by term frequency 
test.tokens.df <- apply(test.tokens.matrix, 1, term.frequency)
str(test.tokens.df)
View(test.tokens.df)
dim(test.tokens.df)
# term frequency and inverse document frequency calculation for train
test.tokens.term_freq.inverse_doc_freq <- apply(test.tokens.df, 2, tf.idf, inverse.document.frequency = train.tokens.idf)
dim(test.tokens.df)

#RANDOM FOREST
# = an ensemble of decision trees. builds and combines 
# multiple decision trees to get more accurate predictions.
# its called random because they choose predictors randomly at a time of training
# and its called forest becaus it takes the output of multiple trees to make a decision.

install.packages("caTools")       
install.packages("randomForest")  
library(caTools)
library(randomForest)

# Fitting Random Forest to our train dataset
train$year <- as.factor(train$year)
str(train)
set.seed(44186)  # Setting seed
classifier_RF = randomForest(x = train[-6], #we write (-6) bcs we have 5 variables
                             y = train$year, 
                             ntree = 500) #number of trees
classifier_RF


# Predicting the train set results
y_pred = predict(classifier_RF, newdata = train[-6])
y_pred
# Plotting model
# error rate must be stabilized with the increase of num of trees :')
plot(classifier_RF)

# Importance plot
importance(classifier_RF)
#order of importance of our variables:
# year 10.9, cast 7.85, name 7.19, genre 6.75, rting 4.63

# variable importance plot
varImpPlot(classifier_RF)



# SVM
# = used for analyzing the data used for classification

library(ggplot2)
qplot(year, rating, data = train)
str(train)

library(e1071)
model <- svm(year~., data= train)
summary(model)

plot(model, data=train, year~rating)