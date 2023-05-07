library(tidyverse)
library(quanteda)
library(ROSE)
library(naivebayes)
library(e1071)
library(caret)


# Import data
setwd('~/Documents/GitHub/text-as-data-sp23/final project')
lyrics <- read.csv('lyrics-dataset-deduped.csv',
                   header=FALSE,
                   stringsAsFactors=FALSE)
colnames(lyrics) <- c('Artist', 'Title', 'Album', 'Date', 'Lyric', 'Year')
#head(lyrics)


# Split dataset to have equal sample of gaga and non-gaga songs
# Add labels for classification
gaga <- subset(lyrics, Artist == 'Lady Gaga')
gaga$label <- 'gaga'
not_gaga <- subset(lyrics, Artist != 'Lady Gaga')
not_gaga$label <- 'not gaga'
#dim(gaga)
#dim(not_gaga)

###### UNDER SAMPLING ######

# Create random sample of non-gaga songs :: "under-sampling"
set.seed(328)
not_gaga_sample <- not_gaga[sample(1:nrow(not_gaga), 161), ]
#dim(not_gaga_sample)


# Combine back into one dataframe for classification
undersample <- rbind(gaga, not_gaga_sample)
#dim(undersample)


## Remove instrumentals/songs without lyrics
undersample <- dplyr::filter(undersample, nchar(undersample$'Lyric') >= 5)
## create corpus
undersample_corpus <- corpus(undersample, text = 'Lyric')


gaga_stopwords <- c('lady', 'gaga')

# Pre-processing helper function
preprocess_text <- function(corpus, punct=TRUE, symbols=TRUE,
                            numbers=TRUE, url=TRUE, lower=TRUE, min_term=10, max_term=1000000,
                            min_doc_freq=0.1, max_doc_freq=0.9, ngram=1) {
  temp_dfm <- corpus %>% tokens(remove_punct = punct,
                                remove_symbols = symbols,
                                remove_numbers = numbers,
                                remove_url = url) %>%
    tokens_ngrams(n = ngram) %>%
    dfm(tolower=lower) %>%
    dfm_remove(c(stopwords('english'), gaga_stopwords)) %>%
    dfm_trim(min_termfreq = min_term,
             max_termfreq = max_term,
             min_docfreq = min_doc_freq,
             max_docfreq = max_doc_freq,
             docfreq_type=c("prop"))
  return(temp_dfm)
}


# classification function with svm and naive bayes

classification <- function(dfm_matrix, labels) {
  # train-test split
  set.seed(328)
  sample <- sample.int(n = nrow(dfm_matrix),
                       size = floor(.8*nrow(dfm_matrix)), 
                       replace = F)
  train_matrix <- dfm_matrix[sample,] # Select everything within our somewhat random sample
  train_labels <- as.factor(labels[sample])
  test_matrix <- dfm_matrix[-sample,] # Select everything not within the sample
  test_labels <- as.factor(labels[-sample])
  
  # SVM
  svm_model <- svm(
    x = train_matrix,
    y = train_labels,
    kernel = "linear")
  
  # naive bayes
  nb_model <- naivebayes::multinomial_naive_bayes(
    x = train_matrix,
    y = train_labels)
  
  # generate predictions
  pred_svm <- predict(svm_model, test_matrix)
  pred_nb <- predict(nb_model, test_matrix)
  
  # confusion matrix + precision & recall
  conf_svm <- confusionMatrix(data=pred_svm, 
                              reference = test_labels,
                              mode = 'prec_recall') 
  
  conf_nb <- confusionMatrix(data=pred_nb, 
                             reference = test_labels, 
                             mode = 'prec_recall') 
  
  return(list(svm = conf_svm, nb = conf_nb))
}



##### CLASSIFICATION MODELS #####

#### without using n-grams ####

## Preprocessing with defaults
## SVM performance
# Accuracy: 0.7077
# Precision: 0.6875
# Recall: 0.7097
## NB performance
# Accuracy: 0.7538
# Precision: 0.7419
# Recall: 0.7419
undersample_dfm <- preprocess_text(undersample_corpus)
#dim(undersample_dfm)
## Convert DFM to matrix
undersample_dfm_matrix <- convert(undersample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(undersample_dfm_matrix, undersample_dfm$label)

## Preprocessing function args for reference
# preprocess_text : function(corpus, stopwords_english=TRUE, punct=TRUE, symbols=TRUE,
 #                            numbers=TRUE, url=TRUE, lower=TRUE, min_term=10, max_term=1000000,
  #                           min_doc_freq=0.1, max_doc_freq=0.9, ngram=1)

## Preprocessing NB PERFORMANCE
# Accuracy: 0.7846
# Precision: 0.7576
# Recall: 0.8065 ## best recall
## SVM performance
# Accuracy: 0.7538
# Precision: 0.7419
# Recall: 0.7419
undersample_dfm <- preprocess_text(undersample_corpus, min_term = 10, max_term = 400)
#dim(undersample_dfm)
## Convert DFM to matrix
undersample_dfm_matrix <- convert(undersample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(undersample_dfm_matrix, undersample_dfm$label)


## Preprocessing NB performance
# Accuracy: 0.7692
# Precision: 0.7353
# Recall: 0.8065 ## best recall
## SVM performance
# Accuracy: 0.7692
# Precision: 0.7667
# Recall: 0.7419
undersample_dfm <- preprocess_text(undersample_corpus, min_term = 10, max_term = 300) #improved SVM, NB poorer
#dim(undersample_dfm)
## Convert DFM to matrix
undersample_dfm_matrix <- convert(undersample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(undersample_dfm_matrix, undersample_dfm$label)


## Preprocessing NB performance
# Accuracy: 0.7846
# Precision: 0.7576
# Recall: 0.8065 ## best recall
## SVM performance
# Accuracy: 0.7538
# Precision: 0.7419
# Recall: 0.7419
undersample_dfm <- preprocess_text(undersample_corpus, min_term = 10, max_term = 350) #better NB worse SVM
#dim(undersample_dfm)
## Convert DFM to matrix
undersample_dfm_matrix <- convert(undersample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(undersample_dfm_matrix, undersample_dfm$label)


## Preprocessing NB performance
# Accuracy: 0.7846
# Precision: 0.7576
# Recall: 0.8065 ## best recall
## SVM performance
# Accuracy: 0.7538
# Precision: 0.7419
# Recall: 0.7419
undersample_dfm <- preprocess_text(undersample_corpus, min_term = 10, max_term = 350) 
#dim(undersample_dfm)
## Convert DFM to matrix
undersample_dfm_matrix <- convert(undersample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(undersample_dfm_matrix, undersample_dfm$label)

# no difference min_term [2,60], then starts to get poorer as it gets higher
# max term [300,400] best performance both SVM and NB



## Preprocessing NB performance
# Accuracy: 0.8154
# Precision: 0.8800
# Recall: 0.7097
## SVM performance POOR
undersample_dfm <- preprocess_text(undersample_corpus, min_term = 10, max_term = 350,
                            min_doc_freq = 0.05, max_doc_freq = 0.9)
#dim(undersample_dfm)
## Convert DFM to matrix
undersample_dfm_matrix <- convert(undersample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(undersample_dfm_matrix, undersample_dfm$label)


## Preprocessing NB performance
# Accuracy: 0.8308 ## best accuracy
# Precision: 0.8846 ## best precision
# Recall: 0.7419
## SVM performance POOR
undersample_dfm <- preprocess_text(undersample_corpus, min_term = 10, max_term = 300,
                            min_doc_freq = 0.05, max_doc_freq = 0.3)
#dim(undersample_dfm)
## Convert DFM to matrix
undersample_dfm_matrix <- convert(undersample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(undersample_dfm_matrix, undersample_dfm$label)


## Preprocessing NB performance
# Accuracy: 0.8308 ## best accuracy
# Precision: 0.8846 ## best precision
# Recall: 0.7419
## SVM performance POOR
undersample_dfm <- preprocess_text(undersample_corpus, min_term = 10, max_term = 350,
                            min_doc_freq = 0.05, max_doc_freq = 0.3)
#dim(undersample_dfm)
## Convert DFM to matrix
undersample_dfm_matrix <- convert(undersample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(undersample_dfm_matrix, undersample_dfm$label)



## Preprocessing NB performance
# Accuracy: 0.8000
# Precision:  0.8000
# Recall: 0.7742 ## improved recall but hurt accuracy & precision
## SVM performance POOR
undersample_dfm <- preprocess_text(undersample_corpus, min_term = 10, max_term = 350,
                            min_doc_freq = 0.05, max_doc_freq = 0.25)
#dim(undersample_dfm)
## Convert DFM to matrix
undersample_dfm_matrix <- convert(undersample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(undersample_dfm_matrix, undersample_dfm$label)


### with stopwords: ### (before I fixed error in preprocessing funct)
# min_term = 10, max_term = 350, min_doc_freq = 0.05, max_doc_freq = 0.3
## NB performance
# Accuracy: 0.8000
# Precision: 0.8214
# Recall: 0.7419


#### with n-grams ####


## tried many variations with only very poor Accuracy and Precision, Recall OK

undersample_dfm <- preprocess_text(undersample_corpus, min_term = 10, max_term = 500,
                            min_doc_freq = 0.3, max_doc_freq = 0.75,
                            ngram = 2)
#dim(undersample_dfm)
## Convert DFM to matrix
undersample_dfm_matrix <- convert(undersample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(undersample_dfm_matrix, undersample_dfm$label)



###### OVER SAMPLING ######

# full dataset
all_lyrics <- rbind(gaga, not_gaga)
#dim(all_lyrics)

# oversampling using ROSE
oversample <- ovun.sample(label~., data = all_lyrics, 
                          method = "over", seed = 328)$data
dim(oversample)

## Remove instrumentals/songs without lyrics
oversample <- dplyr::filter(oversample, nchar(oversample$'Lyric') >= 5)
## create corpus
oversample_corpus <- corpus(oversample, text = 'Lyric')


#### Without n-grams ####

## Preprocessing with defaults
## SVM performance
# Accuracy: 0.8405
# Precision: 0.7843
# Recall: 0.9389
## NB performance
# Accuracy: 0.7742
# Precision: 0.7716
# Recall: 0.7784
oversample_dfm <- preprocess_text(oversample_corpus)
#dim(undersample_dfm)
## Convert DFM to matrix
oversample_dfm_matrix <- convert(oversample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(oversample_dfm_matrix, oversample_dfm$label)


##################### BEST PERFORMANCE OVERALL ##################### 

## SVM
# Accuracy: 0.9651 ## OK THEN!
# Precision: 0.9420 ## WOOOOOOO
# Recall: 0.9913 ## NICE.
## NB
# Accuracy: 0.8901
# Precision: 0.9340
# Recall: 0.8394
oversample_dfm <- preprocess_text(oversample_corpus, min_doc_freq = 0.01)
## Convert DFM to matrix
oversample_dfm_matrix <- convert(oversample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(oversample_dfm_matrix, oversample_dfm$label)


##################### pretty good ##################### 

## SVM
# Accuracy: 0.9477
# Precision: 0.9171
# Recall: 0.9843
## NB
# Accuracy: 0.8378
# Precision: 0.8512
# Recall: 0.8185
oversample_dfm <- preprocess_text(oversample_corpus, min_doc_freq = 0.05)
## Convert DFM to matrix
oversample_dfm_matrix <- convert(oversample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(oversample_dfm_matrix, oversample_dfm$label)


#### with n-grams ####

## SVM performance
# Accuracy: 0.701
# Precision: 0.6620
# Recall: 0.8202
## NB performance
# Accuracy: 0.653
# Precision: 0.6874
# Recall: 0.5602
oversample_dfm <- preprocess_text(oversample_corpus, ngram = 2)
## Convert DFM to matrix
oversample_dfm_matrix <- convert(oversample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(oversample_dfm_matrix, oversample_dfm$label)


##################### the next 2 were pretty good ##################### 

## SVM
# Accuracy: 0.9643
# Precision: 0.9463
# Recall: 0.9843
## NB
# Accuracy: 0.9233
# Precision: 0.9516
# Recall: 0.8918
oversample_dfm <- preprocess_text(oversample_corpus, ngram = 2,
                                  min_doc_freq = 0.01)
## Convert DFM to matrix
oversample_dfm_matrix <- convert(oversample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(oversample_dfm_matrix, oversample_dfm$label)


## SVM
# Accuracy: 0.9189
# Precision: 0.8797
# Recall: 0.9703
## NB
# Accuracy: 0.7847
# Precision: 0.8221
# Recall: 0.7260
oversample_dfm <- preprocess_text(oversample_corpus, ngram = 2,
                                  min_doc_freq = 0.05)
## Convert DFM to matrix
oversample_dfm_matrix <- convert(oversample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(oversample_dfm_matrix, oversample_dfm$label)



###### under-sampling with ROSE ######


# under-sampling using ROSE
undersample_rose <- ovun.sample(label~., data = all_lyrics, 
                          method = "under", seed = 328)$data
dim(undersample_rose)

## Remove instrumentals/songs without lyrics
undersample_rose <- dplyr::filter(undersample_rose, nchar(undersample_rose$'Lyric') >= 5)
## create corpus
undersample_rose_corpus <- corpus(undersample_rose, text = 'Lyric')

#### without using n-grams ####

## SVM performance
# Accuracy: 0.597
# Precision: 0.6129
# Recall: 0.5588
## NB performance
# Accuracy: 0.6866
# Precision: 0.7407
# Recall: 0.5882
undersample_rose_dfm <- preprocess_text(undersample_rose_corpus)
## Convert DFM to matrix
undersample_rose_dfm_matrix <- convert(undersample_rose_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(undersample_rose_dfm_matrix, undersample_rose_dfm$label)


## SVM performance
# Accuracy: 0.7313
# Precision: 0.7500
# Recall: 0.7059
## NB performance
# Accuracy: 0.7761
# Precision: 0.8519
# Recall: 0.6765
undersample_rose_dfm <- preprocess_text(undersample_rose_corpus,
                                        min_doc_freq = 0.01)
## Convert DFM to matrix
undersample_rose_dfm_matrix <- convert(undersample_rose_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(undersample_rose_dfm_matrix, undersample_rose_dfm$label)


#### with n-grams ####

## SVM performance
# Accuracy: 0.6567
# Precision: 0.6667
# Recall: 0.6471
## NB performance
# Accuracy: 0.6866
# Precision: 0.7826
# Recall: 0.5294
undersample_rose_dfm <- preprocess_text(undersample_rose_corpus, ngram = 2)
## Convert DFM to matrix
undersample_rose_dfm_matrix <- convert(undersample_rose_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(undersample_rose_dfm_matrix, undersample_rose_dfm$label)


## SVM performance
# Accuracy: 0.5224
# Precision: 0.5385
# Recall: 0.4118
## NB performance
# Accuracy: 0.7463
# Precision: 0.9474
# Recall: 0.5294
undersample_rose_dfm <- preprocess_text(undersample_rose_corpus,
                                        ngram = 2,
                                        min_doc_freq = 0.01)
## Convert DFM to matrix
undersample_rose_dfm_matrix <- convert(undersample_rose_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(undersample_rose_dfm_matrix, undersample_rose_dfm$label)



### n-grams once again disappoint us!




###### let's try BOTH under and over-sampling together ######


rose_both <- ovun.sample(label~., data = all_lyrics, 
                                method = "both", seed = 328)$data
dim(rose_both)

## Remove instrumentals/songs without lyrics
rose_both <- dplyr::filter(rose_both, nchar(rose_both$'Lyric') >= 5)
## create corpus
rose_both_corpus <- corpus(rose_both, text = 'Lyric')

#### without using n-grams ####

## SVM performance
# Accuracy: 0.8713
# Precision: 0.8314
# Recall: 0.9387
## NB performance
# Accuracy: 0.7789
# Precision: 0.7876
# Recall: 0.7774
rose_both_dfm <- preprocess_text(rose_both_corpus)
## Convert DFM to matrix
rose_both_dfm_matrix <- convert(rose_both_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(rose_both_dfm_matrix, rose_both_dfm$label)


##################### next 2 have good performance ##################### 

## SVM performance
# Accuracy: 0.9472
# Precision: 0.9238
# Recall: 0.9774
## NB performance
# Accuracy: 0.8201
# Precision: 0.8190
# Recall: 0.8323
rose_both_dfm <- preprocess_text(rose_both_corpus,
                                 min_doc_freq = 0.05)
## Convert DFM to matrix
rose_both_dfm_matrix <- convert(rose_both_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(rose_both_dfm_matrix, rose_both_dfm$label)

## SVM performance
# Accuracy: 0.9488
# Precision: 0.9139
# Recall: 0.9935
## NB performance
# Accuracy: 0.9043
# Precision: 0.9257
# Recall: 0.8839
rose_both_dfm <- preprocess_text(rose_both_corpus,
                                 min_doc_freq = 0.01)
## Convert DFM to matrix
rose_both_dfm_matrix <- convert(rose_both_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(rose_both_dfm_matrix, rose_both_dfm$label)


#### with n-grams ####

## SVM performance
# Accuracy: 0.6896
# Precision: 0.6572
# Recall: 0.8226
## NB performance
# Accuracy: 0.6188
# Precision: 0.6502
# Recall: 0.5516
rose_both_dfm <- preprocess_text(rose_both_corpus, ngram = 2)
## Convert DFM to matrix
rose_both_dfm_matrix <- convert(rose_both_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(rose_both_dfm_matrix, rose_both_dfm$label)

## SVM performance
# Accuracy: 0.9092
# Precision: 0.8632
# Recall: 0.9774
## NB performance
# Accuracy: 0.7706
# Precision: 0.7979
# Recall: 0.7387
rose_both_dfm <- preprocess_text(rose_both_corpus,
                                 ngram = 2,
                                 min_doc_freq = 0.05)
## Convert DFM to matrix
rose_both_dfm_matrix <- convert(rose_both_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(rose_both_dfm_matrix, rose_both_dfm$label)

## SVM performance
# Accuracy: 0.9521
# Precision: 0.9271
# Recall: 0.9839
## NB performance
# Accuracy: 0.9125
# Precision: 0.9241
# Recall: 0.9032
rose_both_dfm <- preprocess_text(rose_both_corpus,
                                 ngram = 2,
                                 min_doc_freq = 0.01)
## Convert DFM to matrix
rose_both_dfm_matrix <- convert(rose_both_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(rose_both_dfm_matrix, rose_both_dfm$label)

