library(tidyverse)
library(quanteda)
library(ROSE)
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

# full dataset
all_lyrics <- rbind(gaga, not_gaga)

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


# classification function
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
  
  # Probability table for features
  prob_table <- tables(nb_model)
  # Top features for Lady Gaga
  gaga_top_feat <- prob_table[order(prob_table[,1],decreasing=TRUE),]

  return(gaga_top_feat)
}


## OVER SAMPLING

# oversampling using ROSE
oversample <- ovun.sample(label~., data = all_lyrics, 
                          method = "over", seed = 328)$data
#dim(oversample)

## Remove instrumentals/songs without lyrics
oversample <- dplyr::filter(oversample, nchar(oversample$'Lyric') >= 5)
## create corpus
oversample_corpus <- corpus(oversample, text = 'Lyric')


## UNDER SAMPLING

# under-sampling using ROSE
undersample_rose <- ovun.sample(label~., data = all_lyrics, 
                                method = "under", seed = 328)$data
#dim(undersample_rose)

## Remove instrumentals/songs without lyrics
undersample_rose <- dplyr::filter(undersample_rose, nchar(undersample_rose$'Lyric') >= 5)
## create corpus
undersample_rose_corpus <- corpus(undersample_rose, text = 'Lyric')



## BOTH

rose_both <- ovun.sample(label~., data = all_lyrics, 
                         method = "both", seed = 328)$data
#dim(rose_both)

## Remove instrumentals/songs without lyrics
rose_both <- dplyr::filter(rose_both, nchar(rose_both$'Lyric') >= 5)
## create corpus
rose_both_corpus <- corpus(rose_both, text = 'Lyric')


## testing the best models to see if low min_docfreq is overfitting


## SVM
# Accuracy: 0.9651 
# Precision: 0.9420
# Recall: 0.9913
## NB
# Accuracy: 0.8901
# Precision: 0.9340
# Recall: 0.8394
oversample_dfm <- preprocess_text(oversample_corpus, min_doc_freq = 0.01)
## Convert DFM to matrix
oversample_dfm_matrix <- convert(oversample_dfm, to='matrix')
## Classification: gaga or not gaga?
classification(oversample_dfm_matrix, oversample_dfm$label)


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


