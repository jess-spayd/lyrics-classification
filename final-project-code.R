library(tidyverse)
library(quanteda)
library(ROSE)
library(e1071)
library(caret)
library(quanteda.textplots)
library(quanteda.textstats)
library(wordcloud2)


# Import data
setwd('~/Documents/GitHub/text-as-data-sp23/final project')
lyrics <- read.csv('lyrics-dataset-deduped.csv',
                   header=FALSE,
                   stringsAsFactors=FALSE)
colnames(lyrics) <- c('Artist', 'Title', 'Album', 'Date', 'Lyric', 'Year')
#head(lyrics)


# Split dataset to have equal sample of gaga and non-gaga songs
## Add labels for classification
gaga <- subset(lyrics, Artist == 'Lady Gaga')
gaga$label <- 'gaga'
not_gaga <- subset(lyrics, Artist != 'Lady Gaga')
not_gaga$label <- 'not gaga'
#dim(gaga)
#dim(not_gaga)

# full dataset
all_lyrics <- rbind(gaga, not_gaga)

# oversampling using ROSE
oversample <- ovun.sample(label~., data = all_lyrics, 
                          method = "over", seed = 328)$data
#dim(oversample)

## Remove instrumentals/songs without lyrics
oversample <- dplyr::filter(oversample, nchar(oversample$'Lyric') >= 5)
## create corpus
data_corpus <- corpus(oversample, text = 'Lyric')

# to remove "Lady" and "Gaga"
gaga_stopwords <- c('lady', 'gaga', 'pre')

# Pre-processing
data_dfm <- data_corpus %>% 
  tokens(remove_punct = TRUE,
         remove_symbols = TRUE,
         remove_numbers = TRUE,
         remove_url = TRUE) %>%
  dfm(tolower=TRUE) %>%
  dfm_remove(c(stopwords('english'), gaga_stopwords)) %>%
  dfm_wordstem() %>%
  dfm_trim(min_termfreq = 15)
dim(data_dfm)
# Convert DFM to matrix
data_dfm_matrix <- convert(data_dfm, to='matrix')


# Train-Test split
set.seed(328)
sample <- sample.int(n = nrow(data_dfm_matrix),
                     size = floor(.8*nrow(data_dfm_matrix)), 
                     replace = F)
train_matrix <- data_dfm_matrix[sample,]
train_labels <- as.factor(data_dfm$label[sample])
test_matrix <- data_dfm_matrix[-sample,] 
test_labels <- as.factor(data_dfm$label[-sample])

# Train SVM model
svm_model <- svm(
  x = train_matrix,
  y = train_labels,
  kernel = "linear")

# Generate predictions
pred_svm <- predict(svm_model, test_matrix)

# Confusion Matrix and Scores
confusionMatrix(data=pred_svm, 
                reference = test_labels,
                mode = 'prec_recall')


##### VISUALIZATIONS #####

## original distribution
ggplot(all_lyrics, aes(y=fct_rev(fct_infreq(as.factor(Artist))))) +
  geom_bar() +
  labs(title='Figure 2: Artists in Original Dataset',
       x= 'Number of Songs',
       y= 'Artist') +
  theme(legend.position = 'none',
        panel.grid.major.x = element_line(color = 'lightgray'), 
        panel.grid.minor.x = element_line(color = 'lightgray'),
        panel.grid.major.y = element_blank(), 
        panel.grid.minor.y = element_blank(),
        panel.background = element_blank(), 
        axis.line = element_line(colour = "black"))



## oversample distribution


ggplot(oversample, aes(x=label)) +
  geom_histogram(stat='count') +
  stat_count(geom = "text", colour = "black", size = 3.5,
             aes(label = ..count..),position=position_stack(vjust=1.05)) +
  labs(title='Figure 3: Distribution of Labels in Final Dataset',
       x= 'Label',
       y= 'Number of Songs') +
  theme(legend.position = 'none',
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))


ggplot(oversample, aes(y=fct_rev(fct_infreq(as.factor(Artist))))) +
  geom_bar()+
  labs(title='Figure 4: Distribution of Artists in Final Dataset',
       x= 'Artist',
       y= 'Number of Songs') +
  theme(legend.position = 'none',
        panel.grid.major.x = element_line(color = 'lightgray'), 
        panel.grid.minor.x = element_line(color = 'lightgray'),
        panel.grid.major.y = element_blank(), 
        panel.grid.minor.y = element_blank(),
        panel.background = element_blank(), 
        axis.line = element_line(colour = "black"))



gaga_dfm_new <- dfm_subset(data_dfm, label == 'gaga')
gaga_dfm_freq <- textstat_frequency(gaga_dfm_new, n=100)

### top features for gaga songs
gaga_dfm_freq %>%
  dplyr::arrange(desc(frequency)) %>%
  slice(1:20) %>%
  ggplot(., aes(x=reorder(feature, +frequency), y=frequency, fill=feature)) + 
  geom_point() + 
  geom_segment( aes(x=feature, xend=feature, y=0, yend=frequency)) +
  coord_flip() +
  labs(title='Figure 6: Top Features in Lady Gaga Lyrics (Over-Sampling)',
       x= 'Feature',
       y= 'Frequency') +
  theme(legend.position = 'none',
        panel.grid.major.x = element_line(color = 'lightgray'), 
        panel.grid.minor.x = element_line(color = 'lightgray'),
        panel.grid.major.y = element_blank(), 
        panel.grid.minor.y = element_blank(),
        panel.background = element_blank(), 
        axis.line = element_line(colour = "black"))


## features in most songs

gaga_dfm_freq %>%
  dplyr::arrange(desc(docfreq)) %>%
  slice(1:20) %>%
  ggplot(., aes(x=reorder(feature, +docfreq), y=docfreq, fill=feature)) + 
  geom_point() + 
  geom_segment( aes(x=feature, xend=feature, y=0, yend=docfreq)) +
  coord_flip() +
  labs(title='Figure 8: Features that Appear in Many Lady Gaga Songs (Over-Sampling)',
       x= 'Feature',
       y= 'Document Frequency') +
  theme(legend.position = 'none',
        panel.grid.major.x = element_line(color = 'lightgray'), 
        panel.grid.minor.x = element_line(color = 'lightgray'),
        panel.grid.major.y = element_blank(), 
        panel.grid.minor.y = element_blank(),
        panel.background = element_blank(), 
        axis.line = element_line(colour = "black"))


## word cloud & top features in original dataset
gaga <- dplyr::filter(gaga, nchar(gaga$'Lyric') >= 5)
gaga_corpus <- corpus(gaga, text = 'Lyric')
gaga_dfm <- gaga_corpus %>% 
  tokens(remove_punct = TRUE,
         remove_symbols = TRUE,
         remove_numbers = TRUE,
         remove_url = TRUE) %>%
  dfm(tolower=TRUE) %>%
  dfm_remove(c(stopwords('english'), gaga_stopwords))

# wordcloud
gaga_dfm_orig <- textstat_frequency(gaga_dfm, n=100)
wordcloud2(gaga_dfm_orig, 
           size = 0.5, 
           shape = 'diamond',
           fontFamily = 'Arial',
           color='darkturquoise',
           minRotation = 0,
           maxRotation = 0)


### top features for gaga songs
gaga_dfm_orig %>%
  dplyr::arrange(desc(frequency)) %>%
  slice(1:20) %>%
  ggplot(., aes(x=reorder(feature, +frequency), y=frequency, fill=feature)) + 
  geom_point() + 
  geom_segment( aes(x=feature, xend=feature, y=0, yend=frequency)) +
  coord_flip() +
  labs(title='Figure 5: Top Features in Lady Gaga Lyrics (Original Dataset)',
       x= 'Feature',
       y= 'Frequency') +
  theme(legend.position = 'none',
        panel.grid.major.x = element_line(color = 'lightgray'), 
        panel.grid.minor.x = element_line(color = 'lightgray'),
        panel.grid.major.y = element_blank(), 
        panel.grid.minor.y = element_blank(),
        panel.background = element_blank(), 
        axis.line = element_line(colour = "black"))


## features in most songs

gaga_dfm_orig %>%
  dplyr::arrange(desc(docfreq)) %>%
  slice(1:20) %>%
  ggplot(., aes(x=reorder(feature, +docfreq), y=docfreq, fill=feature)) + 
  geom_point() + 
  geom_segment( aes(x=feature, xend=feature, y=0, yend=docfreq)) +
  coord_flip() +
  labs(title='Figure 7: Features that Appear in Many Lady Gaga Songs (Original Dataset)',
       x= 'Feature',
       y= 'Document Frequency')  +
  theme(legend.position = 'none',
        panel.grid.major.x = element_line(color = 'lightgray'), 
        panel.grid.minor.x = element_line(color = 'lightgray'),
        panel.grid.major.y = element_blank(), 
        panel.grid.minor.y = element_blank(),
        panel.background = element_blank(), 
        axis.line = element_line(colour = "black"))

