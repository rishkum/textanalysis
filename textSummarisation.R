

######3 Load Libraries ########
library(XML)
library(RCurl)
library(rvest)
library(pdftools)
library(tidyverse)
#Inser Url
my_urls <- c("https://www.bankofengland.co.uk/-/media/boe/files/speech/2018/cyborg-supervision-speech-by-james-proudman.pdf?la=en&hash=6FFE5A1D19EAA76681DB615D9054C53DB8823AB4.pdf")

#declare saving name
save_here <- paste0("document", ".pdf")

#run loop: useful for multiple files
mapply(download.file, my_urls, save_here)
file <- "document_.pdf"
#Load file as txt
df <- pdf_text(pdf = "document_.pdf")

text_df <- tibble( text = df)
text_df <- text_df %>% rownames_to_column('sections') %>% filter(!(sections == 1|
                                                        sections == 7|
                                                        sections == 9)) 
                                                        
library(tidytext)

text_df <- text_df %>%
  unnest_tokens(word, text)

#Load data properly
text_df <- pdf_text("document_.pdf") %>% strsplit(split = "\n")
text_df <- text_df[2:(length(text_df)-2)]
text_df <- unlist(text_df)
text_df <- trimws(text_df, "l")
text_df <- as_tibble(text_df)

# remove page numers and footers
text_df$rem <- grepl("^[0-9]+$", text_df$value)
# remove page citations
text_df$rem2 <- grepl("\\(.*\\)$", text_df$value)
# remove headings
text_df$rem3 <- grepl("^[XVI]+[.XVI]", text_df$value) 

text_df <- text_df %>% filter(!(rem==T| rem2==T | rem3 == T))

dat.merged <- text_df %>%
  dplyr::group_by(rem) %>%
  dplyr::summarise(value = paste(value, collapse = ""))
print(head(dat.merged))

top_3 = lexRankr::lexRank(text_df$value,
                          #only 1 article; repeat same docid for all of input vector
                          docId = rep(1, length(text_df$value)),
                          #return 3 sentences to mimick /u/autotldr's output
                          n = 5,
                          continuous = TRUE)

#reorder the top 3 sentences to be in order of appearance in article
order_of_appearance = order(as.integer(gsub("_","",top_3$sentenceId)))
#extract sentences in order of appearance
ordered_top_3 = top_3[order_of_appearance, "sentence"]
ordered_top_3

data(stop_words)
tidy_books <- text_df %>%
  anti_join(stop_words)

grepl("^[0-9]", text_df) %>% sum

tidy_books %>% count(word, sort = TRUE) 


library(ggplot2)

tidy_books %>%
  count(word, sort = TRUE) %>%
  filter(n > 5) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col() +
  xlab(NULL) +
  coord_flip()

nrc_joy <- get_sentiments("nrc")

tidy_books %>%
  filter(book == "Emma") %>%
  inner_join(nrc_joy) %>%
  count(word, sort = TRUE)

                                                 

# load file as corpus
library(tm)
corp <- Corpus(URISource(file),
               readerControl = list(reader = readPDF))

opinions.tdm <- TermDocumentMatrix(corp, 
                                   control = 
                                     list(removePunctuation = TRUE,
                                          stopwords = TRUE,
                                          tolower = TRUE,
                                          stemming = TRUE,
                                          removeNumbers = TRUE,
                                          bounds = list(global = c(1, Inf)))) 
inspect(opinions.tdm[1:10,]) 

corp <- tm_map(corp, removePunctuation, ucp = TRUE)


opinions.tdm <- TermDocumentMatrix(corp, 
                                   control = 
                                     list(stopwords = TRUE,
                                          tolower = TRUE,
                                          stemming = TRUE,
                                          removeNumbers = TRUE,
                                          bounds = list(global = c(1, Inf)))) 

ft <- findFreqTerms(opinions.tdm, lowfreq = 5, highfreq = Inf)
ft.tdm <- as.matrix(opinions.tdm[ft,])
sort(apply(ft.tdm, 1, sum), decreasing = TRUE)


tdm <- tm::DocumentTermMatrix(corp) 
tdm.tfidf <- tm::weightTfIdf(tdm)

tdm.tfidf <- tm::removeSparseTerms(tdm.tfidf, 0.999) 
tfidf.matrix <- as.matrix(tdm.tfidf) 
# Cosine distance matrix (useful for specific clustering algorithms) 
dist.matrix = proxy::dist(opinions.tdm, method = "cosine")
truth.K =4

clustering.kmeans <- kmeans(dist.matrix, truth.K) 
clustering.hierarchical <- hclust(dist.matrix, method = "ward.D2") 
clustering.dbscan <- dbscan::hdbscan(dist.matrix, minPts = 10)


points <- cmdscale(dist.matrix, k = 2) 
palette <- colorspace::diverge_hcl(truth.K) # Creating a color palette 
previous.par <- par(mfrow=c(2,2), mar = rep(1.5, 4)) 


master.cluster <- clustering.kmeans$cluster 
slave.hierarchical <- cutree(clustering.hierarchical, k = truth.K) 
slave.dbscan <- clustering.dbscan$cluster 
stacked.clustering <- rep(NA, length(master.cluster))  
names(stacked.clustering) <- 1:length(master.cluster) 
for (cluster in unique(master.cluster)) { 
  indexes = which(master.cluster == cluster, arr.ind = TRUE) 
  slave1.votes <- table(slave.hierarchical[indexes]) 
  slave1.maxcount <- names(slave1.votes)[which.max(slave1.votes)]   
  slave1.indexes = which(slave.hierarchical == slave1.maxcount, arr.ind = TRUE) 
  slave2.votes <- table(slave.dbscan[indexes]) 
  slave2.maxcount <- names(slave2.votes)[which.max(slave2.votes)]   
  stacked.clustering[indexes] <- slave2.maxcount 
}

plot(points, main = 'K-Means clustering', col = as.factor(master.cluster), 
     mai = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), 
     xaxt = 'n', yaxt = 'n', xlab = '', ylab = '') 
plot(points, main = 'Hierarchical clustering', col = as.factor(slave.hierarchical), 
     mai = c(0, 0, 0, 0), mar = c(0, 0, 0, 0),  
     xaxt = 'n', yaxt = 'n', xlab = '', ylab = '') 
plot(points, main = 'Density-based clustering', col = as.factor(slave.dbscan), 
     mai = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), 
     xaxt = 'n', yaxt = 'n', xlab = '', ylab = '') 
plot(points, main = 'Stacked clustering', col = as.factor(stacked.clustering), 
     mai = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), 
     xaxt = 'n', yaxt = 'n', xlab = '', ylab = '') 
