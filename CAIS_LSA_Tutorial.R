##############################################################
##############################################################
## First example. Banking and financial services complaints ##
##############################################################
##############################################################


# Financial complaint data downloaded from "https://catalog.data.gov/dataset/consumer-complaint-database"
# Only complaints with more than 200 words in their complaint narrative (43038 complaints)
# Take 1/18 of those.

# Section 3.1
# Load required code libraries

library(LSAfun)
library(lsa)

#########################
# Replace the [...] with the path in which the directory MiniComplaints is to be created.
source_dir = '[...]'
# source_dir = 'C:/Users/username/Desktop/MiniComplaints' # Windows example
# source_dir = '~/Desktop/MiniComplaints' # Mac/Unix example 

#########################
data(stopwords_en) 
print(stopwords_en)  # The complete list of stop words can be shown by using Print. 
# or just entering 
stopwords_en

#########################
TDM <- textmatrix(source_dir, stopwords=c(stopwords_en, "xx", "xxxx"), stemming=TRUE, removeNumber=F, minGlobFreq=2) 
# Optionally, we can show the content of the matrix TDM by just typing the dataset name
TDM

#########################
summary.textmatrix(TDM)

#########################
TDM2 <- lw_tf(TDM) * gw_idf(TDM) 
TDM2

#########################
miniLSAspace <- lsa(TDM2, dims=dimcalc_share()) 
as.textmatrix(miniLSAspace) 

#########################
# This command will show the value-weighted matrix of Terms
tk2 = t(miniLSAspace$sk * t(miniLSAspace$tk))
tk2

#########################
# This will show the matrix of Documents
miniLSAspace$dk

#########################
# Because the $sk matrix only has values on the diagonal, R stores it as a numeric vector. 
miniLSAspace$sk

#########################
miniLSAspace3 <- lsa(TDM2, dims=3) 
tk3 = t(miniLSAspace3$sk * t(miniLSAspace3$tk)) 
tk3 

#########################
# The two lines of code must be run together. The first line of code creates a plot of the first two 
# dimensions of $tk, marking the dots as red dots. The second line superimposes term names. 
plot(tk2[,1], y= tk2[,2], col="red", cex=.50, main="TK Plot")
text(tk2[,1], y= tk2[,2], labels=rownames(tk2) , cex=.70)
# This can be done with the documents too. The added parameter cex determines text size. 
plot(dk2[,1], y= dk2[,2], col="blue", pch="+", main="DK Plot")
text(dk2[,1], y= dk2[,2], labels=rownames(dk2), cex=.70)

#########################
# Section 3.2
#
# Create a cosine similarity between two Terms
myCo <- costring('loan','chang', tvectors= tk2, breakdown=TRUE)
myCo                          # Typing the name of an object prints its value
myCo <- costring('loan','due', tvectors= miniLSAspace$tk, breakdown=T)
myCo

#########################
myDocs <- rownames(dk2)
myDocs
myTerms <- rownames(tk2)
myTerms

#########################
myTerms2 <- rownames(tk2)
myCosineSpace2 <- multicos(myTerms2, tvectors=tk2, breakdown=TRUE)
myCosineSpace2

#########################
# Save the cosine space (the user should define the path within file="...")
write.csv(myCosineSpace2, file="C:/Users/.../CosineResults.csv")

#########################
# This provides us with a similarity matrix between documents
myCosineSpace3 <- multicos(myDocs, tvectors=dk2, breakdown=F)
myCosineSpace3

#########################
neighbors("credit", n=5, tvectors=tk3, breakdown=TRUE)

#########################
plot_neighbors("credit", n=20, tvectors= tk3)  

#########################
words <- c("credit","card", "time", "supervisor") 
plot_wordlist(words,tvectors=tk3,dims=2)

#########################
plot_wordlist(words,tvectors= tk3, method="PCA", dims=3,connect.lines="all")

#########################
associate(tk3, "credit", measure="cosine", threshold=0.95)

#########################
X <- c('credit', 'supervisor')
Y <- c('mortgag', 'account')
myCo <- costring(X,Y, tvectors=miniLSAspace$tk, breakdown=TRUE)
myCo

#########################
mcTerms <- multicos(c('credit', 'supervisor', 'mortgag', 'account'), tvectors= miniLSAspace$tk, breakdown=F)
mcTerms 

#########################
trans_tk <- t(as.matrix(tk3))
trans_dk <- t(as.matrix(dk3))
cor(trans_tk, use="complete.obs", method="spearman")
cor(trans_dk, use="pairwise.complete.obs", method="kendall")

#########################
?coherence

#########################
install.packages("wordcloud", dependencies = TRUE)
library(wordcloud)

#########################
Term_count <-apply(TDM2,1,sum)
TCT <- t(Term_count)
wordcloud(myTerms, TCT, min.freq=1, random.order=FALSE, color=brewer.pal(8, "Dark2"))


#########################
# Section 3.3
#
# Load required code libraries
library(cluster)
library(tm)
library(LSAfun)

#########################
# Replace the [...] with the path in to the FinancialComplaints directory.
source_dir = '[...]'
# source_dir = 'C:/Users/username/Desktop/FinancialComplaints' # Windows example
# source_dir = '~/Desktop/FinancialComplaints' # Mac/Unix example
# We shall now create a corpus in memory 
raw_corpus <- VCorpus(doc_source, readerControl=list(language='en'))

#########################
stoplist <- c(stopwords("en"), "xx", "xxxx","xx/xx/xxxx","xxxx/xxxx/", "xxxxxxxxxxxx","xxxxxxxx")
tdm <- TermDocumentMatrix(raw_corpus, 
	control=list(removePunctuation = TRUE,
		removeNumbers = TRUE,
		tolower = TRUE,
		stopwords = stoplist, 
		stemming = TRUE, # snowball stemmer
		weighting = function(x) weightTfIdf(x, normalize = FALSE), # Weight with tf-idf
		bounds=list(global=c(5,Inf)))) # Keep only 5 or more appearances, to accelerate 
			# space creation for purposes of this tutorial
# The tdm matrix is very sparse
tdm

#########################
# Still, it may be very sparse, but inspecting it we can show the occasional non-zero value
inspect(tdm[10:20,11:19])

#########################
findFreqTerms(tdm, 3000)

#########################
myLSAspace <- lsa(tdm, dims=dimcalc_share());
dim(myLSAspace$tk)  # Check how many rows/columns the tk matrix has 
myLSAtk = t(myLSAspace$sk * t(myLSAspace$tk))
plot_neighbors("trust",n=20,tvectors= myLSAtk[,1:70])  # Use only the first 70 dimensions

#########################
install.packages("gplots", dependencies = TRUE)
library(gplots)

# Extract the closest words to “trust” (a list of their distances as a named vector).
words<-neighbors("trust",n=20,tvectors= myLSAtk[,1:70])  

# Extract the actual words, and find the distances in the space.
myCosineSpace2 <- multicos(names(words), tvectors= myLSAtk[,1:70], breakdown=TRUE)
heatmap.2(myCosineSpace2)

#########################
costring("trust believ", "reconcil loss", tvectors= myLSAspace$tk[,1:75], breakdown=T) 
costring("trust believ", "fraud prevent", tvectors= myLSAspace$tk[,1:75], breakdown=T)

#########################
# Section 3.4
#
if (!require("tm")) {
   install.packages("tm", dependencies = TRUE)
   library(tm)
   }
if (!require("RSpectra")) {
   install.packages("RSpectra", dependencies = TRUE)
   library(RSpectra)
   }
if (!require("LSAfun")) {
   install.packages("LSAfun", dependencies = TRUE)
   library(LSAfun)
    }
if (!require("gplots")) {
   install.packages("gplots", dependencies = TRUE)
   library(gplots)
   }

#########################
doc_source <- DirSource('C:/.../StackExchange')

#########################
raw_corpus <- VCorpus(doc_source, readerControl=list(language='en'))

#########################
remove_nonletter <- function(text) { return(gsub('[^a-z\\s\\-]+', ' ', text))}

#########################
p_corpus <- tm_map(raw_corpus, content_transformer(tolower))p_corpus <- tm_map(p_corpus, content_transformer(removeWords), tm::stopwords('en'))p_corpus <- tm_map(p_corpus, content_transformer(remove_nonletter))p_corpus <- tm_map(p_corpus, stemDocument)

#########################
tdm <- TermDocumentMatrix(p_corpus, control = list(bounds = list(global = c(10, Inf))))

#########################
sparse_tdm <- Matrix::sparseMatrix(i = tdm$i, j = tdm$j, x = tdm$v, dims = c(tdm$nrow, tdm$ncol))

#########################
dimnames(sparse_tdm) <- dimnames(tdm)

#########################
doc_count <- dim(sparse_tdm)[[2]]log_doc_count <- log2(doc_count)

#########################
weighted_tdm <- sparse_tdmweighted_tdm@x <- vapply(sparse_tdm@x, function(x) log2(x+.00001), numeric(1))

#########################
gf <- Matrix::rowSums(sparse_tdm)names(gf) <- dimnames(sparse_tdm)$Terms

#########################
partial_entropy <- function(tf, gf) {	p <- tf/gf	return((p*log2(p))/log_doc_count)}

#########################
word_entropy <- numeric(dim(sparse_tdm)[[1]])names(word_entropy) <- dimnames(sparse_tdm)$Termsfor(i in 1:dim(sparse_tdm)[[1]]){    word_row <- sparse_tdm[i,]    non_zero_frequencies <- word_row[which(word_row>0)]    word_entropy[i] <- 1.0 + sum(mapply(partial_entropy, non_zero_frequencies, gf=gf[i]))}

#########################
weighted_tdm <- sweep(weighted_tdm, 1, word_entropy, '*')

#########################
space <- svds(weighted_tdm, 300)                                                            su_mat <- space$d * space$usvt_mat <- space$d * Matrix::t(space$v)#Assign namesdimnames(su_mat) <- list(dimnames(weighted_tdm)[[1]], 1:300)   dimnames(svt_mat) <- list(1:300, dimnames(weighted_tdm)[[2]])  

#########################
plot_neighbors("python",n=20,tvectors= su_mat)plot_neighbors("java",n=20,tvectors= su_mat)plot_neighbors("javascript",n=20,tvectors= su_mat)

#########################
if (!require("wordcloud")) {   install.packages("wordcloud", dependencies = TRUE)   library(wordcloud)   }Term_count <-apply(su_mat,1,sum)TCT <- t(Term_count)myTerms <- rownames(su_mat)wordcloud(myTerms, TCT, min.freq=1, random.order=FALSE, color=brewer.pal(8, "Dark2"))

#########################
costring("package answer", "JAVA", tvectors= su_mat, breakdown=TRUE)costring("package answer", "Python", tvectors= su_mat, breakdown=TRUE)

#########################
pseudo <- 'Tell me about overflow problems'pseudo <- tolower(pseudo)pseudo <- removeWords(pseudo, tm::stopwords('en'))pseudo <- remove_nonletter(pseudo)pseudo <- stemDocument(PlainTextDocument(pseudo))pseudo <- termFreq(pseudo)pseudo <- vapply(pseudo, function(x) log2(x+.00001), numeric(1))pseudo <- mapply(function(x, y) x*y, pseudo, word_entropy[names(pseudo)])pseudo <- colSums(pseudo * su_mat[names(pseudo),])

#########################
neighbors(pseudo, 20, tvectors=su_mat)neighbors(pseudo, 20, tvectors=Matrix::t(svt_mat))

#########################
ls()rm(doc_source)	

#########################
