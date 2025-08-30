# Load required libraries
library("caret")      # for machine learning utilities and model training
library("quanteda")   # for text mining and NLP preprocessing
library(ggplot2)      # for plotting

# Step 1: Load dataset (choose sms_spam.csv file)
data.raw <- read.csv(file.choose(), stringsAsFactors = FALSE, fileEncoding = "UTF-8")

# Step 2: Convert target column "type" (ham/spam) into a factor
data.raw$type <- as.factor(data.raw$type)

# Check class distribution (ham vs spam)
prop.table(table(data.raw$type))

# Step 3: Separate ham and spam datasets
indexes <- which(data.raw$type == "ham")
ham <- data.raw[indexes, ]
spam <- data.raw[-indexes, ]

# Add a new variable "TextLength" = number of characters in each message
spam$TextLength <- nchar(spam$text)
ham$TextLength <- nchar(ham$text)

# View length statistics for each class
summary(ham$TextLength)
summary(spam$TextLength)

# Add TextLength column to original dataset
data.raw$TextLength <- nchar(data.raw$text)

# Step 4: Plot histogram of text lengths by class
ggplot(data.raw, aes(x = TextLength, fill = type)) + 
  theme_bw() + 
  geom_histogram(binwidth = 5) + 
  labs(y = "Text Count", x = "Length of Text", 
       title = "Distribution of Text Lengths with Class Labels")

# Step 5: Train-Test Split (70% train, 30% test)
set.seed(32984)
indexes <- createDataPartition(data.raw$type, times = 1, p = 0.7, list = FALSE)
train.raw <- data.raw[indexes, ]
test.raw <- data.raw[-indexes, ]

# Step 6: Tokenization and preprocessing (training set)
train.tokens <- tokens(train.raw$text, 
                       what = "word", 
                       remove_numbers = TRUE, 
                       remove_punct = TRUE, 
                       remove_symbols = TRUE, 
                       remove_hyphens = TRUE, 
                       ngrams = 1)

# Convert all tokens to lowercase
train.tokens <- tokens_tolower(train.tokens)

# Remove stopwords
train.tokens <- tokens_select(train.tokens, stopwords(), selection = "remove")

# Apply stemming (reduce words to their root form)
train.tokens <- tokens_wordstem(train.tokens, language = "english")

# Convert tokens to Document-Feature Matrix (DFM)
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)

# Convert DFM to dataframe
train.tokens.df <- convert(train.tokens.dfm, to = "data.frame")

# Add target variable back
train.tokens.df <- cbind(type = train.raw$type, train.tokens.df)

# Ensure column names are valid and unique
names(train.tokens.df) <- make.names(names(train.tokens.df))
train.tokens.df <- train.tokens.df[, !duplicated(colnames(train.tokens.df))]

# Step 7: Model Training (Decision Tree with cross-validation)
set.seed(48743)
folds <- createMultiFolds(train.tokens.df$type, k = 10, times = 1)
traincntrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, index = folds)

# Train decision tree (rpart)
rpart_model <- train(type ~ ., 
                     data = train.tokens.df, 
                     method = "rpart", 
                     trControl = traincntrl, 
                     tuneLength = 7)

# Step 8: Preprocess Test Set (same as training)
test.tokens <- tokens(test.raw$text, 
                      what = "word", 
                      remove_numbers = TRUE, 
                      remove_punct = TRUE, 
                      remove_symbols = TRUE, 
                      remove_hyphens = TRUE, 
                      ngrams = 1)

test.tokens <- tokens_tolower(test.tokens)
test.tokens <- tokens_select(test.tokens, stopwords(), selection = "remove")
test.tokens <- tokens_wordstem(test.tokens, language = "english")

# Convert to DFM
test.tokens.dfm <- dfm(test.tokens, tolower = FALSE)

# Match test features with training features
test.tokens.dfm <- dfm_match(test.tokens.dfm, featnames(train.tokens.dfm))

# Convert test DFM to dataframe
test.tokens.df <- convert(test.tokens.dfm, to = "data.frame")

# Add target variable back
test.tokens.df <- cbind(type = test.raw$type, test.tokens.df)

# Ensure column names are valid and unique
names(test.tokens.df) <- make.names(names(test.tokens.df))
test.tokens.df <- test.tokens.df[, !duplicated(colnames(test.tokens.df))]

# Step 9: Predictions
trainresult <- predict(rpart_model, newdata = train.tokens.df)
testresult <- predict(rpart_model, newdata = test.tokens.df)

# Step 10: Evaluate performance
confusionMatrix(table(trainresult, train.tokens.df$type))   # Training accuracy
confusionMatrix(table(testresult, test.tokens.df$type))     # Test accuracy
