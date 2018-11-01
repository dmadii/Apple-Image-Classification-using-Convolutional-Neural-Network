
library(keras)
library(EBImage)

setwd('D://Apple100')
save_in <- ("D://Run100/")

images <- list.files()
w <- 100
h <- 100
for(i in 1:length(images))
{
  result <- tryCatch({
    # Image name
    imgname <- images[i]
    # Read image
    img <- readImage(imgname)
    # Path to file
    img_resized <- resize(img, w = w, h = h)
    path <- paste(save_in, imgname, sep = "")
    # Save image
    writeImage(img_resized, path, quality = 70)
    # Print status
    print(paste("Done",i,sep = " "))},
    # Error function
    error = function(e){print(e)})
}

setwd("D://Run100")
# Read Images
images <- list.files()
images
summary(images)
list_of_images = lapply(images, readImage)
head(list_of_images)
display(list_of_images[[15]])
tail(list_of_images)


#create train
train <- list_of_images[c(1:40, 51:90)]
str(train)
display(train[[20]])

#create test
test <- list_of_images[c(41:50, 91:100)]
test
display(test[[1]])


par(mfrow = c(10,10))
for (i in 1:80) plot(train[[i]])

# Resize & combine
str(train)
for (i in 1:80) {train[[i]] <- resize(train[[i]], 32, 32)}
for (i in 1:20) {test[[i]] <- resize(test[[i]], 32, 32)}
for (f in 1:80) {print(dim(train[[f]]))}
train <- combine(train)
str(train)
x <- tile(train, 5)
display(x, title='Pictures')
test <- combine(test)
y <- tile(test, 2)
display(y, title = 'Pics')

# Reorder dimension
train <- aperm(train, c(4,1,2,3))
test <- aperm(test, c(4,1,2,3))
str(train)
str(test)
rep(0,5)

# Response/Give Label for the Data
trainy <- c(rep(0,40),rep(1,40))
testy <- c(rep(0,10),rep(1,10))

# One hot encoding
trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)

# Model
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = 'relu',
                input_shape = c(32, 32, 3)) %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.01) %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.01) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate=0.01) %>%
  layer_dense(units = 2, activation = 'softmax') %>%
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_sgd(lr = 0.01,
                                    decay = 1e-6,
                                    momentum = 0.9,
                                    nesterov = T),
          metrics = c('accuracy'))
summary(model)

# Fit model
history <- model %>%
  fit(train,
      trainLabels,
      epochs = 50,
      batch_size = 32,
      validation_split = 0.2,
      validation_data = list(test, testLabels))
plot(history)


# Evaluation & Prediction - train data
model %>% evaluate(train, trainLabels)
pred <- model %>% predict_classes(train)
table(Predicted = pred, Actual = trainy)
prob <- model %>% predict_proba(train)
cbind(prob, Predicted_class = pred, Actual = trainy)


# Evaluation & Prediction - test data
model %>% evaluate(test, testLabels)
pred <- model %>% predict_classes(test)
table(Predicted = pred, Actual = testy)
prob <- model %>% predict_proba(test)
cbind(prob, Predicted_class = pred, Actual = testy)


#save model
save_model_hdf5(model,filepath='D://Run100.hdf5')


#Validation
setwd('D:/Uji/')
model=load_model_hdf5(filepath='D://Run100.hdf5')
images <- list.files()
images
summary(images)
list_of_images = lapply(images, readImage)
head(list_of_images)
display(list_of_images[[3]])

# Get the image as a matrix
testt <- list_of_images
for (i in 1:10) {testt[[i]] <- resize(testt[[i]], 32, 32)}
for (i in 1:10) {testt[[i]] <- toRGB(testt[[i]])}
fixx <- combine(testt)
y <- tile(fixx, 2)
display(y, title = 'Pics')
str(fixx)
Uji <- aperm(fixx, c(4, 1, 2, 3))
str(Uji)
testy <- c(rep(0,5), rep(1,5))

# One hot encoding
testLabels <- to_categorical(testy)
pred <- model %>% predict_classes(Uji)
model %>% evaluate(Uji, testLabels)
table(Predicted = pred, Actual = testy)
prob <- model %>% predict_proba(Uji)
prob
colnames(prob)<- c('Bagus','Busuk')
Madi<-cbind(prob, Predicted_class = pred, Actual = testy)
Madi

