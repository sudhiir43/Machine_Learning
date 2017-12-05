rm(list = ls())
setwd("E:\\AVdatafest\\BigMart")


# Loading train and test datasets
traindata <- read.csv("train.csv")
#newtrain = train
testdata <- read.csv("test.csv") 

traindata1 = traindata[,-1]
#rm(train2)
str(traindata1)
summary(traindata1)

(sum(is.na(traindata)))
#imputing missing values using KNN imputation.
library(DMwR)
traindata2 = knnImputation(traindata,k=10)
sum(is.na(traindata2))

#################################
traindata3 = traindata2
#converting the fat content variables into factors assigning 1 or 2
traindata3$Item_Fat_Content= ifelse(traindata3$Item_Fat_Content=="LF",1,ifelse(traindata3$Item_Fat_Content=="reg",2,ifelse(traindata3$Item_Fat_Content=="Low Fat",1,2)))
str(traindata3$Item_Fat_Content)
traindata3$Item_Fat_Content = as.factor(as.character(traindata3$Item_Fat_Content))
traindata3$Outlet_Establishment_Year<-as.factor(traindata3$Outlet_Establishment_Year)
#############################################
########################important.....!!!!!!!

testset = read.csv("test.csv")
summary(testset)
train4 = testset
test = testset
###############################
train4$Item_Fat_Content= ifelse(train4$Item_Fat_Content=="LF",1,ifelse(train4$Item_Fat_Content=="reg",2,ifelse(train4$Item_Fat_Content=="Low Fat",1,2)))
str(train4$Item_Fat_Content)
train4$Item_Fat_Content = as.factor(as.character(train4$Item_Fat_Content))
testset = train4
##############################

sum(is.na(testset))
testset = knnImputation(testset,k=3)


traindata3$target = seq(1,nrow(traindata3),1)
traindata3$target = traindata3$Item_Outlet_Sales/traindata3$Item_MRP
traindata3= traindata3[,-12]
traindata3$Outlet_Size=as.factor(traindata3$Outlet_Size)

str(traindata3)
sum(is.na(traindata3$Outlet_Size))
str(traindata3$Outlet_Size)
traindata3$Outlet_Size=as.factor(traindata3$Outlet_Size)
####################linear regression-experimenting model############  
first = lm((target) ~ (Item_Visibility*Outlet_Size*Outlet_Size)+
             Item_Weight/ Item_Visibility + 
             Item_MRP+ Outlet_Identifier,data = traindata3)
summary(first)
second = lm((target) ~ (Item_Visibility*Outlet_Size*Outlet_Size) 
            + Item_MRP*Item_MRP+Outlet_Identifier
            ,data = traindata3)
#plot(second, color = "blue")

predictsec<-predict(second, testset)
predictsecond1<-predictsec*test$Item_MRP

write.csv(predictsecond1, "linear.csv")

#######Model different########
str(newdatamod)
str(traindata3)
str(testset)

newdatamod$Item_Weight<-traindata3$Item_Weight
newdatamod$Outlet_Location_Type<-traindata3$Outlet_Location_Type


#####model gave 1140 rmse###
library(gbm)
set.seed(1000)
modelgbm <- gbm((target)~( (Item_Visibility*Item_Visibility*Outlet_Size)+ (Item_MRP*Item_MRP*Item_MRP)+
                            Outlet_Identifier*Outlet_Location_Type), data=traindata3, n.trees=100, distribution="gaussian",
                interaction.depth=3, bag.fraction=0.5, train.fraction=1, shrinkage=0.1,
                keep.data=TRUE)

pred <- predict(modelgbm,testset,n.trees=100)

pred2<-pred*testnewdata$Item_MRP
#read the sameple file##
sample<-read.csv("Samplesubmission.csv")
names(sample)
sample$Item_Outlet_Sales<-pred2
write.csv(sample,"gbm.csv")
length(unique(traindata3$Outlet_Identifier))

