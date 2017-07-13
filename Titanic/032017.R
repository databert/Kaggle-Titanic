# Load packages
library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm

train <- read.csv('train.csv', stringsAsFactors = F)
test  <- read.csv('test.csv', stringsAsFactors = F)

full  <- bind_rows(train, test) # bind training & test data

# check data
str(full)

#Make Survival from integer to boolean
full$Survived <- as.logical(full$Survived)
levels(full$Survived) <- c("Not survived", "Survived")

# Grab title from passenger names
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

# Titles with very low cell counts to be combined to "rare" level
special_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Master')

# Also reassign mlle, ms, and mme accordingly
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% special_title]  <- 'Special'

###### Feature engineering ########

# Incorporate importance of Special title

full$Special<-full$Title=='Special'

# Create a family size variable including the passenger themselves
full$Fsize <- full$SibSp + full$Parch + 1

# Discretize family size
full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

#Take care of missing data
full$Embarked[c(62, 830)] <- 'C'

full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)

#Take care of missing age value using mice model 

# Set a random seed
set.seed(129)

# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 

# Save the complete output 
mice_output <- complete(mice_mod)

# Replace Age variable from the mice model.
full$Age <- mice_output$Age

# Show new number of missing Age values
sum(is.na(full$Age))

# Create the column child, and indicate whether child or adult
full$Child[full$Age <= 15] <- 'Child'
full$Child[full$Age > 15] <- 'Adult'

# Adding Mother variable
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$SibSp > 0 & full$Child=='Adult' & full$Title != 'Miss'] <- 'Mother'

# Make variables factors into factors

full$Child  <- factor(full$Child)
full$Mother <- factor(full$Mother)
full$Special <- factor(full$Special)

factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Surname','Family','FsizeD')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

# Split the data back into a train set and a test set
train <- full[1:891,]
test <- full[892:1309,]

# Set a random seed
set.seed(754)

# Build the model (note: not all possible variables are used)
fol <- formula(Survived ~ Age + Sex + Fsize + Child + Mother + Special)   

#Execute randomForest model
rf_model <- randomForest(fol, data = train)

# Show model error
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

#Visualize feature importance
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), Importance = importance)
