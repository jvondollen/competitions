library(ggplot2)


train.raw <- read.csv('c:/github/kaggle/titanic/train.csv', stringsAsFactors = F)
test.raw <- read.csv('c:/github/kaggle/titanic/test.csv', stringsAsFactors = F)



# Quick single feature plots
#~~~~~~~~~~~~~~~~~~~~~~~~~~~
barplot(table(train.raw$Survived),
        names.arg = c("Perished", "Survived"),
        main="Survived (passenger fate)", col="black")
barplot(table(train.raw$Pclass), 
        names.arg = c("first", "second", "third"),
        main="Pclass (passenger traveling class)", col="firebrick")
barplot(table(train.raw$Sex), main="Sex (gender)", col="darkviolet")
hist(train.raw$Age, main="Age", xlab = NULL, col="brown")
barplot(table(train.raw$SibSp), main="SibSp (siblings + spouse aboard)", 
        col="darkblue")
barplot(table(train.raw$Parch), main="Parch (parents + kids aboard)", 
        col="gray50")
hist(train.raw$Fare, main="Fare (fee paid for ticket[s])", xlab = NULL, 
     col="darkgreen")
barplot(table(train.raw$Embarked), 
        names.arg = c("NA","Cherbourg", "Queenstown", "Southampton"),
        main="Embarked (port of embarkation)", col="sienna")



# Feature plots vs survival
#~~~~~~~~~~~~~~~~~~~~~~~~~~

# Compare passenger class with whether they survived
ggplot(train.raw, aes( as.factor(Pclass), fill = as.logical(Survived))) + geom_bar(position="fill") +
  labs(x="Passenger Class", y="Number of Passengers", fill = "Survived")

# Compare passenger sex with whether they survived
ggplot(train.raw, aes( as.factor(Sex), fill = as.logical(Survived))) + geom_bar(position="fill") +
  labs(x="Passenger's Sex", y="Number of Passengers", fill = "Survived")

# Compare passengers with Siblings and Spouses and whether they survived
ggplot(train.raw, aes( as.factor(SibSp), fill = as.logical(Survived))) + geom_bar(position="fill") +
  labs(x="Passenger Siblings + Spouse", y="Number of Passengers", fill = "Survived")

# Compare passenger listed as parents + kids and whether they survived
ggplot(train.raw, aes( as.factor(Parch), fill = as.logical(Survived))) + geom_bar(position="fill") +
  labs(x="Passenger Parentts and Kids aboard", y="Number of Passengers", fill = "Survived")

# Compare where passenger enbarked and whether they survived
ggplot(train.raw, aes( as.factor(Embarked), fill = as.logical(Survived))) + geom_bar(position="fill") +
  labs(x="Where Passenger Embarked", y="Number of Passengers", fill = "Survived")
