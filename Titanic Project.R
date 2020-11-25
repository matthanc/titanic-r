### title: "Titantic Kaggle Project" ###

# Notes: I followed the following tutorial to complete this project:
# https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

### General Library Setup ###

library(tidyverse)
library(ggplot2)
library(janitor)
library(broom)
library(randomForest)
library(caret)


# Step 1: Load the data

train <- read_csv("https://drive.google.com/u/0/uc?id=1fCrTfJ1niH9BYjqCN7NLwy-KGOq7HaPE&export=download")
test <- read_csv("https://drive.google.com/u/0/uc?id=1THfRMmkigrMJM88WPxbM1PMs3W0-dq9l&export=download")
df <- bind_rows(train, test) #combine train and test tabels


# Step 2: Explore the data

str(df) # this provides a glimpse of the variables and what we have to work with.


# Step 2A: Use Names to infer family groups and title. 

df <- df %>%
  mutate(Survived = as.factor(Survived)) %>%
  mutate(PassengerId = as.factor(PassengerId)) %>%
  mutate(Sex = as.factor(Sex)) %>%
  mutate(Pclass = as.factor(Pclass))

df <- df %>%
  separate(Name, c("Surname", "Extra"), sep = ", ", extra = "merge") %>%
  separate(Extra, c("Title", "Extra"), sep = ". ", extra = "merge" ) %>%
  separate(Extra, c("Name", "MaidenN"), sep ="[(]", extra = "merge") %>%
  separate(MaidenN, c("MaidenN", "Extra"), sep ="[)]", extra = "merge") %>%
  select(-Extra)


df %>% tabyl(Sex, Title) #view table of titles by sex

raretitle <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'th')

df <- df %>% mutate(Title = if_else(Title %in% raretitle, "Rare",
                                    if_else(Title == 'Mlle', 'Miss',
                                            if_else(Title == 'Ms', 'Miss',
                                                    if_else(Title == 'Mme', 'Mrs', df$Title)))))

df %>% tabyl(Sex, Title) #view table of titles by sex
```
# Step 2B: Fix family size variable exploration

df <- df %>% mutate(FamilySize = SibSp + Parch + 1) %>%
  mutate(FamilyUnique = paste(df$Surname, df$FamilySize)) %>%
  mutate(Title = as.factor(Title))

df%>% #graph survival rates by family size
  filter(row(df) == 1:891) %>%
  ggplot(aes(x = FamilySize, fill = Survived)) +
  geom_bar(position = "dodge") +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_minimal()

df <- df %>%
  mutate(FamilySizeCat = if_else(FamilySize == 1, "Single", if_else(FamilySize <= 4, "Small", if_else(FamilySize >= 5, "Large", "Error")))) %>%
  mutate(FamilySizeCat = as.factor(FamilySizeCat))

df%>%
  filter(row(df) == 1:891) %>%
  ggplot(aes(x = FamilySizeCat, fill = Survived)) +
  geom_bar(stat = "count", position = "dodge") +
  labs(x = 'Family Size Category') +
  theme_minimal()

# Step 2C: Cabin data exploration.

df <- df %>%
  mutate(Deck = substring(Cabin, 1, 1))

df%>%
  ggplot(aes(x = Deck)) +
  geom_bar(stat = "count", position = "dodge") +
  labs(x = 'Cabin Deck') +
  theme_minimal()

df <- df %>%
  mutate(CabinData = ifelse(is.na(Cabin), 0, 1))

df%>%
  filter(row(df) == 1:891) %>%
  ggplot(aes(x = CabinData, fill = Survived)) +
  geom_bar(stat = "count", position = "dodge") +
  labs(x = 'Cabin Data') +
  theme_minimal()

# Step 2D: Age data exploration.

df %>%
  ggplot(aes(x = Age)) +
  geom_histogram() +
  ggtitle("Age Histogram: Original Output") +
  theme_minimal()

# Step 3: Imputation of missing Data

df <- df %>%
  mutate(SibSp = as.factor(SibSp)) %>%
  mutate(Parch = as.factor(Parch)) %>%
  mutate(FamilySize = as.factor(FamilySize))

# step 3A: Impuute missing Age data.

agemodel <- df %>%
  filter(!is.na(Age)) %>%
  lm(Age ~ Title + Pclass + FamilySize, data = .)

df <- df %>%
  mutate(Age = ifelse(is.na(Age), predict(agemodel, newdata = df, type = "response"), Age)) %>%
  mutate(Age = ifelse(Age < 0, 0.5, Age))

df %>%
  ggplot(aes(x = Age)) +
  geom_histogram() +
  ggtitle("Age Histogram: Model Output") +
  theme_minimal()

df <- df %>% mutate(AgeClass = ifelse(Age < 18, "Child","Adult")) #now that we have missing age data, we can indicate whether a passenger is a child or not

df %>% tabyl(AgeClass, Survived)

df <- df %>%  #we can also create a new variable indicating whether a passenger is a mother or not
  mutate(Parch = as.numeric(Parch)) %>%
  mutate(Mother = ifelse(Age >= 18 & Sex == "female" & Parch > 0 & Title != "Miss", "Mother", "Not Mother")) %>%
  mutate(Parch = as.factor(Parch))

df %>% tabyl(Mother, Survived)

# Step 3B: Check for more missing values to impute.

df %>%
  gather(key = "key", value = "val") %>%
    mutate(is.missing = is.na(val)) %>%
    group_by(key, is.missing) %>%
    summarise(num.missing = n()) %>%
    filter(is.missing==T) %>%
    select(-is.missing) %>%
    arrange(desc(num.missing))

# Step 3C: Impute Embarked (2) and Fare (1) missing values.

df %>% filter(PassengerId %in% c(62, 830, 1044))

df %>% mutate(Ticket = as.numeric(Ticket)) %>% filter(Ticket > 113000 & Ticket < 114000) %>% tabyl(Embarked)

df <- df %>% mutate(Embarked = ifelse(is.na(Embarked), "S", Embarked))

faremodel <- df %>%
  filter(!is.na(Fare)) %>%
  lm(Fare ~ Title + Pclass + Embarked + Age + Sex, data = .)

df <- df %>%
  mutate(Fare = ifelse(is.na(Fare), predict(faremodel, newdata = df, type = "response"), Fare))

# Step 4: Prediction

df <- df %>%
  mutate(Embarked = as.factor(Embarked)) %>%
  mutate(AgeClass = as.factor(AgeClass)) %>%
  mutate(Mother = as.factor(Mother))

train <- df %>% filter(!is.na(Survived)) #separating data into test and train sets
test <- df %>% filter(is.na(Survived))


set.seed(1354) #set a random seed

survivalmodel <- randomForest(Survived ~ Pclass + Title + Sex + Age + Fare + Embarked + FamilySize + FamilySizeCat + AgeClass + Mother + SibSp + CabinData, data = train) #model

# Step 4A: Review OOB error

err <- survivalmodel$err.rate #grabbing OOB error matrix
oob_err <- err[500, "OOB"]
print(oob_err)
paste0("OOB Accuracy: ", 1 - oob_err) # OOB accuracy

plot(survivalmodel) + 
  legend(x = "right", legend = colnames(err), fill = 1:ncol(err))

#Step 4B: Check variable importance

importance <- importance(survivalmodel)
varimportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[,'MeanDecreaseGini'],2))

rankImportance <- varimportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_minimal()

# Step 5: Export Kaggle submission file.

test <- test %>%
  mutate(Survived = predict(survivalmodel, newdata = test, type = "response"))

submission <- test %>%
  select(PassengerId, Survived)

write_csv(submission, "matthanc_submission.csv")



