import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.model_selection import train_test_split


# Score function
def score_(y_pred, y_test):
    """
    Function return various matrics
    
    Predicted value and actual passed to a built-in 
    function of sklearn library returns accuracy,precision
    ,recall and F1 score.
    """
    accuracy = accuracy_score(y_test, y_pred)
    # Precision
    precision = precision_score(y_test, y_pred, average='weighted',zero_division=1)
    # Recall
    recall = recall_score(y_test, y_pred, average='weighted',zero_division=1)
    # F1-Score
    f1 = f1_score(y_test, y_pred, average='weighted',zero_division=1)
    return accuracy, precision, recall, f1

def assign_val(df):
    """
    Function to assisgn value to new column 
    
    it will consider toss winner and toss decision
    to return the team which batted first.
    """
    if(df['toss_decision']=='bat'):
        return df['toss_winner']
    elif(df['toss_winner'] == df['team1']):
        return df['team2']
    else:
        return df['team1']  

# Plot for learning curve
def LC_plot(method,x,y,line=""):
    """Learning Curve"""
    train_sizes, train_scores, test_scores = learning_curve(method,
                                                            x, y,cv=5)
    # Compute the mean and standard deviation of the training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Visualisatiom
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.title('Learning Curve' + line)
    plt.xlabel('Training Samples')
    plt.ylabel('Score')

    plt.grid()
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='b')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                     alpha=0.1, color='r')
    plt.plot(train_sizes, train_mean, 'o-',
             color='b', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='r',
             label='Cross Validation Score')
    plt.legend(loc='best')
    return plt

def mult_cls_label_enc(clm,x):
    """Function to apply label encoder of multi class features"""
    le = LabelEncoder()
    if(len(clm) > 0):
        for column in clm:
            x[column] = le.fit_transform(x[column])          
    else:
        x = le.fit_transform(x[column])
    return x
       
# Base Folder
base_folder = 'D:\Module\Final Project'

# Read the data
Ipl_ball_data = pd.read_csv(
    'D:\Module\Final Project\Datasets\deliveries.csv')
Ipl_match_data = pd.read_csv(
    'D:\Module\Final Project\Datasets\matches.csv')
Rev_per_match = pd.read_csv('rev_data.csv')

# Pre-Processing
# Dropping Unwanted Columns
# checking all null values

Drp_clm = ['umpire1','umpire2', 'umpire3',
            'player_of_match']
Ipl_match_data = Ipl_match_data.drop(Drp_clm, axis=1)

data = Ipl_match_data.isnull().sum()
print(data)

data = Ipl_ball_data.isnull().sum()
print(data)

# Removing Null value from the dataset
Ipl_match_data.dropna(inplace=True)

# Replacing team names
# For Delhi Capitals
target_1 = "Delhi Daredevils"
replacement_string_1 = "Delhi Capitals"
Ipl_match_data['team1'] = Ipl_match_data['team1'].str.replace(
    target_1, replacement_string_1)
Ipl_match_data['team2'] = Ipl_match_data['team2'].str.replace(
   target_1, replacement_string_1)
Ipl_match_data['winner'] = Ipl_match_data['winner'].str.replace(
    target_1, replacement_string_1)
Ipl_match_data['toss_winner'] = Ipl_match_data['toss_winner'].str.replace(
    target_1, replacement_string_1)
# For sunrisers Hyderabad
target_2 = "Deccan Chargers"
replacement_string_2 = "Sunrisers Hyderabad"
Ipl_match_data['team1'] = Ipl_match_data['team1'].str.replace(
    target_2,replacement_string_2)
Ipl_match_data['team2'] = Ipl_match_data['team2'].str.replace(
    target_2,replacement_string_2)
Ipl_match_data['winner'] = Ipl_match_data['winner'].str.replace(
    target_2,replacement_string_2)
Ipl_match_data['toss_winner'] = Ipl_match_data['toss_winner'].str.replace(
    target_2,replacement_string_2)

#Number of matches played by each team
value_counts_team1 = Ipl_match_data['team1'].value_counts()
value_counts_team2 = Ipl_match_data['team2'].value_counts()
team_match = value_counts_team1 + value_counts_team2

#removing unwanted teams
teams = ['Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Gujarat Lions','Rising Pune Supergiant']
Ipl_match_data.drop(Ipl_match_data[Ipl_match_data['team1'].isin(teams)].index, inplace = True)
Ipl_match_data.drop(Ipl_match_data[Ipl_match_data['team2'].isin(teams)].index, inplace = True)

# Preprocessing rev dataset for visualisation
Rev_per_match['Rev/Match(Thousand)'] = Rev_per_match['Rev/Match(Thousand)'].\
    str.replace(',', '')
Rev_per_match['Rev/Match(Thousand)'] = Rev_per_match['Rev/Match(Thousand)'].\
    astype('float')
Rev_per_match = Rev_per_match.sort_values(
    'Rev/Match(Thousand)', ascending=False)
Rev_per_match = Rev_per_match.replace(',', '')
Rev_per_match['League'] = Rev_per_match['League'].str.extract(r'(\(.*\)?)')

#Fetching useful data from IPL Ball by ball dataset
#Top 10 batsman by number of ball faced
batsman_data = Ipl_ball_data['batsman'].value_counts().iloc[:10]
#Top 10 bowler by number of ball bowled
bowler_data =  Ipl_ball_data['bowler'].value_counts().iloc[:10]

#Extracting all ball which has been gone for 6
six_by_ball = Ipl_ball_data[Ipl_ball_data['batsman_runs']== 6]
#Top 10 batsman with highest number of 6
six_batsman = six_by_ball['batsman'].value_counts().iloc[:10]

#Extracting all ball which gone for 4
four_by_ball = Ipl_ball_data[Ipl_ball_data['batsman_runs']== 4]
#Top 10 batsman with highest number of 4
four_batsman = four_by_ball['batsman'].value_counts().iloc[:10]

#value count of run score in each ball ranging [0-7]
runs_counts = Ipl_ball_data['batsman_runs'].value_counts()
#Value count of each type of dismissal
dissmisal_kind_count = Ipl_ball_data['dismissal_kind'].value_counts()

#Extracting all ball with fall of wicket due to bowler
dismissal_kind = ['run out','retired hurt','obstructing the field','hit wicket']
wicket_by_bowler = Ipl_ball_data[~Ipl_ball_data['dismissal_kind'].isnull()]
wicket_by_bowler = wicket_by_bowler[~wicket_by_bowler['dismissal_kind'].isin(dismissal_kind)]

#Top 10 bowler with highest wickets
wicket_bowl = wicket_by_bowler['bowler'].value_counts().iloc[:10]

# Visualisations

#Misc Visualisation
# Plot the number of match played by each team using Seaborn
plt.figure(figsize=(10,8))
sns.barplot(x=team_match.index, y=team_match.values)
plt.title('Number of match played by each team')
plt.xlabel('Teams')
plt.ylabel('Match Played')
plt.axhline(y=team_match.mean(), color='r', linestyle='--', label='Mean of number of match played')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(base_folder+'\pictures\play.png',dpi=300, transparent=True)
plt.legend()
plt.show()

# Creating barplot for top 5 league by rev/match (in thousands)
plt.figure(figsize=(10, 6))
sns.barplot(x='League', y='Rev/Match(Thousand)', data=Rev_per_match.iloc[:5])
plt.xlabel('League', fontsize=12)
plt.ylabel('Revenue/match', fontsize=12)
plt.title('Top 5 sports league by Rev/Match(In thousands Dollar)')
footer_text = "Source:Wikipedia"
plt.text(0.92, 0.97, footer_text, ha='center', va='center',
          transform=plt.gca().transAxes)
plt.savefig(base_folder+'\pictures\Rev.png',dpi=300, transparent=True)

# Creating countplot for the cities along with number of match played there.
order = Ipl_match_data['city'].value_counts().iloc[:10].index
plt.figure(figsize=(10, 6))
sns.countplot(y='city', data=Ipl_match_data,
              order=order)
plt.xlabel('No of matches')
plt.ylabel('City')
plt.title('Top 10 cities by number of matches played')
plt.savefig(base_folder+'\pictures\cities.png', transparent=True)
plt.show()

#Top 10 batsman with number of ball faced
batsman_data = pd.DataFrame(batsman_data)
plt.figure(figsize=(10,6))
sns.barplot(x='batsman',y=batsman_data.index, data = batsman_data)
plt.title("Top 10 batsman by number of ball faced")
plt.ylabel("Batsman")
plt.xlabel("Ball faced")
plt.show()

#Top 10 bowler with number of ball bowled
bowler_data = pd.DataFrame(bowler_data)
plt.figure(figsize=(10,6))
sns.barplot(x='bowler',y=bowler_data.index, data = bowler_data)
plt.title("Top 10 bowler by number of ball bowled")
plt.ylabel("bowler")
plt.xlabel("Ball bowled")
plt.show()

#batsmans with highest number of 6's
six_batsman = pd.DataFrame(six_batsman)
plt.figure(figsize=(10,6))
sns.barplot(x='batsman',y=six_batsman.index, data = six_batsman)
plt.title("Top 10 batsman by number of 6's")
plt.ylabel("Batsman")
plt.xlabel("Number of 6's")
plt.show()

#batsmans with highest number of 4's
four_batsman = pd.DataFrame(four_batsman)
plt.figure(figsize=(10,6))
sns.barplot(x='batsman',y=four_batsman.index, data = four_batsman)
plt.title("Top 10 batsman by number of 4's")
plt.ylabel("Batsman")
plt.xlabel("Number of 4's")
plt.show()

#bowler with highest number of wickets
wicket_bowl = pd.DataFrame(wicket_bowl)
plt.figure(figsize=(10,6))
sns.barplot(x='bowler',y=wicket_bowl.index, data = wicket_bowl)
plt.title("Top 10 bowler by number of wiclets")
plt.ylabel("Bowler")
plt.xlabel("Number of wickets")
plt.show()

plt.figure(figsize=(10, 6))
colors = sns.color_palette('pastel')[0:10]
plt.title("How the runs scored(in %)")
explode = [0,0,0,0,0,0]
labels= ['Zero','One','Four','Two','Six','Three']
plt.pie(runs_counts[:6],labels=labels,startangle=0,explode= explode,colors=(colors),autopct='%1.1f%%')
plt.legend(loc='lower right', bbox_to_anchor=(-0.2, -0.1, 1, 1),ncol = 3)
plt.show()

plt.figure(figsize=(10, 6))
plt.title("How wickets falls(in %)")
plt.pie(dissmisal_kind_count[:7],startangle=45,labels=dissmisal_kind_count.index[:7],colors=(colors),autopct='%1.1f%%')
plt.legend(loc='lower right', bbox_to_anchor=(0.2, -0.1, 1, 1),ncol = 4)
plt.show()

#team performance
plt.figure(figsize=(10, 6))
colors = sns.color_palette('pastel')[0:10]
team_counts = Ipl_match_data['winner'].value_counts()
#create pie chart
labels = Ipl_match_data['winner'].unique()
plt.pie(team_counts,labels=team_counts.index,colors=colors,autopct='%1.1f%%')
plt.title('Percentage of matches win by each team')
plt.savefig(base_folder+'\pictures\winper.png', transparent=True)
plt.legend(loc='lower right', bbox_to_anchor=(0.5, -0.1, 1, 1),ncol = 4)
plt.show()

plt.figure(figsize=(10, 6))
plt.title("Winning by Runs")
plt.xlabel("Runs")
plt.grid(True)
sns.boxplot(y = 'winner', x = 'win_by_runs', data=Ipl_match_data[Ipl_match_data['win_by_runs']>0], orient = 'h')
plt.savefig(base_folder+'\pictures\winbyrun.png', transparent=True)
plt.show()

plt.figure(figsize=(10, 6))
plt.title("Winning by Wickets")
plt.grid(True)
sns.boxplot(y = 'winner', x = 'win_by_wickets', data=Ipl_match_data[Ipl_match_data['win_by_wickets']>0], orient = 'h')
plt.savefig(base_folder+'\pictures\winbywicket.png', transparent=True)
plt.show()

#Toss
plt.figure(figsize=(10, 6))
sns.countplot( x = 'toss_winner', data = Ipl_match_data)
plt.xticks(rotation='vertical')
plt.xlabel('Teams', fontsize=12)
plt.ylabel('Toss Won', fontsize=12)
plt.title('Toss won by each team in all season')
plt.savefig(base_folder+'\pictures\citiesteam.png', transparent=True)

plt.figure(figsize=(10,6))
t_data = Ipl_match_data[Ipl_match_data['toss_winner'] == Ipl_match_data['winner']]
Yes = len(t_data)/len(Ipl_match_data)*100
No = 100- Yes
plt.pie([Yes,No],labels=['Yes','No'],autopct='%1.1f%%',colors=['blue','Red'],startangle=90)
plt.title("Team winning toss winning matches")
plt.savefig(base_folder+'\pictures\Toss.png', transparent=True)
plt.show()

# Team winning match after choosing to bat
t_data = Ipl_match_data[(Ipl_match_data['toss_winner'] == Ipl_match_data['winner']) 
                        & (Ipl_match_data['toss_decision'] == 'bat')]
Bat= len(t_data)/len(Ipl_match_data)*100
Field = 100 - Bat
plt.figure(figsize=(10, 6))
plt.pie([Bat,Field],labels=['Yes','No'],autopct='%1.1f%%',colors=['blue','Red'],startangle=90)
plt.title("Team winning Match after choosing to bat")
plt.savefig(base_folder+'\pictures\Bat.png', transparent=True)
plt.show()

#Team winning batting first or field first

Ipl_match_data['pl_team']=Ipl_match_data.apply(lambda row: assign_val(row), axis=1)
t_data = Ipl_match_data[(Ipl_match_data['pl_team'] == Ipl_match_data['winner'])]
bat_toss = len(t_data)/len(Ipl_match_data)*100
field_toss = 100-bat_toss
plt.figure(figsize=(10, 6))
plt.pie([bat_toss,field_toss],labels=['Yes','No'],autopct='%1.1f%%',colors=['blue','Red'],startangle=90)
plt.title("Team batting first win match")
plt.savefig(base_folder+'\pictures\Toss_bat.png', transparent=True)
plt.show()

#Venue
venue_needed = ['Hyderabad', 'Bangalore' ,'Mumbai' , 'Kolkata', 'Delhi',
                'Chandigarh','Jaipur', 'Chennai','Nagpur',  'Mohali']
data = Ipl_match_data.drop(Ipl_match_data[~Ipl_match_data['city'].isin(venue_needed)].index, inplace = False)
# Create the pivot table with 'venue' as rows, 'winner' as columns, and count of matches as values
pivot_table = data.pivot_table(index='city', columns='winner', values='team1', aggfunc='count', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='g')
plt.xlabel('Teams')
plt.title('Number of Matches Won by Each Team at Each Indian Venue')
plt.savefig(base_folder+'\pictures\Team_venue.png', transparent=True)
plt.show()

# Data preparation
# Data droping
Model_data = Ipl_match_data.drop(["id", "Season","city","date",'pl_team',"result","win_by_wickets","dl_applied"], axis=1)

# dividing data into x and y features
x = Model_data.drop('winner', axis=1)
y = Model_data['winner']

# Encodder to change categorical values in the numerical values
# One hot encoder
clm=["team1","team2", "toss_winner","toss_decision","venue"]
numerical_x = pd.get_dummies(x, clm, drop_first=True)
numerical_y = pd.get_dummies(y)

# Label encoder
le = LabelEncoder()
x = mult_cls_label_enc(clm, x)
numerical_x_1 = x
numerical_y_1 = le.fit_transform(y)

# Scalling data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numerical_x_1)

x_train, x_test, y_train, y_test = train_test_split(numerical_x_1, numerical_y_1,
                                                    train_size=0.8,
                                                    shuffle=(False))
x_train_ohe, x_test_ohe, y_train_ohe, y_test_ohe = train_test_split(numerical_x, 
                                                                    numerical_y,
                                                    train_size=0.8,
                                                    shuffle=(False))

# Machine learning algoritms
# Random Forest classifier with label encoder
RCF_model = RandomForestClassifier(
    n_estimators=105,min_samples_split=2,
                                    max_features = "auto",random_state=42)

# Random Forest classifier with label encoder
RCF_model.fit(x_train, y_train)
y_pred = RCF_model.predict(x_test)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Accuracy
print("\nRandom Forest Classifier:")
print("-------------------------")
# Accuracy,prescision,recall,F1 score of the model
accuracy, precision, recall, f1 = score_(y_pred, y_test)
print("accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Learning curve plot for label encoder
line =  " With Random forest and label encoder"
plt = LC_plot(RCF_model,x_train,y_train,line)
plt.savefig(base_folder+'\pictures\RandomForest.png', transparent=True)
plt.show()

#Random Forest classifier with one hot encoder
RCF_model_oh = RandomForestClassifier(
    n_estimators=110,min_samples_split=2,
                                    max_features = "auto",random_state=42)
RCF_model_oh.fit(x_train_ohe, y_train_ohe)
y_pred = RCF_model_oh.predict(x_test_ohe)

# Accuracy
print("\nRandom Forest Classifier with onehotencoder:")
print("-------------------------")
# Accuracy,prescision,recall,F1 score of the model
accuracy, precision, recall, f1 = score_(y_pred, y_test_ohe)
print("accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Visualisatiom
line =  " With Random forest and one hot encoder"
plt = LC_plot(RCF_model_oh,x_train_ohe,y_train_ohe,line)
plt.savefig(base_folder+'\pictures\RandomForest_ohe.png', transparent=True)
plt.show()

# decision tree classifier with label encoder
dtc = DecisionTreeClassifier(random_state=(42))
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)

# Accuracy
print("\nDecision Tree Classifier:")
print("-------------------------")
# Accuracy,prescision,recall,F1 score of the model
accuracy, precision, recall, f1 = score_(y_pred, y_test)
print("accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

#Learning Curve with decision tree and label encoder
line =  " With decision tree and label encoder"
plt = LC_plot(dtc,x_train,y_train,line)
plt.savefig(base_folder+'\pictures\DecisionTree.png', transparent=True)
plt.show()

#Decision tree with one hot encoder
dtc_ohe = DecisionTreeClassifier(random_state=(42))
dtc_ohe.fit(x_train_ohe, y_train_ohe)
y_pred = dtc_ohe.predict(x_test_ohe)
# Accuracy
print("\nDecision Tree Classifier with one hot encoder:")
print("-------------------------")
# Accuracy,prescision,recall,F1 score of the model
accuracy, precision, recall, f1 = score_(y_pred, y_test_ohe)
print("accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

#Learning Curve plot for Decision tree with one hot encoder
line =  " With decision tree and one hot encoder"
plt = LC_plot(dtc_ohe,x_train_ohe,y_train_ohe,line)
plt.savefig(base_folder+'\pictures\DecisionTree.png', transparent=True)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(X_scaled, numerical_y_1,
                                                    train_size=0.8,
                                                    shuffle=(False))
# Logestic regression
Lgr = LogisticRegression(solver='lbfgs',multi_class='ovr',random_state=(42))

Lgr.fit(x_train, y_train)
y_pred = Lgr.predict(x_test)
# Accuracy
print("\nLogistic regression:")
print("-------------------------")
# Accuracy,prescision,recall,F1 score of the model
accuracy, precision, recall, f1 = score_(y_pred, y_test)
print("accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

line =  " With logestic regression with label encoder"
plt = LC_plot(Lgr,x_train,y_train,line)
plt.savefig(base_folder+'\pictures\Lgr.png', transparent=True)
plt.show()

# Logestic regression with one hot encoder and label encoder combo
Lgr_ohe_l = LogisticRegression(max_iter=300,solver='lbfgs',multi_class='ovr',random_state=(42))

Lgr_ohe_l.fit(x_train_ohe, y_train)
y_pred = Lgr_ohe_l.predict(x_test_ohe)
# Accuracy
print("\nLogistic regression and one hot encoder combined with label encoder:")
print("-------------------------")
# Accuracy,prescision,recall,F1 score of the model
accuracy, precision, recall, f1 = score_(y_pred, y_test)
print("accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

line =  " With logestic regression and one hot encoder combined with label encoder"
plt = LC_plot(Lgr_ohe_l,x_train_ohe,y_train,line)
plt.savefig(base_folder+'\pictures\Lgr_1.png', transparent=True)
plt.show()


#Support vector machine
svm = SVC(kernel='rbf',random_state=(42))
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
print("\nSupport Vector Machine:")
print("-------------------------")
#Accuracy,prescision,recall,F1 score of the model
accuracy, precision, recall, f1 = score_(y_pred, y_test)
print("accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
line =  " With Support vector machine and label encoder"
plt = LC_plot(svm,x_train,y_train,line)
plt.savefig(base_folder+'\pictures\SVM.png', transparent=True)
plt.show()

#Support vector machine
svm_ohe_l = SVC(kernel='rbf',random_state=(42))
svm_ohe_l.fit(x_train_ohe, y_train)
y_pred = svm_ohe_l.predict(x_test_ohe)
print("\nSupport Vector Machine :")
print("-------------------------")
#Accuracy,prescision,recall,F1 score of the model
accuracy, precision, recall, f1 = score_(y_pred, y_test)
print("accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

line =  " With Support vector machine with the combination of encoders"
plt = LC_plot(svm_ohe_l,x_train_ohe, y_train,line)
plt.savefig(base_folder+'\pictures\SVM_1.png', transparent=True)
plt.show()
