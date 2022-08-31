# Code for 'diabetes_main.py' file.

# Importing the necessary Python modules.
import streamlit as st
import numpy as np
import pandas as pd

# Configure your home page by setting its title and icon that will be displayed in a browser tab.
st.set_page_config(page_title = 'Early Diabetes Prediction Web App',
                    page_icon = 'random',
                    layout = 'wide',
                    initial_sidebar_state = 'auto'
                    )

# Loading the dataset.
@st.cache()
def load_data():
    # Load the Diabetes dataset into DataFrame.

    df = pd.read_csv('https://s3-whjr-curriculum-uploads.whjr.online/b510b80d-2fd6-4c08-bfdf-2a24f733551d.csv')
    df.head()

    # Rename the column names in the DataFrame.
    df.rename(columns = {"BloodPressure": "Blood_Pressure",}, inplace = True)
    df.rename(columns = {"SkinThickness": "Skin_Thickness",}, inplace = True)
    df.rename(columns = {"DiabetesPedigreeFunction": "Pedigree_Function",}, inplace = True)

    df.head() 

    return df

diabetes_df = load_data()
##########################################
# Create the Page Navigator for 'Home', 'Predict Diabetes' and 'Visualise Decision Tree' web pages in 'diabetes_main.py'
# Import the 'diabetes_predict' 'diabetes_home', 'diabetes_plots' Python files
import diabetes_home
import diabetes_predict
import diabetes_plots

# Adding a navigation in the sidebar using radio buttons
# Create the 'pages_dict' dictionary to navigate.
pages_dict = {"Home": diabetes_home, 
           "Predict Diabetes": diabetes_predict, 
           "Visualise Decision Tree": diabetes_plots}
st.sidebar.title("Navigation")
# Add radio buttons in the sidebar for navigation and call the respective pages based on user selection.
user_choice = st.sidebar.radio("Go To:", tuple(pages_dict.keys))
if user_choice == "Home:":
  diabetes_home.app()
else:
  selected_page = pages_dict[user_choice]
  selected_page.app(diabetes_df)
######################################
# Show complete dataset and summary in 'diabetes_home.py'
# Import the streamlit modules.
import streamlit as st

# Define a function 'app()' which accepts 'census_df' as an input.
def app(diabetes_df):
    # Set the title to the home page contents.
    st.title("View Data")
    # Provide a brief description for the web app.
    st.markdown("""Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy.
    There isn’t a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help in reducing the impact of diabetes.
    This Web app will help you to predict whether a person has diabetes or is prone to get diabetes in future by analysing the values of several features using the Decision Tree Classifier.
    """)
    # Add the 'beta_expander' to view full dataset 
    with st.beta_expander("View Dataset"):
      st.table(diabetes_df)

    # Add a checkbox in the first column. Display the column names of 'diabetes_df' on the click of checkbox.
    beta_col1, beta_col2, beta_col3 = st.beta_columns(3)

    with beta_col1:
      if st.checkbox("Show All Columns Names"):
        st.table(list(diabetes_df.columns))

    # Add a checkbox in the second column. Display the column data-types of 'diabetes_df' on the click of checkbox.
    with beta_col2:
      if st.checkbox("View Columns' Datatypes"):
        st.table(diabetes_df.types)
    # Add a checkbox in the third column followed by a selectbox which accepts the column name whose data needs to be displayed.
    with beta_col3:
      if st.checkbox("View Column Data"):
        column_data = st.selectbox("Select Column", tuple(diabetes_df.columns))
        st.write(diabetes_df[column_data])
##############################################
# Import the necessary modules design the Decision Tree classifier
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import GridSearchCV  
from sklearn import tree
from sklearn import metrics
######################################
# Create the 'd_tree_pred' function to predict the diabetes using the Decision Tree classifier
@st.cache()
def d_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age):
    # Split the train and test dataset. 
    feature_columns = list(diabetes_df.columns)

    # Remove the 'Pregnancies', Skin_Thickness' columns and the 'target' column from the feature columns
    feature_columns.remove('Skin_Thickness')
    feature_columns.remove('Pregnancies')
    feature_columns.remove('Outcome')

    X = diabetes_df[feature_columns]
    y = diabetes_df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    dtree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    dtree_clf.fit(X_train, y_train) 
    y_train_pred = dtree_clf.predict(X_train)
    y_test_pred = dtree_clf.predict(X_test)
    # Predict diabetes using the 'predict()' function.
    prediction = dtree_clf.predict([[glucose, bp, insulin, bmi, pedigree, age]])
    prediction = prediction[0]

    score = round(metrics.accuracy_score(y_train, y_train_pred) * 100, 3)

    return prediction, score
############################################################################
def grid_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age):    
    feature_columns = list(diabetes_df.columns)
    # Remove the 'Pregnancies', 'Skin_Thickness' columns and the 'target' column from the feature columns
    feature_columns.remove('Pregnancies')
    feature_columns.remove('Skin_Thickness')
    feature_columns.remove('Outcome')
    X = diabetes_df[feature_columns]
    y = diabetes_df['Outcome']
    # Split the train and test dataset. 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    param_grid = {'criterion':['gini','entropy'], 'max_depth': np.arange(4,21), 'random_state': [42]}

    # Create a grid
    grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'roc_auc', n_jobs = -1)

    # Training
    grid_tree.fit(X_train, y_train)
    best_tree = grid_tree.best_estimator_
    
    # Predict diabetes using the 'predict()' function.
    prediction = best_tree.predict([[glucose, bp, insulin, bmi, pedigree, age]])
    prediction = prediction[0]

    score = round(grid_tree.best_score_ * 100, 3)

    return prediction, score
#############################
# Create the user defined 'app()' function.
def app(diabetes_df):
    st.markdown("<p style='color:red;font-size:25px'>This app uses <b>Decision Tree Classifier</b> for the Early Prediction of Diabetes.", unsafe_allow_html = True) 
    st.subheader("Select Values:")
    
    # Create six sliders with the respective minimum and maximum values of features. 
    # store them in variables 'glucose', 'bp', 'insulin, 'bmi', 'pedigree' and 'age'
    # Write your code here:
    glucose = st.slider("Glucose", float(diabetes_df["Glucose"].min()), float(diabetes_df["Glucose"].max()))
    bp = st.slider("Glucose", float(diabetes_df["BloodPressure"].min()), float(diabetes_df["BloodPressure"].max()))
    insulin = st.slider("Glucose", float(diabetes_df["Insulin"].min()), float(diabetes_df["Insulin"].max()))
    bmi = st.slider("Glucose", float(diabetes_df["BMI"].min()), float(diabetes_df["BMI"].max()))
    pedigree = st.slider("Glucose", float(diabetes_df["Pedigree"].min()), float(diabetes_df["Pedigree"].max()))
    age = st.slider("Age", float(diabetes_df["Age"].min()), float(diabetes_df["Age"].max()))

    st.subheader("Model Selection")

    # Add a single select drop down menu with label 'Select the Classifier'
    predictor = st.selectbox("Select the Decision Tree Classifier",('Decision Tree Classifier', 'GridSearchCV Best Tree Classifier'))

    if predictor == 'Decision Tree Classifier':
        if st.button("Predict"):            
            prediction, score = d_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age)
            st.subheader("Decision Tree Prediction results:")
            if prediction == 1:
                st.info("The person either has diabetes or prone to get diabetes")
            else:
                st.info("The person is free from diabetes")
            st.write("The accuracy score of this model is", score, "%")


    elif predictor == 'GridSearchCV Best Tree Classifier':
        if st.button("Predict"):
            prediction, score = grid_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age)
            st.subheader("Optimised Decision Tree Prediction results:")
            if prediction == 1:
                st.info("The person either has diabetes or prone to get diabetes")
            else:
                st.info("The person is free from diabetes")
            st.write("The best score of this model is", score, "%")
###############################################
# Code for 'diabetes_plot.py' file.
# Import necessary modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import GridSearchCV  
from sklearn import tree
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
import graphviz as graphviz
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image  


# Define a function 'app()' which accepts 'census_df' as an input.
def app(diabetes_df):
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Visualise the Diabetes Prediction Web app ")

    if st.checkbox("Show the correlation heatmap"):
        st.subheader("Correlation Heatmap")
        plt.figure(figsize = (10, 6))
        ax = sns.heatmap(diabetes_df.iloc[:, 1:].corr(), annot = True) # Creating an object of seaborn axis and storing it in 'ax' variable
        bottom, top = ax.get_ylim() # Getting the top and bottom margin limits.
        ax.set_ylim(bottom + 0.5, top - 0.5) # Increasing the bottom and decreasing the top margins respectively.
        st.pyplot()

    st.subheader("Predictor Selection")


    # Add a single select with label 'Select the Classifier'
    plot_select = st.selectbox("Select the Classifier to Visualise the Diabetes Prediction:", ('Decision Tree Classifier', 'GridSearchCV Best Tree Classifier')) 

    if plot_select == 'Decision Tree Classifier':
        # Split the train and test dataset. 
        feature_columns = list(diabetes_df.columns)

        # Remove the 'Pregnancies', 'Skin_Thickness' columns and the 'target' column from the feature columns
        feature_columns.remove('Pregnancies')
        feature_columns.remove('Skin_Thickness')
        feature_columns.remove('Outcome')

        X = diabetes_df[feature_columns]
        y = diabetes_df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

        dtree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        dtree_clf.fit(X_train, y_train) 
        y_train_pred = dtree_clf.predict(X_train)
        y_test_pred = dtree_clf.predict(X_test)

         
        if st.checkbox("Plot confusion matrix"):
            plt.figure(figsize = (10, 6))
            plot_confusion_matrix(dtree_clf, X_train, y_train, values_format = 'd')
            st.pyplot()

        if st.checkbox("Plot Decision Tree"):   
            # Export decision tree in dot format and store in 'dot_data' variable.
            dot_data = tree.export_graphviz(decision_tree = dtree_clf, max_depth = 3, out_file = None, filled = True, rounded = True,
                feature_names = feature_columns, class_names = ['0', '1'])
            # Plot the decision tree using the 'graphviz_chart' function of the 'streamlit' module.
            st.graphviz_chart(dot_data)


    if plot_select == 'GridSearchCV Best Tree Classifier':
        # Split the train and test dataset. 
        feature_columns = list(diabetes_df.columns)

        # Remove the 'Pregnancies', 'Skin_Thickness' columns and the 'target' column from the feature columns
        feature_columns.remove('Pregnancies')
        feature_columns.remove('Skin_Thickness')
        feature_columns.remove('Outcome')

        X = diabetes_df[feature_columns]
        y = diabetes_df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

        param_grid = {'criterion':['gini','entropy'], 'max_depth': np.arange(4,21), 'random_state': [42]}

        # Create a grid
        grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'roc_auc', n_jobs = -1)

        # Training
        grid_tree.fit(X_train, y_train)
        best_tree = grid_tree.best_estimator_

        grid_tree.fit(X_train, y_train) 
        y_train_pred = grid_tree.predict(X_train)
        y_test_pred = grid_tree.predict(X_test)

         
        if st.checkbox("Plot confusion matrix"):
            plt.figure(figsize = (5, 3))
            plot_confusion_matrix(grid_tree, X_train, y_train, values_format = 'd')
            st.pyplot()

        if st.checkbox("Plot Decision Tree"):   
            # Create empty dot file.
            #dot_data = StringIO()
            # Export decision tree in dot format.
            dot_data = tree.export_graphviz(decision_tree = best_tree, max_depth = 3, out_file = None, filled = True, rounded = True,
                feature_names = feature_columns, class_names = ['0', '1'])
            st.graphviz_chart(dot_data)
            
