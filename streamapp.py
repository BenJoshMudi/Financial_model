import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the saved model
model_path = "C:/Users/mudia/OneDrive/Desktop/JJPF/model_saved"  #Loading the model 
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Get unique values for each categorical column
location_type_values = ['Rural', 'Urban']
cellphone_access_values = ['No', 'Yes']
gender_values = ['Male', 'Female']
relationship_with_head_values = [
    'Head of Household', 'Spouse', 'Child', 'Parent', 
    'Other relative', 'Other non-relatives', 'Dont know'
]
marital_status_values = [
    'Married/Living together', 'Divorced/Seperated', 
    'Widowed', 'Single/Never Married', 'Dont know'
]
education_level_values = [
    'No formal education', 'Primary education', 
    'Secondary education', 'Vocational/Specialised training', 
    'Tertiary education', 'Other/Dont know/RTA'
]
job_type_values = [
    'Farming and Fishing', 'Self employed', 
    'Formally employed Government', 'Formally employed Private', 
    'Informally employed', 'Remittance Dependent', 
    'Government Dependent', 'Other Income', 
    'No Income', 'Dont Know/Refuse to answer'
]

# Define mappings for categorical variables
location_type_map = {value: idx for idx, value in enumerate(location_type_values)}
cellphone_access_map = {value: idx for idx, value in enumerate(cellphone_access_values)}
gender_map = {value: idx for idx, value in enumerate(gender_values)}
relationship_with_head_map = {value: idx for idx, value in enumerate(relationship_with_head_values)}
marital_status_map = {value: idx for idx, value in enumerate(marital_status_values)}
education_level_map = {value: idx for idx, value in enumerate(education_level_values)}
job_type_map = {value: idx for idx, value in enumerate(job_type_values)}

# Function to convert categorical inputs to numerical
def categorical_to_numerical(value, category_map):
    return category_map.get(value, 0.0)  # Default to 0.0 if value not found

# Create a function for prediction
def expresso_prediction(input_data):
    # Convert categorical inputs to numerical
    input_data[0] = categorical_to_numerical(input_data[0], location_type_map)
    input_data[1] = categorical_to_numerical(input_data[1], cellphone_access_map)
    input_data[4] = categorical_to_numerical(input_data[4], gender_map)
    input_data[5] = categorical_to_numerical(input_data[5], relationship_with_head_map)
    input_data[6] = categorical_to_numerical(input_data[6], marital_status_map)
    input_data[7] = categorical_to_numerical(input_data[7], education_level_map)
    input_data[8] = categorical_to_numerical(input_data[8], job_type_map)

    # Convert input data to float
    input_data_as_num = np.asarray(input_data, dtype=float).reshape(1, -1)
    
    # Make prediction
    prediction = loaded_model.predict(input_data_as_num)
    
    if prediction[0] == 1:
        return 'The customer has access.'
    else:
        return 'The customer does not have access.'

def main():
    st.title("Bank Account Access Predictor")
    
    # Getting the input data from the user using radio buttons
    location_type = st.radio('Type of location', location_type_values)
    cellphone_access = st.radio('Cellphone access', cellphone_access_values)
    household_size = st.text_input('Number of people living in one house')
    age_of_respondent = st.text_input('Age of the interviewee')
    gender_of_respondent = st.radio('Gender of interviewee:', gender_values)
    relationship_with_head = st.radio('The interviewee’s relationship with the head of the house:', relationship_with_head_values)
    marital_status = st.radio('The marital status of the interviewee:', marital_status_values)
    education_level = st.radio('Highest level of education:', education_level_values)
    job_type = st.radio('Type of job interviewee has:', job_type_values)
    
    diagnosis = ''
    
    if st.button('Check Eligibility'):
        diagnosis = expresso_prediction([
            location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, 
            relationship_with_head, marital_status, education_level, job_type
        ])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
