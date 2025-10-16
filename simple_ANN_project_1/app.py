import streamlit as st           # standard alias for Streamlit
import numpy as np               # numerical operations
import tensorflow as tf          # deep learning models
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder  # preprocessing
import pandas as pd              # dataframes
import pickle                    # saving/loading Python objects


# load the train model 

model=tf.keras.models.load_model("model.h5")

 # load the encoder and scaler 
with open('oneHot_encoder_geo.pkl','rb') as file:
    oneHot_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)



with open('scalar.pkl','rb') as file:
    scalar=pickle.load(file)


## streamlit app
st.title("Customer Churn Prediction")

# 25 User input
geography = st.selectbox('Geography', oneHot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


#prepare data

input_data= pd.DataFrame({
    
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
    
    
    
    
})


#ohe hot encoder "geography"
geo_encodeed=oneHot_encoder_geo.transform([[geography]])
geo_encodeed_df=pd.DataFrame(geo_encodeed,columns=oneHot_encoder_geo.get_feature_names_out(['Geography']))


# concatinat one hot encodeed data with original df

input_data= pd.concat([input_data.reset_index(drop=True), geo_encodeed_df], axis=1)

# scale the input data
input_data_scale=scalar.transform(input_data)

# prediction
prediction=model.predict(input_data_scale)



prediction_prob=prediction[0][0]


if prediction_prob>0.5:
    st.write("the customer is likely to churn.")
    st.write("probility is :",prediction_prob)
else:
    st.write(" the cusomer is not likely to churn")
    st.write("probility is :",prediction_prob)