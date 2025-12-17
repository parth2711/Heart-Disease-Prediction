import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open('model.pkl','rb') as f:
    model=pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)

st.title('Heart Disease Detection')

st.markdown("""
### About the Model
This app uses a kNN model trained on the UCI heart disease dataset.
Predictions and probabilities are shown below.
""")

st.subheader("Enter details of the patient")

age=st.number_input('Age',1,120,55)
sex=st.selectbox('Sex',[0,1],format_func=lambda x:'Female' if x==0 else 'Male')
cp=st.selectbox('Chest Pain Type',[0,1,2,3,4,5],format_func=lambda x:{0:'Typical Angina',1:'Atypical Angina',2:'Non-anginal',3:'Asymptomatic',4:'Typical',5:'Atypical'}[x])
trestbps=st.number_input('Resting Blood Pressure (mm Hg)',80,200,120)
chol=st.number_input('Serum Cholesterol (mg/dl)',100,600,200)
fbs=st.selectbox('Fasting Blood Sugar > 120 mg/dl',[0,1],format_func=lambda x:'No' if x==0 else 'Yes')
restecg=st.selectbox('Resting ECG',[0,1,2],format_func=lambda x:{0:'Normal',1:'LV Hypertrophy',2:'ST-T Abnormality'}[x])
thalach=st.number_input('Maximum Heart Rate Achieved',60,220,150)
exang=st.selectbox('Exercise Induced Angina',[0,1],format_func=lambda x:'No' if x==0 else 'Yes')
oldpeak=st.number_input('ST Depression Induced by Exercise',0.0,6.0,1.0,0.1)
slope=st.selectbox('Slope of Peak Exercise ST Segment',[0,1,2],format_func=lambda x:{0:'Flat',1:'Upsloping',2:'Downsloping'}[x])
ca=st.selectbox('Number of Major Vessels (0–3)',[0,1,2,3])
thal=st.selectbox('Thalassemia',[0,1,2],format_func=lambda x:{0:'Normal',1:'Fixed Defect',2:'Reversible Defect'}[x])

sampleinput=pd.DataFrame([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]],columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalch','exang','oldpeak','slope','ca','thal'])
scaledsample=scaler.transform(sampleinput)
pred=model.predict(scaledsample)[0]
prob=model.predict_proba(scaledsample)[0]

st.divider()
st.subheader("Prediction Result")

if pred==1:
    st.error("Heart disease detected") 
else:
    st.success("No heart disease detected")

probdf=pd.DataFrame({"Condition":["No Heart Disease","Heart Disease"],"Probability":[prob[0],prob[1]]})
st.subheader("Prediction Probabilities")
st.dataframe(probdf.style.format({"Probability":"{:.2%}"}),use_container_width=True)
