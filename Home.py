import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn import preprocessing
from numpy import ravel
# PAGE CONFIG
st.set_page_config(page_title='Home',
                   page_icon='😀',
                   layout="wide")

# Title Display
image = Image.open('img/title.png')
st.image(image,use_column_width=True)
# sub_title = '<p style=" color:#dfaeff; font-size: 40px; text-align: center"><b>AI/ML Capability - Anomaly Detection</b></p>'
# st.markdown(sub_title, unsafe_allow_html=True)

#===================================================================
#===================================================================
# DATA PROCESSING

# Reading in dataset
df = pd.read_csv('fraud_dataset.csv')
# rename columns
df.rename(columns = {'BELNR': 'document_no', 'WAERS': 'currency_key',
           'BUKRS': 'company_code', 'KTOSL': 'gl_key',
          'PRCTR': 'profit_center', 'BSCHL': 'posting_key',
          'HKONT': 'gl_account', 'DMBTR': 'amount_local_currency',
          'WRBTR': 'amount_document_currency'}, inplace=True)


#===================================================================
#===================================================================
## ViSUALIZATIONS

columns = st.columns((1,1))
with columns[0]:
    sub_title = '<p style=" color:#dfaeff; font-size: 25px"><b>Distribution of Target Variable - Label</b></p>'
    st.markdown(sub_title, unsafe_allow_html=True)
    # visualizing distribution
    label = pd.DataFrame(df.label.value_counts().reset_index())
    fig = px.bar(label, x='index', y='label', color = 'index', color_discrete_map={'regular':'#3C567F', 'global': '#dfaeff',
                                                                                                        'local': "#f3e2fe"},
                 # title='Distribution of Data Labels',
                 labels={
                     'label':'Count',
                     'index':'Data Label'
                 })
    newnames = {'regular': 'Regular',
                'local': 'Local (Anomaly)',
                'global': 'Global (Anomaly)'}

    fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                          legendgroup=newnames[t.name],
                                          hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                          )
                       )
    fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=20, height=500)
    st.plotly_chart(fig, use_container_width=True)

with columns[1]:
    sub_title = '<p style=" color:#dfaeff; font-size: 25px"><b>3D Label Distribution</b></p>'
    st.markdown(sub_title, unsafe_allow_html=True)
    # visualizing 3d distribution
    df_3d = pd.read_csv('anomaly_vectors.csv')
    fig = px.scatter_3d(df_3d, x='x', y='y', z='z', color='label', size_max=8,
                        labels={
                            'x': 'Principal Component 1',
                            'y': 'Principal Component 2',
                            'z': 'Principal Component 3'
                        },
                        color_discrete_map={
                            'Regular': '#3C567F',
                            'Global (Anomaly)': '#dfaeff',
                            'Local (Anomaly)': '#f3e2fe'
                        }, symbol='label')

    fig.update_layout(legend=dict(font=dict(
        size=16,
        color="#f3e2fe"
    )), height = 500)

    st.plotly_chart(fig, use_container_width=True)
    st.write('\n')
    st.write('Note: Click on legend or data points to interact with 3D plot')
# ===================================================================
# ===================================================================
columns = st.columns((1,1))
with columns[0]:
    options = ['Currency Key', 'Company Code', 'GL Key', 'Profit Center', 'Posting Key', 'GL Account']
    selected = st.selectbox("Select Categorical Feature:", options=options)
    if selected == 'Currency Key':
        st.write('There are 76 unique currency keys among this dataset.')
        # storing them into dataframe
        currency_df = pd.DataFrame(df['currency_key'].value_counts().head(10).reset_index())
        fig = px.bar(currency_df, x='index', y='currency_key', color='currency_key', color_continuous_scale='Purp',
                     labels={
                         'label': 'Count',
                         'index': 'Currency Key'
                     })
        fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=16,
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

    if selected == 'Company Code':
        st.write('There are 158 unique company codes among this dataset.')
        company_df = pd.DataFrame(df['company_code'].value_counts().head(20).reset_index())
        fig = px.bar(company_df, x='index', y='company_code', color='company_code', color_continuous_scale='Purp',
                     labels={
                         'label': 'Count',
                         'index': 'Company Code'
                     })
        fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=16,
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

    if selected == 'GL Key':
        st.write('There are 79 unique GL keys among this dataset.')
        glkey_df = pd.DataFrame(df['gl_key'].value_counts().head(10).reset_index())
        fig = px.bar(glkey_df, x='index', y='gl_key', color='gl_key', color_continuous_scale='Purp',
                     labels={
                         'label': 'Count',
                         'index': 'GL Keys'
                     })
        fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=16,
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

    if selected == 'Profit Center':
        st.write('There are 157 unique profit centers among this dataset.')
        prof_df = pd.DataFrame(df['profit_center'].value_counts().head(20).reset_index())
        fig = px.bar(prof_df, x='index', y='profit_center', color='profit_center', color_continuous_scale='Purp',
                     labels={
                         'label': 'Count',
                         'index': 'Profit Centers'
                     })
        fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=16,
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

    if selected == 'Posting Key':
        st.write('There are 73 unique posting keys among this dataset.')
        pk_df = pd.DataFrame(df['posting_key'].value_counts().head(8).reset_index())
        fig = px.bar(pk_df, x='index', y='posting_key', color='posting_key', color_continuous_scale='Purp',
                     labels={
                         'label': 'Count',
                         'index': 'Posting Keys'
                     })
        fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=16,
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

    if selected == 'GL Account':
        st.write('There are 73 unique GL accounts among this dataset.')
        gl_account_df = pd.DataFrame(df['gl_account'].value_counts().head(8).reset_index())
        fig = px.bar(gl_account_df, x='index', y='gl_account', color='gl_account', color_continuous_scale='Purp',
                     labels={
                         'label': 'Count',
                         'index': 'GL Accounts'
                     })
        fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=16,
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

with columns[1]:
    image = Image.open('img/overview.png')
    st.image(image, use_column_width=True)


# ===================================================================
# ===================================================================
# # MODEL
# df = pd.read_csv('anomaly_feature_engineering.csv')
# sub_title = '<p style=" color:#dfaeff; font-size: 25px"><b>Decision Tree Classifier</b></p>'
# st.markdown(sub_title, unsafe_allow_html=True)
# # Define X,y
# X = df.iloc[:,1:-1]
# y = df.iloc[:,-1:]
# le = preprocessing.LabelEncoder()
# y = le.fit_transform(ravel(y))
# # Splitting data
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0,stratify=y)
# clf = DecisionTreeClassifier(random_state=0)
# clf.fit(X_train, y_train)
# ypred=clf.predict(X_test)
# colors = ['#3C567F', '#dfaeff', '#f3e2fe']
# fig, ax = plt.subplots(figsize=(12,8), dpi = 300)
# plot_tree(clf,  filled=True, fontsize=7, rounded=True)
# st.pyplot(fig)
