import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
import plotly.express as px

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

tab1, tab2=st.tabs(['Anomaly Detection', 'Supplier Segmentation'])
# DATA PROCESSING
with tab1:
    # Reading in dataset
    df = pd.read_csv('fraud_dataset.csv')
    # rename columns
    df.rename(columns = {'BELNR': 'document_no', 'WAERS': 'currency_key',
               'BUKRS': 'company_code', 'KTOSL': 'gl_key',
              'PRCTR': 'profit_center', 'BSCHL': 'posting_key',
              'HKONT': 'gl_account', 'DMBTR': 'amount_local_currency',
              'WRBTR': 'amount_document_currency'}, inplace=True)

    df['amount_local_currency'] = np.log(df['amount_local_currency']) + 1
    df['amount_document_currency'] = np.log(df['amount_document_currency'] + 1)

    from sklearn.preprocessing import MinMaxScaler

    m_scaler1 = MinMaxScaler()
    # local
    local_scaled = m_scaler1.fit_transform(df['amount_local_currency'].values.reshape(-1, 1))
    df['amount_local_currency'] = pd.DataFrame(local_scaled)

    m_scaler2 = MinMaxScaler()

    # document
    document_scaled = m_scaler2.fit_transform(df['amount_document_currency'].values.reshape(-1, 1))
    df['amount_document_currency'] = pd.DataFrame(document_scaled)

    from sklearn.preprocessing import StandardScaler

    # local
    sc1 = StandardScaler()

    local_scaled = sc1.fit_transform(df['amount_local_currency'].values.reshape(-1, 1))
    df['amount_local_currency'] = pd.DataFrame(local_scaled)

    # document
    sc2 = StandardScaler()

    docu_scaled = sc2.fit_transform(df['amount_document_currency'].values.reshape(-1, 1))
    df['amount_document_currency'] = pd.DataFrame(docu_scaled)

    # ===================================================================
    # ===================================================================
    columns = st.columns((1,1))
    with columns[0]:
        options = ['Currency Key', 'Company Code', 'GL Key', 'Profit Center', 'Posting Key', 'GL Account', 'Local Currency Amount', 'Document Currency Amount']
        selected = st.selectbox("Select Feature:", options=options)
        if selected == 'Currency Key':
            sub_columns = st.columns((2,1))
            with sub_columns[0]:
                st.write('There are 76 unique currency keys among this dataset.')
                # storing them into dataframe
                currency_df = pd.DataFrame(df['currency_key'].value_counts().head(10).reset_index())
                fig = px.bar(currency_df, x='index', y='currency_key', color='currency_key', color_continuous_scale='Purp',
                             labels={
                                 'currency_key': 'Count',
                                 'index': 'Currency Key'
                             })
                fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=16,
                                  height=500)
                st.plotly_chart(fig, use_container_width=True)
            with sub_columns[1]:
                st.write('\n')
                fig = px.scatter(df, x='label', y='currency_key', color='label', color_discrete_map={
                    'regular': '#3C567F',
                    'global': '#dfaeff',
                    'local': '#f3e2fe'
                },
                                 labels={
                                     'currency_key': 'Currenecy Key',
                                     'label': 'Data Label',
                                 })
                newnames = {'regular': 'Regular',
                            'local': 'Local (Anomaly)',
                            'global': 'Global (Anomaly)'}

                fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                                      legendgroup=newnames[t.name],
                                                      hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                                      )
                                   )
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


        if selected == 'Local Currency Amount':
            fig = px.box(df, x='label', y="amount_local_currency", color='label', color_discrete_map={
                'regular': '#3C567F',
                'global': '#dfaeff',
                'local': '#f3e2fe'
            })
            newnames = {'regular': 'Regular',
                        'local': 'Local (Anomaly)',
                        'global': 'Global (Anomaly)'}

            fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                                  legendgroup=newnames[t.name],
                                                  hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                                  )
                               )
            fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=20,
                              height=500)
            st.plotly_chart(fig, use_container_width=True)

        if selected == 'Document Currency Amount':
            fig = px.box(df, x='label', y="amount_document_currency", color='label', color_discrete_map={
                'regular': '#3C567F',
                'global': '#dfaeff',
                'local': '#f3e2fe'
            })
            newnames = {'regular': 'Regular',
                        'local': 'Local (Anomaly)',
                        'global': 'Global (Anomaly)'}

            fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                                  legendgroup=newnames[t.name],
                                                  hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                                  )
                               )
            fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=20,
                              height=500)
            st.plotly_chart(fig, use_container_width=True)

    with columns[1]:
        image = Image.open('img/overview.png')
        st.image(image, use_column_width=True)
        st.write('\n')


    # ===================================================================
    # ===================================================================

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

#=============================================================================================
#=============================================================================================
with tab2:
    df = pd.read_csv('supplier_segmentation.csv')
    df['num_manual_edit'] = df['num_manual_edit'].astype(object)

    columns = st.columns((1,1))
    with columns[0]:
        # PROBLEM STATEMENT
        sub_title = '<p style=" color:#dfaeff; font-size: 25px"><b>Problem Statement</b></p>'
        st.markdown(sub_title, unsafe_allow_html=True)
        image = Image.open('img/problem_statement.png')
        st.image(image, use_column_width=True)

        # DATASET OVERVIEW
        sub_title = '<p style=" color:#dfaeff; font-size: 25px"><b>Dataset Overview</b></p>'
        st.markdown(sub_title, unsafe_allow_html=True)
        image = Image.open('img/overview_supplier.png')
        st.image(image, use_column_width=True)



    with columns[1]:
        sub_title = '<p style=" color:#dfaeff; font-size: 25px"><b>Supplier Segmentation based on Invoice Data</b></p>'
        st.markdown(sub_title, unsafe_allow_html=True)
        st.write('Hover over data point to see supplier information and cause of manual edit.')
        # visualization
        fig = px.scatter_3d(df, x='sum', y='count', z='num_manual_edit', color='num_manual_edit', color_discrete_map={
            0: '#8ecae6',
            1: '#dfaeff'
        }, size_max=6,
                            labels={
                                'sum': 'Generated Profit from Suppliers',
                                'count': 'Frequency of Interactions with Suppliers',
                                'num_manual_edit': 'No. of Manual Edit'
                            },
                            hover_data=['Supplier_ID', 'cause'])

        fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=16,
                          height=900)
        st.plotly_chart(fig, use_container_width=True)