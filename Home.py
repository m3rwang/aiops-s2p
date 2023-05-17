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

tab1, tab2=st.tabs(['   Anomaly Detection   ', '   Supplier Segmentation'])
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
                currency_df = pd.DataFrame(df['currency_key'].value_counts().head(10))
                currency_df['Currency Key'] = currency_df.index
                fig = px.bar(currency_df, x='Currency Key', y='currency_key', color='currency_key', color_continuous_scale='Purp',
                             labels={
                                 'currency_key': 'Count'
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
            fig = px.bar(company_df, x=company_df.index, y='company_code', color='company_code', color_continuous_scale='Purp',
                         labels={
                             'label': 'Count',
                             '_index': 'Company Code'
                         })
            fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=16,
                              height=500)
            st.plotly_chart(fig, use_container_width=True)

        if selected == 'GL Key':
            st.write('There are 79 unique GL keys among this dataset.')
            glkey_df = pd.DataFrame(df['gl_key'].value_counts().head(10).reset_index())
            fig = px.bar(glkey_df, x=glkey_df.index, y='gl_key', color='gl_key', color_continuous_scale='Purp',
                         labels={
                             'label': 'Count',
                             '_index': 'GL Keys'
                         })
            fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=16,
                              height=500)
            st.plotly_chart(fig, use_container_width=True)

        if selected == 'Profit Center':
            st.write('There are 157 unique profit centers among this dataset.')
            prof_df = pd.DataFrame(df['profit_center'].value_counts().head(20).reset_index())
            fig = px.bar(prof_df, x=prof_df.index, y='profit_center', color='profit_center', color_continuous_scale='Purp',
                         labels={
                             'label': 'Count',
                             '_index': 'Profit Centers'
                         })
            fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=16,
                              height=500)
            st.plotly_chart(fig, use_container_width=True)

        if selected == 'Posting Key':
            st.write('There are 73 unique posting keys among this dataset.')
            pk_df = pd.DataFrame(df['posting_key'].value_counts().head(8).reset_index())
            fig = px.bar(pk_df, x=pk_df.index, y='posting_key', color='posting_key', color_continuous_scale='Purp',
                         labels={
                             'label': 'Count',
                             '_index': 'Posting Keys'
                         })
            fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=16,
                              height=500)
            st.plotly_chart(fig, use_container_width=True)

        if selected == 'GL Account':
            st.write('There are 73 unique GL accounts among this dataset.')
            gl_account_df = pd.DataFrame(df['gl_account'].value_counts().head(8).reset_index())
            fig = px.bar(gl_account_df, x=gl_account_df.index, y='gl_account', color='gl_account', color_continuous_scale='Purp',
                         labels={
                             'label': 'Count',
                             '_index': 'GL Accounts'
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
        label = pd.DataFrame(df.label.value_counts())
        label['Label'] = label.index.map({'regular': 'Regular',
                                             'local': 'Local (Anomaly)',
                                             'global': 'Global (Anomaly)'})

        fig = px.bar(label, x='Label', y='label', color = 'Label', color_discrete_map={'Regular':'#3C567F', 'Local (Anomaly)': '#dfaeff',
                                                                                                            'Global (Anomaly)': "#f3e2fe"},
                     # title='Distribution of Data Labels',
                     labels={
                         'label':'Count',
                         'Label':'Data Label'
                     })

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
        image = Image.open('img/detail.png')
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

    #======================================================================
    #======================================================================
    # CLUSTERING
    columns = st.columns((1,1))
    with columns[0]:
        sub_title = '<p style=" color:#dfaeff; font-size: 25px"><b>Supplier Segmentation Clustering Result Utilizing KMean Algorithm</b></p>'
        st.markdown(sub_title, unsafe_allow_html=True)
        cluster = pd.read_csv('cluster.csv')
        # visualization
        fig = px.scatter_3d(cluster, x='sum', y='count', z='num_manual_edit', color='label',
                            color_discrete_map={
                                'Cluster 1': '#3C567F',
                                'Cluster 2': '#8ecae6',
                                'Cluster 3': '#f3e2fe'
                            },
                            size_max=6,
                            labels={
                                'sum': 'Generated Profit from Suppliers',
                                'count': 'Frequency of Interactions with Suppliers',
                                'num_manual_edit': 'No. of Manual Edit'
                            },
                            hover_data=['Supplier_ID', 'cause'])
        fig.update_layout(paper_bgcolor="#202020", plot_bgcolor='#202020', font_color='#f3e2fe', font_size=16,
                          height=650)
        st.plotly_chart(fig, use_container_width=True)
    # EVALUATION

    with columns[1]:
        sub_title = '<p style=" color:#dfaeff; font-size: 25px"><b>KMean Algorithm Evaluation</b></p>'
        st.markdown(sub_title, unsafe_allow_html=True)
        elbow = pd.read_csv('elbow.csv')
        fig = px.line(elbow, x='K', y='Inertia', markers=True)
        fig.add_annotation(x=2.6, y=3900,
                           text='"Elbow"',
                           showarrow=False)
        fig.update_layout(title=dict(text='Elbow Method: Change in Inertia as K Increases', font=dict(size=20)),
                          font_color='#f3e2fe', font_size=20, paper_bgcolor="#202020", plot_bgcolor='#202020', height=550)
        st.plotly_chart(fig, use_container_width=True)

        st.write("**Elbow** method gives us an idea on what a good K number of clusters would be based on the sum of squared distance (SSE) between data points and their assigned clusters’ centroids."
                 "\n"
                 "We pick K at the spot where SSE starts to flatten out and forming an elbow."
                 "\n"
                 "In this case, we select K=3 as our cluster number based on the 'elbow' as indicated above.")

    sub_title = '<p style=" color:#dfaeff; font-size: 25px"><b>Interactive Visualization to Select Suppliers</b></p>'
    st.markdown(sub_title, unsafe_allow_html=True)


    from bokeh.plotting import output_notebook
    output_notebook()  # set default; alternative is output_file()

    from bokeh.io import output_notebook  # prevent opening separate tab with graph


    # arranging bokeh layout objects

    from bokeh.layouts import row
    from bokeh.layouts import grid

    # ColumnDataSource for mapping between column name and list of data & CustomJS callbacks
    from bokeh.models import CustomJS, ColumnDataSource
    from bokeh.models import Button  # for saving data
    from bokeh.events import ButtonClick  # for saving data

    from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
    from bokeh.models import HoverTool
    from bokeh.plotting import figure
    from bokeh.models import Legend, LegendItem

    # reading the data to variables
    df = pd.read_csv('bokeh.csv')
    x = df['sum'].to_list()
    y = df['count'].to_list()
    z = df['num_manual_edit'].to_list()
    d = df['Supplier_ID'].to_list()
    f = df['cause'].to_list()

    colormap = {0: 'red', 1: 'green', 2: 'blue'}
    colors = [colormap[x] for x in df['clusters']]
    e = colors

    # create first subplot
    plot_width = 550
    plot_height = 600

    s1 = ColumnDataSource(data=dict(x=x, y=y, z=z, d=d, e=e, f=f))

    fig01 = figure(
        width=plot_width,
        height=plot_height,
        tools=["lasso_select", "reset", "save"],
        title="Select Interesting Suppliers in this Plot",
    )

    fig01.circle("x", "y", source=s1, alpha=0.9, color="e")

    # color legend
    rc = fig01.rect(x=0, y=0, height=1, width=1, color=["red", "blue", "green"])
    rc.visible = False

    # let us define Cluster information and use it as a legend for the first plot
    cluster_name = ["Cluster_0: profit= low, interaction: low, no of edits: 0",
                    "Cluster_1: profit= medium, interaction: medium, no of edits: 0",
                    "Cluster_2: profit= high, interaction: high, no of edits: 0,1"]
    legend = Legend(items=[
        LegendItem(label=cluster_name[i], renderers=[rc], index=i) for i in range(0, len(cluster_name))
    ], location="top_left")
    fig01.add_layout(legend, 'below')

    # let us create an HoverTool and define what needs to be displayed
    tooltips = [
        ("Profit_value:", "@x"),
        ("Frequency_of_interaction:", "@y"),
        ("Number_of_Manual_Edits", "@z"),
        ("Supplier_Id", "@d"),
        ("Cause", "@f"),
    ]

    fig01.add_tools(HoverTool(tooltips=tooltips))


    # create second subplot
    s2 = ColumnDataSource(data=dict(x=[], y=[], z=[], d=[], e=[], f=[]))

    fig02 = figure(
        width=500,
        height=600,
        x_range=(0, 13),
        y_range=(0, 10),
        tools=["box_zoom", "wheel_zoom", "reset", "save"],
        title="These are your selected Suppliers",
    )

    fig02.circle("x", "y", source=s2, alpha=0.9, color="e")

    # let us create a dynamic table
    columns = [
        TableColumn(field="x", title="Profit value"),
        TableColumn(field="y", title="Frequency of Interactions"),
        TableColumn(field="z", title="Num_Edit"),
        TableColumn(field="d", title="Supplier_Id"),
        TableColumn(field="f", title="Cause"),
    ]

    table = DataTable(
        source=s2,
        columns=columns,
        width=600,
        height=150,
        sortable=True,
        selectable=True,
        editable=True,
    )
    # Inspired from online Bokeh tutorials

    # let us link subplots using CustomJS

    s1.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(s1=s1, s2=s2, table=table),
            code="""
            var inds = cb_obj.indices;
            var d1 = s1.data;
            var d2 = s2.data;
            d2['x'] = []
            d2['y'] = []
            d2['z'] = []
            d2['d'] = []
            d2['e'] = []
            d2['f'] = []
            for (var i = 0; i < inds.length; i++) {
                d2['x'].push(d1['x'][inds[i]])
                d2['y'].push(d1['y'][inds[i]])
                d2['z'].push(d1['z'][inds[i]])
                d2['d'].push(d1['d'][inds[i]])
                d2['e'].push(d1['e'][inds[i]])
                d2['f'].push(d1['f'][inds[i]])
            }
            s2.change.emit();
            table.change.emit();

            var inds = source_data.selected.indices;
            var data = source_data.data;
            var out = "x, y, z, d, e, f \\n";
            for (i = 0; i < inds.length; i++) {
                out += data['x'][inds[i]] + "," + data['y'][inds[i]] + "," + data['z'][inds[i]]+ "," + data['d'][inds[i]] + "," + data['f'][inds[i]] + "\\n";
            }
            var file = new Blob([out], {type: 'text/plain'});

        """,
        ),
    )


    # let us create our save button

    savebutton = Button(label="Save", button_type="success")
    savebutton.js_on_event(ButtonClick, CustomJS(
        args=dict(source_data=s1),
        code="""
            var inds = source_data.selected.indices;
            var data = source_data.data;
            var out = "Profit_value, Frequency_of_interaction, Number_of_Manual_Edits, Supplier_Id, Cause \\n";
            for (var i = 0; i < inds.length; i++) {
                out += data['x'][inds[i]] + "," + data['y'][inds[i]] + "," + data['z'][inds[i]] + "," + data['d'][inds[i]] + "," + data['f'][inds[i]] + "\\n";
            }
            var file = new Blob([out], {type: 'text/plain'});
            var elem = window.document.createElement('a');
            elem.href = window.URL.createObjectURL(file);
            elem.download = 'selected-data.txt';
            document.body.appendChild(elem);
            elem.click();
            document.body.removeChild(elem);
            """
    )
                           )

    # let us create an HoverTool and define what needs to be displayed
    tooltips = [
        ("Profit_value:", "@x"),
        ("Frequency_of_interaction:", "@y"),
        ("Number_of_Manual_Edits", "@z"),
        ("Supplier_Id", "@d"),
        ("Cause", "@f"),
    ]

    fig02.add_tools(HoverTool(tooltips=tooltips))

    # let us create a layout and display all our objects together
    layout = grid([fig01, fig02, table, savebutton], ncols=2)
    st.bokeh_chart(layout)










css = '''
<style>
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    text-align: center; font-size: 32px
    }
    
</style>
'''

st.markdown(css, unsafe_allow_html=True)

