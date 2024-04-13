
import pandas as pd
import numpy as np 
import streamlit as st

from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

from model import model1, model2
from calc import get_dataframes

prob_df, feat_df = get_dataframes()

st.set_option('deprecation.showPyplotGlobalUse', False)

api_key_mapbox = 'pk.eyJ1IjoiYmluaXQxMyIsImEiOiJjbHRyaWQ3dzcwZWNrMnBrOHNtZzVmOTd4In0.c_HGh6IUkJJRBQTiYX9RSw'

df = pd.read_csv('repository/Telecom_churn_final')
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
col_cal = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies','Streaming Music','Unlimited Data','Paperless Billing','Payment Method','Paperless Billing','Contract','Internet Service']
category_mapping = {'Competitor': 1, 'Dissatisfaction': 2, 'Attitude': 3, 'Price': 4, 'Other': 5}

churn_data = df[df['Customer Status'] == 'Churned']
pie_data = df[(df['Customer Status'] == 'Stayed') | (df['Customer Status'] == 'Joined')]

st.set_page_config(layout="wide")
def streamlit_menu():

    selected = option_menu(
        menu_title=None,  # required
        options=["Home | Descriptive ", "Predictive Analytics", "Prescriptive Analytics"],  # required
        icons=["bi bi-activity", "bi bi-clipboard-data", "bi bi-pie-chart"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#C6DDB6"},
            "icon": {"color": "#072810", "font-size": "25px"},
            "nav-link": {
                "font-size": "25px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#355F10"},
        },
    )
    return selected

select  = streamlit_menu()


if select == "Home | Descriptive ":
    col1, col2, col3 = st.columns([1,0.13,1])
    col1_bg_color = "#c1d6c1"
    col2_bg_color = "#e4f2e4"

    with col2:
        icon_url = "repository/3242288-200.png"
        st.image(icon_url, width=75)

    col1.markdown(
        f'<div style="background-image: linear-gradient(to right, #428142, #AAD4AA, 80%, transparent); font-size: 20px; padding: 8px; text-align: center;">Current User | New User &nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;\
        4720|454 </div>',
        unsafe_allow_html=True)
    with col1:
        col_1, col_2 = st.columns(2)
        with col_1:
            colors = ['#A3DD9B', '#EBF1EA']
            color = ['#A3DD9B', '#C0ECBA']
            trace = go.Pie(labels=pie_data['Gender'].unique(), values=pie_data['Gender'].value_counts(), hole=0.4, marker=dict(colors=color))
            layout = go.Layout(title=dict(text='Demographics', x=0.3, y=0.9, xanchor='center', yanchor='top'),
                               legend=dict(orientation='v', x=1.5, y=1, font=dict(size=7)), height=300,width=300)
            fig = go.Figure(data=[trace], layout=layout)
            st.plotly_chart(fig, use_container_width=True)

        with col_2:
            st.markdown("\n\n\n")
            senior_data = ((pie_data[pie_data['Age'] > 65]['Age'].count() / pie_data['Age'].count()) * 100)
            senior_datab = "{:.2f}%".format(senior_data)
            st.markdown("""<div style='background-color: #c1d6c1; padding: 10px; border-radius: 5px'>
                    <h2 style='text-align: center'>{}</h2>
                    <p style='font-size: 24px; text-align: center'>Senior Citizen</p></div>""".format(senior_datab), unsafe_allow_html=True)

            st.markdown("\n")
            married_data = ((pie_data[pie_data['Married'] == 'Yes']['Married'].count() / pie_data['Married'].count()) * 100)
            married_data = "{:.2f}%".format(married_data)
            st.markdown(
                    """<div style='background-color: #c1d6c1; padding: 2px; border-radius: 2px'>
                       <h2 style='text-align: center'>{}</h2>
                       <p style='font-size: 24px; text-align: center'>Partner</p></div>""".format(married_data),
                    unsafe_allow_html=True)

        with st.expander('Data Description'):
            cols1, cols2 = st.columns(2)
            with cols1:
                button_stayed = st.button(label='Stayed')
            with cols2:
                button_joined = st.button(label ='Joined')
            if button_stayed:
                stayed_des = pie_data[pie_data['Customer Status']=='Stayed'].drop(columns=['Zip Code','Latitude','Longitude']).describe().transpose()
                stayed_des = stayed_des.apply(pd.to_numeric, errors='ignore')
                stayed_des = round(stayed_des, 2)
                st.write(stayed_des.style.background_gradient(cmap='Greens_r'))
                print(stayed_des)
            if button_joined:
                stayed_des_join = pie_data[pie_data['Customer Status'] == 'Joined'].drop(columns=['Zip Code', 'Latitude', 'Longitude']).describe().transpose().round(2)
                stayed_des_join = stayed_des_join.apply(pd.to_numeric, errors='ignore')
                stayed_des_join = round(stayed_des_join, 2)
                st.write(stayed_des_join.style.background_gradient(cmap='Greens_r'))

        st.markdown('\n\n\n')
        st.markdown(f'<div style = "background-image: linear-gradient(to right, #428142, #AAD4AA); font-size: 20px; padding: 8px; text-align: center;">Users Categorical Features</div>',unsafe_allow_html=True)
        selected_category = st.selectbox('', col_cal)
        colors = ['#379037', '#71B971', '#9DD39D']
        val_user = pie_data[selected_category].value_counts().values
        total = sum(val_user)
        percentage_user = [f'{(v / total) * 100:.2f}%' for v in val_user]
        trace = go.Bar(x=pie_data[selected_category].value_counts().index,
                       y=pie_data[selected_category].value_counts().values,
                       marker=dict(color=colors), hovertext=percentage_user)
        layout = go.Layout()

        fig = go.Figure(data=[trace], layout=layout)
        st.plotly_chart(fig, use_container_width=True)


    col3.markdown(
        f'<div style="background-image: linear-gradient(to left, #428142, #AAD4AA, 80%, transparent); font-size: 20px; padding: 8px; text-align: center;">Churn Customer  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;\
        &nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp  1869</div>',
        unsafe_allow_html=True)
    with col3:
        cols_1, cols_2 = st.columns(2)

        with cols_1:
            colors = ['#EBF1EA','#F1F6F1']
            trace = go.Pie(labels=churn_data['Gender'].unique(), values=churn_data['Gender'].value_counts(), hole=0.4, marker=dict(colors=color))
            layout = go.Layout(title=dict(text='Demographics', x=0.3, y=0.9, xanchor='center', yanchor='top'),
                               legend=dict(orientation='v', x=1.5, y=1, font=dict(size=7)), height=300,width=300)
            fig = go.Figure(data=[trace], layout=layout)
            st.plotly_chart(fig, use_container_width=True)

        with cols_2:
            st.markdown("\n\n\n")
            churn_senior_data = ((churn_data[churn_data['Age'] > 65]['Age'].count() / churn_data['Age'].count()) * 100)
            churn_senior_datab = "{:.2f}%".format(churn_senior_data)
            st.markdown("""<div style='background-color: #e4f2e4; padding: 10px; border-radius: 5px'>
                    <h2 style='text-align: center'>{}</h2>
                    <p style='font-size: 24px; text-align: center'>Senior Citizen</p></div>""".format(churn_senior_datab), unsafe_allow_html=True)

            st.markdown("\n")
            churn_married_data = ((churn_data[churn_data['Married'] == 'Yes']['Married'].count() / churn_data['Married'].count()) * 100)
            churn_married_data = "{:.2f}%".format(churn_married_data)
            st.markdown(
                    """<div style='background-color: #e4f2e4; padding: 2px; border-radius: 2px'>
                       <h2 style='text-align: center'>{}</h2>
                       <p style='font-size: 24px; text-align: center'>Partner</p></div>""".format(churn_married_data),
                    unsafe_allow_html=True)

        with st.expander('Data Description'):
            stayed_des = churn_data.drop(columns=['Zip Code','Latitude','Longitude']).describe().transpose().round(2)
            st.write(stayed_des.style.background_gradient(cmap='Greens_r'))

        st.markdown('\n\n\n')
        st.markdown(
            f'<div style = "background-image: linear-gradient(to left, #428142, #AAD4AA, 100%, transparent); font-size: 20px; padding: 8px; text-align: center;"> Churn User Categorical Feature </div>',
            unsafe_allow_html=True)
        selected_category_churn = st.selectbox('', col_cal, key=2)
        colors = ['#379037', '#71B971', '#9DD39D']
        val_churn = churn_data[selected_category_churn].value_counts().values
        total = sum(val_churn)
        percentages = [f'{(v / total) * 100:.2f}%' for v in val_churn]

        trace = go.Bar(x=churn_data[selected_category_churn].value_counts().index,
                       y=churn_data[selected_category_churn].value_counts().values, marker=dict(color=colors),
                       hovertext=percentages)
        layout = go.Layout()
        fig = go.Figure(data=[trace], layout=layout)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('\n\n')
    st.markdown(f'<div style = "background-image: linear-gradient(to right, #428142, #AAD4AA, 50%, transparent); font-size: 20px; padding: 8px; text-align: center;"> Numerical Distribution </div>',
    unsafe_allow_html=True)

    num_list = ['Age', 'Tenure in Months', 'Monthly Charge', 'Total Refunds','Avg Monthly GB Download','Total Extra Data Charges', 'Total Long Distance Charges','Total Revenue']
    num_select = st.selectbox('', num_list,)
    colss1, colss2, colss3 =  st.columns([1, 1, 1])

    with colss1:
        fig = px.histogram(pie_data, x=num_select, color='Customer Status', barmode='overlay',
                           color_discrete_map={'Joined': '#4FBB4F', 'Stayed': '#C2E7C2'})
        fig.update_layout(title=dict(text='Existing User', x=0.5, xanchor='center',font=dict(size=20)), xaxis_title=num_select,
                          yaxis_title='Frequency', margin=dict(t=120),)
        st.plotly_chart(fig, use_container_width=True)

    with colss2:
        cat_col = ['Married','Phone Service','Internet Service','Streaming Music','Unlimited Data','Paperless Billing', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies']
        num_cat_select = st.selectbox('', cat_col)
        num_data = df.groupby([num_cat_select, 'Customer Status']).agg({'Customer Status': 'count'}).rename(columns={'Customer Status': 'count'}).reset_index()
        fig = px.sunburst(num_data, path=[num_cat_select, 'Customer Status'], values='count', color='Customer Status',
                           title=' ', color_discrete_map={'No':'#1A5C1A', 'Yes':'#108C10',
                                                          'Churned':'#94D494', 'Joined':'#C3E7C3','Stayed':'#5CAD5C'})

        fig.update_layout(plot_bgcolor='seagreen', title_font_family='Calibri Black', title_font_color='#221f1f',
                          title_font_size=12, title_x=0.5, margin=dict(t=1),)
        fig.update_traces(textinfo='label + percent parent')
        st.plotly_chart(fig, use_container_width=True)

    with colss3:
        fig = px.histogram(churn_data, x=num_select, color_discrete_sequence=['rgb(197, 227, 197)'])
        fig.update_layout(title=dict(text='Churned Customer' ,x=0.5, xanchor='center',font=dict(size=20)),xaxis_title=num_select,
                          yaxis_title='Frequency',margin=dict(t=120),)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('\n\n')
    st.markdown(f'<div style="background-image: linear-gradient(to right, #428142, #AAD4AA, 50%, transparent); font-size: 20px; padding: 8px; text-align: center;"> Correlation | Map | Churn Reason </div>',
    unsafe_allow_html=True)
    st.markdown('\n\n\n\n')

    dff = pd.read_csv('repository/Telecom_churn_semi')
    corr = dff.drop(columns=['Unnamed: 0','Unnamed: 0.1','Customer ID', 'Gender', 'Married', 'City', 'Zip Code', 'Offer', 'Internet Type', 'Contract',
                  'Payment Method', 'Customer Status', 'Churn Category', 'Churn Reason']).corr()
    plt.figure(figsize=(30, 16))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    ax = sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.3,
                     cmap='Greens', vmin=-1, vmax=1)
    st.pyplot()

    pred_Col1, pred_Col2 = st.columns([2.25,1.75])
    with pred_Col1:
        px.set_mapbox_access_token(api_key_mapbox)
        colors_ = ['rgb(36, 105, 36)', 'rgb(79, 152, 79)', 'rgb(187, 225, 187)']
        fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Customer Status",
                                size="Tenure in Months", color_discrete_sequence=colors_, size_max=20, zoom=10,
                                hover_data=['Age', 'City', 'Avg Monthly Long Distance Charges',
                                            'Avg Monthly GB Download', 'Unlimited Data', 'Contract','Tenure in Months'])
        fig.update_layout(title=dict(text='Customer Status Map', x=0.5, y=0.95, xanchor='center',
                                     yanchor='top', font=dict(size=20)), )
        st.plotly_chart(fig, use_container_width=True)

    with pred_Col2:
        col1_bg_color = "#538C53"
        col2_bg_color = "#4FBB4F"
        col3_bg_color = "#C2E7C2"

        values_ = list(range(20))
        sorted_reason_counts = churn_data['Churn Reason'].value_counts().sort_values(ascending=True)

        trace = go.Bar(x=sorted_reason_counts.values,
                       y=sorted_reason_counts.index, orientation='h',
                       marker=dict(color=values_, colorscale='Greens'))
        layout = go.Layout(margin=dict(t=50))
        fig = go.Figure(data=[trace], layout=layout)
        st.plotly_chart(fig, use_container_width=True)


if select == "Predictive Analytics":

    col_pr1, col_pr2, col_pr3 = st.columns([1,0.15,1])
    with col_pr2:
        icon_url = "repository/3242288-200.png"
        st.image(icon_url, width=75)

    with col_pr3:
        prediction_placeholder = st.empty()
        st.session_state['prediction_result'] = ''
        st.session_state['Churn_reason'] = ''
        prediction_placeholder.markdown(f"""<div style='background-color: #c1d6c1; padding: 100px; border-radius: 5px'>
                                                <h2 style='text-align: center'>Prediction</h2>
                                                <p style='font-size: 24px; text-align: center'>{st.session_state['prediction_result']}  |  {st.session_state['Churn_reason']}</p></div>""",
                                        unsafe_allow_html=True)

        st.markdown('\n\n')

        predict_btn = st.button('Predict', use_container_width=True)
        pr_col1, pr_col2, pr_col3, pr_col4 =  st.columns(4)
        sec_col1, sec_col2 = st.columns(2)
        with pr_col1:
            gender = st.selectbox('Gender',['Male','Female'])
            udata = st.selectbox('Unlimited Data', ['No','Yes',])
            backup = st.selectbox('Online Backup',['Yes','No'])
            movies = st.selectbox('Streaming Movies',['Yes','No'])

        with sec_col1:
            age = int(st.number_input('Age'))
            revenue = float(st.number_input('Total Revenue'))
        with sec_col2:
            tenure = int(st.number_input('Tenure (Month)'))
            month_charge = float(st.number_input('Monthly Charge'))

        with pr_col2:
            service = st.selectbox('Phone Service',['No','Yes',])
            contract = st.selectbox('Contract', ['Month-to-Month', 'One Year','Two Year'])
            protection = st.selectbox('Device Protection', ['No','Yes',])
            tech_support = st.selectbox('Tech Support', ['Yes', 'No'])
        with pr_col3:
            married = st.selectbox('Married',['Yes','No'])
            internet_service = st.selectbox('Internet Service',['No','Yes',])
            billing = st.selectbox('Paperless Billing',['Yes','No'])
            mult_line = st.selectbox('Multiple Lines',['No','Yes'])

        with pr_col4:
            dependent = st.selectbox('Dependents', ['No','Yes',])
            music = st.selectbox('Streaming Music',['Yes','No'])
            security = st.selectbox('Online Security', ['No','Yes',])
            tv = st.selectbox('Streaming TV', ['Yes','No'])
        payment = st.selectbox('Payment Method', ['Bank Withdrawal', 'Credit Card', 'Mailed Check'])

        pdata = {
            'Gender': [gender], 'Age': [age], 'Married': [married], 'Number of Dependents': [dependent],
            'Tenure in Months': [tenure],
            'Phone Service': [service], 'Internet Service': [internet_service], 'Streaming Music': [music], 'Unlimited Data': [udata],
            'Contract': [contract], 'Paperless Billing': [billing], 'Payment Method': [payment],
            'Monthly Charge': [month_charge],
            'Total Revenue': [revenue], 'MultipleLines': [mult_line], 'TechSupport': [tech_support], 'OnlineBackup': [backup],
            'DeviceProtection': [protection], 'OnlineSecurity': [security], 'StreamingTV': [tv], 'StreamingMovies': [movies]}

        pred_data = pd.DataFrame(pdata)

        if predict_btn:
            churn_pred = model1.predict(pred_data.values)

            if churn_pred[0]==1:
                churn_pred2 = model2.predict(pred_data.values)[0]
                for key, value in category_mapping.items():
                    if value == churn_pred2:
                        st.session_state['prediction_result'] = "Churn" if churn_pred == 1 else "On Going Customer"
                        st.session_state['Churn_reason'] = key
                        prediction_placeholder.markdown(f"""<div style='background-color: #c1d6c1; padding: 100px; border-radius: 5px'>
                                                                                        <h2 style='text-align: center'>Churn Prediction</h2>
                                                                                        <p style='font-size: 24px; text-align: center'>{st.session_state['prediction_result']}  |  {st.session_state['Churn_reason']}</p></div>""",
                                                        unsafe_allow_html=True)

            else:
                st.session_state['prediction_result'] = "On Going Customer"
                prediction_placeholder.markdown(f"""<div style='background-color: #c1d6c1; padding: 100px; border-radius: 5px'>
                                                                <h2 style='text-align: center'>Churn Prediction</h2>
                                                                <p style='font-size: 24px; text-align: center'>{st.session_state['prediction_result']}</p></div>""",
                                                    unsafe_allow_html=True)


    st.markdown('\n\n\n\n\n\n')

    with col_pr1:

        st.markdown(
            f'<div style = "background-image: linear-gradient(to right, #428142, #AAD4AA, 80%, transparent); font-size: 20px; padding: 8px; text-align: center;">User Multi-Variate Observation</div>',
            unsafe_allow_html=True)

        sel_cols = ['Number of Dependents','Phone Service','Internet Service','Streaming Music','Unlimited Data',
                    'Paperless Billing','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                    'StreamingTV','StreamingMovies','Contract']
        sel_cols_map = {'Yes': 1, 'No': 0}
        contract_map = {'Month-Month':0, 'One Year':0.5, 'Two Year':1}
        mod_pred_data = pred_data[sel_cols]
        radar_data = mod_pred_data.replace(sel_cols_map)
        radar_data = radar_data.replace(contract_map)


        fig = px.line_polar(mod_pred_data, radar_data.values.reshape(-1), theta=sel_cols, line_close=True)
        fig.update_traces(fill='toself', line_color='green')
        st.plotly_chart(fig, use_container_width=True)



        st.markdown(
            f'<div style = "background-image: linear-gradient(to right, #428142, #AAD4AA, 50%, transparent); font-size: 20px; padding: 8px; text-align: center;">Churn Category</div>',
            unsafe_allow_html=True)

        category = churn_data['Churn Category'].value_counts().reset_index()
        total = category.iloc[0:, 1].sum()
        category['percentage'] = (category.iloc[0:, 1] / total) * 100
        category['percentage'] = category['percentage'].round(2)
        category['hover_text'] = category.apply(lambda row: f"{row['percentage']:.2f}%", axis=1)
        hover_text = category['hover_text']
        fig = go.Figure(go.Treemap(labels=category['Churn Category'], values=category['count'], parents=[None] * 5,
                                   hovertext=hover_text, root_color="white",
                                   marker=dict(
                                       colors=['rgb(110, 179, 110)'] + ['rgb(146,199,146)'] + ['rgb(168,213,168)'] + [
                                           'rgb(191, 228, 191)'] * 2)))
        fig.update_layout(margin=dict(t=40, l=25, r=25, b=25))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div style = "background-image: linear-gradient(to right, #428142, #AAD4AA, 50%, transparent); font-size: 20px; padding: 8px; text-align: center;">Contract | Internet Service | Customer Status Relation </div>',
        unsafe_allow_html=True)

    temp1 = df.groupby(['Contract', 'Internet Service']).size().reset_index()
    temp1.columns = ['source', 'target', 'value']
    temp2 = df.groupby(['Internet Service', 'Customer Status']).size().reset_index()
    temp2.columns = ['source', 'target', 'value']
    sank_data = pd.concat([temp1, temp2], axis=0)
    u_list = list(pd.unique(sank_data[['source', 'target']].values.ravel('K')))
    mapping_dict = {k: v for v, k in enumerate(u_list)}

    sank_data['source'] = sank_data['source'].map(mapping_dict)
    sank_data['target'] = sank_data['target'].map(mapping_dict)
    sank_list = sank_data.to_dict(orient='list')

    color_green = ['rgba(24, 104, 63, 0.70)','rgba(24, 104, 63, 0.70)','rgba(24, 104, 63, 0.70)',
                   'rgba(39, 146, 93, 0.70)','rgba(39, 146, 93, 0.70)','rgba(163, 213, 188, 0.70)',
                   'rgba(163, 213, 188, 0.70)','rgba(163, 213, 188, 0.70)']
    cgreen = ['rgba(0, 128, 0, 0.5)','rgba(137, 201, 168, 0.2)','rgba(13, 138, 13, 0.509)',
              'rgba(19, 143, 19, 0.513)','rgba(26, 148, 26, 0.518)','rgba(32, 153, 32, 0.522)',
                'rgba(39, 158, 39, 0.527)','rgba(45, 163, 45, 0.531)','rgba(52, 168, 52, 0.536)',
                'rgba(137, 201, 168, 0.4)','rgba(65, 178, 65, 0.545)','rgba(72, 183, 72, 0.55)']

    fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20,
                                              line=dict(color='black', width=0.5), label=u_list, color=color_green),
                                    link=dict(source=sank_list['source'],
                                              target=sank_list['target'],
                                              value=sank_list['value'], color=cgreen,
                                              ))])
    fig.update_layout(margin=dict(t=50),)
    st.plotly_chart(fig, use_container_width=True)



    st.markdown(
        f'<div style = "background-image: linear-gradient(to right, #428142, #AAD4AA, 50%, transparent); font-size: 20px; padding: 8px; text-align: center;">Feature Importance</div>',
        unsafe_allow_html=True)

    val_col = list(range(40))
    trace = go.Bar(x=feat_df['Importance'].values,
                   y=feat_df.index, orientation='h',
                   marker=dict(color=val_col, colorscale='Greens'))
    layout = go.Layout(margin=dict(t=50), height=800)
    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig, use_container_width=True)

if select == "Prescriptive Analytics":

    pres_col1, pres_col2 = st.columns([1.5,2.5])
    cust_placeholder = st.empty()
    image_path = 'repository/211746.png'

    with pres_col1:

        cust_id = st.selectbox('Customer ID', prob_df['Customer ID'], )

        st.markdown('\n\n\n\n\n')

    with pres_col2:

        with open('repository/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        prob_val = f"{prob_df[prob_df['Customer ID']==cust_id]['Probs'].values[0]}"
        age_val =  prob_df[prob_df['Customer ID']==cust_id]['Age'].values[0]
        tenure_mn = f"{prob_df[prob_df['Customer ID']==cust_id]['Tenure in Months'].values[0]} months"
        download = prob_df[prob_df['Customer ID']==cust_id]['Avg Monthly GB Download'].values[0]
        pr_city = prob_df[prob_df['Customer ID']==cust_id]['City'].values[0]
        pr_contract = prob_df[prob_df['Customer ID']==cust_id]['Contract'].values[0]
        pr_revenue = prob_df[prob_df['Customer ID']==cust_id]['Total Revenue'].values[0]
        st.markdown(f"""<div style='background-color: #c1d6c1; padding: 25px; border-radius: 5px'>
                    <h2 class={'change-prob' if float(prob_val) > 90.0 else 'change'} style='text-align: center; font=size: 36px'>{'Churn Percentage'} : {prob_val}%</h2>
                    <div style='margin-top: 25px'></div>
                    """ , unsafe_allow_html=True)
        st.markdown('\n\n')
        st.markdown(f"""<div style='background-color: #c1d6c1; padding: 25px; border-radius: 5px'>
                    <p style='font-size: 20px; text-align: left'><b>Age</b>: {age_val} | <b>City</b>: {pr_city} | <b>Tenure</b>: {tenure_mn}</p>
                    <p style='font-size: 20px; text-align: left'><b>Avg. Download(M/GB)</b>: {download}</p>
                    <p class= {'pr-contract' if pr_contract == 'Month-to-Month' else 'contract'} style='font-size: 20px; text-align: left'><b>Contract</b>: {pr_contract}</p>
                    <p style='font-size: 20px; text-align: left'><b>Total Revenue</b>: {pr_revenue}</p>
                    """ , unsafe_allow_html=True)


    with pres_col1:

        if prob_df[prob_df['Customer ID'] == cust_id]['Probs'].values[0] > 65 and not \
        prob_df[prob_df['Customer ID'] == cust_id]['Churn Reason'].isna().any():
            pres_reason = prob_df[prob_df['Customer ID'] == cust_id]['Churn Reason'].values[0]
            st.markdown(f"""<div style='background-color: #c1d6c1; padding: 80px; border-radius: 5px'>
                                        <h2 style='text-align: left; font-size: 18px'>Churned| {pres_reason}</h2>
                                        <div style='margin-top: 20px'></div>
                                        """, unsafe_allow_html=True)
        elif prob_df[prob_df['Customer ID'] == cust_id]['Probs'].values[0] > 65 and \
                prob_df[prob_df['Customer ID'] == cust_id]['Churn Reason'].isna().any():
            pres_reason1 = np.random.choice(prob_df['Churn Reason'].value_counts().index)
            st.markdown(f"""<div style='background-color: #c1d6c1; padding: 80px; border-radius: 5px'>
                                                <h2 style='text-align: left; font-size: 18px'> Churned | {pres_reason1}</h2>
                                                <div style='margin-top: 20px'></div>
                                                """, unsafe_allow_html=True)
        else:
            if prob_df[prob_df['Customer ID'] == cust_id]['Probs'].values[0] < 50 and \
                    prob_df[prob_df['Customer ID'] == cust_id]['Probs'].values[0] >= 30:
                st.markdown(f"""<div style='background-color: #c1d6c1; padding: 80px; border-radius: 5px'>
                                        <h2 style='text-align: left; font-size: 18px'>{'Please Check Customer Status|Bio and take appropriate action Asap'}</h2>
                                        <div style='margin-top: 20px'></div>
                                        """, unsafe_allow_html=True)
            elif prob_df[prob_df['Customer ID'] == cust_id]['Probs'].values[0] < 30 and \
                    prob_df[prob_df['Customer ID'] == cust_id]['Probs'].values[0] >= 20:
                if pr_contract == 'Month-to-Month':
                    st.markdown(f"""<div style='background-color: #c1d6c1; padding: 80px; border-radius: 5px'>
                                                    <h2 style='text-align: left; font-size: 18px'>{'Keep Monitoring and Convert Contract to Yearly'}</h2>
                                                    <div style='margin-top: 25px'></div>
                                                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style='background-color: #c1d6c1; padding: 80px; border-radius: 5px'>
                                                                    <h2 style='text-align: left; font-size: 18px'>{'Keep Monitoring Customer Status'}</h2>
                                                                    <div style='margin-top: 25px'></div>
                                                                    """, unsafe_allow_html=True)

            else:
                if pr_contract == 'Month-to-Month':
                    st.markdown(f"""<div style='background-color: #c1d6c1; padding: 80px; border-radius: 5px'>
                                                                <h2 style='text-align: left; font-size: 18px'>{'Solid Customer but convert contract to Yearly'}</h2>
                                                                <div style='margin-top: 20px'></div>
                                                                """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""<div style='background-color: #c1d6c1; padding: 80px; border-radius: 5px'>
                                                                                <h2 style='text-align: left; font-size: 18px'>{'Solid Customer'}</h2>
                                                                                <div style='margin-top: 20px'></div>


                                                                        """, unsafe_allow_html=True)

    st.markdown("\n\n\n\n")

    if pr_contract == 'Month-to-Month' and float(prob_val) < 75.0:
        st.write('Promote these Package to the USER ASAP')
        tdata = [['Yearly Bundle Package', '50 GB/month|speed 5Mbps', 100, 100, 600, '-', 'Free', 'Unlimited'],
                ['Yearly Roaming Package', '60 GB/month|speed 5Mbps', 100, 100, 700, 'National|International', 'Free',
                 'Unlimited'],
                ['5G Max Yearly Package', '70 GB/month|speed 5Mbps', 100, 100, 800, 'N/A', 'Free', 'Unlimited']]
        st.write(pd.DataFrame(tdata,
                              columns = ['Type','Internet','Phone Call','SMS', 'Price', 'Roaming','StreamingTV','Telco Super WIFI'], index=range(1, len(tdata)+1)))
        st.write('<span style="font-size: smaller;">*Phone call and SMS are free for the first 100; thereafter, normal charges apply.</span>', unsafe_allow_html=True)
        st.write(
            '<span style="font-size: smaller;">**Internet speed will drop to 1Mbps after Usage limit is over</span>',
            unsafe_allow_html=True)
