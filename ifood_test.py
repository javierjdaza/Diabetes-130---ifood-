import pandas as pd
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy
import warnings
from PIL import Image


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import streamlit as st

warnings.filterwarnings('ignore')

# ============
# FUCTIONS
# ============
st.set_page_config(page_title='Diabetes 130 US hospitals for years 1999–2008',layout='wide',page_icon=":brain:",initial_sidebar_state="collapsed")

# @st.cache(show_spinner=False)
def load_dataframe()->pd.DataFrame:

    
    df = pd.read_csv('./diabetic_data.csv')#, nrows=100)
    return df
    
def text_center(texto:str):

    text_html = f'<div style="text-align: center; font-weight : bold;margin-bottom: 0px"> {texto} </div>'
    st.markdown(text_html, unsafe_allow_html=True)

def text_center_light(texto:str):

    text_html = f'<div style="text-align: center;margin-bottom: 0px"> {texto} </div>'
    st.markdown(text_html, unsafe_allow_html=True)

text_html = '<div style="text-align: center; font-weight : bold;margin-bottom: 0px"> Authentication </div>'
st.sidebar.markdown(text_html, unsafe_allow_html=True)
st.sidebar.write(' ')

bootstrap = st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">', unsafe_allow_html=True)
icons = st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css">', unsafe_allow_html=True)
st.title('🩺 Diabetes 130 US hospitals for years 1999–2008 ')
st.write('---')
col1,col2,col3,col4 = st.columns(4)
with col1:
    st.markdown('Assessment: <b>Data Scientist - iFood Colombia</b>', unsafe_allow_html=True)
with col2:
    st.markdown('Author: <b>Javier Javier Daza Olivella</b>', unsafe_allow_html=True)
with col3:
    st.markdown('Email: <b>javierjdaza@gmail.com</b>', unsafe_allow_html=True)
with col4:
    st.markdown('GitHub: <b>https://github.com/javierjdaza</b>', unsafe_allow_html=True)

st.write('---')

hide_streamlit_style = """<style>footer{visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Start the magic
with st.spinner('Loading Diabetes 130 US hospitals Data Frame...'):
    df = load_dataframe()

st.markdown('<h3>Importing Libraries: <b>Data Processing & Data Visualization</b></h3>', unsafe_allow_html=True)
st.code("""
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import scipy
        import warnings
        warnings.filterwarnings('ignore')
        """)
st.write('---')
#------------------------------------------------

st.markdown('<h3>Importing Libraries: <b>Model Algorithms</b></h3>', unsafe_allow_html=True)

st.code("""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn import metrics
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import roc_curve, auc
        from sklearn.metrics import classification_report

        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        """)
st.write('---')

#------------------------------------------------
st.markdown('<h3>Importing Libraries: <b>Model Algorithms</b></h3>', unsafe_allow_html=True)

st.code("""
        df = pd.read_csv('./diabetic_data.csv')
        """)
st.dataframe(df) # see *

#------------------------------------------------
st.write('---')
st.markdown('<h3>Drop Duplicates based on: <b>patient_nbr</b></h3>', unsafe_allow_html=True)
st.code("""
        df.dropna(inplace = True)
        len_after_rm_duplicates = len(df)
        df.drop_duplicates(['patient_nbr'], keep = 'first', inplace = True)
        print(f'Duplicates: {len_after_rm_duplicates - len(df)}')
        """)

df.dropna(inplace = True)
len_after_rm_duplicates = len(df)
df.drop_duplicates(['patient_nbr'], keep = 'first', inplace = True)
st.write(f'Duplicates: {len_after_rm_duplicates - len(df)}')

#------------------------------------------------
st.write('---')
st.markdown('<h3>Converting age feature: <b>to numeric</b></h3>', unsafe_allow_html=True)
st.code("""
        # converting age feature to numeric
        def replace_age(row):
            l_value = int(row.replace('[','').replace(')','').split('-')[0])
            h_value = int(row.replace('[','').replace(')','').split('-')[1])
            return (l_value + h_value )//2
            
        df['age'] = df['age'].apply(lambda x : replace_age(x))

        """)
# converting age feature to numeric
def replace_age(row):
    l_value = int(str(row).replace('[','').replace(')','').split('-')[0])
    h_value = int(str(row).replace('[','').replace(')','').split('-')[1])
    return (l_value + h_value )//2
    
    
df['age'] = df['age'].apply(lambda x : replace_age(x))

st.write('---')
#------------------------------------------------

st.markdown('<h3>How Many null values are?</h3>', unsafe_allow_html=True)
st.code("""
        df_replace_question = df.replace('?',np.nan)
        null_values = df_replace_question.isnull().sum().to_frame().reset_index()
        null_values.columns = ['column_name','null_values']
        null_values['total_rows'] = int(len(df_replace_question))
        null_values['p_null_values'] = round((null_values['null_values']/null_values['total_rows'])*100,2)
        null_values.sort_values(by = ['p_null_values'], inplace = True, ascending = False)
        """)
df_replace_question = df.replace('?',np.nan)
null_values = df_replace_question.isnull().sum().to_frame().reset_index()
null_values.columns = ['column_name','null_values']
null_values['total_rows'] = int(len(df_replace_question))
null_values['p_null_values'] = round((null_values['null_values']/null_values['total_rows'])*100,2)
null_values.sort_values(by = ['p_null_values'], inplace = True, ascending = False)
st.dataframe(null_values)

st.write('---')
#------------------------------------------------

st.markdown("""
* <b>'weight'</b> feature has lots of missing values (96%) so -> <b>GOOD BYE</b><br/>
* For all 'Diagnosis' features we can use most common value to fill missing value
* For 'medical_specialty' i decide to fill nan with 'no information'
""", unsafe_allow_html=True)

st.write('---')
#------------------------------------------------

st.markdown("<h3><b>Sayonara</b> 'weight' and 'Unknown/Invalid gender'</h3>", unsafe_allow_html=True)
st.code("""
        df = df[~(df.gender == 'Unknown/Invalid')]
        del df['weight']
        """)
df = df[~(df.gender == 'Unknown/Invalid')]
del df['weight']
st.write('---')
#------------------------------------------------
st.markdown('<h2 style = "color: #A9333A;"><b>Dealing with Null values</b></h2>', unsafe_allow_html=True)
st.write(' ')
st.write(' ')

st.markdown("<h3>Get the <b>most common</b> value for <b>'Diagnosis'</b></h3>", unsafe_allow_html=True)

st.code("""
        diag_1 = df['diag_1'].mode()[0]
        diag_2 = df['diag_2'].mode()[0]
        diag_3 = df['diag_3'].mode()[0]
        df['diag_1'] = df['diag_1'].apply(lambda x : str(x).replace('?',diag_1))
        df['diag_2'] = df['diag_1'].apply(lambda x : str(x).replace('?',diag_2))
        df['diag_3'] = df['diag_3'].apply(lambda x : str(x).replace('?',diag_3))
        """)
diag_1 = df['diag_1'].mode()[0]
diag_2 = df['diag_2'].mode()[0]
diag_3 = df['diag_3'].mode()[0]
df['diag_1'] = df['diag_1'].apply(lambda x : str(x).replace('?',diag_1))
df['diag_2'] = df['diag_1'].apply(lambda x : str(x).replace('?',diag_2))
df['diag_3'] = df['diag_3'].apply(lambda x : str(x).replace('?',diag_3))

st.write('---')
#------------------------------------------------

st.markdown("<h3>'medical_specialty' null values to <b>'no information'</b></h3>", unsafe_allow_html=True)

st.code("""
        df['medical_specialty'].fillna('no information', inplace = True)
        """)
df['medical_specialty'].fillna('no information', inplace = True)

st.write('---')
#------------------------------------------------

st.markdown("<h3>Get rid <b>'Race'</b> null values</h3>", unsafe_allow_html=True)

st.code("""
        df.dropna(subset = ['race'], inplace = True)
        """)
df.dropna(subset = ['race'], inplace = True)

st.write('---')
#------------------------------------------------

st.markdown("<h3>Get rid <b>unnecessary features</b> ‘encounter_id’, ‘patient_nbr’, 'payer_code'</h3>", unsafe_allow_html=True)

st.code("""
        features_drop_list = ['encounter_id', 'patient_nbr','payer_code', 'medical_specialty', 'repaglinide', 'nateglinide', 'chlorpropamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone','metformin-pioglitazone', 'acetohexamide', 'tolbutamide']
        df.drop(features_drop_list, axis=1,inplace=True)
        """)
features_drop_list = ['encounter_id', 'patient_nbr','payer_code', 'medical_specialty', 'repaglinide', 'nateglinide', 'chlorpropamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone','metformin-pioglitazone', 'acetohexamide', 'tolbutamide']
df.drop(features_drop_list, axis=1,inplace=True)


st.write('---')
#------------------------------------------------

st.markdown("<h3>'Gender' ,'diabetesMed' string to <b>dummy</b></h3>", unsafe_allow_html=True)

st.code("""
        df['gender'] = df['gender'].replace('Male', 1)
        df['gender'] = df['gender'].replace('Female', 0)
        df['diabetesMed']=df['diabetesMed'].replace('Yes', 1)
        df['diabetesMed']=df['diabetesMed'].replace('No', 0)
        """)
df['gender'] = df['gender'].replace('Male', 1)
df['gender'] = df['gender'].replace('Female', 0)
df['diabetesMed']=df['diabetesMed'].replace('Yes', 1)
df['diabetesMed']=df['diabetesMed'].replace('No', 0)

st.write('---')
#------------------------------------------------

st.markdown("<h3>Transformation readmitted to: <b>predict variable</b></h3>", unsafe_allow_html=True)

st.code("""
        df['readmitted'] = df['readmitted'].apply(lambda x : 0 if (x == '>30' or x == 'NO') else 1)
        """)
df['readmitted'] = df['readmitted'].apply(lambda x : 0 if (x == '>30' or x == 'NO') else 1)

st.write('---')
#------------------------------------------------

st.markdown("<h3>Transformation categorical variables: <b>('up' , 'down', 'steady' etc) to numerical</b></h3>", unsafe_allow_html=True)

st.code("""
        def up_down_transf(row):
    
            if row == 'Up':
                return 3
            elif row == 'Down':
                return 1
            elif row == 'Steady':
                return 2
            else:
                return -0

        def max_glu_trasnf(row):
            if row == '>200':
                return 2
            elif row == '>300':
                return 3
            elif row == 'Norm':
                return 1
            else:
                return 0


        def a1cresult_trasnf(row):
            
            if row == '>7':
                return 2
            elif row == '>8':
                return 3
            elif row == 'Norm':
                return 1
            else:
                return 0
            
            
        for col in ["metformin",  "glimepiride",  "glipizide", "glyburide", "pioglitazone", "rosiglitazone","insulin"]:
            df[col] = df[col].apply(lambda x : up_down_transf(x))


        df['change'] = df['change'].apply(lambda x : 1 if x == 'Ch'else 0)


        df['diabetesMed'] = df['diabetesMed'].apply(lambda x : 0 if x == 'No'else 1)


        df['max_glu_serum'] = df['max_glu_serum'].apply(lambda x : max_glu_trasnf(x))

        df['A1Cresult'] = df['A1Cresult'].apply(lambda x : a1cresult_trasnf(x))
        """)
def up_down_transf(row):
    
    if row == 'Up':
        return 3
    elif row == 'Down':
        return 1
    elif row == 'Steady':
        return 2
    else:
        return -0

def max_glu_trasnf(row):
    if row == '>200':
        return 2
    elif row == '>300':
        return 3
    elif row == 'Norm':
        return 1
    else:
        return 0


def a1cresult_trasnf(row):
    
    if row == '>7':
        return 2
    elif row == '>8':
        return 3
    elif row == 'Norm':
        return 1
    else:
        return 0
    
    
for col in ["metformin",  "glimepiride",  "glipizide", "glyburide", "pioglitazone", "rosiglitazone","insulin"]:
    df[col] = df[col].apply(lambda x : up_down_transf(x))


df['change'] = df['change'].apply(lambda x : 1 if x == 'Ch'else 0)
df['diabetesMed'] = df['diabetesMed'].apply(lambda x : 0 if x == 'No'else 1)
df['max_glu_serum'] = df['max_glu_serum'].apply(lambda x : max_glu_trasnf(x))
df['A1Cresult'] = df['A1Cresult'].apply(lambda x : a1cresult_trasnf(x))
st.write('---')
#------------------------------------------------

st.write('---')
#------------------------------------------------

st.code("""
        plt.figure()
        sns.set_theme(style="whitegrid")
        ax = sns.countplot(x = 'readmitted', data = df, hue = 'readmitted')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels = ['Not Admitted', 'Admitted'])
        plt.figure()
        """)
        
plt1,plt2 = st.columns(2)
with plt1:
    fig = plt.figure(figsize = (10, 5))
    sns.set_theme(style="whitegrid")
    ax = sns.countplot(x = 'readmitted', data = df, hue = 'readmitted')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels = ['Not Admitted', 'Admitted'])
    st.pyplot(fig)
with plt2:
    st.dataframe(df['readmitted'].value_counts(dropna=False).to_frame())
#------------------------------------------------
st.write('---')
#------------------------------------------------
st.markdown('<h2 style = "color: #A9333A;"><b>Research</b></h2>', unsafe_allow_html=True)
st.write(' ')
st.write('''
They are coded in ICD-9 code (https://en.wikipedia.org/wiki/List_of_ICD-9_codes), resulting in hundreds of distinct categories. One way to simplify this, is by grouping every ICD-9 code value into one of 18 generic health problems, depending on the first 3 digits as following:
List of ICD-9 codes 001–139: infectious and parasitic diseases\n
List of ICD-9 codes 140–239: neoplasms\n
List of ICD-9 codes 240–279: endocrine, nutritional and metabolic diseases, and immunity disorders\n
List of ICD-9 codes 280–289: diseases of the blood and blood-forming organs\n
List of ICD-9 codes 290–319: mental disorders\n
List of ICD-9 codes 320–389: diseases of the nervous system and sense organs\n
List of ICD-9 codes 390–459: diseases of the circulatory system\n
List of ICD-9 codes 460–519: diseases of the respiratory system\n
List of ICD-9 codes 520–579: diseases of the digestive system\n
List of ICD-9 codes 580–629: diseases of the genitourinary system\n
List of ICD-9 codes 630–679: complications of pregnancy, childbirth, and the puerperium\n
List of ICD-9 codes 680–709: diseases of the skin and subcutaneous tissue\n
List of ICD-9 codes 710–739: diseases of the musculoskeletal system and connective tissue\n
List of ICD-9 codes 740–759: congenital anomalies\n
List of ICD-9 codes 760–779: certain conditions originating in the perinatal period\n
List of ICD-9 codes 780–799: symptoms, signs, and ill-defined conditions\n
List of ICD-9 codes 800–999: injury and poisoning\n
List of ICD-9 codes E and V codes: external causes of injury and supplemental classification\n
''')


st.write('---')
#------------------------------------------------

st.markdown("<h3>Transformation: <b>'Diagnosis' </b></h3>", unsafe_allow_html=True)

st.code("""
        
        df.loc[df['diag_1'].str.contains('V',na=False,case=False), 'diag_1'] = 0
        df.loc[df['diag_1'].str.contains('E',na=False,case=False), 'diag_1'] = 0
        df.loc[df['diag_2'].str.contains('V',na=False,case=False), 'diag_2'] = 0
        df.loc[df['diag_2'].str.contains('E',na=False,case=False), 'diag_2'] = 0
        df.loc[df['diag_3'].str.contains('V',na=False,case=False), 'diag_3'] = 0
        df.loc[df['diag_3'].str.contains('E',na=False,case=False), 'diag_3'] = 0

        df['diag_1'] = df['diag_1'].replace('?',-1)
        df['diag_2'] = df['diag_2'].replace('?',-1)
        df['diag_3'] = df['diag_3'].replace('?',-1)
        df['diag_1'] = df['diag_1'].astype(float)
        df['diag_2'] = df['diag_2'].astype(float)
        df['diag_3'] = df['diag_3'].astype(float)

        
        df['diag_1'].loc[(df['diag_1']>=1) & (df['diag_1']< 140)] = 1
        df['diag_1'].loc[(df['diag_1']>=140) & (df['diag_1']< 240)] = 2
        df['diag_1'].loc[(df['diag_1']>=240) & (df['diag_1']< 280)] = 3
        df['diag_1'].loc[(df['diag_1']>=280) & (df['diag_1']< 290)] = 4
        df['diag_1'].loc[(df['diag_1']>=290) & (df['diag_1']< 320)] = 5
        df['diag_1'].loc[(df['diag_1']>=320) & (df['diag_1']< 390)] = 6
        df['diag_1'].loc[(df['diag_1']>=390) & (df['diag_1']< 460)] = 7
        df['diag_1'].loc[(df['diag_1']>=460) & (df['diag_1']< 520)] = 8
        df['diag_1'].loc[(df['diag_1']>=520) & (df['diag_1']< 580)] = 9
        df['diag_1'].loc[(df['diag_1']>=580) & (df['diag_1']< 630)] = 10
        df['diag_1'].loc[(df['diag_1']>=630) & (df['diag_1']< 680)] = 11
        df['diag_1'].loc[(df['diag_1']>=680) & (df['diag_1']< 710)] = 12
        df['diag_1'].loc[(df['diag_1']>=710) & (df['diag_1']< 740)] = 13
        df['diag_1'].loc[(df['diag_1']>=740) & (df['diag_1']< 760)] = 14
        df['diag_1'].loc[(df['diag_1']>=760) & (df['diag_1']< 780)] = 15
        df['diag_1'].loc[(df['diag_1']>=780) & (df['diag_1']< 800)] = 16
        df['diag_1'].loc[(df['diag_1']>=800) & (df['diag_1']< 1000)] = 17
        df['diag_1'].loc[(df['diag_1']==-1)] = 0

        df['diag_2'].loc[(df['diag_2']>=1) & (df['diag_2']< 140)] = 1
        df['diag_2'].loc[(df['diag_2']>=140) & (df['diag_2']< 240)] = 2
        df['diag_2'].loc[(df['diag_2']>=240) & (df['diag_2']< 280)] = 3
        df['diag_2'].loc[(df['diag_2']>=280) & (df['diag_2']< 290)] = 4
        df['diag_2'].loc[(df['diag_2']>=290) & (df['diag_2']< 320)] = 5
        df['diag_2'].loc[(df['diag_2']>=320) & (df['diag_2']< 390)] = 6
        df['diag_2'].loc[(df['diag_2']>=390) & (df['diag_2']< 460)] = 7
        df['diag_2'].loc[(df['diag_2']>=460) & (df['diag_2']< 520)] = 8
        df['diag_2'].loc[(df['diag_2']>=520) & (df['diag_2']< 580)] = 9
        df['diag_2'].loc[(df['diag_2']>=580) & (df['diag_2']< 630)] = 10
        df['diag_2'].loc[(df['diag_2']>=630) & (df['diag_2']< 680)] = 11
        df['diag_2'].loc[(df['diag_2']>=680) & (df['diag_2']< 710)] = 12
        df['diag_2'].loc[(df['diag_2']>=710) & (df['diag_2']< 740)] = 13
        df['diag_2'].loc[(df['diag_2']>=740) & (df['diag_2']< 760)] = 14
        df['diag_2'].loc[(df['diag_2']>=760) & (df['diag_2']< 780)] = 15
        df['diag_2'].loc[(df['diag_2']>=780) & (df['diag_2']< 800)] = 16
        df['diag_2'].loc[(df['diag_2']>=800) & (df['diag_2']< 1000)] = 17
        df['diag_2'].loc[(df['diag_2']==-1)] = 0

        df['diag_3'].loc[(df['diag_3']>=1) & (df['diag_3']< 140)] = 1
        df['diag_3'].loc[(df['diag_3']>=140) & (df['diag_3']< 240)] = 2
        df['diag_3'].loc[(df['diag_3']>=240) & (df['diag_3']< 280)] = 3
        df['diag_3'].loc[(df['diag_3']>=280) & (df['diag_3']< 290)] = 4
        df['diag_3'].loc[(df['diag_3']>=290) & (df['diag_3']< 320)] = 5
        df['diag_3'].loc[(df['diag_3']>=320) & (df['diag_3']< 390)] = 6
        df['diag_3'].loc[(df['diag_3']>=390) & (df['diag_3']< 460)] = 7
        df['diag_3'].loc[(df['diag_3']>=460) & (df['diag_3']< 520)] = 8
        df['diag_3'].loc[(df['diag_3']>=520) & (df['diag_3']< 580)] = 9
        df['diag_3'].loc[(df['diag_3']>=580) & (df['diag_3']< 630)] = 10
        df['diag_3'].loc[(df['diag_3']>=630) & (df['diag_3']< 680)] = 11
        df['diag_3'].loc[(df['diag_3']>=680) & (df['diag_3']< 710)] = 12
        df['diag_3'].loc[(df['diag_3']>=710) & (df['diag_3']< 740)] = 13
        df['diag_3'].loc[(df['diag_3']>=740) & (df['diag_3']< 760)] = 14
        df['diag_3'].loc[(df['diag_3']>=760) & (df['diag_3']< 780)] = 15
        df['diag_3'].loc[(df['diag_3']>=780) & (df['diag_3']< 800)] = 16
        df['diag_3'].loc[(df['diag_3']>=800) & (df['diag_3']< 1000)] = 17
        df['diag_3'].loc[(df['diag_3']==-1)] = 0
        """)

st.write('---')
#start by setting all values containing E or V into 0 (as one category)
df.loc[df['diag_1'].str.contains('V',na=False,case=False), 'diag_1'] = 0
df.loc[df['diag_1'].str.contains('E',na=False,case=False), 'diag_1'] = 0
df.loc[df['diag_2'].str.contains('V',na=False,case=False), 'diag_2'] = 0
df.loc[df['diag_2'].str.contains('E',na=False,case=False), 'diag_2'] = 0
df.loc[df['diag_3'].str.contains('V',na=False,case=False), 'diag_3'] = 0
df.loc[df['diag_3'].str.contains('E',na=False,case=False), 'diag_3'] = 0

#setting all missing values into -1
df['diag_1'] = df['diag_1'].replace('?',-1)
df['diag_2'] = df['diag_2'].replace('?',-1)
df['diag_3'] = df['diag_3'].replace('?',-1)
df['diag_1'] = df['diag_1'].astype(float)
df['diag_2'] = df['diag_2'].astype(float)
df['diag_3'] = df['diag_3'].astype(float)

# Now we will reduce the number of categories in diag features according to ICD-9 code
#(Missing values will be grouped as E & V values)
df['diag_1'].loc[(df['diag_1']>=1) & (df['diag_1']< 140)] = 1
df['diag_1'].loc[(df['diag_1']>=140) & (df['diag_1']< 240)] = 2
df['diag_1'].loc[(df['diag_1']>=240) & (df['diag_1']< 280)] = 3
df['diag_1'].loc[(df['diag_1']>=280) & (df['diag_1']< 290)] = 4
df['diag_1'].loc[(df['diag_1']>=290) & (df['diag_1']< 320)] = 5
df['diag_1'].loc[(df['diag_1']>=320) & (df['diag_1']< 390)] = 6
df['diag_1'].loc[(df['diag_1']>=390) & (df['diag_1']< 460)] = 7
df['diag_1'].loc[(df['diag_1']>=460) & (df['diag_1']< 520)] = 8
df['diag_1'].loc[(df['diag_1']>=520) & (df['diag_1']< 580)] = 9
df['diag_1'].loc[(df['diag_1']>=580) & (df['diag_1']< 630)] = 10
df['diag_1'].loc[(df['diag_1']>=630) & (df['diag_1']< 680)] = 11
df['diag_1'].loc[(df['diag_1']>=680) & (df['diag_1']< 710)] = 12
df['diag_1'].loc[(df['diag_1']>=710) & (df['diag_1']< 740)] = 13
df['diag_1'].loc[(df['diag_1']>=740) & (df['diag_1']< 760)] = 14
df['diag_1'].loc[(df['diag_1']>=760) & (df['diag_1']< 780)] = 15
df['diag_1'].loc[(df['diag_1']>=780) & (df['diag_1']< 800)] = 16
df['diag_1'].loc[(df['diag_1']>=800) & (df['diag_1']< 1000)] = 17
df['diag_1'].loc[(df['diag_1']==-1)] = 0

df['diag_2'].loc[(df['diag_2']>=1) & (df['diag_2']< 140)] = 1
df['diag_2'].loc[(df['diag_2']>=140) & (df['diag_2']< 240)] = 2
df['diag_2'].loc[(df['diag_2']>=240) & (df['diag_2']< 280)] = 3
df['diag_2'].loc[(df['diag_2']>=280) & (df['diag_2']< 290)] = 4
df['diag_2'].loc[(df['diag_2']>=290) & (df['diag_2']< 320)] = 5
df['diag_2'].loc[(df['diag_2']>=320) & (df['diag_2']< 390)] = 6
df['diag_2'].loc[(df['diag_2']>=390) & (df['diag_2']< 460)] = 7
df['diag_2'].loc[(df['diag_2']>=460) & (df['diag_2']< 520)] = 8
df['diag_2'].loc[(df['diag_2']>=520) & (df['diag_2']< 580)] = 9
df['diag_2'].loc[(df['diag_2']>=580) & (df['diag_2']< 630)] = 10
df['diag_2'].loc[(df['diag_2']>=630) & (df['diag_2']< 680)] = 11
df['diag_2'].loc[(df['diag_2']>=680) & (df['diag_2']< 710)] = 12
df['diag_2'].loc[(df['diag_2']>=710) & (df['diag_2']< 740)] = 13
df['diag_2'].loc[(df['diag_2']>=740) & (df['diag_2']< 760)] = 14
df['diag_2'].loc[(df['diag_2']>=760) & (df['diag_2']< 780)] = 15
df['diag_2'].loc[(df['diag_2']>=780) & (df['diag_2']< 800)] = 16
df['diag_2'].loc[(df['diag_2']>=800) & (df['diag_2']< 1000)] = 17
df['diag_2'].loc[(df['diag_2']==-1)] = 0

df['diag_3'].loc[(df['diag_3']>=1) & (df['diag_3']< 140)] = 1
df['diag_3'].loc[(df['diag_3']>=140) & (df['diag_3']< 240)] = 2
df['diag_3'].loc[(df['diag_3']>=240) & (df['diag_3']< 280)] = 3
df['diag_3'].loc[(df['diag_3']>=280) & (df['diag_3']< 290)] = 4
df['diag_3'].loc[(df['diag_3']>=290) & (df['diag_3']< 320)] = 5
df['diag_3'].loc[(df['diag_3']>=320) & (df['diag_3']< 390)] = 6
df['diag_3'].loc[(df['diag_3']>=390) & (df['diag_3']< 460)] = 7
df['diag_3'].loc[(df['diag_3']>=460) & (df['diag_3']< 520)] = 8
df['diag_3'].loc[(df['diag_3']>=520) & (df['diag_3']< 580)] = 9
df['diag_3'].loc[(df['diag_3']>=580) & (df['diag_3']< 630)] = 10
df['diag_3'].loc[(df['diag_3']>=630) & (df['diag_3']< 680)] = 11
df['diag_3'].loc[(df['diag_3']>=680) & (df['diag_3']< 710)] = 12
df['diag_3'].loc[(df['diag_3']>=710) & (df['diag_3']< 740)] = 13
df['diag_3'].loc[(df['diag_3']>=740) & (df['diag_3']< 760)] = 14
df['diag_3'].loc[(df['diag_3']>=760) & (df['diag_3']< 780)] = 15
df['diag_3'].loc[(df['diag_3']>=780) & (df['diag_3']< 800)] = 16
df['diag_3'].loc[(df['diag_3']>=800) & (df['diag_3']< 1000)] = 17
df['diag_3'].loc[(df['diag_3']==-1)] = 0
#------------------------------------------------

st.write('---')
#------------------------------------------------

st.markdown("<h3><b>Chi-Squared Test</b></h3>", unsafe_allow_html=True)

st.code("""
        categorical_features = ['gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id','diag_1', 'diag_2', 'diag_3', 'readmitted']
        rejected_features = []
        for col in categorical_features : 
            df_crosstab = pd.crosstab(df['readmitted'],  df[col], margins = False) 

            stat, p, dof, expected = scipy.stats.chi2_contingency(df_crosstab)
            if p < 0.4 :
                print(p, col, 'is significant')
            else:
                print(p, col, 'is not significant')
                rejected_features.append(col)
        """)
categorical_features = ['gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id','diag_1', 'diag_2', 'diag_3', 'readmitted']
rejected_features = []
for col in categorical_features : 
    df_crosstab = pd.crosstab(df['readmitted'],  df[col], margins = False) 

    stat, p, dof, expected = scipy.stats.chi2_contingency(df_crosstab)
    if p < 0.4 :
        print(p, col, 'is significant')
    else:
        print(p, col, 'is not significant')
        rejected_features.append(col)
st.caption('''
    >>
    0.5346920045915386 gender is not significant
    0.01750064068640562 admission_type_id is significant
    1.3232043949603843e-295 discharge_disposition_id is significant
    0.36205128123276453 admission_source_id is significant
    3.131780022573835e-22 diag_1 is significant
    3.131780022573835e-22 diag_2 is significant
    3.407310052111208e-10 diag_3 is significant
    0.0 readmitted is significant
                ''')
st.write('---')
#------------------------------------------------

st.markdown("<h3><b>Spearman Correlation Coefficient</b></h3>", unsafe_allow_html=True)

st.code("""
        numeric_features = ['age','time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'A1Cresult','max_glu_serum','metformin','glimepiride','glipizide','glyburide','pioglitazone','rosiglitazone','insulin', 'change', 'diabetesMed']

        for col in numeric_features :
            rho , pval = scipy.stats.spearmanr(df['readmitted'], df[col])
            if pval < 0.4 : 
                print(col, 'is significant')
            else : 
                print(col, 'is not significant')
                rejected_features.append(col)
        """)
numeric_features = ['age','time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'A1Cresult','max_glu_serum','metformin','glimepiride','glipizide','glyburide','pioglitazone','rosiglitazone','insulin', 'change', 'diabetesMed']

for col in numeric_features :
    rho , pval = scipy.stats.spearmanr(df['readmitted'], df[col])
    if pval < 0.4 : 
        print(col, 'is significant')
    else : 
        print(col, 'is not significant')
        rejected_features.append(col)
st.caption('''
    >>
    age is significant
    time_in_hospital is significant
    num_lab_procedures is significant
    num_procedures is not significant
    num_medications is significant
    number_outpatient is significant
    number_emergency is significant
    number_inpatient is significant
    number_diagnoses is significant
    A1Cresult is significant
    max_glu_serum is significant
    metformin is significant
    glimepiride is not significant
    glipizide is significant
    glyburide is significant
    pioglitazone is not significant
    rosiglitazone is not significant
    insulin is significant
    change is significant
    diabetesMed is not significant
                ''')
st.write('---')
#------------------------------------------------

st.markdown('<h2 style = "color: #A9333A;"><b>EDA according to features</b></h2>', unsafe_allow_html=True)
st.write(' ')

st.markdown("<h3><b>discharge_disposition_id</b></h3>", unsafe_allow_html=True)

st.code("""
    fig = plt.figure(figsize = (10, 5))
    sns.countplot(x  = 'discharge_disposition_id', data = df, hue = 'readmitted')
    """)
p1,p2 = st.columns(2)

with p1:
    fig = plt.figure(figsize = (10, 5))
    sns.countplot(x  = 'discharge_disposition_id', data = df, hue = 'readmitted')
    st.pyplot(fig)
    st.caption('Clearly if discharge disposition id is 1,2,6 and 22 the patient will readmit.')


st.write('---')
#------------------------------------------------
st.markdown("<h3><b>number_inpatient</b></h3>", unsafe_allow_html=True)

st.code("""
    fig = plt.figure(figsize = (10, 5)) 
    sns.countplot(x  = 'number_inpatient', data = df, hue = 'readmitted')
    """)
p1,p2 = st.columns(2)

with p1:
    fig = plt.figure(figsize = (10, 5)) 
    sns.countplot(x  = 'number_inpatient', data = df, hue = 'readmitted')
    st.pyplot(fig)
    st.caption('If patient has not admitted in the past or admitted very few times the chance that he will readmit is very low')

st.write('---')
#------------------------------------------------
st.markdown("<h3><b>admission_source_id</b></h3>", unsafe_allow_html=True)

st.code("""
    fig = plt.figure(figsize = (10, 5)) 
    sns.countplot(x = 'admission_source_id', hue = 'readmitted', data = df)
    """)
p1,p2 = st.columns(2)

with p1:
    fig = plt.figure(figsize = (10, 5)) 
    sns.countplot(x = 'admission_source_id', hue = 'readmitted', data = df)
    st.pyplot(fig)

st.write('---')
#------------------------------------------------
st.markdown("<h3><b>Age</b></h3>", unsafe_allow_html=True)

st.code("""
    fig = plt.figure(figsize = (10, 5)) 
    sns.countplot(x = 'age', hue = 'readmitted', data = df)
    """)
p1,p2 = st.columns(2)

with p1:
    fig = plt.figure(figsize = (10, 5)) 
    sns.countplot(x = 'age', hue = 'readmitted', data = df)
    st.pyplot(fig)

st.write('---')

#------------------------------------------------
st.markdown("<h3><b>Race</b></h3>", unsafe_allow_html=True)

st.code("""
    fig = plt.figure(figsize = (10, 5)) 
    ax = fig.add_subplot(111)

    sns.countplot(df[df.readmitted == 1].race.values)
    ax.set_title('Readmitted Patient')

    fig = plt.figure(figsize = (10, 5)) 
    ax = fig.add_subplot(111)
    sns.countplot(df[df.readmitted == 0].race.values)
    ax.set_title('Not Readmitted Patient')
    """)
p1,p2 = st.columns(2)

with p1:
    fig = plt.figure(figsize = (10, 5)) 
    ax = fig.add_subplot(111)

    sns.countplot(df[df.readmitted == 1].race.values)
    ax.set_title('Readmitted Patient')

    fig = plt.figure(figsize = (10, 5)) 
    ax = fig.add_subplot(111)
    sns.countplot(df[df.readmitted == 0].race.values)
    ax.set_title('Not Readmitted Patient')
    st.pyplot(fig)

st.write('---')

#------------------------------------------------
st.markdown("<h3><b>Correlation Matrix</b></h3>", unsafe_allow_html=True)

st.code("""
    matrix = np.triu(df.corr())
    fig, ax = plt.subplots(figsize=(16,16))
    sns.heatmap(df.corr(), annot=True, ax=ax, fmt='.1g', vmin=-1, vmax=1, center= 0, mask=matrix, cmap='RdBu_r')
    plt.show()
    """)
p1,p2 = st.columns(2)

with p1:
    matrix = np.triu(df.corr())
    fig, ax = plt.subplots(figsize=(16,16))
    sns.heatmap(df.corr(), annot=True, ax=ax, fmt='.1g', vmin=-1, vmax=1, center= 0, mask=matrix, cmap='RdBu_r')
    plt.show()
    st.pyplot(fig)

st.write('---')

#------------------------------------------------
st.markdown("<h3><b>Drugs Dosage Changes</b></h3>", unsafe_allow_html=True)

st.code("""
    fig, ax = plt.subplots(1, 7,figsize=(20, 4), subplot_kw=dict(aspect="equal"))
    ax[0].pie(df['metformin'].value_counts(), autopct='%1.0f%%', labels=['No', 'Steady', 'Up', 'Down'])
    ax[0].set_title('metformin Dosage')
    ax[1].pie(df['glimepiride'].value_counts(), autopct='%1.0f%%', labels=['No', 'Steady', 'Up', 'Down'])
    ax[1].set_title('glimepiride Dosage')
    ax[2].pie(df['glipizide'].value_counts(), autopct='%1.0f%%', labels=['No', 'Steady', 'Up', 'Down'])
    ax[2].set_title('glipizide Dosage')
    ax[3].pie(df['glyburide'].value_counts(), autopct='%1.0f%%', labels=['No', 'Steady', 'Up', 'Down'])
    ax[3].set_title('glyburide Dosage')
    ax[4].pie(df['pioglitazone'].value_counts(), autopct='%1.0f%%', labels=['No', 'Steady', 'Up', 'Down'])
    ax[4].set_title('pioglitazone Dosage')
    ax[5].pie(df['rosiglitazone'].value_counts(), autopct='%1.0f%%', labels=['No', 'Steady', 'Up', 'Down'])
    ax[5].set_title('rosiglitazone Dosage')
    ax[6].pie(df['insulin'].value_counts(), autopct='%1.0f%%', labels=['No', 'Steady', 'Down', 'Up'])
    ax[6].set_title('insulin Dosage')

    fig.suptitle('Drugs Dosage Changes During Encounter')
    plt.show()
    """)

fig, ax = plt.subplots(1, 7,figsize=(20, 4), subplot_kw=dict(aspect="equal"))
ax[0].pie(df['metformin'].value_counts(), autopct='%1.0f%%', labels=['No', 'Steady', 'Up', 'Down'])
ax[0].set_title('metformin Dosage')
ax[1].pie(df['glimepiride'].value_counts(), autopct='%1.0f%%', labels=['No', 'Steady', 'Up', 'Down'])
ax[1].set_title('glimepiride Dosage')
ax[2].pie(df['glipizide'].value_counts(), autopct='%1.0f%%', labels=['No', 'Steady', 'Up', 'Down'])
ax[2].set_title('glipizide Dosage')
ax[3].pie(df['glyburide'].value_counts(), autopct='%1.0f%%', labels=['No', 'Steady', 'Up', 'Down'])
ax[3].set_title('glyburide Dosage')
ax[4].pie(df['pioglitazone'].value_counts(), autopct='%1.0f%%', labels=['No', 'Steady', 'Up', 'Down'])
ax[4].set_title('pioglitazone Dosage')
ax[5].pie(df['rosiglitazone'].value_counts(), autopct='%1.0f%%', labels=['No', 'Steady', 'Up', 'Down'])
ax[5].set_title('rosiglitazone Dosage')
ax[6].pie(df['insulin'].value_counts(), autopct='%1.0f%%', labels=['No', 'Steady', 'Down', 'Up'])
ax[6].set_title('insulin Dosage')

fig.suptitle('Drugs Dosage Changes During Encounter')
plt.show()
st.pyplot(fig)
st.write('---')
#------------------------------------------------

st.markdown('<h2 style = "color: #A9333A;"><b>DATA PREPROCESSING</b></h2>', unsafe_allow_html=True)
st.write(' ')

st.code("""
    df = pd.concat([df,pd.get_dummies(df['race'], prefix='race')], axis=1).drop(['race'],axis=1)
    y = df['readmitted']
    X = df.drop(['readmitted'], axis=1)

    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=101)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=101)
    sc_X = StandardScaler()

    Xsc_train = sc_X.fit_transform(X_train)
    Xsc_val = sc_X.fit_transform(X_val)
    Xsc_test = sc_X.fit_transform(X_test)
    """)

df = pd.concat([df,pd.get_dummies(df['race'], prefix='race')], axis=1).drop(['race'],axis=1)
y = df['readmitted']
X = df.drop(['readmitted'], axis=1)

X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=101)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=101)
sc_X = StandardScaler()

Xsc_train = sc_X.fit_transform(X_train)
Xsc_val = sc_X.fit_transform(X_val)
Xsc_test = sc_X.fit_transform(X_test)

st.write('---')

#------------------------------------------------
st.markdown('<h2 style = "color: #A9333A;"><b>Models</b></h2>', unsafe_allow_html=True)
st.write(' ')

st.code("""
    #SVM
    svm = SVC() 
    svm.fit(Xsc_train,y_train)
    svm_pred = svm.predict(Xsc_val)

    #Gradient Boosting
    gbm = GradientBoostingClassifier() 
    gbm.fit(X_train,y_train)
    gbm_pred = gbm.predict(X_val)

    #Naive Bayes
    nbm = GaussianNB()
    nbm.fit(X_train,y_train)
    nbm_pred = nbm.predict(X_val)

    #KNN
    knn = KNeighborsClassifier() 
    knn.fit(Xsc_train,y_train)
    knn_pred = knn.predict(Xsc_val)

    #Randon Forest
    rfm = RandomForestClassifier() 
    rfm.fit(X_train,y_train)
    rfm_pred = rfm.predict(X_val)
    """)

model_selected = st.selectbox('Select The Model',['Give the best Model','SVM','Gradient Boosting','Naive Bayes','KNN','Random Forest'])

if model_selected == 'SVM':
    svm = SVC() 
    svm.fit(Xsc_train,y_train)
    svm_pred = svm.predict(Xsc_val)
    results = [
    {
        'Model':'SVM',
       'f1_score': metrics.f1_score(y_val, svm_pred,average='micro')
    }
    ]
    st.dataframe(pd.DataFrame(results))

elif model_selected == 'Gradient Boosting':
    gbm = GradientBoostingClassifier() 
    gbm.fit(X_train,y_train)
    gbm_pred = gbm.predict(X_val)
    results = [
    
    {
        'Model':'Gradient Boost',
       'f1_score': metrics.f1_score(y_val, gbm_pred,average='micro')
    }
    ]
    st.dataframe(pd.DataFrame(results))

elif model_selected == 'Naive Bayes':
    nbm = GaussianNB()
    nbm.fit(X_train,y_train)
    nbm_pred = nbm.predict(X_val)
    results = [
    
    {
    "Model":'Gaussian Naive Bayes',
    'f1_score':metrics.f1_score(y_val, nbm_pred,average='micro')
    }
    ]
    st.dataframe(pd.DataFrame(results))

elif model_selected == 'KNN':
    knn = KNeighborsClassifier() 
    knn.fit(Xsc_train,y_train)
    knn_pred = knn.predict(Xsc_val)
    results = [

    {
        'Model':'K-Nearest Neighbor',
       'f1_score':metrics.f1_score(y_val, knn_pred,average='micro')
      },
    ]
    st.dataframe(pd.DataFrame(results))
elif model_selected == 'Random Forest':
    rfm = RandomForestClassifier() 
    rfm.fit(X_train,y_train)
    rfm_pred = rfm.predict(X_val)
    results = [
    {
        'Model':'Random Forest',
       'f1_score': metrics.f1_score(y_val, rfm_pred,average='micro')
    }
    ]
    st.dataframe(pd.DataFrame(results))



else:
    results = [{'Model': 'Gaussian Naive Bayes', 'f1_score': 0.8748951244523165},
                {'Model': 'K-Nearest Neighbor', 'f1_score': 0.9060315092756596},
                {'Model': 'SVM', 'f1_score': 0.9126503216183462},
                {'Model': 'Random Forest', 'f1_score': 0.9126503216183462},
                {'Model': 'Gradient Boost', 'f1_score': 0.9128367670364501}]
    results_df = pd.DataFrame(results)
    st.write(' ')
    results_df.sort_values('f1_score', ascending=False,inplace=True)
    st.dataframe(results_df)
    winner = results_df.head(1).to_dict(orient='records')[0]['Model']
    st.markdown(f'<h2><b>The Winner: {winner}</b></h2>', unsafe_allow_html=True)
    # results_df[results_df['f1_score'] == results_df['f1_score'].max()]

    st.markdown("<h3><b>Results</b></h3>", unsafe_allow_html=True)

    st.code("""
    print('Gaussian Naive Bayes:')
    print('---------------------------------')
    print('F1 Score        : ', metrics.f1_score(y_val, nbm_pred,average='micro'))
    print('Confusion Matrix:\n ', confusion_matrix(y_val, nbm_pred))

    print('K-Nearest Neighbor:')
    print('---------------------------------------')
    print('F1 Score        : ', metrics.f1_score(y_val, knn_pred,average='micro'))
    print('Confusion Matrix:\n ', confusion_matrix(y_val, knn_pred))

    print('SVM:')
    print('------------------------')
    print('F1 Score        : ', metrics.f1_score(y_val, svm_pred,average='micro'))
    print('Confusion Matrix:\n ', confusion_matrix(y_val, svm_pred))

    print('Random Forest:')
    print('----------------------------------')
    print('F1 Score        : ', metrics.f1_score(y_val, rfm_pred,average='micro'))
    print('Confusion Matrix:\n ', confusion_matrix(y_val, rfm_pred))

    print('Gradient Boost:')
    print('-----------------------------------')
    print('F1 Score        : ', metrics.f1_score(y_val, gbm_pred,average='micro'))
    print('Confusion Matrix:\n ', confusion_matrix(y_val, gbm_pred))
            """)


    st.write('---')
    st.caption('''
    Gaussian Naive Bayes:
    ---------------------------------
    F1 Score        :  0.8748951244523165
    Confusion Matrix:
    [[9250  540]
    [ 802  135]]

    K-Nearest Neighbor:
    ---------------------------------------
    F1 Score        :  0.9060315092756596
    Confusion Matrix:
    [[9708   82]
    [ 926   11]]

    SVM:
    ------------------------
    F1 Score        :  0.9126503216183462
    Confusion Matrix:
    [[9790    0]
    [ 937    0]]

    Random Forest:
    ----------------------------------
    F1 Score        :  0.9126503216183462
    Confusion Matrix:
    [[9790    0]
    [ 937    0]]

    Gradient Boost:
    -----------------------------------
    F1 Score        :  0.9128367670364501
    Confusion Matrix:
    [[9788    2]
    [ 933    4]]
 ''')
    st.markdown('<h2 style = "color: #A9333A;"><b>Feature Importance Score</b></h2>', unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')
    st.code('''
        predictors = [x for x in X_train.columns]
        feat_imp = pd.Series(gbm.feature_importances_, predictors).sort_values(ascending=False)
        fig = plt.figure(figsize=(12, 6))
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
            ''')
    
    col1,col2 = st.columns(2)
    with col1:
        image = Image.open(f'./pictures/f_importance.png')
        st.image(image,use_column_width = True)