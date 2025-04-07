import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, Any

# Configuração inicial
st.set_page_config(
    page_title="Bank Churn Analytics Pro",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
DATA_PATH = Path(__file__).parent / "churn.csv"
MODEL_PATH = Path(__file__).parent / "models"
COUNTRIES = {
    'France': 'EUR', 'Germany': 'EUR', 'Spain': 'EUR', 'Brasil': 'BRL',
    'Argentina': 'ARS', 'Estados Unidos': 'USD', 'Canadá': 'CAD',
    'México': 'MXN', 'Reino Unido': 'GBP', 'Japão': 'JPY', 'China': 'CNY',
    'Índia': 'INR', 'Austrália': 'AUD', 'Suíça': 'CHF'
}
CURRENCY_SYMBOLS = {
    'EUR': '€', 'BRL': 'R$', 'USD': 'US$', 'CAD': 'C$', 'MXN': 'MX$',
    'GBP': '£', 'JPY': '¥', 'CNY': '¥', 'INR': '₹', 'AUD': 'A$', 'CHF': 'CHF'
}
VALID_COUNTRIES = list(COUNTRIES.keys())
PRIMARY_COLOR = "#2A2A2A"
SECONDARY_COLOR = "#00CC96"
BACKGROUND_COLOR = "#121212"
TEXT_COLOR = "#FFFFFF"
COLOR_PALETTE = ["#00CC96", "#2A2A2A", "#6C757D", "#ADB5BD"]

# Traduções (mínimas para brevidade; expanda conforme necessário)
LANGUAGES = {
    "PT-BR": {
        "title": "Análise Preditiva de Churn Bancário",
        "subtitle": "Sistema Inteligente de Previsão de Rotatividade de Clientes",
        "metrics": {
            "rf": "Modelo de Floresta Aleatória",
            "nn": "Modelo de Rede Neural",
            "desc": "Probabilidade de Cancelamento",
            "help_rf": "Algoritmo baseado em múltiplas árvores de decisão",
            "help_nn": "Rede neural profunda com 3 camadas ocultas"
        },
        "inputs": {
            "title": "Parâmetros do Cliente",
            "credit_score": "Pontuação de Crédito (300-850)",
            "age": "Idade do Cliente",
            "tenure": "Tempo como Cliente (anos)",
            "balance": "Saldo em Conta",
            "country": "País de Origem",
            "currency": "Moeda",
            "gender": "Gênero",
            "products": "Produtos Contratados",
            "card": "Possui Cartão de Crédito",
            "active": "Membro Ativo",
            "salary": "Salário Estimado"
        },
        "charts": {
            "main": "Fatores Determinantes para Churn",
            "age": "Distribuição de Idade dos Clientes",
            "corr": "Correlação entre Variáveis",
            "scatter": "Relação Saldo vs Idade"
        },
        "help": {
            "risk1": "Valores acima de 50% indicam alto risco",
            "risk2": "Considere ações preventivas acima de 30%"
        }
    },
    "EN": {
        "title": "Bank Churn Predictive Analytics",
        "subtitle": "Intelligent Customer Attrition Prediction System",
        "metrics": {
            "rf": "Random Forest Model",
            "nn": "Neural Network Model",
            "desc": "Cancellation Probability",
            "help_rf": "Algorithm based on multiple decision trees",
            "help_nn": "Deep neural network with 3 hidden layers"
        },
        "inputs": {
            "title": "Customer Parameters",
            "credit_score": "Credit Score (300-850)",
            "age": "Customer Age",
            "tenure": "Tenure (years)",
            "balance": "Account Balance",
            "country": "Country",
            "currency": "Currency",
            "gender": "Gender",
            "products": "Number of Products",
            "card": "Has Credit Card",
            "active": "Active Member",
            "salary": "Estimated Salary"
        },
        "charts": {
            "main": "Key Churn Drivers",
            "age": "Customer Age Distribution",
            "corr": "Variable Correlation",
            "scatter": "Balance vs Age Relationship"
        },
        "help": {
            "risk1": "Values above 50% indicate high risk",
            "risk2": "Consider preventive actions above 30%"
        }
    },
    "ES": {
        "title": "Análisis Predictivo de Abandono Bancario",
        "subtitle": "Sistema Inteligente de Predicción de Rotación de Clientes",
        "metrics": {
            "rf": "Modelo de Bosque Aleatorio",
            "nn": "Modelo de Red Neuronal",
            "desc": "Probabilidad de Cancelación",
            "help_rf": "Algoritmo basado en múltiples árboles de decisión",
            "help_nn": "Red neuronal profunda con 3 capas ocultas"
        },
        "inputs": {
            "title": "Parámetros del Cliente",
            "credit_score": "Puntuación de Crédito (300-850)",
            "age": "Edad del Cliente",
            "tenure": "Tiempo como Cliente (años)",
            "balance": "Saldo en Cuenta",
            "country": "País de Origen",
            "currency": "Moneda",
            "gender": "Género",
            "products": "Productos Contratados",
            "card": "Tiene Tarjeta de Crédito",
            "active": "Miembro Activo",
            "salary": "Salario Estimado"
        },
        "charts": {
            "main": "Factores Clave de Abandono",
            "age": "Distribución de Edad de Clientes",
            "corr": "Correlación de Variables",
            "scatter": "Relación Saldo vs Edad"
        },
        "help": {
            "risk1": "Valores superiores al 50% indican alto riesgo",
            "risk2": "Considere acciones preventivas arriba del 30%"
        }
    }
}

# Estilo
def set_app_style():
    custom_css = f"""
    <style>
    .stApp {{background-color: {BACKGROUND_COLOR}; color: {TEXT_COLOR}; font-family: 'Segoe UI', sans-serif;}}
    h1, h2, h3 {{color: {SECONDARY_COLOR}; border-bottom: 2px solid {PRIMARY_COLOR}; padding-bottom: 8px;}}
    [data-testid="stSidebar"] {{background: {PRIMARY_COLOR}; border-right: 1px solid {SECONDARY_COLOR};}}
    [data-testid="stMetric"] {{background: {PRIMARY_COLOR}; color: {TEXT_COLOR}; border-radius: 15px; padding: 20px; border: 1px solid {SECONDARY_COLOR};}}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Funções de dados
@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error("Arquivo de dados 'churn.csv' não encontrado.")
        st.stop()
    
    try:
        df = pd.read_csv(DATA_PATH, dtype={
            'CreditScore': 'int32', 'Geography': 'category', 'Gender': 'category',
            'Age': 'int32', 'Tenure': 'int32', 'Balance': 'float64',
            'NumOfProducts': 'int32', 'HasCrCard': 'bool', 'IsActiveMember': 'bool',
            'EstimatedSalary': 'float64', 'Exited': 'bool'
        })
        required_columns = {'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                           'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                           'EstimatedSalary', 'Exited'}
        if missing := required_columns - set(df.columns):
            raise ValueError(f"Colunas faltantes: {', '.join(missing)}")
        
        df['Geography'] = df['Geography'].apply(lambda x: x if x in VALID_COUNTRIES else 'Other')
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        st.stop()

@st.cache_data
def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    df_clean = df.drop(columns=['CustomerId', 'Surname'], errors='ignore')
    categorical_cols = ['Geography', 'Gender']
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                     'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), categorical_cols)
        ])
    
    X = df_clean.drop('Exited', axis=1)
    y = df_clean['Exited'].astype(int)
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor

# Modelagem
@st.cache_resource
def train_or_load_models(X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestClassifier, Sequential]:
    MODEL_PATH.mkdir(exist_ok=True)
    rf_path = MODEL_PATH / "rf_model.pkl"
    nn_path = MODEL_PATH / "nn_model.h5"
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if rf_path.exists():
        rf_model = joblib.load(rf_path)
    else:
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, rf_path)
    
    if nn_path.exists():
        keras_model = load_model(nn_path)
    else:
        keras_model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(1, activation='sigmoid')
        ])
        keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        keras_model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2, verbose=0)
        keras_model.save(nn_path)
    
    return rf_model, keras_model

# Componentes de UI
def create_inputs(lang: Dict[str, Any]) -> pd.DataFrame:
    with st.sidebar:
        st.markdown(f"<h2 style='color:{SECONDARY_COLOR};'>🔍 {lang['inputs']['title']}</h2>", unsafe_allow_html=True)
        
        selected_country = st.selectbox(lang['inputs']['country'], options=VALID_COUNTRIES)
        currency_symbol = CURRENCY_SYMBOLS[COUNTRIES[selected_country]]
        
        inputs = {
            'CreditScore': st.slider(lang['inputs']['credit_score'], 300, 850, 650),
            'Age': st.slider(lang['inputs']['age'], 18, 100, 40),
            'Tenure': st.slider(lang['inputs']['tenure'], 0, 10, 3),
            'Balance': st.number_input(f"{lang['inputs']['balance']} ({currency_symbol})",
                                     min_value=0.0, value=5000.0, step=100.0),
            'Geography': selected_country if selected_country in VALID_COUNTRIES else 'Other',
            'Gender': st.radio(lang['inputs']['gender'], ['Male', 'Female'], horizontal=True),
            'NumOfProducts': st.selectbox(lang['inputs']['products'], [1, 2, 3, 4]),
            'HasCrCard': st.checkbox(lang['inputs']['card'], True),
            'IsActiveMember': st.checkbox(lang['inputs']['active'], True),
            'EstimatedSalary': st.number_input(f"{lang['inputs']['salary']} ({currency_symbol})",
                                             min_value=0.0, value=5000.0, step=100.0)
        }
        return pd.DataFrame([inputs])

# Visualizações
def create_main_chart(preprocessor: ColumnTransformer, rf_model: RandomForestClassifier, lang: Dict[str, Any]) -> go.Figure:
    feature_names = [name.replace('num__', '').replace('cat__', '') for name in preprocessor.get_feature_names_out()]
    importance = rf_model.feature_importances_
    fig = px.bar(x=importance, y=feature_names, title=f"<b>{lang['charts']['main']}</b>",
                 color=importance, color_continuous_scale=COLOR_PALETTE)
    fig.update_layout(plot_bgcolor=BACKGROUND_COLOR, paper_bgcolor=BACKGROUND_COLOR, font={'color': TEXT_COLOR},
                      xaxis_title="Importância Relativa", yaxis={'categoryorder': 'total ascending'})
    return fig

def create_age_chart(df: pd.DataFrame, lang: Dict[str, Any]) -> go.Figure:
    fig = px.histogram(df, x='Age', nbins=20, title=f"<b>{lang['charts']['age']}</b>",
                       color_discrete_sequence=[SECONDARY_COLOR])
    fig.update_layout(plot_bgcolor=BACKGROUND_COLOR, paper_bgcolor=BACKGROUND_COLOR, font={'color': TEXT_COLOR})
    return fig

def create_corr_chart(df: pd.DataFrame, lang: Dict[str, Any]) -> go.Figure:
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                                    colorscale=COLOR_PALETTE, zmin=-1, zmax=1))
    fig.update_layout(title=f"<b>{lang['charts']['corr']}</b>", plot_bgcolor=BACKGROUND_COLOR,
                      paper_bgcolor=BACKGROUND_COLOR, font={'color': TEXT_COLOR})
    return fig

def create_scatter_chart(df: pd.DataFrame, lang: Dict[str, Any]) -> go.Figure:
    fig = px.scatter(
        df,
        x='Age',
        y='Balance',
        color='Exited',
        title=f"<b>{lang['charts']['scatter']}</b>",
        color_continuous_scale=COLOR_PALETTE
    )
    fig.update_layout(
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font={'color': TEXT_COLOR}
    )
    return fig

# Aplicação principal
def main():
    set_app_style()
    
    # Verifica se LANGUAGES é um dicionário
    if not isinstance(LANGUAGES, dict):
        st.error("Erro: LANGUAGES deve ser um dicionário, não um set ou outro tipo.")
        st.stop()
    
    lang_option = st.selectbox("", options=list(LANGUAGES.keys()), index=0)
    lang = LANGUAGES[lang_option]
    
    st.markdown(f"""
        <div style="text-align:center; padding:30px; background:{PRIMARY_COLOR}; 
                    border-radius:15px; margin-bottom:30px; border: 2px solid {SECONDARY_COLOR};">
            <h1 style="color:{SECONDARY_COLOR};">{lang['title']}</h1>
            <p style="color:{TEXT_COLOR};">{lang['subtitle']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.spinner('Carregando dados e modelos...'):
        try:
            df = load_dataset()
            X, y, preprocessor = preprocess_data(df)
            rf_model, keras_model = train_or_load_models(X, y)
        except Exception as e:
            st.error(f"Erro ao carregar dados ou modelos: {str(e)}")
            st.stop()
    
    input_df = create_inputs(lang)
    
    try:
        processed_input = preprocessor.transform(input_df)
        rf_pred = rf_model.predict_proba(processed_input)[0][1] * 100
        nn_pred = keras_model.predict(processed_input, verbose=0)[0][0] * 100
        
        st.markdown(f"### {lang['metrics']['desc']}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(lang['metrics']['rf'], f"{rf_pred:.1f}%", help=lang['metrics']['help_rf'])
        with col2:
            st.metric(lang['metrics']['nn'], f"{nn_pred:.1f}%", help=lang['metrics']['help_nn'])
        
        st.markdown(f"""
            <div style='background:{PRIMARY_COLOR}; padding:20px; border-radius:10px; margin:20px 0;'>
                <h4 style='color:{SECONDARY_COLOR};'>📌 {lang['help']['risk1']}</h4>
                <p style='color:{TEXT_COLOR};'>{lang['help']['risk2']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            lang['charts']['main'], lang['charts']['age'],
            lang['charts']['corr'], lang['charts']['scatter']
        ])
        
        with tab1:
            st.plotly_chart(create_main_chart(preprocessor, rf_model, lang), use_container_width=True)
        with tab2:
            st.plotly_chart(create_age_chart(df, lang), use_container_width=True)
        with tab3:
            st.plotly_chart(create_corr_chart(df, lang), use_container_width=True)
        with tab4:
            st.plotly_chart(create_scatter_chart(df, lang), use_container_width=True)
        
    except ValueError as ve:
        st.error(f"Erro nos dados de entrada: {str(ve)}")
    except Exception as e:
        st.error(f"Erro inesperado: {str(e)}")

if __name__ == "__main__":
    main()
