import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Taylor & Estadística | Dashboard",
    page_icon="∫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para darle un toque más "App" y menos "Script"
st.markdown("""
<style>
    /* Color de fondo general */
    .main {
        background-color: #f5f5f5;
    }
    
    /* ESTILO DE LAS TARJETAS DE MÉTRICAS */
    div[data-testid="stMetric"], .stMetric {
        background-color: #ffffff !important;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #2c3e50 !important; /* <--- ESTO ARREGLA EL TEXTO INVISIBLE */
    }
    
    /* Forzar color de las etiquetas (labels) dentro de la métrica */
    div[data-testid="stMetric"] label {
        color: #2c3e50 !important;
    }
    
    /* Forzar color del valor numérico */
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #2c3e50 !important;
    }

    /* Títulos */
    h1, h2, h3 {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LÓGICA MATEMÁTICA (BACKEND)
# -----------------------------------------------------------------------------

@st.cache_data # Decorador para optimizar rendimiento
def factorial(n):
    return math.factorial(n)

def taylor_exponencial_engine(x, n_terms):
    """
    Calcula e^x manualmente y devuelve el historial de convergencia.
    Retorna: (valor_final, lista_de_valores_por_iteracion)
    """
    suma = 0
    historial = []
    for n in range(n_terms):
        term = (x ** n) / factorial(n)
        suma += term
        historial.append(suma)
    return suma, historial

def prob_binomial(k, n, p):
    """P(X=k) Binomial estándar"""
    if k > n: return 0
    comb = math.comb(n, k)
    return comb * (p**k) * ((1-p)**(n-k))

def prob_poisson_taylor(k, lam, n_terms_taylor=50):
    """
    P(X=k) Poisson INTEGRADO con Taylor.
    Calcula e^-lambda usando nuestra propia serie.
    """
    # Aquí está el "truco" del proyecto: Usar Taylor para la parte exponencial
    e_minus_lam, _ = taylor_exponencial_engine(-lam, n_terms_taylor)
    return (lam**k * e_minus_lam) / factorial(k)

# -----------------------------------------------------------------------------
# 3. INTERFAZ: BARRA LATERAL (CONTROLES)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Panel de Control")
    
    st.markdown("### 1. Configuración Taylor")
    x_input = st.number_input("Exponente (x) para e^x", value=-5.0, step=0.5)
    n_terms = st.slider("Precisión (N términos)", 1, 100, 20)
    
    st.markdown("---")
    
    st.markdown("### 2. Configuración Estadística")
    st.info("Simula la convergencia Binomial -> Poisson")
    n_ensayos = st.number_input("Ensayos (n)", min_value=10, max_value=200, value=50)
    p_exito = st.slider("Probabilidad (p)", 0.01, 0.50, 0.10, step=0.01)
    
    # Cálculo automático de Lambda
    lam_calc = n_ensayos * p_exito
    st.markdown(f"**Lambda Resultante ($\lambda = np$):** `{lam_calc:.2f}`")
    
    st.markdown("---")
    st.caption("Proyecto Universitario | Cálculo & Probabilidad")

# -----------------------------------------------------------------------------
# 4. INTERFAZ: ÁREA PRINCIPAL
# -----------------------------------------------------------------------------

st.title("Integración de Series de Taylor y Distribuciones Discretas")
st.markdown("Este dashboard interactivo demuestra cómo los métodos numéricos (**Series de Taylor**) fundamentan el cálculo de probabilidades en modelos estocásticos (**Poisson**).")

# Pestañas para organizar el contenido
tab1, tab2, tab3 = st.tabs(["📘 Análisis Matemático (Taylor)", "📊 Convergencia Estadística", "💻 Código Fuente"])

# --- TAB 1: TAYLOR ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Cálculo Numérico")
        st.write(f"Aproximando $f(x) = e^{{{x_input}}}$")
        
        val_taylor, historial = taylor_exponencial_engine(x_input, n_terms)
        val_real = math.exp(x_input)
        error_abs = abs(val_real - val_taylor)
        
        # Métricas grandes
        st.metric(label="Valor Real (Python Math)", value=f"{val_real:.10f}")
        st.metric(label=f"Aproximación Taylor (N={n_terms})", value=f"{val_taylor:.10f}", delta=f"Error: {error_abs:.2e}", delta_color="inverse")
        
        st.latex(r"e^x \approx \sum_{n=0}^{N} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \dots")

    with col2:
        st.subheader("Visualización de Convergencia")
        # Gráfica: Cómo cambia el valor a medida que agregamos términos
        fig_conv = go.Figure()
        
        # Línea del valor real (objetivo)
        fig_conv.add_trace(go.Scatter(
            x=list(range(n_terms)), 
            y=[val_real]*n_terms,
            mode='lines',
            name='Valor Real (e^x)',
            line=dict(color='green', dash='dash')
        ))
        
        # Línea de la aproximación paso a paso
        fig_conv.add_trace(go.Scatter(
            x=list(range(n_terms)), 
            y=historial,
            mode='lines+markers',
            name='Serie de Taylor',
            line=dict(color='blue')
        ))
        
        fig_conv.update_layout(
            title="Evolución de la aproximación por número de términos",
            xaxis_title="Número de términos (n)",
            yaxis_title="Valor Calculado",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_conv, use_container_width=True)

# --- TAB 2: ESTADÍSTICA ---
with tab2:
    st.subheader(f"Comparativa: Binomial(n={n_ensayos}, p={p_exito}) vs Poisson(λ={lam_calc:.2f})")
    
    col_stats_1, col_stats_2 = st.columns([3, 1])
    
    # Generar datos
    rango_k = range(int(lam_calc * 3) + 2) # Mostrar rango relevante según lambda
    data_stats = []
    
    for k in rango_k:
        b_val = prob_binomial(k, n_ensayos, p_exito)
        p_val = prob_poisson_taylor(k, lam_calc, n_terms) # Usamos el n_terms del sidebar
        data_stats.append({
            "k": k,
            "Binomial": b_val,
            "Poisson (Taylor)": p_val,
            "Diff": abs(b_val - p_val)
        })
    
    df_stats = pd.DataFrame(data_stats)
    
    with col_stats_1:
        # Gráfica de Barras Comparativa
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            x=df_stats['k'], 
            y=df_stats['Binomial'],
            name='Binomial',
            marker_color='rgba(52, 152, 219, 0.7)'
        ))
        fig_bars.add_trace(go.Scatter(
            x=df_stats['k'], 
            y=df_stats['Poisson (Taylor)'],
            name='Poisson (Calculada con Taylor)',
            mode='lines+markers',
            line=dict(color='rgba(231, 76, 60, 1)', width=3)
        ))
        
        fig_bars.update_layout(
            title="Distribución de Probabilidad (PMF)",
            xaxis_title="Éxitos (k)",
            yaxis_title="Probabilidad",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        st.plotly_chart(fig_bars, use_container_width=True)
        
    with col_stats_2:
        st.write("### Análisis de Error")
        mae = df_stats['Diff'].mean()
        max_err = df_stats['Diff'].max()
        
        st.metric("Error Medio Absoluto", f"{mae:.5f}")
        st.metric("Diferencia Máxima", f"{max_err:.5f}")
        
        if mae < 0.01:
            st.success("✅ **Convergencia Exitosa**")
        else:
            st.warning("⚠️ **Divergencia Detectada** (Ajusta n o p)")
            
        with st.expander("Ver Tabla de Datos"):
            st.dataframe(df_stats, height=300)

# --- TAB 3: CÓDIGO ---
with tab3:
    st.markdown("### Lógica de Implementación")
    st.markdown("""
    A continuación se muestra cómo se implementó la función de Poisson utilizando 
    nuestra propia función de Taylor para calcular $e^{-\lambda}$.
    """)
    st.code("""
def taylor_exponencial_engine(x, n_terms):
    '''Calcula e^x mediante sumatoria manual'''
    suma = 0
    for n in range(n_terms):
        term = (x ** n) / math.factorial(n)
        suma += term
    return suma

def prob_poisson_taylor(k, lam, n_terms):
    '''
    Calcula Poisson usando Taylor para el componente exponencial.
    Formula: P(k) = (lambda^k * e^-lambda) / k!
    '''
    # Aquí llamamos a nuestra función manual
    e_menos_lam_taylor = taylor_exponencial_engine(-lam, n_terms)
    
    return (lam**k * e_menos_lam_taylor) / math.factorial(k)
    """, language="python")



    # -----------------------------------------------------------------------------
# 5. FOOTER (Integrantes y Datos del Proyecto)
# -----------------------------------------------------------------------------
def mostrar_footer():
    st.markdown("---")
    col_izq, col_der = st.columns([1, 1])
    
    with col_izq:
        st.markdown("""
        ### 🏛️ Datos
        **Instituto Politécnico Nacional (IPN)** *Escuela Superior de Cómputo (ESCOM)* **Materia:** Cálculo Aplicado  
        **Grupo:** 2CM5  
        **Profesora:** Claudia Garcia Blanquel  
        """)
        
    with col_der:
        st.markdown("### 👥 Integrantes del Equipo")
        # SUSTITUYE AQUÍ LOS NOMBRES REALES
        st.write("1. Bravo Calderón Osvaldo Samuel")
        st.write("2. Juarez Cortés Daniel")
        st.write("3. Mendoza David Alberto")

# Llamamos a la función al final del script
mostrar_footer()