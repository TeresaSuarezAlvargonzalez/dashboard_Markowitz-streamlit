
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco
from datetime import date
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import requests



# Configuración general
st.set_page_config(page_title="Dashboard de Inversión", layout="wide")


st.markdown("""
<style>
/* Forzar spellcheck false en textarea con !important y otros hacks */
textarea {
  -webkit-text-security: none !important;
  spellcheck: false !important;
  caret-color: auto !important;
}
</style>
""", unsafe_allow_html=True)


st.title("📈 Dashboard de Cartera de Inversión")
# Parámetros de la cartera

tickers_input = st.text_area("Introduce los tickers separados por coma:", "PLTR, ASML, GOOGL, MC.PA, MRNA, SLB")

# Procesar la entrada en lista de tickers, limpiando espacios
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

# Selector para elegir años (1 a 20)
años = st.slider("Selecciona el número de años para el rango:", min_value=1, max_value=20, value=4)

fecha_fin = date.today()
fecha_inicio = fecha_fin - relativedelta(years=años)

st.markdown(f"Periodo analizado: **{fecha_inicio}** a **{fecha_fin}**")

# Función para descargar datos históricos
@st.cache_data
def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=False, threads=True)
    
    # Si hay múltiples niveles (por tener varios tickers)
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = {}
        for ticker in tickers:
            try:
                adj_close[ticker] = data[ticker]['Adj Close']
            except KeyError:
                st.warning(f"⚠️ No se encontraron datos de 'Adj Close' para {ticker}")
        df_adj_close = pd.DataFrame(adj_close)
        return df_adj_close.dropna(axis=1, how='all')  # Elimina columnas completamente vacías
    else:
        # Para el caso de un solo ticker
        if 'Adj Close' in data.columns:
            return data['Adj Close'].to_frame(name=tickers[0])
        else:
            raise ValueError("No se encontró 'Adj Close' en los datos descargados.")

# Descargar los datos y mostrar tabla
data = get_data(tickers, fecha_inicio, fecha_fin)
# Descargar los datos y mostrar tabla
data = get_data(tickers, fecha_inicio, fecha_fin)

# Comprobar si algún ticker fue descartado por falta de datos
tickers_con_datos = data.columns.tolist()
tickers_sin_datos = [ticker for ticker in tickers if ticker not in tickers_con_datos]

if tickers_sin_datos:
    st.warning(f"⚠️ No se encontraron datos suficientes para los siguientes tickers en el periodo seleccionado: {', '.join(tickers_sin_datos)}")

st.subheader("📊 Precios ajustados")

st.subheader("📊 Precios ajustados")
st.dataframe(data.tail())



# Normalizar inversión inicial a 1€ por ticker
normalized = data / data.iloc[0]

# Selector dinámico de tickers a mostrar en el gráfico
tickers_seleccionados = st.multiselect(
    "Selecciona los tickers a mostrar en la evolución de inversión",
    options=normalized.columns.tolist(),
    default=normalized.columns.tolist()
)

if tickers_seleccionados:
    # Filtrar solo los tickers seleccionados
    normalized_filtrado = normalized[tickers_seleccionados]

    # Convertir a formato 'long' para plotly
    df_long = normalized_filtrado.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Valor (€)')
    
    # Crear gráfico de líneas con Plotly Express
    fig = px.line(
        df_long,
        x='Date',
        y='Valor (€)',
        color='Ticker',
        title='Evolución de 1€ invertido en cada acción',
        labels={'Date': 'Fecha'}
    )

    # Diseño para fondo oscuro y texto blanco
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend_title_text='Ticker',
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=50, b=40, l=60, r=20),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Selecciona al menos un ticker para mostrar la evolución.")




# Calcular rentabilidad diaria
daily_returns = data.pct_change().dropna()

# Calcular métricas por activo
annual_returns = daily_returns.mean() * 252
annual_volatility = daily_returns.std() * np.sqrt(252)

# Obtener PER (Price to Earnings Ratio) para cada ticker
def get_per(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("trailingPE", np.nan)
    except:
        return np.nan

with st.spinner("📥 Obteniendo datos de PER..."):
    per_values = {ticker: get_per(ticker) for ticker in data.columns}


# Definir tasa libre de riesgo anual (por ejemplo 0.5% = 0.005)
risk_free_rate = 0.005

# Cálculo Sharpe Ratio (usando volatilidad anual)
sharpe_ratio = (annual_returns - risk_free_rate) / annual_volatility

# Cálculo Sortino Ratio
# Primero calculamos la desviación estándar solo de los retornos negativos (downside)
negative_returns = daily_returns.copy()
negative_returns[negative_returns > 0] = 0
downside_std = negative_returns.std() * np.sqrt(252)

sortino_ratio = (annual_returns - risk_free_rate) / downside_std


# Crear DataFrame con todas las métricas
metrics = pd.DataFrame({
    "Rentabilidad anual (%)": annual_returns * 100,
    "Volatilidad anual (%)": annual_volatility * 100,
    "Shape Ratio": sharpe_ratio,
    "Sortino Ratio": sortino_ratio,
    "PER": pd.Series(per_values)
}).sort_values("Rentabilidad anual (%)", ascending=False)


# Mostrar tabla actualizada
st.subheader("📈 Rentabilidad, Volatilidad, PER, Sharpe y Sortino")
st.dataframe(metrics.style.format({
    "Rentabilidad anual (%)": "{:.2f}",
    "Volatilidad anual (%)": "{:.2f}",
    "PER": "{:.2f}",
    "Sharpe Ratio": "{:.2f}",
    "Sortino Ratio": "{:.2f}",
}))



df_resultados_individuales = pd.DataFrame({
    "Ticker": data.columns,
    "Rendimiento Anual (%)": annual_returns * 100,
    "Volatilidad Anual (%)": annual_volatility * 100,
    "Sharpe": sharpe_ratio,
    "Sortino": sortino_ratio
}).set_index("Ticker").reset_index()

# Ordenamos para cada métrica
df_rend = df_resultados_individuales.sort_values(by='Rendimiento Anual (%)', ascending=True)
df_vol = df_resultados_individuales.sort_values(by='Volatilidad Anual (%)', ascending=True)
df_sharpe = df_resultados_individuales.sort_values(by='Sharpe', ascending=True)
df_sortino = df_resultados_individuales.sort_values(by='Sortino', ascending=True)

df_resultados_individuales = pd.DataFrame({
    "Ticker": data.columns,
    "Rendimiento Anual (%)": annual_returns * 100,
    "Volatilidad Anual (%)": annual_volatility * 100,
    "Sharpe": sharpe_ratio,
    "Sortino": sortino_ratio
}).set_index("Ticker").reset_index()

df_rend = df_resultados_individuales.sort_values(by='Rendimiento Anual (%)', ascending=True)
df_vol = df_resultados_individuales.sort_values(by='Volatilidad Anual (%)', ascending=True)
df_sharpe = df_resultados_individuales.sort_values(by='Sharpe', ascending=True)
df_sortino = df_resultados_individuales.sort_values(by='Sortino', ascending=True)

def get_max_abs(series):
    return max(abs(series.min()), abs(series.max()))

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Rendimiento Anual (%)", "Volatilidad Anual (%)", "Ratio de Sharpe", "Ratio de Sortino"),
    horizontal_spacing=0.1, vertical_spacing=0.15
)

# Rendimiento Anual (bicolor)
max_abs_rend = get_max_abs(df_rend['Rendimiento Anual (%)'])
fig.add_trace(go.Bar(
    x=df_rend['Rendimiento Anual (%)'],
    y=df_rend['Ticker'],
    orientation='h',
    marker=dict(
        color=df_rend['Rendimiento Anual (%)'],
        colorscale=[[0, '#e74c3c'], [0.5, '#f0f0f0'], [1, '#2ecc71']],
        cmin=-max_abs_rend,
        cmax=max_abs_rend,
    ),
    # Quitamos text (etiquetas fijas)
    hovertemplate='%{y}: %{x:.2f}%<extra></extra>'
), row=1, col=1)

# Volatilidad (rojo unidireccional)
max_vol = df_vol['Volatilidad Anual (%)'].max()
fig.add_trace(go.Bar(
    x=df_vol['Volatilidad Anual (%)'],
    y=df_vol['Ticker'],
    orientation='h',
    marker=dict(
        color=df_vol['Volatilidad Anual (%)'],
        colorscale=[[0, '#f0f0f0'], [1, '#e74c3c']],
        cmin=0,
        cmax=max_vol,
    ),
    hovertemplate='%{y}: %{x:.2f}%<extra></extra>'
), row=1, col=2)

# Sharpe (bicolor)
max_abs_sharpe = get_max_abs(df_sharpe['Sharpe'])
fig.add_trace(go.Bar(
    x=df_sharpe['Sharpe'],
    y=df_sharpe['Ticker'],
    orientation='h',
    marker=dict(
        color=df_sharpe['Sharpe'],
        colorscale=[[0, '#e74c3c'], [0.5, '#f0f0f0'], [1, '#2ecc71']],
        cmin=-max_abs_sharpe,
        cmax=max_abs_sharpe,
    ),
    hovertemplate='%{y}: %{x:.2f}<extra></extra>'
), row=2, col=1)

# Sortino (bicolor)
max_abs_sortino = get_max_abs(df_sortino['Sortino'])
fig.add_trace(go.Bar(
    x=df_sortino['Sortino'],
    y=df_sortino['Ticker'],
    orientation='h',
    marker=dict(
        color=df_sortino['Sortino'],
        colorscale=[[0, '#e74c3c'], [0.5, '#f0f0f0'], [1, '#2ecc71']],
        cmin=-max_abs_sortino,
        cmax=max_abs_sortino,
    ),
    hovertemplate='%{y}: %{x:.2f}<extra></extra>'
), row=2, col=2)

fig.update_layout(
    autosize=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(t=80, b=40, l=70, r=70),
    showlegend=False,
    font=dict(color='white')
)

for i in range(1, 5):
    fig.update_xaxes(visible=False, row=(i-1)//2+1, col=(i-1)%2+1)
    fig.update_yaxes(showgrid=False, row=(i-1)//2+1, col=(i-1)%2+1)

for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color='white', size=20, family='Arial', weight='bold')

st.subheader("📉 Comparación Visual de Métricas Financieras")
st.plotly_chart(fig, use_container_width=True)


def get_risk_metrics(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "Beta": info.get("beta", np.nan),
            "Debt to Equity": info.get("debtToEquity", np.nan),
            "Current Ratio": info.get("currentRatio", np.nan),
            "Total Debt": info.get("totalDebt", np.nan),
            # "Equity" se calcula después
        }
    except Exception as e:
        st.warning(f"⚠️ Error al obtener datos de {ticker}: {e}")
        return {
            "Beta": np.nan,
            "Debt to Equity": np.nan,
            "Current Ratio": np.nan,
            "Total Debt": np.nan,
        }

tickers = list(data.columns)  # O tu lista de tickers

with st.spinner("📥 Obteniendo métricas financieras..."):
    risk_metrics = {ticker: get_risk_metrics(ticker) for ticker in tickers}
    risk_metrics_df = pd.DataFrame(risk_metrics).T

# Calcular Equity: Total Debt / (Debt to Equity / 100)
# Evitar división por cero o NaN
risk_metrics_df["Equity"] = risk_metrics_df.apply(
    lambda row: row["Total Debt"] / (row["Debt to Equity"] / 100)
    if pd.notnull(row["Total Debt"]) and pd.notnull(row["Debt to Equity"]) and row["Debt to Equity"] != 0
    else np.nan,
    axis=1
)

def highlight_debt_to_equity(val):
    if val >= 200:
        return 'background-color: #ff6666; color: black'  # rojo suave
    elif val >= 150:
        return 'background-color: #ffeb99; color: black'  # amarillo suave
    else:
        return 'background-color: #d0f0c0; color: black'  # verde suave

def highlight_current_ratio(val):
    if val < 1:
        return 'background-color: #ff6666; color: black'
    elif val > 3:
        return 'background-color: #ffeb99; color: black'
    else:
        return 'background-color: #d0f0c0; color: black'

def highlight_beta(val):
    if val > 2:
        return 'background-color: #ff6666; color: black'
    elif val > 1.5:
        return 'background-color: #ffeb99; color: black'
    else:
        return 'background-color: #d0f0c0; color: black'

st.subheader("⚠️ Métricas de riesgo y endeudamiento")

# Mostrar solo las métricas clave
df_mostrar = risk_metrics_df[["Beta", "Debt to Equity", "Current Ratio"]]

st.dataframe(
    df_mostrar.style.format({
        "Beta": "{:.2f}",
        "Debt to Equity": "{:.2f}",
        "Current Ratio": "{:.2f}",
    })
    .applymap(highlight_debt_to_equity, subset=["Debt to Equity"])
    .applymap(highlight_current_ratio, subset=["Current Ratio"])
    .applymap(highlight_beta, subset=["Beta"])
)


# Sección de la matriz de correlación
st.subheader("🔗 Matriz de Correlaciones entre Activos")

# Calcular matriz de correlación
correlation_matrix = daily_returns.corr().round(2)
tickers_corr = correlation_matrix.columns.tolist()

# Crear heatmap con estilo moderno
fig_corr = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=tickers_corr,
    y=tickers_corr,
    colorscale='RdBu',
    zmin=-1,
    zmax=1,
    colorbar=dict(title="Correlación"),
    hovertemplate="Activo 1: %{y}<br>Activo 2: %{x}<br>Correlación: %{z}<extra></extra>"
))

# Añadir anotaciones sobre cada celda
for i in range(len(tickers_corr)):
    for j in range(len(tickers_corr)):
        fig_corr.add_annotation(
            x=tickers_corr[j],
            y=tickers_corr[i],
            text=str(correlation_matrix.iloc[i, j]),
            showarrow=False,
            font=dict(color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black", size=12),
            xanchor="center",
            yanchor="middle"
        )

# Layout estilizado
fig_corr.update_layout(
    title="📊 Matriz de Correlaciones entre Activos",
    title_font_size=24,
    autosize=True,
    height=800,
    margin=dict(l=60, r=60, t=100, b=60),
    xaxis=dict(showgrid=False, tickangle=-45),
    yaxis=dict(showgrid=False),
    plot_bgcolor='rgba(0,0,0,0)',
)

# Mostrar en Streamlit
st.plotly_chart(fig_corr, use_container_width=True)

prices = data

mean_returns = daily_returns.mean() * 252

# Convertir a DataFrame para plotear
mean_returns_df = mean_returns.reset_index()
mean_returns_df.columns = ['Activo', 'Retorno Anual']

# Gráfico con Plotly
fig_retornos = px.scatter(
    mean_returns_df,
    x='Activo',
    y='Retorno Anual',
    title='Retorno Histórico Anualizado por Activo',
    labels={'Retorno Anual': 'Retorno Anual (%)'},
    template='plotly_white'
)

# Añadir línea horizontal roja en y=0
fig_retornos.add_shape(
    type="line",
    x0=-1,
    x1=len(mean_returns_df['Activo']),  # un poco después del último tick
    y0=0,
    y1=0,
    line=dict(color="red", width=1),
    xref="x",
    yref="y"
)

fig_retornos.update_traces(marker=dict(size=30, color='rgba(54, 162, 235, 0.6)'))
fig_retornos.update_layout(
    xaxis_tickangle=-45,
    yaxis_tickformat=".2%",
    height=500
)

st.plotly_chart(fig_retornos, use_container_width=True)

API_KEY = "e2b86727fa874751911c95e690357262"

tickers = ["PLTR", "SLB", "ASML", "GOOGL","MC.PA", "MRNA"]
selected_ticker = st.selectbox("Selecciona una acción", tickers)

def get_news_newsapi(query):
    url = ("https://newsapi.org/v2/everything?"
           f"q={query}&"
           "sortBy=publishedAt&"
           "language=en&"
           "pageSize=10&"
           f"apiKey={API_KEY}")
    response = requests.get(url)
    data = response.json()
    if data.get("status") != "ok":
        st.error("Error al obtener noticias")
        return []
    articles = data.get("articles", [])
    news_list = []
    for article in articles:
        title = article.get("title")
        url = article.get("url")
        news_list.append((title, url))
    return news_list

st.subheader(f"Noticias recientes para {selected_ticker}")
news = get_news_newsapi(selected_ticker)

if news:
    for title, url in news:
        st.markdown(f"- [{title}]({url})")
else:
    st.write("No se encontraron noticias para este ticker.")


@st.cache_data(show_spinner=True)
def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)
    if data.empty:
        raise ValueError("No se han descargado datos para los tickers y fechas indicados.")
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            data = data.loc[:, 'Adj Close']
        elif 'Close' in data.columns.levels[0]:
            data = data.loc[:, 'Close']
        else:
            raise ValueError("Los datos no contienen ni 'Adj Close' ni 'Close'.")
    else:
        if 'Adj Close' in data.columns:
            data = data[['Adj Close']]
        elif 'Close' in data.columns:
            data = data[['Close']]
        else:
            raise ValueError("Los datos no contienen ni 'Adj Close' ni 'Close'.")
    if isinstance(tickers, str):
        tickers = [tickers]
    if len(tickers) == 1:
        data.columns = [tickers[0]]
    return data.dropna(how='all')

@st.cache_data(show_spinner=True)
def frontera_markowitz_streamlit(tickers, num_portfolios, años):
    fecha_fin = date.today()
    fecha_inicio = fecha_fin - relativedelta(years=años)
    data = get_data(tickers, fecha_inicio, fecha_fin)
    daily_returns = data.pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    num_assets = len(tickers)

    results = np.zeros((5, num_portfolios))  # 5 filas: retorno, volatilidad, sharpe, sortino, max retorno
    weights_record = []

    # Para Sortino: asumimos retorno objetivo cero
    target_return = 0

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        port_return = np.sum(mean_returns * weights) * 252
        port_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

        # Cálculo Sharpe
        sharpe_ratio = port_return / port_stddev if port_stddev != 0 else 0

        # Cálculo Sortino (usamos desviación negativa anualizada)
        port_returns_series = daily_returns.dot(weights)
        downside_returns = port_returns_series[port_returns_series < target_return]
        downside_stddev = np.sqrt((downside_returns**2).mean()) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = port_return / downside_stddev if downside_stddev != 0 else 0

        results[0, i] = port_return
        results[1, i] = port_stddev
        results[2, i] = sharpe_ratio
        results[3, i] = sortino_ratio
        results[4, i] = port_return  # para máximo retorno, se usa retorno directamente

    results_df = pd.DataFrame(results.T, columns=['Retorno Anual', 'Volatilidad Anual', 'Sharpe', 'Sortino', 'Retorno'])
    weights_df = pd.DataFrame(weights_record, columns=tickers)
    df_final = pd.concat([results_df, weights_df], axis=1)

    idx_sharpe = df_final['Sharpe'].idxmax()
    idx_vol = df_final['Volatilidad Anual'].idxmin()
    idx_sortino = df_final['Sortino'].idxmax()
    idx_max_return = df_final['Retorno'].idxmax()

    return df_final, idx_sharpe, idx_vol, idx_sortino, idx_max_return, data

# --- INTERFAZ STREAMLIT ---
st.title("Frontera Eficiente de Markowitz Dinámica")

tickers_input = st.text_input("Introduce los tickers separados por coma, ejemplo: AAPL, MSFT, GOOGL", "PLTR, ASML, GOOGL, MC.PA, MRNA, SLB")
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
años = st.slider("Años para análisis", 1, 10, 3)
num_portfolios = st.slider("Número de carteras simuladas para la frontera", 1000, 10000, 5000, step=500)

if st.button("Calcular frontera eficiente"):
    with st.spinner("Calculando frontera eficiente..."):
        try:
            df_frontera, idx_sharpe, idx_vol, idx_sortino, idx_max_return, data = frontera_markowitz_streamlit(tickers, num_portfolios, años)
            st.success("Cálculo finalizado!")

            # Datos SP500 para comparación
            sp500_ticker = '^GSPC'
            sp500_data = get_data(sp500_ticker, data.index.min(), data.index.max())
            sp500_returns = sp500_data.pct_change().dropna()
            sp500_mean_return = sp500_returns.mean().values[0] * 252
            sp500_volatility = sp500_returns.std().values[0] * np.sqrt(252)
            sp500_sharpe = sp500_mean_return / sp500_volatility

            # --- Gráfico frontera eficiente (solo máximo Sharpe y mínima volatilidad) ---
            fig1 = go.Figure()

            fig1.add_trace(go.Scatter(
                x=df_frontera['Volatilidad Anual'],
                y=df_frontera['Retorno Anual'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=df_frontera['Sharpe'],
                    colorscale='Viridis',
                    colorbar=dict(title='Sharpe Ratio'),
                    showscale=True,
                    opacity=0.6,
                ),
        
                hovertemplate=(
                    'Volatilidad: %{x:.2%}<br>' +
                    'Retorno: %{y:.2%}<br>' +
                    'Sharpe: %{marker.color:.2f}<br>' +
                    '<extra></extra>'
                ),
                showlegend=False
            ))

            # Máximo Sharpe (solo en gráfico)
            fig1.add_trace(go.Scatter(
                x=[df_frontera.loc[idx_sharpe, 'Volatilidad Anual']],
                y=[df_frontera.loc[idx_sharpe, 'Retorno Anual']],
                mode='markers+text',
                marker=dict(color='red', size=18, symbol='star'),
                text=['Máximo Sharpe'],
                textposition='top center',
                showlegend=False,
                hovertemplate=(
                    'Máximo Sharpe<br>' +
                    'Volatilidad: %{x:.2%}<br>' +
                    'Retorno: %{y:.2%}<br>' +
                    '<extra></extra>'
                )
            ))

            # Mínima Volatilidad (solo en gráfico)
            fig1.add_trace(go.Scatter(
                x=[df_frontera.loc[idx_vol, 'Volatilidad Anual']],
                y=[df_frontera.loc[idx_vol, 'Retorno Anual']],
                mode='markers+text',
                marker=dict(color='blue', size=16, symbol='x'),
                text=['Mínima Volatilidad'],
                textposition='bottom center',
                showlegend=False,
                hovertemplate=(
                    'Mínima Volatilidad<br>' +
                    'Volatilidad: %{x:.2%}<br>' +
                    'Retorno: %{y:.2%}<br>' +
                    '<extra></extra>'
                )
            ))

            fig1.update_layout(
                title='Frontera Eficiente - Carteras simuladas',
                xaxis_title='Volatilidad Anual',
                yaxis_title='Retorno Anual',
                template='plotly_white',
                width=900,
                height=600,
                hovermode='closest'
            )
            st.plotly_chart(fig1, use_container_width=True)

            # --- Tabla con carteras clave: Sharpe, Volatilidad, Sortino, Máximo Retorno ---
            st.write("### Cartera con métricas clave")

            def mostrar_cartera(titulo, cartera):
                st.subheader(titulo)
                cols = st.columns(4)
                cols[0].metric("Retorno Anual", f"{cartera['Retorno Anual']:.2%}")
                cols[1].metric("Volatilidad Anual", f"{cartera['Volatilidad Anual']:.2%}")
                cols[2].metric("Sharpe Ratio", f"{cartera['Sharpe']:.2f}")
                cols[3].metric("Sortino Ratio", f"{cartera['Sortino']:.2f}")

                # Pesos en porcentaje redondeados
                pesos = (cartera[tickers] * 100).round(2)
                pesos = pesos[pesos > 0]  # Filtrar los activos con peso 0
                # Tabla compacta y moderna
                st.markdown("### 🧮 Distribución de Activos (%)")
                st.dataframe(
                    pesos.to_frame(name="Peso (%)").T.style.format("{:.2f}")

                )
                # Gráfico de pastel moderno
                fig = px.pie(
                    names=pesos.index,
                    values=pesos.values,
                    title="",
                    hole=0.3,
                    color_discrete_sequence=px.colors.qualitative.Pastel  # cambio clave
                )
                fig.update_traces(
                    textinfo="none",
                    hovertemplate="%{label}: %{percent:.1%} (%{value:.2f}%)<extra></extra>",
                    marker=dict(line=dict( width=2))  # separador blanco
                )
                fig.update_layout(
                    showlegend=True,
                    legend_title="Activos",
                    legend=dict(orientation="h", y=-0.2),
                    margin=dict(t=20, b=20, l=0, r=0),
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True, key=f"pie_{titulo}")

            mostrar_cartera("📌 Cartera con Mayor Sharpe", df_frontera.loc[idx_sharpe])
            mostrar_cartera("📌 Cartera con Menor Volatilidad", df_frontera.loc[idx_vol])
            mostrar_cartera("📌 Cartera con Mayor Sortino", df_frontera.loc[idx_sortino])
            mostrar_cartera("📌 Cartera con Máximo Retorno", df_frontera.loc[idx_max_return])

            # --- Segundo gráfico: frontera + SP500 sin leyenda, pero con marcas ---
            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                x=df_frontera['Volatilidad Anual'],
                y=df_frontera['Retorno Anual'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=df_frontera['Sharpe'],
                    colorscale='Viridis',
                    showscale=False,
                    opacity=0.6,
                ),
                name='Portafolios Simulados',
                hovertemplate=(
                    'Volatilidad: %{x:.2%}<br>' +
                    'Retorno: %{y:.2%}<br>' +
                    'Sharpe: %{marker.color:.2f}<br>' +
                    '<extra></extra>'
                )
            ))

            fig2.add_trace(go.Scatter(
                x=[df_frontera.loc[idx_sharpe, 'Volatilidad Anual']],
                y=[df_frontera.loc[idx_sharpe, 'Retorno Anual']],
                mode='markers+text',
                marker=dict(color='red', size=18, symbol='star'),
                text=['Máximo Sharpe'],
                textposition='top center',
                showlegend=False,
                hovertemplate=(
                    'Máximo Sharpe<br>' +
                    'Volatilidad: %{x:.2%}<br>' +
                    'Retorno: %{y:.2%}<br>' +
                    '<extra></extra>'
                )
            ))

            fig2.add_trace(go.Scatter(
                x=[df_frontera.loc[idx_vol, 'Volatilidad Anual']],
                y=[df_frontera.loc[idx_vol, 'Retorno Anual']],
                mode='markers+text',
                marker=dict(color='blue', size=16, symbol='x'),
                text=['Mínima Volatilidad'],
                textposition='bottom center',
                showlegend=False,
                hovertemplate=(
                    'Mínima Volatilidad<br>' +
                    'Volatilidad: %{x:.2%}<br>' +
                    'Retorno: %{y:.2%}<br>' +
                    '<extra></extra>'
                )
            ))

            fig2.add_trace(go.Scatter(
                x=[sp500_volatility],
                y=[sp500_mean_return],
                mode='markers+text',
                marker=dict(color='green', size=16, symbol='diamond'),
                text=['S&P 500'],
                textposition='bottom right',
                showlegend=False,
                hovertemplate=(
                    f'S&P 500<br>Volatilidad: {sp500_volatility:.2%}<br>Retorno: {sp500_mean_return:.2%}<br>Sharpe: {sp500_sharpe:.2f}<extra></extra>'
                )
            ))

            fig2.update_layout(
                title='Frontera Eficiente + S&P 500',
                xaxis_title='Volatilidad Anual',
                yaxis_title='Retorno Anual',
                template='plotly_white',
                width=900,
                height=600,
                hovermode='closest',
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Calculamos los retornos diarios de las carteras óptimas
            returns_daily = data.pct_change().dropna()

            weights_sharpe = df_frontera.loc[idx_sharpe, tickers].values
            weights_vol = df_frontera.loc[idx_vol, tickers].values
            weights_sortino = df_frontera.loc[idx_sortino, tickers].values 

            returns_sharpe = (returns_daily * weights_sharpe).sum(axis=1)
            returns_vol = (returns_daily * weights_vol).sum(axis=1)
            returns_sortino = (returns_daily * weights_sortino).sum(axis=1)

            returns_sp500 = sp500_returns.squeeze()  # Serie simple

            # Sincronizamos índices para evitar errores
            returns_sp500 = returns_sp500.reindex(returns_sharpe.index).dropna()
            returns_sharpe = returns_sharpe.loc[returns_sp500.index]
            returns_vol = returns_vol.loc[returns_sp500.index]
            returns_sortino = returns_sortino.loc[returns_sp500.index] 

            # Calculamos evolución acumulada partiendo de 1000€
            start_capital = 1000
            evolution_sharpe = start_capital * (1 + returns_sharpe).cumprod()
            evolution_vol = start_capital * (1 + returns_vol).cumprod()
            evolution_sortino = start_capital * (1 + returns_sortino).cumprod()  
            evolution_sp500 = start_capital * (1 + returns_sp500).cumprod()

            fig3 = go.Figure()

            fig3.add_trace(go.Scatter(
                x=evolution_sharpe.index,
                y=evolution_sharpe.values,
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.8)'),
                name='Cartera Máximo Sharpe'
            ))

            fig3.add_trace(go.Scatter(
                x=evolution_vol.index,
                y=evolution_vol.values,
                mode='lines',
                line=dict(color='rgba(0, 0, 255, 0.8)'),
                name='Cartera Mínima Volatilidad'
            ))

            fig3.add_trace(go.Scatter(
                x=evolution_sortino.index,
                y=evolution_sortino.values,
                mode='lines',
                line=dict(color='rgba(255, 165, 0, 0.8)'),
                name='Cartera Máximo Sortino'
            ))

            fig3.add_trace(go.Scatter(
                x=evolution_sp500.index,
                y=evolution_sp500.values,
                mode='lines',
                line=dict(color='rgba(0, 128, 0, 0.8)'),
                name='SP500'
            ))

            fig3.update_layout(
                title='Evolución de 1000€ invertidos en carteras y SP500',
                xaxis_title='Fecha',
                yaxis_title='Valor (€)',
                template='plotly_white',
                width=900,
                height=600
            )

            st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:
            st.error(f"Ha ocurrido un error: {e}")
            # ...

        except Exception as e:
            st.error(f"Error al calcular la frontera eficiente: {e}")

