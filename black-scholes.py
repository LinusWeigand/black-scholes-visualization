import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm

st.set_page_config(
    page_title="A Journey Through Option Pricing",
    layout="centered",
)

st.title("A Journey Through Option Pricing")

st.sidebar.header("Core Model Parameters")
strike_price = st.sidebar.slider("Strike Price ($)", 40.0, 60.0, 50.0, 0.5)
volatility_param = st.sidebar.slider("Volatility (%)", 10, 60, 30, 1)
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.25) / 100.0

def create_layout(title, xaxis_title, yaxis_title, zaxis_title):
    """Creates a consistent layout for all 3D plots."""
    return go.Layout(
        title={'text': title, 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        scene=dict(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            zaxis_title=zaxis_title,
            camera=dict(eye=dict(x=1.9, y=-1.9, z=1.3)),
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        height=600,
        margin=dict(l=40, r=40, b=50, t=80)
    )

x_underlying = np.linspace(20, 80, 80)
y_time = np.linspace(1, 365, 80)
X_std, Y_std = np.meshgrid(x_underlying, y_time)
time_scaling_std = Y_std / 365.0
volatility_std = volatility_param / 100.0
epsilon = 1e-9

d1_std = (np.log(X_std / strike_price) + (risk_free_rate + 0.5 * volatility_std ** 2) * time_scaling_std) / (volatility_std * np.sqrt(time_scaling_std) + epsilon)
d2_std = d1_std - volatility_std * np.sqrt(time_scaling_std)


st.header("1. Long Call Option Price")
st.markdown("This surface shows the theoretical price of a **long call option** based on the underlying asset's price and the time remaining until expiration. It represents the smoothed-out present value of the final payoff.")

Z_price_call = (X_std * norm.cdf(d1_std) - strike_price * np.exp(-risk_free_rate * time_scaling_std) * norm.cdf(d2_std))
fig_price_call = go.Figure(data=[go.Surface(z=Z_price_call, x=X_std, y=Y_std, colorscale='plasma', cmin=0)])
fig_price_call.update_layout(create_layout('Long Call Price', 'Underlying Price ($)', 'Days to Expiration', 'Option Price ($)'))
st.plotly_chart(fig_price_call, use_container_width=True)


st.header("2. Long Put Option Price (vs. Volatility & Time)")
st.markdown("This plot offers a different perspective, showing the price of a **long put option** as a function of **volatility** and time. Higher volatility increases the chance of the stock finishing below the strike, thus increasing the put's value.")

put_plot_underlying = st.slider("Underlying Price for Put Plot ($)", 20.0, 80.0, 45.0, 0.5)
y_vol = np.linspace(5, 80, 80)
x_time_put = np.linspace(1, 365, 80)
X_put, Y_put = np.meshgrid(x_time_put, y_vol)

time_scaling_put = X_put / 365.0
volatility_put = Y_put / 100.0
d1_put = (np.log(put_plot_underlying / strike_price) + (risk_free_rate + 0.5 * volatility_put ** 2) * time_scaling_put) / (volatility_put * np.sqrt(time_scaling_put) + epsilon)
d2_put = d1_put - volatility_put * np.sqrt(time_scaling_put)

Z_price_put = (strike_price * np.exp(-risk_free_rate * time_scaling_put) * norm.cdf(-d2_put) - put_plot_underlying * norm.cdf(-d1_put))

fig_price_put = go.Figure(data=[go.Surface(z=Z_price_put, x=X_put, y=Y_put, colorscale='magma', cmin=0)])
fig_price_put.update_layout(create_layout('Long Put Price', 'Days to Expiration', 'Volatility (%)', 'Option Price ($)'))
st.plotly_chart(fig_price_put, use_container_width=True)


st.header("3. Long Call / Put Gamma")
st.markdown("Gamma measures the rate of change of Delta. It's identical for both calls and puts. A high Gamma indicates that the option's directional exposure (Delta) is highly sensitive to moves in the underlying asset.")
Z_gamma = norm.pdf(d1_std) / (X_std * volatility_std * np.sqrt(time_scaling_std) + epsilon)
fig_gamma = go.Figure(data=[go.Surface(z=Z_gamma, x=X_std, y=Y_std, colorscale='inferno')])
fig_gamma.update_layout(create_layout('Long Gamma', 'Underlying Price ($)', 'Days to Expiration', 'Gamma'))
st.plotly_chart(fig_gamma, use_container_width=True)

st.header("4. Short Call / Put Gamma")
st.markdown("For a short option position, Gamma is simply the negative of the long position. A seller of an option has **negative Gamma**, meaning their directional exposure worsens as the underlying moves against them. This is a key risk for option sellers.")
Z_short_gamma = -Z_gamma
fig_short_gamma = go.Figure(data=[go.Surface(z=Z_short_gamma, x=X_std, y=Y_std, colorscale='inferno_r')])
fig_short_gamma.update_layout(create_layout('Short Gamma', 'Underlying Price ($)', 'Days to Expiration', 'Gamma'))
st.plotly_chart(fig_short_gamma, use_container_width=True)


st.header("5. Long Call / Put Vega")
st.markdown("Vega measures sensitivity to a 1% change in volatility and is identical for calls and puts. Higher uncertainty increases the chance of a large price move, making the option more valuable. Vega is highest for at-the-money options with a long time until expiration.")
Z_vega = X_std * norm.pdf(d1_std) * np.sqrt(time_scaling_std) * 0.01
fig_vega = go.Figure(data=[go.Surface(z=Z_vega, x=X_std, y=Y_std, colorscale='viridis')])
fig_vega.update_layout(create_layout('Long Vega', 'Underlying Price ($)', 'Days to Expiration', 'Vega'))
st.plotly_chart(fig_vega, use_container_width=True)


st.header("6. Long Call / Put Vanna")
st.markdown("""
Vanna is a **second-order Greek** that measures how an option's Delta changes in response to a change in **volatility**. It's also the sensitivity of Vega to a change in the underlying price.
- **Positive Vanna:** For a call, when volatility increases, Vanna tells you that your Delta will increase (the option becomes more sensitive to price moves).
- **Negative Vanna:** For a put, when volatility increases, your Delta becomes less negative (moves closer to zero).
""")
Z_vanna = -norm.pdf(d1_std) * d2_std / volatility_std
fig_vanna = go.Figure(data=[go.Surface(z=Z_vanna, x=X_std, y=Y_std, colorscale='twilight')])
fig_vanna.update_layout(create_layout('Vanna', 'Underlying Price ($)', 'Days to Expiration', 'Vanna'))
st.plotly_chart(fig_vanna, use_container_width=True)

st.header("7. Long Call / Put Option Zomma")
st.markdown("""
Zomma is another **second-order Greek**, measuring the rate of change of **Gamma** with respect to a change in **volatility**. It essentially tells you how 'sticky' your Gamma is. A positive Zomma means that as volatility increases, your Gamma also increases, which is typically favorable for a long option holder.
""")
Z_zomma = Z_gamma * ((d1_std * d2_std - 1) / (volatility_std + epsilon))
fig_zomma = go.Figure(data=[go.Surface(z=Z_zomma, x=X_std, y=Y_std, colorscale='cividis')])
fig_zomma.update_layout(create_layout('Zomma', 'Underlying Price ($)', 'Days to Expiration', 'Zomma'))
st.plotly_chart(fig_zomma, use_container_width=True)

