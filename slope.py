#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from PIL import Image
from mpmath import mp, cos, sin, tan, radians, sqrt, exp, cosh, sinh, invertlaplace
import matplotlib.pyplot as plt
from numpy import linspace, array, polyfit
import torch
import joblib

st.set_page_config(layout="wide")   # wide layout to cover most of the page width
col1, col2, col3 = st.columns([0.4, 0.2, 0.4])   # three columns, covering 40%, 20%, 40% of the page width

with col1:
    image = Image.open('slope.png')
    st.image(image)
    button = st.button('Evaluate', use_container_width = True)

with col2:
    t_in = st.text_input('t (from 10 to 1000)', '10')
    ε_in = st.text_input('ε (from 0.05 to 0.3)', '0.1')
    κ_in = st.text_input('κ (from 0.1 to 0.5)', '0.1')
    λ_in = st.text_input('λ (from 10 to 100)', '10')
    θ_in = st.text_input('θ (deg, from -3 to 3)', '1')
    L_in = st.text_input('L (from 1 to 10)', '5')
    if button:
        t = float(t_in)
        ε = float(ε_in)
        κ = float(κ_in)
        λ = float(λ_in)
        θ = float(θ_in)
        L = float(L_in)
    else:
        input()   # wait for user input

    # ==============================
    # PART A: "EXPERIMENT" EMULATION
    # Use the defined parameters to evaluate hf and hm profiles, 
    # and the aquifer specific discharge profile over time

    # mpmath stuff
    mp.dps = 15   # precision
    mp.pretty = True   # avoid lots of digits
    cosθ = cos(radians(θ))
    sinθ = sin(radians(θ))
    tanθ = tan(radians(θ))
    # end of mpmath stuff
    V = sinθ
    nx = 20   # 20 points should be enough to capture the resulting hf and hm curves
    xs = linspace(0, L, num = nx)
    # t values that will provide the rate Qs* vs. time curve 
    # (starts from t=20, to avoid large decrease after t=10)
    ts_rate = [20, 30, 50, 75, 100, 200, 300, 400, 700, 1000]
    # x values to capture the hf derivarive at x=0
    nx_rate = 11   # 11 points from x=0 to x=0.1 should be enough
    xs_rate = linspace(0, 0.1, num = nx_rate)

    # Akylas and Koussis (2007) approach for Da (must import sympy)
    # Da = sp.symbols('Da')
    # Da = sp.nsolve(Da - V * (L - tanθ * L**2 / 2) / 
    #              (1 - sp.exp(-L * V / Da)), Da, (0.1, 2.))
 
    # Basha (2021) approach for Da (some 2% difference with Akylas and Koussis)
    Da = cosθ * (1 - tanθ * L / 2)

    # helpers
    a = -V / (2 * Da)
    Ψ = lambda p: p + λ * κ * Da * p / (κ * Da + λ * p)
    c = lambda p: sqrt(a**2 + Ψ(p)/Da)

    # hf and hm evaluation
    hf = []
    hm = []

    for x in xs:
        # Laplace transform of hf
        Lhf = lambda p: exp(-V*x/Da) / p - (ε / p) * exp(a*x) * \
                        ((c(p)*cosh(c(p)*L - c(p)*x) - \
                        a*sinh(c(p)*L - c(p)*x)) / (c(p)*cosh(c(p)*L) - a*sinh(c(p)*L)))
        # Laplace transform of hm
        Lhm = lambda p: λ * exp(-V*x/Da) / (λ*p + κ*Da) + κ * Da * Lhf(p) / (λ*p + κ*Da)
        hf.append([float(invertlaplace(Lhf, t))])
        hm.append([float(invertlaplace(Lhm, t))])
    
    # evaluate rate for each time value ts_rate
    Qs = []
    for t_rate in ts_rate:
        # hf derivative at x=0 evaluation
        hfs_rate = []   # list of hf values for the xs values
        for x_rate in xs_rate:
            # Laplace transform of hf
            Lhf = lambda p: exp(-V*x_rate/Da) / p - (ε / p) * exp(a*x_rate) * \
                            ((c(p)*cosh(c(p)*L - c(p)*x_rate) - \
                            a*sinh(c(p)*L - c(p)*x_rate)) / (c(p)*cosh(c(p)*L) - a*sinh(c(p)*L)))
            hf_rate = invertlaplace(Lhf, t_rate)
            hfs_rate.append(float(hf_rate))

        # find derivative at x=0, approximating hf by line
        slope = polyfit(xs_rate, hfs_rate, 1)[0]
                        
        # evaluate rate Qs* at x=0
        hf0 = hfs_rate[0]
        Qs_value = -(Da * slope + hf0 * sinθ) / cosθ
        Qs.append(float(Qs_value))

    # approximate curve of rates Qs by a third-degree polynomial
    [Qs_a, Qs_b, Qs_c, Qs_d] = polyfit(ts_rate, Qs, 3)
 
    # =====================
    # PART B: MODELING PART
    # Use the "geometric" parameters (ε, L, θ) and the Qs coefficients
    # to evaluate κ and λ from neural network. 
    # Then, use ε, L, θ, and evaluated κ and λ to obtain modeled hf and hm 

    X = array([ε, θ, L, Qs_a, Qs_b, Qs_c, Qs_d]).reshape(1, 7)
    # load scaling parameters
    x_scaler = joblib.load("./x_scaler.joblib")
    y_scaler = joblib.load("./y_scaler.joblib")
    # scale the input data (do not use .fit_transform here)
    X = x_scaler.transform(X)
    # scaling done, convert to tensor for the model
    X = torch.FloatTensor(X)
    # load the model
    model = torch.load('Analytical.pth')
    model.eval()   # evaluate model predictions for the input
    y = model(X).detach().numpy().reshape(1, 2)
    y = y_scaler.inverse_transform(y) 
    κ = float(y[0][0])
    λ = float(y[0][1])

    # modeled hf and hm evaluation
    hf_model = []
    hm_model = []

    for x in xs:
        # Laplace transform of hf
        Lhf = lambda p: exp(-V*x/Da) / p - (ε / p) * exp(a*x) * \
                        ((c(p)*cosh(c(p)*L - c(p)*x) - \
                        a*sinh(c(p)*L - c(p)*x)) / (c(p)*cosh(c(p)*L) - a*sinh(c(p)*L)))
        # Laplace transform of hm
        Lhm = lambda p: λ * exp(-V*x/Da) / (λ*p + κ*Da) + κ * Da * Lhf(p) / (λ*p + κ*Da)
        hf_model.append([float(invertlaplace(Lhf, t))])
        hm_model.append([float(invertlaplace(Lhm, t))])
    
# ======
# OUTPUT
with col3:
    fig = plt.figure()
    plt.plot(xs, hf, 'bo', label = 'hf exp')
    plt.plot(xs, hm, 'ro', label = 'hm exp')
    plt.ylabel('hf, hm')
    plt.xlabel('x')
    plt.plot(xs, hf_model, 'b-', mfc = 'none', label = 'hf model')
    plt.plot(xs, hm_model, 'r-', mfc = 'none', label = 'hm model')

    # show the plot
    plt.legend()
    plt.grid()
    st.pyplot(fig)
    
    # report modeled κ, λ
    st.write("Evaluated κ = %.3f" % κ)
    st.write("Evaluated λ = %.3f" % λ)
 
