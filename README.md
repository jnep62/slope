# slope
Code of a Streamlit Python applet to support the publication "On the flow behaviour of unconfined dual porosity aquifers with sloping base" by Konstantinos N. Moutsopoulos, John N. E. Papaspyros, Antonis D. Koussis, Frédérick Delay, and Marwan Fahs, submitted to Advances in Water Research.

First, collection of experimental values is emulated: the user defines values for the problem parameters (time, ε, κ, λ, θ, and L), and clicks the "Evaluate" button. The model provides the profiles of hf and hm, and the specific discharge of the aquifer. In turn, this specific discharge is fitted by a third-degree polynomial for time larger than t*=20 (to avoid steep gradients close to t*=0), and the four coefficients of this polynomial are determined. This concludes the "experimental" emulation.

Next, there is a modeling phase. The three "geometrical" parameters of the field (ε, θ, L) and the four coefficients of the polynomial are used as inputs of a pytorch neural network that delivers approximations for the "double porosity" parameters κ and λ. Finally, these values of κ and λ, together with the original values of t, ε, θ, and L, are used by the model to provide "modeled" profiles of hf and hm. All profiles are plotted. The values of estimated κ and λ are also reported.

The agreement between "experimental" and modeled profiles is nice, implying that specific discharge can be used to provide values of κ and λ in practical applications.

