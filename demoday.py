import streamlit as st
import pandas as pd
import numpy as np
import csv
import io
import warnings

import plotly.express as px
import plotly.graph_objects as go

from scipy.interpolate import griddata

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer
from pymoo.core.callback import Callback
from pymoo.core.mixed import (
    MixedVariableSampling,
    MixedVariableMating,
    MixedVariableDuplicateElimination
)
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(layout="wide")

st.markdown(
    """
    <h1 style='text-align: center;'> Smart Offhore Design Platform</h1>
    <h4 style='text-align: center; color: gray;'>
    Reducing cost & risk for floating offshore wind
    </h4>
    """,
    unsafe_allow_html=True
)
st.divider()

st.markdown("""
Welcome to the Mooring System Optimization Platform.

To begin, please:
1. Define the ranges for your decision variables using the panel on the left.
2. Upload your dataset used to train the surrogate model.
3. Click 'Run Optimization' to execute the optimization and view results.
""")

# --------------------------------------------------
# SIDEBAR – PARAMETERS
# --------------------------------------------------
st.sidebar.header("⚙️ Design Variable Ranges")

param_ranges = {
    "chain_length_min": int(st.sidebar.number_input("Min Chain Length (m)", 200)),
    "chain_length_max": int(st.sidebar.number_input("Max Chain Length (m)", 350)),
    "chain_mass_min": int(st.sidebar.number_input("Min Chain Mass (kg/m)", 107)),
    "chain_mass_max": int(st.sidebar.number_input("Max Chain Mass (kg/m)", 285)),
    "pretension_min": int(st.sidebar.number_input("Min Pretension (kN)", 75)),
    "pretension_max": int(st.sidebar.number_input("Max Pretension (kN)", 225)),
    "buoy_dist_min": int(st.sidebar.number_input("Min Buoy-Hull Distance (m)", 50)),
    "buoy_dist_max": int(st.sidebar.number_input("Max Buoy-Hull Distance (m)", 100)),
}

# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------
st.subheader("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV dataset used to train the surrogate model",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset uploaded successfully")

    with st.expander("Preview dataset"):
        st.dataframe(df.head())

    # --------------------------------------------------
    # COST & CONSTRAINT FUNCTIONS
    # --------------------------------------------------
    chain_price = 23.80
    rope_price = 15.24

    def cost_function(chain_length, pretension, chain_mass):
        return (chain_length * chain_mass * chain_price) + (pretension * rope_price)

    def constraints(chain_length, chain_mass, pretension, buoy_dist):
        X = df[['Chain_lengths', 'Chain_mass', 'PreTensions', 'Buoy_Hull_Distances']].values
        y = df['Output_mean_X'].values

        interp = griddata(
            X, y,
            [[chain_length, chain_mass, pretension, buoy_dist]],
            method="linear"
        )

        if interp is None or np.isnan(interp[0]):
            return 0.0

        return float(interp[0])

    # --------------------------------------------------
    # OPTIMIZATION PROBLEM
    # --------------------------------------------------
    class FloatingWindTurbineProblem(ElementwiseProblem):
        def __init__(self):
            vars = {
                "Chain_lengths": Integer(bounds=(param_ranges["chain_length_min"], param_ranges["chain_length_max"])),
                "Chain_mass": Integer(bounds=(param_ranges["chain_mass_min"], param_ranges["chain_mass_max"])),
                "PreTensions": Integer(bounds=(param_ranges["pretension_min"], param_ranges["pretension_max"])),
                "Buoy_Hull_Distances": Integer(bounds=(param_ranges["buoy_dist_min"], param_ranges["buoy_dist_max"])),
            }
            super().__init__(vars=vars, n_obj=2, n_ieq_constr=1)

        def _evaluate(self, x, out, *args, **kwargs):
            cost = cost_function(
                float(x["Chain_lengths"]),
                float(x["PreTensions"]),
                float(x["Chain_mass"])
            )

            offset = constraints(
                float(x["Chain_lengths"]),
                float(x["Chain_mass"]),
                float(x["PreTensions"]),
                float(x["Buoy_Hull_Distances"])
            )

            out["F"] = [cost, -offset]
            out["G"] = -offset - 12

    # --------------------------------------------------
    # CALLBACK
    # --------------------------------------------------
    class MyCallback(Callback):
        def __init__(self):
            super().__init__()
            self.data["best_cost"] = []
            self.data["best_offset"] = []

        def notify(self, algorithm):
            F = algorithm.pop.get("F")
            self.data["best_cost"].append(F[:, 0].min())
            self.data["best_offset"].append(F[:, 1].max())

    # --------------------------------------------------
    # RUN OPTIMIZATION
    # --------------------------------------------------
    st.subheader(" Run Optimization")

    if st.button("Run Optimization"):
        with st.spinner("SOD is exploring thousands of designs..."):

            problem = FloatingWindTurbineProblem()

            algorithm = NSGA2(
                pop_size=10,
                sampling=MixedVariableSampling(),
                mating=MixedVariableMating(
                    eliminate_duplicates=MixedVariableDuplicateElimination()
                ),
                eliminate_duplicates=MixedVariableDuplicateElimination()
            )

            termination = RobustTermination(
                MultiObjectiveSpaceTermination(tol=0.05, n_skip=10),
                period=25
            )

            callback = MyCallback()

            res = minimize(
                problem,
                algorithm,
                seed=1,
                termination=termination,
                callback=callback,
                save_history=True
            )

        st.success(" Optimization completed")


        # -------------------------------
        # Prepare population history for animation
        # -------------------------------



        pop_history = []

        for algo in res.history:
            gen_pop = []
            X_gen = algo.pop.get("X")  # list of dicts

            for ind in X_gen:
                gen_pop.append({
                    "Chain_lengths": ind["Chain_lengths"],
                    "Chain_mass": ind["Chain_mass"],
                    "Buoy_Hull_Distances": ind["Buoy_Hull_Distances"],
                    "PreTensions": ind["PreTensions"]
                })

            pop_history.append(gen_pop)


        # -------------------------------
        # 4D Decision Variables animation
        # -------------------------------
        import plotly.graph_objects as go
       

        st.subheader(" How the SOD Explores the Decision Variables")

        frames = []

        # Collect global ranges (important for smooth animation)
        all_x, all_y, all_z, all_c = [], [], [], []

        for gen in pop_history:
            for ind in gen:
                all_x.append(ind["Chain_lengths"])
                all_y.append(ind["Chain_mass"])
                all_z.append(ind["Buoy_Hull_Distances"])
                all_c.append(ind["PreTensions"])

        # Build animation frames
        for gen_idx, gen in enumerate(pop_history):

            x = [ind["Chain_lengths"] for ind in gen]
            y = [ind["Chain_mass"] for ind in gen]
            z = [ind["Buoy_Hull_Distances"] for ind in gen]
            c = [ind["PreTensions"] for ind in gen]

            frames.append(
                go.Frame(
                    data=[
                        go.Scatter3d(
                            x=x,
                            y=y,
                            z=z,
                            mode="markers",
                            marker=dict(
                                size=6,
                                color=c,
                                colorscale="Turbo",
                                opacity=0.75,
                                colorbar=dict(title="Pretension (kN)")
                            )
                        )
                    ],
                    name=str(gen_idx)
                )
            )

        # Initial figure
        fig = go.Figure(
            data=frames[0].data,
            frames=frames,
            layout=go.Layout(
                title="SOD Exploring 4-D Decision Variables",
                height=600,
                scene=dict( aspectmode="cube",
                    xaxis=dict(
                        title="Chain Length (m)",
                        range=[min(all_x)*0.95, max(all_x)*1.05]
                    ),
                    yaxis=dict(
                        title="Chain Mass (kg/m)",
                        range=[min(all_y)*0.95, max(all_y)*1.05]
                    ),
                    zaxis=dict(
                        title="Buoy–Hull Distance (m)",
                        range=[min(all_z)*0.95, max(all_z)*1.05]
                    ),
                ),
                template="plotly_white",
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [{
                        "label": "▶ Play Optimization",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": 500, "redraw": True},
                            "fromcurrent": True
                        }]
                    }]
                }]
            )
        )

        st.plotly_chart(fig, use_container_width=True)



        # -------------------------------
        # Result Design Space animation
        # -------------------------------


        st.subheader(" How the SOD Explores the Design Space")

        frames = []
        all_points = []

        for gen, h in enumerate(res.history):
            F_gen = h.pop.get("F")
            df_gen = pd.DataFrame(F_gen, columns=["Cost", "Neg Offset"])
            df_gen["X-Offset"] = -df_gen["Neg Offset"]
            df_gen["Generation"] = gen
            all_points.append(df_gen)

        df_anim = pd.concat(all_points, ignore_index=True)

        fig = px.scatter(
            df_anim,
            x="Cost",
            y="X-Offset",
            animation_frame="Generation",
            range_x=[df_anim["Cost"].min()*0.95, df_anim["Cost"].max()*1.05],
            range_y=[df_anim["X-Offset"].min()*0.95, df_anim["X-Offset"].max()*1.05],
            title="SOD Exploring Mooring Designs Over Time",
            labels={"X-Offset": "Platform Offset (m)"},
        )

        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(template="plotly_white", height=500)

        st.plotly_chart(fig, use_container_width=True)








        # --------------------------------------------------
        # STATISTICS
        # --------------------------------------------------
        n_generations = len(res.history)
        n_scenarios = sum(len(h.pop) for h in res.history)

        st.subheader(" Optimization Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("Scenarios Evaluated", f"{n_scenarios:,}")
        c2.metric("Iterations", n_generations)
        c3.metric("Scenarios per Iterations", algorithm.pop_size)

        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------
        df_results = pd.DataFrame(res.F, columns=["Cost", "Neg Offset"])
        df_results["X-Offset"] = -df_results["Neg Offset"]
        df_results.drop(columns="Neg Offset", inplace=True)

        best = df_results.loc[df_results["Cost"].idxmin()]

        st.subheader(" Best Design Found")

        c1, c2 = st.columns(2)
        c1.metric("Minimum Cost", f"${best['Cost']:,.0f}")
        c2.metric("Platform Offset", f"{best['X-Offset']:.2f} m")

        baseline_cost = 1_250_000  # example baseline
        best_cost = df_results['Cost'].min()
        st.metric(
            "Best Design vs Baseline",
            f"${best_cost:,.0f}",
            delta=f"-${baseline_cost - best_cost:,.0f}"
        )

        # --------------------------------------------------
        # PARETO FRONT
        # --------------------------------------------------
        st.subheader(" Cost vs Performance Trade-Off")

        fig = px.scatter(
            df_results,
            x="Cost",
            y="X-Offset",
            color="Cost",
            color_continuous_scale="Viridis",
            title="Pareto-Optimal Mooring Designs"
        )

        fig.update_traces(marker=dict(size=10))
        fig.update_layout(template="plotly_white", height=500)

        st.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------
        # CONVERGENCE
        # --------------------------------------------------
        # conv_df = pd.DataFrame({
        #     "Generation": range(len(callback.data["best_cost"])),
        #     "Best Cost": callback.data["best_cost"]
        # })

        # fig2 = px.line(
        #     conv_df,
        #     x="Generation",
        #     y="Best Cost",
        #     title="Optimization Convergence"
        # )

        # fig2.update_layout(template="plotly_white")
        # st.plotly_chart(fig2, use_container_width=True)



        st.subheader(" Cost Reduction Over Optimization")

        conv_df = pd.DataFrame({
            "Generation": range(len(callback.data["best_cost"])),
            "Best Cost": callback.data["best_cost"]
        })

        fig = px.line(
            conv_df,
            x="Generation",
            y="Best Cost",
            markers=True,
            title="SOD Converging Toward Lower-Cost Designs"
        )

        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)



        # st.subheader("⚓ Visualizing the Optimized Mooring Configuration")

        # best_x = res.X[df_results["Cost"].idxmin()]

        # chain_length = best_x["Chain_lengths"]
        # buoy_dist = best_x["Buoy_Hull_Distances"]

        # fig = go.Figure()

        # fig.add_trace(go.Scatter(
        #     x=[0, 0],
        #     y=[0, chain_length],
        #     mode="lines",
        #     line=dict(width=6),
        #     name="Mooring Line"
        # ))

        # fig.add_trace(go.Scatter(
        #     x=[0],
        #     y=[chain_length - buoy_dist],
        #     mode="markers",
        #     marker=dict(size=20),
        #     name="Buoy"
        # ))

        # fig.update_layout(
        #     title="Optimized Mooring Geometry (Conceptual)",
        #     yaxis=dict(title="Depth (m)", autorange="reversed"),
        #     xaxis=dict(visible=False),
        #     height=500,
        #     template="plotly_white"
        # )

        # st.plotly_chart(fig, use_container_width=True)



        # --------------------------------------------------
        # DOWNLOADS
        # --------------------------------------------------
        st.divider()
        st.subheader("📥 Export Results")

        buffer = io.StringIO()
        df_results.to_csv(buffer, index=False)

        st.download_button(
            "Download Pareto Solutions",
            buffer.getvalue(),
            file_name="pareto_solutions.csv",
            mime="text/csv"
        )
