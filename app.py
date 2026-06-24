import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="SBJ Force Plate Data Analysis Tool", layout="wide")

st.title("SBJ Force Plate Data Analysis Tool")

st.caption(
    "This application was developed by Dr Jason Tallis, Coventry University. "
    "For technical support, enquiries, or potential research collaboration opportunities, "
    "please contact: Jason.Tallis@coventry.ac.uk"
)

st.markdown("### Overview")
st.markdown(
    """
This tool is designed to analyse standing broad jump (SBJ) force plate data and calculate:

- Body mass and body weight estimates from quiet standing force
- Movement threshold and movement onset
- Contact time, unweighting time, braking time, and propulsive time
- Vertical and horizontal impulse and power variables
- Resultant force, force angle, and force ratio metrics
- Centre of pressure (CoP) path, velocity, and AP/ML displacement
- Normalised waveform data (101 data points) for later waveform/SPM-style analysis
"""
)

st.markdown("### Instructions")
st.markdown(
    """
- Upload a **CSV or TXT** file containing **6 columns in this exact order**: `Fx, Fy, Fz, Mx, My, Mz`
- If the participant stood on the plate in the opposite horizontal orientation, tick the **Swap X/Y** checkbox
- Use the slider to select an analysis window that includes **a minimum of 1 second of quiet standing before the start of the jump**
- After reviewing the raw Fz/Fy visualisation, optionally tick **Invert Fy data for analysis** if the horizontal force direction needs reversing
- Set the **movement threshold percentage of SD** (for example: `500` means `BW - 5 × SD`)
- Click **Run Analysis**
- Review the results in the **summary table** and the **figures in tabs**
- Export:
  - **Summary CSV** for discrete outcomes
  - **Curves CSV** for 101-point time-normalised waveform data
"""
)

# ============================================================
# SESSION STATE
# ============================================================
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

# ============================================================
# ORIENTATION CHECKBOX
# ============================================================
swap_xy = st.checkbox(
    "Swap X/Y force and moment channels (use if Fx is actually Fy, and Mx is actually My)"
)

# ============================================================
# FILE UPLOAD
# ============================================================
file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])

if file is not None:

    # --------------------------------------------------------
    # LOAD FILE ROBUSTLY
    # --------------------------------------------------------
    try:
        df = pd.read_csv(file, header=None, sep=None, engine="python")
    except Exception:
        file.seek(0)
        try:
            df = pd.read_csv(file, header=None)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, header=None, delimiter=r"\s+")

    if df.shape[1] != 6:
        st.error("The uploaded file must contain exactly 6 columns in this order: Fx, Fy, Fz, Mx, My, Mz.")
        st.stop()

    df.columns = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    # --------------------------------------------------------
    # OPTIONAL RELABELLING OF X/Y CHANNELS
    # --------------------------------------------------------
    if swap_xy:
        df[["Fx", "Fy", "Mx", "My"]] = df[["Fy", "Fx", "My", "Mx"]].to_numpy()
        st.warning("X/Y channels swapped: Fx↔Fy and Mx↔My are being used for all subsequent analysis.")

    # --------------------------------------------------------
    # SAMPLING + THRESHOLD PERCENT
    # --------------------------------------------------------
    fs = st.number_input("Sampling Frequency (Hz)", min_value=1.0, value=1000.0, step=1.0)
    dt = 1.0 / fs
    df["time"] = np.arange(len(df)) * dt

    threshold_percent = st.number_input(
        "Movement threshold (% of SD subtracted from body weight; 500 = BW - 5×SD)",
        min_value=0.0,
        value=500.0,
        step=10.0
    )

    # ========================================================
    # BASE FIGURE
    # SHOWN BEFORE OPTIONAL Fy INVERSION
    # ========================================================
    base_fig = go.Figure()

    base_fig.add_trace(go.Scatter(
        x=df["time"], y=df["Fz"],
        mode="lines",
        name="Fz (N)",
        yaxis="y1"
    ))

    base_fig.add_trace(go.Scatter(
        x=df["time"], y=df["Fy"],
        mode="lines",
        name="Fy (N)",
        yaxis="y2"
    ))

    start_time, end_time = st.slider(
        "Select analysis window (include a minimum of 1 s quiet standing before the start of the jump)",
        min_value=float(df["time"].min()),
        max_value=float(df["time"].max()),
        value=(float(df["time"].min()), float(df["time"].max()))
    )

    base_fig.add_vrect(
        x0=start_time, x1=end_time,
        fillcolor="green", opacity=0.15, line_width=0
    )

    base_fig.update_layout(
        title="Raw Force-Time Curve",
        xaxis_title="Time (s)",
        yaxis=dict(title="Fz (N)"),
        yaxis2=dict(title="Fy (N)", overlaying="y", side="right"),
        legend_title="Series"
    )

    st.plotly_chart(base_fig, use_container_width=True)

    # ========================================================
    # OPTIONAL Fy INVERSION AFTER RAW DATA VISUALISATION
    # ========================================================
    invert_fy = st.checkbox(
        "Invert Fy data for analysis",
        help=(
            "Tick this if the Fy signal is in the opposite horizontal direction. "
            "If selected, all subsequent horizontal-force calculations, figures, "
            "summary outcomes, and exported curves will use inverted Fy values."
        )
    )

    if invert_fy:
        df["Fy"] = -df["Fy"]
        st.warning(
            "Fy has been inverted for all subsequent analysis, figures, summary outcomes, and exported waveform data."
        )

    if st.button("Run Analysis"):
        st.session_state.run_analysis = True

    # ========================================================
    # ANALYSIS
    # ========================================================
    if st.session_state.run_analysis:

        # ----------------------------------------------------
        # SELECTED TRIAL WINDOW
        # ----------------------------------------------------
        trial_df = df[(df["time"] >= start_time) & (df["time"] <= end_time)].copy()

        if len(trial_df) < 20:
            st.error("The selected analysis window is too short for reliable analysis.")
            st.stop()

        # ----------------------------------------------------
        # BASELINE / BODY WEIGHT / THRESHOLD
        # ----------------------------------------------------
        n_baseline = min(1000, len(trial_df))
        baseline = trial_df["Fz"].iloc[:n_baseline]

        BW = baseline.mean()       # N
        SD = baseline.std()        # N
        mass = BW / 9.81           # kg

        # CORRECTED MOVEMENT THRESHOLD USED THROUGHOUT
        threshold = BW - ((threshold_percent / 100.0) * SD)  # N

        # ----------------------------------------------------
        # CORE KINETICS / KINEMATICS
        # ----------------------------------------------------
        trial_df["net_Fz"] = trial_df["Fz"] - BW
        trial_df["accel"] = trial_df["net_Fz"] / mass

        trial_df["velocity"] = np.cumsum(trial_df["accel"]) * dt
        trial_df["velocity"] -= trial_df["velocity"].iloc[0]

        trial_df["disp"] = np.cumsum(trial_df["velocity"]) * dt
        trial_df["power"] = trial_df["net_Fz"] * trial_df["velocity"]

        trial_df["accel_Fy"] = trial_df["Fy"] / mass
        trial_df["vel_Fy"] = np.cumsum(trial_df["accel_Fy"]) * dt
        trial_df["vel_Fy"] -= trial_df["vel_Fy"].iloc[0]

        trial_df["power_Fy"] = trial_df["Fy"] * trial_df["vel_Fy"]

        # Resultant based on net vertical and raw/inverted horizontal Fy
        trial_df["resultant"] = np.sqrt(trial_df["net_Fz"]**2 + trial_df["Fy"]**2)

        # Angle based on raw Fz and raw/inverted Fy
        trial_df["angle"] = np.degrees(np.arctan2(trial_df["Fy"], trial_df["Fz"]))

        # Force ratio = resultant / net vertical force
        trial_df["ratio"] = np.where(
            np.abs(trial_df["net_Fz"]) > 1e-8,
            trial_df["resultant"] / trial_df["net_Fz"],
            np.nan
        )

        # ----------------------------------------------------
        # EVENT DETECTION
        # ----------------------------------------------------
        # Onset = first run of consecutive samples below threshold
        min_duration_below_threshold = 0.02   # 20 ms
        n_consecutive = max(1, int(min_duration_below_threshold * fs))

        below_thresh = trial_df["Fz"] < threshold
        rolling_below = below_thresh.rolling(window=n_consecutive).sum()

        onset_candidates = trial_df.index[rolling_below >= n_consecutive]

        if len(onset_candidates) == 0:
            st.error("Movement onset could not be detected: Fz never stayed below the movement threshold long enough.")
            st.stop()

        onset_idx = onset_candidates[0] - (n_consecutive - 1)
        onset_time = trial_df.loc[onset_idx, "time"]

        trial_post_onset = trial_df[trial_df["time"] >= onset_time].copy()
        takeoff_candidates = trial_post_onset[
            (trial_post_onset["Fz"] < 10) &
            (trial_post_onset["velocity"] > 0)
        ]

        if takeoff_candidates.empty:
            st.error("Take-off could not be detected: no point with Fz < 10 N and positive vertical velocity was found after onset.")
            st.stop()

        takeoff_idx = takeoff_candidates.index[0]
        takeoff_time = trial_df.loc[takeoff_idx, "time"]

        contact_time = takeoff_time - onset_time

        # ----------------------------------------------------
        # RAW ANALYSIS DATAFRAME = onset → take-off
        # ----------------------------------------------------
        analysis_raw = trial_df[
            (trial_df["time"] >= onset_time) &
            (trial_df["time"] <= takeoff_time)
        ].copy().reset_index(drop=True)

        if len(analysis_raw) < 10:
            st.error("The onset-to-take-off window is too short for reliable analysis.")
            st.stop()

        # ----------------------------------------------------
        # PHASE DETECTION
        # ----------------------------------------------------
        vel_arr = analysis_raw["velocity"].values
        time_arr = analysis_raw["time"].values

        min_idx = analysis_raw["velocity"].idxmin()
        min_time = analysis_raw.loc[min_idx, "time"]

        prop_start = None
        prop_start_idx = None

        for i in range(1, len(vel_arr)):
            if vel_arr[i - 1] < 0 and vel_arr[i] >= 0:
                prop_start = time_arr[i]
                prop_start_idx = i
                break

        if prop_start is None:
            st.error("Propulsion start could not be detected: no negative-to-positive velocity zero crossing was found.")
            st.stop()

        # Phase timings
        T_unw = min_time - onset_time
        T_break = prop_start - min_time
        T_prop = takeoff_time - prop_start

        breaking_df = analysis_raw[
            (analysis_raw["time"] > min_time) &
            (analysis_raw["time"] <= prop_start)
        ].copy()

        prop_df = analysis_raw[
            analysis_raw["time"] > prop_start
        ].copy()

        if len(breaking_df) == 0 or len(prop_df) == 0:
            st.error("Phase segmentation failed: braking or propulsive phase contains no samples.")
            st.stop()

        # ----------------------------------------------------
        # DISCRETE OUTCOMES
        # ----------------------------------------------------
        min_disp = float(analysis_raw["disp"].min())  # m

        Fz_break_imp = float(breaking_df["net_Fz"].mean() * T_break)  # Ns
        Fz_prop_imp = float(prop_df["net_Fz"].mean() * T_prop)        # Ns

        Fz_avg_break = float(breaking_df["power"].mean())             # W
        Fz_avg_prop = float(prop_df["power"].mean())                  # W

        Fy_imp = float(np.trapezoid(analysis_raw["Fy"], analysis_raw["time"]))  # Ns
        Fy_power = float(analysis_raw["power_Fy"].mean())                       # W

        avg_ratio = float(
            analysis_raw["ratio"].replace([np.inf, -np.inf], np.nan).mean()
        )

        avg_vec = float(prop_df["resultant"].mean())                  # N
        peak_vec = float(prop_df["resultant"].max())                  # N
        peak_vec_time = float(
            prop_df.loc[prop_df["resultant"].idxmax(), "time"] - onset_time
        )

        avg_angle = float(prop_df["angle"].mean())                    # deg
        peak_angle = float(prop_df["angle"].max())                    # deg
        peak_angle_time = float(
            prop_df.loc[prop_df["angle"].idxmax(), "time"] - onset_time
        )

        time_diff = float(peak_angle_time - peak_vec_time)

        takeoff_vel = float(trial_df.loc[takeoff_idx, "velocity"])    # m/s
        takeoff_vel_Fy = float(trial_df.loc[takeoff_idx, "vel_Fy"])   # m/s

        time_in_air = (2 * takeoff_vel) / 9.81                        # s
        jump_dist = abs(takeoff_vel_Fy * time_in_air * 100)           # cm

        # ----------------------------------------------------
        # COP — STRICTLY onset → take-off
        # ----------------------------------------------------
        analysis_raw["COP_AP"] = np.where(
            analysis_raw["Fz"] > 20,
            -analysis_raw["My"] / analysis_raw["Fz"],
            np.nan
        )

        analysis_raw["COP_ML"] = np.where(
            analysis_raw["Fz"] > 20,
            analysis_raw["Mx"] / analysis_raw["Fz"],
            np.nan
        )

        analysis_raw["COP_path_inst"] = np.nan
        analysis_raw["COP_vel_inst"] = np.nan

        cop_valid = analysis_raw.dropna(subset=["COP_AP", "COP_ML"]).copy()

        if len(cop_valid) >= 2:
            dx = np.diff(cop_valid["COP_AP"])
            dy = np.diff(cop_valid["COP_ML"])

            cop_step = np.sqrt(dx**2 + dy**2)
            cop_step = np.insert(cop_step, 0, 0.0)

            cop_path = float(np.nansum(cop_step) * 100)               # cm
            cop_velocity = float(cop_path / contact_time)             # cm/s

            cop_x_disp = float((cop_valid["COP_AP"].max() - cop_valid["COP_AP"].min()) * 100)
            cop_y_disp = float((cop_valid["COP_ML"].max() - cop_valid["COP_ML"].min()) * 100)

            cop_valid["COP_path_inst"] = np.cumsum(cop_step) * 100
            cop_valid["COP_vel_inst"] = (cop_step / dt) * 100

            analysis_raw = analysis_raw.drop(columns=["COP_path_inst", "COP_vel_inst"]).merge(
                cop_valid[["time", "COP_path_inst", "COP_vel_inst"]],
                on="time",
                how="left"
            )

        else:
            cop_path = np.nan
            cop_velocity = np.nan
            cop_x_disp = np.nan
            cop_y_disp = np.nan

        # ----------------------------------------------------
        # STORE EVENT Y VALUES BEFORE NORMALISATION
        # ----------------------------------------------------
        onset_Fz = float(trial_df.loc[onset_idx, "Fz"])
        min_Fz = float(analysis_raw.loc[min_idx, "Fz"])

        prop_row = int((analysis_raw["time"] - prop_start).abs().idxmin())
        prop_Fz = float(analysis_raw.loc[prop_row, "Fz"])

        takeoff_Fz = float(trial_df.loc[takeoff_idx, "Fz"])

        # ----------------------------------------------------
        # FIGURES
        # ----------------------------------------------------
        force_fig = go.Figure()

        force_fig.add_trace(go.Scatter(
            x=trial_df["time"], y=trial_df["Fz"],
            mode="lines",
            name="Fz (N)",
            yaxis="y1"
        ))

        force_fig.add_trace(go.Scatter(
            x=trial_df["time"], y=trial_df["Fy"],
            mode="lines",
            name="Fy (N)",
            yaxis="y2"
        ))

        force_fig.add_trace(go.Scatter(
            x=[onset_time], y=[onset_Fz],
            mode="markers",
            name="Onset",
            marker=dict(size=10, color="blue")
        ))

        force_fig.add_trace(go.Scatter(
            x=[min_time], y=[min_Fz],
            mode="markers",
            name="Start Breaking",
            marker=dict(size=10, color="orange")
        ))

        force_fig.add_trace(go.Scatter(
            x=[prop_start], y=[prop_Fz],
            mode="markers",
            name="Start Propulsion",
            marker=dict(size=10, color="green")
        ))

        force_fig.add_trace(go.Scatter(
            x=[takeoff_time], y=[takeoff_Fz],
            mode="markers",
            name="Take-off",
            marker=dict(size=10, color="red")
        ))

        force_fig.add_hline(y=threshold, line_dash="dash", line_color="purple")

        force_fig_title = "Force-Time Curve with Events"
        if invert_fy:
            force_fig_title += " — Fy Inverted"

        force_fig.update_layout(
            title=force_fig_title,
            xaxis_title="Time (s)",
            yaxis=dict(title="Fz (N)"),
            yaxis2=dict(title="Fy (N)", overlaying="y", side="right"),
            legend_title="Series"
        )

        vel_fig = go.Figure()

        vel_fig.add_trace(go.Scatter(
            x=analysis_raw["time"], y=analysis_raw["velocity"],
            mode="lines",
            name="Vertical Velocity (m/s)",
            yaxis="y1"
        ))

        vel_fig.add_trace(go.Scatter(
            x=analysis_raw["time"], y=analysis_raw["vel_Fy"],
            mode="lines",
            name="Horizontal Velocity (m/s)",
            yaxis="y2"
        ))

        vel_fig.update_layout(
            title="Velocity-Time Curve",
            xaxis_title="Time (s)",
            yaxis=dict(title="Vertical Velocity (m/s)"),
            yaxis2=dict(title="Horizontal Velocity (m/s)", overlaying="y", side="right")
        )

        power_fig = go.Figure()

        power_fig.add_trace(go.Scatter(
            x=analysis_raw["time"], y=analysis_raw["power"],
            mode="lines",
            name="Vertical Power (W)",
            yaxis="y1"
        ))

        power_fig.add_trace(go.Scatter(
            x=analysis_raw["time"], y=analysis_raw["power_Fy"],
            mode="lines",
            name="Horizontal Power (W)",
            yaxis="y2"
        ))

        power_fig.update_layout(
            title="Power-Time Curve",
            xaxis_title="Time (s)",
            yaxis=dict(title="Vertical Power (W)"),
            yaxis2=dict(title="Horizontal Power (W)", overlaying="y", side="right")
        )

        disp_fig = go.Figure()

        disp_fig.add_trace(go.Scatter(
            x=analysis_raw["time"], y=analysis_raw["disp"],
            mode="lines",
            name="Displacement (m)"
        ))

        disp_fig.update_layout(
            title="Displacement-Time Curve",
            xaxis_title="Time (s)",
            yaxis_title="Displacement (m)"
        )

        resultant_fig = go.Figure()

        resultant_fig.add_trace(go.Scatter(
            x=analysis_raw["time"], y=analysis_raw["resultant"],
            mode="lines",
            name="Resultant Force (N)"
        ))

        resultant_fig.update_layout(
            title="Resultant Force",
            xaxis_title="Time (s)",
            yaxis_title="Resultant Force (N)"
        )

        angle_fig = go.Figure()

        angle_fig.add_trace(go.Scatter(
            x=analysis_raw["time"], y=analysis_raw["angle"],
            mode="lines",
            name="Vector Angle (deg)"
        ))

        angle_fig.update_layout(
            title="Vector Angle",
            xaxis_title="Time (s)",
            yaxis_title="Angle (deg)"
        )

        cop_fig = go.Figure()

        cop_fig.add_trace(go.Scatter(
            x=analysis_raw["time"], y=analysis_raw["COP_AP"] * 100,
            mode="lines",
            name="COP AP (cm)",
            yaxis="y1"
        ))

        cop_fig.add_trace(go.Scatter(
            x=analysis_raw["time"], y=analysis_raw["COP_ML"] * 100,
            mode="lines",
            name="COP ML (cm)",
            yaxis="y2"
        ))

        cop_fig.update_layout(
            title="COP Components",
            xaxis_title="Time (s)",
            yaxis=dict(title="COP AP (cm)"),
            yaxis2=dict(title="COP ML (cm)", overlaying="y", side="right")
        )

        # ----------------------------------------------------
        # TABS
        # ----------------------------------------------------
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            ["Force", "Velocity", "Power", "Displacement", "Resultant", "Angle", "COP"]
        )

        with tab1:
            st.plotly_chart(force_fig, use_container_width=True)

        with tab2:
            st.plotly_chart(vel_fig, use_container_width=True)

        with tab3:
            st.plotly_chart(power_fig, use_container_width=True)

        with tab4:
            st.plotly_chart(disp_fig, use_container_width=True)

        with tab5:
            st.plotly_chart(resultant_fig, use_container_width=True)

        with tab6:
            st.plotly_chart(angle_fig, use_container_width=True)

        with tab7:
            st.plotly_chart(cop_fig, use_container_width=True)

        # ----------------------------------------------------
        # SUMMARY TABLE
        # ----------------------------------------------------
        summary = pd.DataFrame({
            "Metric": [
                "Body Mass",
                "Body Weight",
                "Movement Threshold",
                "Contact Time",
                "Jump Distance",
                "Minimum Displacement",
                "Unweighting Time",
                "Breaking Time",
                "Propulsive Time",
                "Fz Breaking Impulse",
                "Fz Propulsive Impulse",
                "Fz Average Breaking Power",
                "Fz Average Propulsive Power",
                "Fy Net Impulse",
                "Fy Average Power",
                "Average Force Ratio",
                "Average Propulsive Resultant Force",
                "Peak Propulsive Resultant Force",
                "Time of Peak Propulsive Resultant Force",
                "Average Propulsive Force Angle",
                "Peak Propulsive Force Angle",
                "Time of Peak Propulsive Force Angle",
                "Peak Angle–Magnitude Time Difference",
                "COP Path Length",
                "COP Velocity",
                "COP AP Displacement",
                "COP ML Displacement"
            ],
            "Value": [
                mass,
                BW,
                threshold,
                contact_time,
                jump_dist,
                min_disp,
                T_unw,
                T_break,
                T_prop,
                Fz_break_imp,
                Fz_prop_imp,
                Fz_avg_break,
                Fz_avg_prop,
                Fy_imp,
                Fy_power,
                avg_ratio,
                avg_vec,
                peak_vec,
                peak_vec_time,
                avg_angle,
                peak_angle,
                peak_angle_time,
                time_diff,
                cop_path,
                cop_velocity,
                cop_x_disp,
                cop_y_disp
            ],
            "Units": [
                "kg",
                "N",
                "N",
                "s",
                "cm",
                "m",
                "s",
                "s",
                "s",
                "Ns",
                "Ns",
                "W",
                "W",
                "Ns",
                "W",
                "-",
                "N",
                "N",
                "s from onset",
                "deg",
                "deg",
                "s from onset",
                "s",
                "cm",
                "cm/s",
                "cm",
                "cm"
            ],
            "Description": [
                "Estimated body mass from quiet standing force",
                "Mean quiet standing vertical force",
                "Movement threshold defined as BW - (% of SD)",
                "Duration from movement onset to take-off",
                "Predicted horizontal jump distance from take-off velocities",
                "Minimum centre-of-mass displacement during the analysed movement",
                "From movement onset to minimum vertical velocity",
                "From minimum vertical velocity to first vertical velocity zero-crossing",
                "From first vertical velocity zero-crossing to take-off",
                "Mean net vertical force × braking time",
                "Mean net vertical force × propulsive time",
                "Mean vertical power during braking phase",
                "Mean vertical power during propulsive phase",
                "Integrated horizontal force from onset to take-off",
                "Mean horizontal power from onset to take-off",
                "Mean ratio of resultant force to net vertical force from onset to take-off",
                "Mean resultant force during propulsive phase",
                "Peak resultant force during propulsive phase",
                "Time of peak resultant force, normalised to movement onset",
                "Mean force vector angle during propulsive phase",
                "Peak force vector angle during propulsive phase",
                "Time of peak force angle, normalised to movement onset",
                "Peak angle timing minus peak magnitude timing",
                "Total COP path length from onset to take-off",
                "Average COP velocity from onset to take-off",
                "Range of COP in the anterior-posterior direction",
                "Range of COP in the medio-lateral direction"
            ]
        })

        st.subheader("Summary Outcomes")

        if invert_fy:
            st.info("Fy-dependent outcomes were calculated using inverted Fy data.")

        st.dataframe(summary, use_container_width=True)

        # ----------------------------------------------------
        # NORMALISED EXPORT — 101-point waveform only
        # ----------------------------------------------------
        export_cols = [
            "Fz", "Fy",
            "net_Fz",
            "accel", "velocity", "disp", "power",
            "accel_Fy", "vel_Fy", "power_Fy",
            "resultant", "angle", "ratio",
            "COP_AP", "COP_ML", "COP_path_inst", "COP_vel_inst"
        ]

        export_source = analysis_raw[export_cols].copy()

        old_x = np.linspace(0, 100, len(export_source))
        new_x = np.linspace(0, 100, 101)

        export_df = pd.DataFrame({"Normalized_Percent": new_x})

        for col in export_cols:
            y = export_source[col].astype(float).values
            y = pd.Series(y).interpolate(limit_direction="both").values
            export_df[col] = np.interp(new_x, old_x, y)

        # ----------------------------------------------------
        # DOWNLOAD BUTTONS
        # ----------------------------------------------------
        st.download_button(
            "Download Summary (CSV)",
            summary.to_csv(index=False),
            "summary.csv"
        )

        st.download_button(
            "Download Curves (101-point normalized CSV)",
            export_df.to_csv(index=False),
            "curves_101pts.csv"
        )
