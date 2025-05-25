from typing import Literal

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from colorspacious import cspace_convert
from matplotlib.axes import Axes
from numpy.typing import NDArray

AVAILABLE_COLORMAPS = plt.colormaps()

CVD_TYPES = {
    "None": None,
    "Protanopia (Type 1 color blindness)": "protanomaly",
    "Deuteranopia (Type 2 color blindness)": "deuteranomaly",
    "Tritanopia (Type 3 color blindness)": "tritanomaly",
    "Monochrome": "mono",
}


def simulate_cvd(
    colors: NDArray,
    cvd_type: Literal["protanomaly", "deuteranomaly", "tritanomaly", "mono"] | None,
) -> NDArray:
    if cvd_type is None:
        return colors

    colors = np.clip(colors / 255.0, 0, 1)

    if cvd_type == "mono":
        gray = (
            0.2126 * colors[..., 0] + 0.7152 * colors[..., 1] + 0.0722 * colors[..., 2]
        )
        colors = np.stack([gray] * 3, axis=-1)
    else:
        try:
            colors = cspace_convert(
                colors[:, :3],
                {"name": "sRGB1+CVD", "cvd_type": cvd_type, "severity": 100},
                "sRGB1",
            )
        except Exception as e:
            st.warning(
                f"Error during CVD simulation ({cvd_type}): {e}. Original colors will be used."
            )

    colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
    return colors


def draw_colormap(
    ax: Axes,
    cmap_name: str,
    num_steps: int = 256,
    is_discrete: bool = False,
    cvd_type: str | None = None,
):
    try:
        cmap = mpl.colormaps[cmap_name]
    except ValueError:
        ax.text(
            0.5,
            0.5,
            f"'{cmap_name}'\nnot found",
            ha="center",
            va="center",
            color="red",
            fontsize=8,
        )
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(cmap_name, fontsize=9, pad=3)
        return

    rgb = cmap(np.linspace(0, 1, 256))
    rgb = (rgb * 255).astype(np.uint8)
    simulated = simulate_cvd(rgb, cvd_type)

    if is_discrete:
        indices = np.linspace(0, 255, num_steps, dtype=int)

        for i, idx in enumerate(indices):
            color = mcolors.to_rgba(simulated[idx] / 255.0)
            ax.add_patch(
                plt.Rectangle((i / num_steps, 0), 1 / num_steps, 1, facecolor=color)
            )
    else:
        ax.imshow(simulated[np.newaxis, :, :], aspect="auto")

    ax.set_title(cmap_name, fontsize=9, pad=3)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


st.set_page_config(layout="wide", page_title="Matplotlib Colormap Viewer")
st.title("🎨 Matplotlib Colormap Viewer")
st.markdown("""
Displays all available colormaps in Matplotlib.
Simulates different types of color vision deficiencies and allows switching between continuous and discrete display modes.
""")

st.sidebar.header("⚙️ Display Settings")
cvd_label = st.sidebar.selectbox(
    "👁️ Color Vision Deficiency Simulation:",
    options=list(CVD_TYPES.keys()),
    index=0,
    help="Simulates the selected type of color vision deficiency.",
)
cvd_type = CVD_TYPES[cvd_label]

display_mode = st.sidebar.radio(
    "📊 Display Mode:",
    ("Continuous Gradient", "Discrete Steps"),
    index=0,
    horizontal=True,
    help="Choose between smooth gradient or step-based color representation.",
)
is_discrete = display_mode == "Discrete Steps"

num_steps = (
    st.sidebar.slider(
        "🎨 Number of Discrete Steps:",
        min_value=2,
        max_value=100,
        value=20,
        step=1,
        help="Specify how many steps to use for discrete mode.",
    )
    if is_discrete
    else 256
)

search_term = st.sidebar.text_input(
    "🔍 Search Colormap:",
    placeholder="e.g. viridis, coolwarm, _r",
    help="Enter part of the colormap name to filter the list (case insensitive).",
)

cols_per_row = st.sidebar.select_slider(
    "📏 Number of Columns per Row:",
    options=[1, 2, 3, 4, 5, 6],
    value=3,
    help="Adjust the number of colormaps displayed per row.",
)

colormaps = (
    [c for c in AVAILABLE_COLORMAPS if search_term.lower() in c.lower()]
    if search_term
    else AVAILABLE_COLORMAPS
)

if not colormaps:
    st.warning(f"No colormaps found matching the search term '{search_term}'.")
else:
    st.info(
        f"Displaying {len(colormaps)} colormaps (Settings: {cvd_label}, {display_mode}{f', {num_steps} steps' if is_discrete else ''})"
    )

    for i in range(0, len(colormaps), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, cmap_name in enumerate(colormaps[i : i + cols_per_row]):
            with cols[j]:
                show = st.checkbox(
                    f"🎨 {cmap_name}", value=True, key=f"show_{cmap_name}"
                )
                if not show:
                    continue

                fig, ax = plt.subplots(figsize=(5, 0.6))
                try:
                    draw_colormap(ax, cmap_name, num_steps, is_discrete, cvd_type)
                except Exception as e:
                    st.error(f"Error displaying colormap '{cmap_name}': {e}")
                    ax.text(
                        0.5,
                        0.5,
                        "Display Error",
                        ha="center",
                        va="center",
                        color="red",
                        fontsize=8,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(cmap_name, fontsize=9, pad=3)

                st.pyplot(fig, clear_figure=True)
                cmap = plt.get_cmap(cmap_name)
                colors = (
                    [
                        mcolors.to_hex(cmap(i / (num_steps - 1)))
                        for i in range(num_steps)
                    ]
                    if is_discrete
                    else [mcolors.to_hex(cmap(i / 255)) for i in range(0, 256)]
                )
                with st.expander("📋 Show HEX Color Codes"):
                    st.code('["' + '", "'.join(colors) + '"]', language="text")
                plt.close(fig)


st.sidebar.markdown("---")
st.sidebar.markdown("### About this App")
st.sidebar.info("""
This viewer was created to help explore the wide variety of colormaps in Matplotlib visually and to support accessibility-focused color choices.

- **Color Vision Deficiency Simulation**: Powered by the `colorspacious` library.
- **Display Modes**: Switch between continuous gradient and discrete steps.
- **Search & Layout**: Quickly find the desired colormap and customize display layout.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 kage1020")
